#![allow(clippy::type_complexity)]

mod ppo;
mod threading;
mod util;

pub use ppo::ppo_learner::PPOLearnerConfig;
pub use rlgym;
pub use rlgym::rocketsim_rs;
pub use rocketsim_rs::glam_ext::glam;
pub use tch;
pub use util::avg_tracker::AvgTracker;
pub use util::report::Report;

use ppo::{
    exp_buf::{ExperienceBuffer, ExperienceTensors},
    ppo_learner::PPOLearner,
};
use rlgym::{
    rocketsim_rs::glam_ext::GameStateA, Action, Env, Obs, Reward, SharedInfoProvider, StateSetter,
    Terminal, Truncate,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Instant,
};
use tch::{no_grad_guard, Device, IndexOp, Kind, Tensor};
use threading::{agent_mngr::AgentManager, trajectory::Trajectory};
use util::{
    compute::{self, NonBlockingTransfer},
    running_stat::{WStatConfig, WelfordRunningStat},
};

#[derive(Debug, Clone)]
pub struct LearnerConfig {
    pub num_threads: NonZeroUsize,
    pub num_games_per_thread: NonZeroUsize,
    pub render: bool,
    /// If render is true, this will attempt to launch the RLViser
    /// If this is false, RLViser will not be launched
    pub try_launch_exe: bool,
    /// Set to 0 to disable
    pub timestep_limit: u64,
    pub exp_buffer_size: u64,
    pub standardize_returns: bool,
    // pub standardize_obs: bool,
    pub max_returns_per_stats_inc: u32,
    pub steps_per_obs_stats_inc: u32,
    /// Actions with the highest probability are always chosen, instead of being more likely
    /// This will make your bot play better, but is horrible for learning
    /// Trying to run a PPO learn iteration with deterministic mode will throw an exception
    pub deterministic: bool,
    /// The number of ***additional*** timesteps to may be collected during the learning phase
    ///
    /// Why? This helps when one thread is running slower than the others,
    /// with the other threads having an extra buffer size to continue collecting
    /// and make up for the slower thread
    ///
    /// If collection_during_learn is `true`,
    /// this should probably be set to near `0` as the algorithm will naturally overflow.
    /// A high value may cause runaway memory usage.
    ///
    /// If `false`, `batch_size / 2` is recommended
    pub collection_timesteps_overflow: u64,
    /// How many ticks to skip before updating checking for
    /// an updated policy/if the collection should pause
    ///
    /// Lower values mean more frequent checks with a
    /// final timestep closer to the requested number
    /// and ensuring the the latest policy is being ran,
    /// but more overhead from loading atomics
    ///
    /// Realistically, checking every tick is excessive!
    pub check_update_frequency: u64,
    /// Collect additional steps during the learning phase
    ///
    /// Note that, once the learning phase completes and the policy is updated, these additional steps are from the old policy
    ///
    /// WARNING: Do NOT enable this if training on CPU!
    pub collection_during_learn: bool,
    pub ppo: PPOLearnerConfig,
    pub gae_lambda: f32,
    pub gae_gamma: f32,
    /// Clip range for normalized rewards, set 0 to disable
    pub reward_clip_range: f32,
    /// Set to a directory with numbered subfolders, the learner will load the subfolder with the highest number
    /// If the folder is empty or does not exist, loading is skipped
    /// Set empty to disable loading entirely
    pub checkpoint_load_folder: PathBuf,
    /// Checkpoints are saved here as timestep-numbered subfolders
    /// (e.g. a checkpoint at 20,000 steps will save to a subfolder called "20000")
    ///
    /// Set empty to disable saving
    pub checkpoint_save_folder: PathBuf,
    /// Appends the unix time to checkpointSaveFolder
    pub save_folder_add_unix_timestamp: bool,
    /// Save every timestep
    /// Set to zero to just use timestepsPerIteration
    pub timesteps_per_save: u64,
    pub random_seed: i64,
    /// Checkpoint storage limit before old checkpoints are deleted, set to -1 to disable
    pub checkpoints_to_keep: u32,
    /// Auto will use your CUDA GPU if available
    pub device: Device,
    // pub send_metrics: bool,
    // pub metrics_project_name: String,
    // pub metrics_group_name: String,
    // pub metrics_run_name: String,
    // pub skill_tracker_config: SkillTrackerConfig,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            num_threads: NonZeroUsize::new(8).unwrap(),
            num_games_per_thread: NonZeroUsize::new(16).unwrap(),
            render: false,
            try_launch_exe: false,
            timestep_limit: 0,
            exp_buffer_size: 100_000,
            standardize_returns: true,
            // standardize_obs: false,
            max_returns_per_stats_inc: 150,
            steps_per_obs_stats_inc: 5,
            deterministic: false,
            check_update_frequency: 15,
            collection_timesteps_overflow: 25_000,
            collection_during_learn: false,
            ppo: PPOLearnerConfig::default(),
            gae_lambda: 0.95,
            gae_gamma: 0.99,
            reward_clip_range: 10.0,
            checkpoint_load_folder: PathBuf::from("checkpoints"),
            checkpoint_save_folder: PathBuf::from("checkpoints"),
            save_folder_add_unix_timestamp: false,
            timesteps_per_save: 5_000_000,
            random_seed: 123,
            checkpoints_to_keep: 5,
            device: Device::cuda_if_available(),
            // send_metrics: true,
            // metrics_project_name: "rlgymppo-rs".to_string(),
            // metrics_group_name: "unnamed-runs".to_string(),
            // metrics_run_name: "rlgymppo-rs-run".to_string(),
            // skill_tracker_config: SkillTrackerConfig::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Stats {
    cumulative_timesteps: u64,
    cumulative_model_updates: u64,
    epoch: u64,
    return_stat: WStatConfig,
    // todo, for wandb metrics reporting
    // run_id: String,
}

pub struct Learner {
    config: LearnerConfig,
    // obs_size: usize,
    // action_size: usize,
    exp_buffer: ExperienceBuffer,
    ppo: PPOLearner,
    agent_mngr: AgentManager,
    return_stat: WelfordRunningStat,
    total_timesteps: u64,
    total_epochs: u64,
}

impl Learner {
    pub fn new<F, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
        create_env_fn: F,
        step_callback: C,
        config: LearnerConfig,
    ) -> Self
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI> + Send + Clone + 'static,
        C: Fn(&mut Report, &SI, &GameStateA) + Clone + Send + 'static,
        SS: StateSetter<SI>,
        SIP: SharedInfoProvider<SI>,
        OBS: Obs<SI>,
        ACT: Action<SI, Input = Vec<i32>>,
        REW: Reward<SI>,
        TERM: Terminal<SI>,
        TRUNC: Truncate<SI>,
    {
        tch::manual_seed(config.random_seed);

        tch::set_num_interop_threads(1);
        tch::set_num_threads(1);

        let (obs_size, action_size) = {
            println!("Creating test environment to determine OBS size and action amount...");
            let env = create_env_fn();
            let obs_size = env.get_obs_space(0);
            let action_size = env.get_action_space(0);

            println!("\tOBS size: {obs_size}");
            println!("\tAction amount: {action_size}");

            (obs_size, action_size)
        };

        println!("Creating experience buffer...");
        let exp_buffer = ExperienceBuffer::new(config.exp_buffer_size, config.random_seed as u64);

        println!("Creating learner...");
        let ppo = PPOLearner::new(obs_size, action_size, config.ppo.clone(), config.device);

        println!("Creating agent manager...");
        let mut agent_mngr = AgentManager::new(
            ppo.get_policy(),
            config.ppo.batch_size + config.collection_timesteps_overflow,
            config.deterministic,
            config.collection_during_learn,
            config.check_update_frequency,
            config.device,
        );

        println!("Creating {} agents...", config.num_threads);
        agent_mngr.create_agents(
            create_env_fn,
            step_callback.clone(),
            config.num_threads.get(),
            config.num_games_per_thread.get(),
            config.render,
            config.try_launch_exe,
        );

        Self {
            config,
            // obs_size,
            // action_size,
            exp_buffer,
            ppo,
            agent_mngr,
            return_stat: WelfordRunningStat::default(),
            total_timesteps: 0,
            total_epochs: 0,
        }
    }

    pub fn learn(&mut self) {
        println!("Starting...");
        self.agent_mngr.start();

        let mut steps_since_save = 0;

        let mut overall_time = Instant::now();
        while self.config.timestep_limit == 0 || self.total_timesteps < self.config.timestep_limit {
            let mut report = Report::default();

            let collection_time = Instant::now();
            let timesteps = self
                .agent_mngr
                .collect_timesteps(self.config.ppo.batch_size);

            let timesteps_collected = timesteps.len();
            let collection_time_elapsed = collection_time.elapsed();
            let overall_time_elapsed = overall_time.elapsed();
            overall_time = Instant::now();

            self.total_timesteps += self.config.ppo.batch_size;

            if self.config.deterministic {
                println!("Deterministic mode is enabled, skipping learning phase");
                continue;
            }

            if self.config.ppo.policy_lr == 0. && self.config.ppo.critic_lr == 0. {
                println!("Learning rate is 0, skipping learning phase");
                continue;
            }

            self.add_new_experience(timesteps, &mut report);

            self.agent_mngr.get_metrics(&mut report);

            report["Timesteps Collected"] = timesteps_collected.into();
            report["Overall Steps per Second"] =
                (self.config.ppo.batch_size as f64 / overall_time_elapsed.as_secs_f64()).into();
            report["Collected Steps per Second"] =
                (self.config.ppo.batch_size as f64 / collection_time_elapsed.as_secs_f64()).into();
            report["Cumulative Timesteps"] = self.total_timesteps.into();

            println!("{report}");

            steps_since_save += self.config.ppo.batch_size;
            if steps_since_save >= self.config.timesteps_per_save {
                self.save();
                steps_since_save = 0;
            }
        }

        println!("Timestep limit reached, stopping...");
        self.agent_mngr.stop();
    }

    fn load_stats<P: AsRef<Path>>(&mut self, path: P) {
        let path = path.as_ref().join("stats.toml");

        let contents = fs::read_to_string(&path).unwrap();
        let stats: Stats = toml::from_str(&contents).unwrap();

        self.total_timesteps = stats.cumulative_timesteps;
        self.ppo.cumulative_model_updates = stats.cumulative_model_updates;
        self.total_epochs = stats.epoch;
        self.return_stat = stats.return_stat.into();
    }

    fn save_stats<P: AsRef<Path>>(&self, path: P) {
        let path = path.as_ref().join("stats.toml");

        let stats = Stats {
            cumulative_timesteps: self.total_timesteps,
            cumulative_model_updates: self.ppo.cumulative_model_updates,
            epoch: self.total_epochs,
            return_stat: self.return_stat.config().clone(),
        };

        fs::write(&path, toml::to_string(&stats).unwrap()).unwrap();
    }

    pub fn load(&mut self) {
        println!("Loading model checkpoint...");
        let folder = self.config.checkpoint_load_folder.clone();

        if !folder.exists() {
            println!("Checkpoint folder does not exist, skipping load");
            return;
        }

        let mut checkpoint_name = None;
        let mut max_timesteps = 0;
        for entry in fs::read_dir(folder).unwrap().flatten() {
            let path = entry.path();
            if path.is_dir() {
                let dirname = path.file_name().unwrap().to_str();
                let timesteps = dirname.unwrap().parse::<u64>().unwrap();
                if timesteps > max_timesteps {
                    checkpoint_name = Some(path);
                    max_timesteps = timesteps;
                }
            }
        }

        let Some(checkpoint_name) = checkpoint_name else {
            println!("No checkpoint found in folder, skipping load");
            return;
        };

        self.load_stats(&checkpoint_name);
        self.ppo.load(checkpoint_name);
    }

    pub fn save(&self) {
        println!("Saving model checkpoint...");
        let mut folder = self.config.checkpoint_save_folder.clone();
        folder.push(self.total_timesteps.to_string());

        fs::create_dir_all(&folder).unwrap();

        self.save_stats(&folder);
        self.ppo.save(folder);
    }

    fn add_new_experience(&mut self, timesteps: Trajectory, report: &mut Report) {
        let _no_grad = no_grad_guard();

        let data = timesteps.into_inner();
        let count = data.states.size()[0] as usize;

        // Construct input to the value function estimator that includes the final state (which an action was not taken in)
        let val_input = Tensor::cat(
            &[
                data.states.shallow_clone(),
                data.next_states.i(count as i64 - 1).unsqueeze(0),
            ],
            0,
        )
        .no_block_to(self.config.device);

        let val_preds_tensor = self
            .ppo
            .get_value_net()
            .forward(&val_input, self.config.deterministic)
            .to(Device::Cpu)
            .flatten(0, -1);
        let val_preds = tensor_to_f32_vec(&val_preds_tensor);

        let ret_std = if self.config.standardize_returns {
            self.return_stat.get_std()[0]
        } else {
            1.0
        };

        let (advantages, value_targets, returns) = compute::gae(
            tensor_to_f32_vec(&data.rewards.view([-1])),
            tensor_to_f32_vec(&data.dones.view([-1])),
            tensor_to_f32_vec(&data.truncateds.view([-1])),
            val_preds,
            self.config.gae_gamma,
            self.config.gae_lambda,
            ret_std,
            self.config.reward_clip_range,
        );

        let avg_ret = returns.iter().copied().map(f32::abs).sum::<f32>() / returns.len() as f32;
        report["Avg Return"] = (avg_ret / ret_std).into();

        report["Avg Advantage"] = advantages.abs().mean(Kind::Float).double_value(&[]).into();
        report["Avg Val Target"] = value_targets
            .abs()
            .mean(Kind::Float)
            .double_value(&[])
            .into();

        if self.config.standardize_returns {
            let num_to_increment = returns
                .len()
                .min(self.config.max_returns_per_stats_inc as usize);
            self.return_stat.increment(&returns, num_to_increment);
        }

        let exp_tensors = ExperienceTensors {
            states: data.states,
            actions: data.actions,
            log_probs: data.log_probs,
            rewards: data.rewards,
            next_states: data.next_states,
            dones: data.dones,
            truncateds: data.truncateds,
            values: value_targets,
            advantages,
        };

        self.exp_buffer.submit_experience(exp_tensors);

        self.ppo.learn(&mut self.exp_buffer, report);
        self.agent_mngr.update_policy(self.ppo.get_policy());
        self.total_epochs += u64::from(self.config.ppo.epochs);
    }
}

fn tensor_to_f32_vec(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dim(), 1);
    let tensor = tensor.to(Device::Cpu).detach().to_kind(Kind::Float);
    Vec::<f32>::try_from(tensor).unwrap()
}
