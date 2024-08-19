mod ppo;
mod threading;
mod util;

pub use ppo::ppo_learner::PPOLearnerConfig;
pub use rlgym_rs;
pub use rlgym_rs::rocketsim_rs;
pub use tch;

use ppo::{exp_buf::ExperienceBuffer, ppo_learner::PPOLearner};
use rlgym_rs::{Action, Env, Obs, Reward, StateSetter, Terminal, Truncate};
use std::{num::NonZeroUsize, path::PathBuf, time::Instant};
use tch::Device;
use threading::agent_mngr::AgentManager;
use util::report::Report;

#[derive(Debug, Clone)]
pub struct LearnerConfig {
    pub num_threads: NonZeroUsize,
    pub num_games_per_thread: NonZeroUsize,
    pub render: bool,
    /// Set to 0 to disable
    pub timestep_limit: u64,
    pub exp_buffer_size: i64,
    pub standardize_returns: bool,
    // pub standardize_obs: bool,
    pub max_returns_per_stats_inc: u32,
    pub steps_per_obs_stats_inc: u32,
    /// Actions with the highest probability are always chosen, instead of being more likely
    /// This will make your bot play better, but is horrible for learning
    /// Trying to run a PPO learn iteration with deterministic mode will throw an exception
    pub deterministic: bool,
    /// Collect additional steps during the learning phase
    /// Note that, once the learning phase completes and the policy is updated, these additional steps are from the old policy
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
    pub device_type: Device,
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
            timestep_limit: 0,
            exp_buffer_size: 100 * 1000,
            standardize_returns: true,
            // standardize_obs: false,
            max_returns_per_stats_inc: 150,
            steps_per_obs_stats_inc: 5,
            deterministic: false,
            collection_during_learn: false,
            ppo: PPOLearnerConfig::default(),
            gae_lambda: 0.95,
            gae_gamma: 0.99,
            reward_clip_range: 10.0,
            checkpoint_load_folder: PathBuf::from("checkpoints"),
            checkpoint_save_folder: PathBuf::from("checkpoints"),
            save_folder_add_unix_timestamp: false,
            timesteps_per_save: 500 * 1000,
            random_seed: 123,
            checkpoints_to_keep: 5,
            device_type: Device::cuda_if_available(),
            // send_metrics: true,
            // metrics_project_name: "rlgymppo-rs".to_string(),
            // metrics_group_name: "unnamed-runs".to_string(),
            // metrics_run_name: "rlgymppo-rs-run".to_string(),
            // skill_tracker_config: SkillTrackerConfig::default(),
        }
    }
}

pub struct Learner<F, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    F: Fn() -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Send + Clone + 'static,
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    create_env_fn: F,
    config: LearnerConfig,
    obs_size: usize,
    action_size: usize,
    exp_buffer: ExperienceBuffer,
    ppo: PPOLearner,
    agent_mngr: AgentManager,
    total_timesteps: u64,
    total_epochs: u64,
}

impl<F, SS, OBS, ACT, REW, TERM, TRUNC, SI> Learner<F, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    F: Fn() -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Send + Clone + 'static,
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(create_env_fn: F, config: LearnerConfig) -> Self {
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
        let exp_buffer = ExperienceBuffer::new(
            config.exp_buffer_size,
            config.random_seed as u64,
            config.device_type,
        );

        println!("Creating learner...");
        let ppo = PPOLearner::new(
            obs_size,
            action_size,
            config.ppo.clone(),
            config.device_type,
        );

        println!("Creating agent manager...");
        let mut agent_mngr = AgentManager::new(
            ppo.get_policy(),
            config.ppo.batch_size + config.ppo.batch_size / 2,
            config.deterministic,
            config.collection_during_learn,
            config.device_type,
        );

        println!("Creating {} agents...", config.num_threads);
        agent_mngr.create_agents(
            create_env_fn.clone(),
            config.num_threads.get(),
            config.num_games_per_thread.get(),
        );

        Self {
            create_env_fn,
            config,
            obs_size,
            action_size,
            exp_buffer,
            ppo,
            agent_mngr,
            total_timesteps: 0,
            total_epochs: 0,
        }
    }

    pub fn learn(&mut self) {
        self.agent_mngr.start();

        // let device = self.config.device_type;
        // let deterministic = self.config.deterministic;

        let mut steps_since_save = 0;
        let mut current_time = Instant::now();

        let mut timestep_collection_time = Instant::now();
        while self.config.timestep_limit == 0 || self.total_timesteps < self.config.timestep_limit {
            let mut report = Report::default();

            let timesteps = self
                .agent_mngr
                .collect_timesteps(self.config.ppo.batch_size);

            let timestep_collection_elapsed = timestep_collection_time.elapsed();
            timestep_collection_time = Instant::now();

            let steps_per_second =
                timesteps.len() as f64 / timestep_collection_elapsed.as_secs_f64();
            println!(
                "Collected {} timesteps in {:.2} seconds ({steps_per_second:.0} overall sps)",
                timesteps.len(),
                timestep_collection_elapsed.as_secs_f64()
            );

            self.total_timesteps += timesteps.len() as u64;

            if self.config.ppo.policy_lr == 0. && self.config.ppo.critic_lr == 0. {
                println!("Learning rate is 0, skipping learning phase");
                continue;
            }

            self.agent_mngr.update_policy(self.ppo.get_policy());

            self.total_epochs += 1;
        }

        self.agent_mngr.stop();
    }
}
