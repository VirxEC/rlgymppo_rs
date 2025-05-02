#![allow(clippy::type_complexity)]
mod agent;
mod base;
mod environment;

pub mod utils;

pub use burn::backend;
pub use agent::config::PpoLearnerConfig;
pub use rlgym::{self, glam, rocketsim_rs};

use agent::{Ppo, model::Actic};
use burn::{
    module::{AutodiffModule, Module},
    tensor::backend::AutodiffBackend,
};
use environment::batch_sim::{BatchSim, BatchSimConfig};
use rand::{SeedableRng, rngs::SmallRng};
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::{
    io::{Read, stdin},
    path::PathBuf,
    sync::mpsc::{Sender, channel},
    thread,
    time::Instant,
};
use utils::{
    Report,
    running_stat::Stats,
    serde::{load_latest_model, save_model},
};

enum HumanInput {
    Save,
    Quit,
}

fn stdin_reader(s: Sender<HumanInput>) {
    let mut buffer = [0; 1];
    while stdin().read_exact(&mut buffer).is_ok() {
        match char::from(buffer[0]).to_ascii_lowercase() {
            'q' => {
                println!("Finishing iteration, saving, then exiting...");
                s.send(HumanInput::Quit).unwrap();
                return;
            }
            's' => {
                println!("Saving model after this iteration...");
                s.send(HumanInput::Save).unwrap();
            }
            _ => {}
        }
    }
}

pub struct LearnerConfig<B: AutodiffBackend> {
    /// Hyperparameters for the PPO learner.
    pub ppo: PpoLearnerConfig,
    /// Where to load/save checkpoints.
    /// If None, defaults to "checkpoints".
    /// If the path does not exist, it will be created.
    pub checkpoints_folder: PathBuf,
    /// The device to use for training.
    /// Will default to the default device from the given backend.
    pub device: B::Device,
    /// The layer sizes for the policy network.
    pub policy_layer_sizes: Vec<usize>,
    /// The layer sizes for the critic network.
    pub critic_layer_sizes: Vec<usize>,
    /// The maximum number of checkpoints to keep.
    /// If None, all checkpoints will be kept.
    pub checkpoints_limit: Option<usize>,
    /// The number of timesteps to run before saving a checkpoint.
    pub timesteps_per_save: u64,
    /// The number of games to run per thread.
    /// Increasing this will increase GPU utilization
    /// and the utilization of 1 cpu thread.
    pub num_games_per_thread: usize,
    /// The size of the experience replay buffer.
    pub exp_buffer_size: usize,
}

impl<B: AutodiffBackend> Default for LearnerConfig<B> {
    fn default() -> Self {
        Self {
            ppo: PpoLearnerConfig::default(),
            checkpoints_folder: PathBuf::from("checkpoints"),
            device: B::Device::default(),
            policy_layer_sizes: vec![256; 2],
            critic_layer_sizes: vec![256; 2],
            checkpoints_limit: None,
            timesteps_per_save: 1_000_000,
            num_games_per_thread: 4,
            exp_buffer_size: 60_000,
        }
    }
}

impl<B: AutodiffBackend> LearnerConfig<B> {
    pub fn init<F, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
        self,
        create_env: F,
        step_callback: C,
    ) -> Learner<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
        C: Fn(&mut Report, &mut SI, &GameStateA) + Clone,
        SS: StateSetter<SI>,
        SIP: SharedInfoProvider<SI>,
        OBS: Obs<SI>,
        ACT: Action<SI, Input = usize>,
        REW: Reward<SI>,
        TERM: Terminal<SI>,
        TRUNC: Truncate<SI>,
    {
        assert_ne!(
            self.policy_layer_sizes.len(),
            0,
            "policy_layer_sizes must not be empty"
        );
        assert_ne!(
            self.critic_layer_sizes.len(),
            0,
            "critic_layer_sizes must not be empty"
        );
        assert_ne!(
            self.timesteps_per_save, 0,
            "timesteps_per_save must not be 0"
        );
        assert!(
            self.exp_buffer_size >= self.ppo.batch_size,
            "exp_buffer_size must be greater than or equal to ppo.batch_size"
        );

        let env = (create_env)();
        let obs_space = env.get_obs_space();
        let action_space = env.get_action_space();

        let model = Actic::<B>::new(
            obs_space,
            action_space,
            self.policy_layer_sizes,
            self.critic_layer_sizes,
            &self.device,
        );

        println!("# parameters in actor: {}", model.actor.num_params());
        println!("# parameters in critic: {}", model.critic.num_params());

        let batch_sim = BatchSim::new(
            create_env,
            step_callback,
            BatchSimConfig {
                num_games: self.num_games_per_thread,
                buffer_size: self.exp_buffer_size,
            },
            self.device.clone(),
        );

        Learner {
            ppo: self.ppo.init(self.device.clone()),
            rng: SmallRng::from_os_rng(),
            metrics: Report::default(),
            stats: Stats::default(),
            device: self.device,
            model: Some(model),
            checkpoints_folder: self.checkpoints_folder,
            checkpoints_limit: self.checkpoints_limit,
            timesteps_per_save: self.timesteps_per_save,
            last_save_timestep: 0,
            batch_sim,
        }
    }
}

pub struct Learner<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: AutodiffBackend,
    C: Fn(&mut Report, &mut SI, &GameStateA) + Clone,
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    ppo: Ppo<B>,
    rng: SmallRng,
    metrics: Report,
    stats: Stats,
    device: B::Device,
    model: Option<Actic<B>>,
    checkpoints_folder: PathBuf,
    checkpoints_limit: Option<usize>,
    timesteps_per_save: u64,
    last_save_timestep: u64,
    batch_sim: BatchSim<B::InnerBackend, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
}

impl<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    Learner<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: AutodiffBackend,
    C: Fn(&mut Report, &mut SI, &GameStateA) + Clone,
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    /// Load the previously saved model.
    /// Does nothing if the model can't be loaded, this is safe to call unconditionally.
    pub fn load(&mut self) {
        let mut model = self
            .model
            .take()
            .expect("load() can't be called after train()");

        (model, self.stats) = load_latest_model(model, &self.checkpoints_folder, &self.device);
        self.model = Some(model);
    }

    /// Train the model, and automatically saves it before exiting.
    pub fn learn(&mut self) {
        let mut model = self.model.take().expect("train() can only be called once");

        let (s, r) = channel();

        thread::spawn(move || {
            stdin_reader(s);
        });

        println!("Running for the first time. This might be slow at first...");
        println!("Press Q to quit, and S to save then continue (must confirm by pressing enter)\n");
        'train: loop {
            let collect_start = Instant::now();
            let memory = self.batch_sim.run(model.valid());
            let collect_elapsed = collect_start.elapsed().as_secs_f64();

            self.stats.cumulative_timesteps += memory.len() as u64;
            let num_steps = memory.len() as f64;

            let train_start = Instant::now();
            model = self.ppo.train(
                model,
                memory,
                &mut self.rng,
                &mut self.metrics,
                &mut self.stats,
            );
            memory.clear();

            let train_elapsed = train_start.elapsed().as_secs_f64();
            let overall_elapsed = collect_start.elapsed().as_secs_f64();

            self.metrics.clear();
            self.metrics += self.batch_sim.get_metrics();
            self.metrics[".Episode Length"] = num_steps.into();
            self.metrics[".Collection time"] = collect_elapsed.into();
            self.metrics[".Collected SPS"] = (num_steps / collect_elapsed).into();
            self.metrics[".Training time"] = train_elapsed.into();
            self.metrics[".Overall time"] = overall_elapsed.into();
            self.metrics[".Overall SPS"] =
                (num_steps / collect_start.elapsed().as_secs_f64()).into();
            self.metrics[".Cumulative steps"] = self.stats.cumulative_timesteps.into();
            self.metrics[".Cumulative epochs"] = self.stats.cumulative_epochs.into();
            self.metrics[".Cumulative updates"] = self.stats.cumulative_model_updates.into();

            println!("{}", self.metrics);
            self.metrics.clear();

            for input in r.try_iter() {
                match input {
                    HumanInput::Quit => {
                        break 'train;
                    }
                    HumanInput::Save => {
                        save_model(
                            model.valid(),
                            &self.stats,
                            &self.checkpoints_folder,
                            self.checkpoints_limit,
                        );
                    }
                }
            }

            if self.stats.cumulative_timesteps - self.last_save_timestep > self.timesteps_per_save {
                save_model(
                    model.valid(),
                    &self.stats,
                    &self.checkpoints_folder,
                    self.checkpoints_limit,
                );
                self.last_save_timestep = self.stats.cumulative_timesteps;
            }

            if self.stats.cumulative_model_updates % 10 == 0 {
                println!(
                    "Press Q to quit, and S to save then continue (must confirm by pressing enter)\n"
                );
            }
        }

        save_model(
            model,
            &self.stats,
            &self.checkpoints_folder,
            self.checkpoints_limit,
        );

        println!("Exiting.")
    }
}
