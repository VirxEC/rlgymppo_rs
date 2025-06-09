#![allow(clippy::type_complexity)]
mod agent;
mod base;
mod environment;

pub mod utils;

pub use agent::config::PpoLearnerConfig;
pub use burn::backend;
pub use rlgym::{self, glam, rocketsim_rs};

use agent::{Ppo, model::Actic};
use burn::{
    module::{AutodiffModule, Module},
    tensor::backend::AutodiffBackend,
};
use environment::{
    render::{Renderer, RendererControls},
    thread_sim::ThreadSim,
};
use parking_lot::{Condvar, Mutex};
use rand::{SeedableRng, rngs::SmallRng};
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::{
    io::{Read, stdin},
    path::PathBuf,
    sync::{
        Arc,
        mpsc::{Sender, channel},
    },
    thread::{self, JoinHandle},
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
    ToggleRender,
    ToggleDeterministic,
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
            'r' => {
                println!("Renderer will be toggled after this iteration...");
                s.send(HumanInput::ToggleRender).unwrap();
            }
            'd' => {
                println!(
                    "Toggling deterministic mode for the rendered model after this iteration..."
                );
                s.send(HumanInput::ToggleDeterministic).unwrap();
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
    /// The device to use for rendering.
    /// Will default to the default device from the given backend.
    pub render_device: B::Device,
    /// The layer sizes for the policy network.
    pub policy_layer_sizes: Vec<usize>,
    /// The layer sizes for the critic network.
    pub critic_layer_sizes: Vec<usize>,
    /// The maximum number of checkpoints to keep.
    /// If None, all checkpoints will be kept.
    pub checkpoints_limit: Option<usize>,
    /// The number of timesteps to run before saving a checkpoint.
    pub timesteps_per_save: u64,
    /// The number of threads to use for collecting data.
    pub num_threads: usize,
    /// The number of games to run per thread.
    /// Increasing this will increase GPU utilization
    /// and the utilization of 1 cpu thread.
    pub num_games_per_thread: usize,
    /// The size of the experience replay buffer.
    pub exp_buffer_size: usize,
    /// The number of additional iterations (episodes) to run training for,
    /// exiting after that.
    /// `None` means run indefinitely.
    pub num_additional_iterations: Option<u64>,
    /// If true, one extra instance will be launched
    /// and RLViser will be used to visualize training.
    pub render: bool,
    /// Whether to try to launch the rlviser executable.
    /// If false, RLViser needs to be started manually.
    pub try_launch_rlviser: bool,
    /// The name of the project to use for logging to wandb.
    /// If None, metrics will not be logged to wandb.
    pub wandb_project_name: Option<String>,
}

impl<B: AutodiffBackend> Default for LearnerConfig<B> {
    fn default() -> Self {
        Self {
            ppo: PpoLearnerConfig::default(),
            checkpoints_folder: PathBuf::from("checkpoints"),
            device: B::Device::default(),
            render_device: B::Device::default(),
            policy_layer_sizes: vec![256; 2],
            critic_layer_sizes: vec![256; 2],
            checkpoints_limit: None,
            timesteps_per_save: 1_000_000,
            num_threads: 8,
            num_games_per_thread: 32,
            exp_buffer_size: 60_000,
            num_additional_iterations: None,
            render: false,
            try_launch_rlviser: true,
            wandb_project_name: None,
        }
    }
}

impl<B: AutodiffBackend> LearnerConfig<B> {
    pub fn init<F, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
        self,
        create_env: F,
        step_callback: C,
    ) -> Learner<B>
    where
        F: Fn(Option<usize>) -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
            + Clone
            + Send
            + 'static,
        C: Fn(&mut Report, &mut SI, &GameStateA) + Clone + Send + 'static,
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

        let env = (create_env)(None);
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

        let renderer_controls = Arc::new((
            Mutex::new(RendererControls::new(self.render)),
            Condvar::new(),
        ));

        let renderer = {
            let create_env = create_env.clone();
            let step_callback = step_callback.clone();
            let renderer_controls = renderer_controls.clone();

            thread::spawn(move || {
                Renderer::new(
                    (create_env)(None),
                    step_callback,
                    self.try_launch_rlviser,
                    renderer_controls,
                    self.render_device,
                )
                .run();
            })
        };

        let thread_sim = ThreadSim::new(
            create_env,
            step_callback,
            self.ppo.batch_size,
            self.exp_buffer_size,
            self.num_threads,
            self.num_games_per_thread,
            self.device.clone(),
        );

        Learner {
            ppo: self.ppo.init(self.device.clone()),
            rng: SmallRng::from_os_rng(),
            stats: Stats::default(),
            device: self.device,
            model,
            wandb_project_name: self.wandb_project_name,
            checkpoints_folder: self.checkpoints_folder,
            checkpoints_limit: self.checkpoints_limit,
            timesteps_per_save: self.timesteps_per_save,
            last_save_timestep: 0,
            num_additional_iterations: self.num_additional_iterations,
            renderer_controls: renderer_controls.clone(),
            renderer,
            collector: thread_sim,
        }
    }
}

pub struct Learner<B: AutodiffBackend> {
    ppo: Ppo<B>,
    rng: SmallRng,
    stats: Stats,
    device: B::Device,
    model: Actic<B>,
    wandb_project_name: Option<String>,
    checkpoints_folder: PathBuf,
    checkpoints_limit: Option<usize>,
    timesteps_per_save: u64,
    last_save_timestep: u64,
    num_additional_iterations: Option<u64>,
    renderer_controls: Arc<(Mutex<RendererControls<B::InnerBackend>>, Condvar)>,
    renderer: JoinHandle<()>,
    collector: ThreadSim<B::InnerBackend>,
}

impl<B: AutodiffBackend> Learner<B> {
    /// Load the previously saved model.
    /// Does nothing if the model can't be loaded, this is safe to call unconditionally.
    pub fn load(&mut self) {
        (self.model, self.stats) =
            load_latest_model(self.model.clone(), &self.checkpoints_folder, &self.device);
    }

    fn print_controls_prompt() {
        println!("Press Q to quit, S to quick save, R to toggle rendering,");
        println!("and D to toggle deterministic mode for the renderer.");
        println!("!!! Must be confirmed by pressing enter. !!!\n");
    }

    fn handle_input(&self, input: HumanInput) -> bool {
        match input {
            HumanInput::Quit => {
                return false;
            }
            HumanInput::Save => {
                save_model(
                    self.model.valid(),
                    &self.stats,
                    &self.checkpoints_folder,
                    self.checkpoints_limit,
                );
            }
            HumanInput::ToggleRender => {
                let (controls, start_renderer) = &*self.renderer_controls;
                let mut guard = controls.lock();
                let render = !guard.render;
                guard.render = render;
                drop(guard);

                if render {
                    println!("Starting renderer...");
                    start_renderer.notify_all();
                } else {
                    println!("Stopping renderer...");
                }
            }
            HumanInput::ToggleDeterministic => {
                let (controls, _) = &*self.renderer_controls;
                let mut guard = controls.lock();
                guard.deterministic = !guard.deterministic;
                println!("Rendering deterministic: {}", guard.deterministic);
                drop(guard);
            }
        }

        true
    }

    /// Train the model, and automatically saves it before exiting.
    pub fn learn(mut self) {
        #[cfg(not(feature = "wandb"))]
        assert_eq!(
            self.wandb_project_name, None,
            "'wandb' feature is not enabled, but wandb_project_name is set. \
             Enable the 'wandb' feature in Cargo.toml to use Weights & Biases logging."
        );

        #[cfg(feature = "wandb")]
        let mut wandb = if let Some(project_name) = self.wandb_project_name.as_ref() {
            use crate::utils::running_stat::WandbRun;

            let mut settings = wandb::settings::Settings::default();
            settings.set_project(project_name.clone());

            let run = if let Some(wandb_run) = self.stats.wandb_run.as_ref() {
                settings.proto.entity = Some(wandb_run.entity.clone());

                let sess = wandb::session::Session::new(settings).unwrap();
                sess.init_run(Some(wandb_run.run_id.clone())).unwrap()
            } else {
                let sess = wandb::session::Session::new(settings).unwrap();
                let run = sess.init_run(None).unwrap();
                self.stats.wandb_run = Some(WandbRun {
                    run_id: run.settings.proto.run_id.as_ref().unwrap().clone(),
                    entity: run.settings.proto.entity.as_ref().unwrap().clone(),
                });

                run
            };

            Some(run)
        } else {
            None
        };

        let (s, r) = channel();

        thread::spawn(move || {
            stdin_reader(s);
        });

        println!("Running for the first time. This might be slow at first...");
        Self::print_controls_prompt();

        let inital_cumulative_updates = self.stats.cumulative_model_updates;
        'train: while self
            .num_additional_iterations
            .is_none_or(|n| self.stats.cumulative_model_updates - inital_cumulative_updates < n)
        {
            let collect_start = Instant::now();

            let nodiff_actor = self.model.actor.valid();

            // update the model the renderer is using
            {
                let (controls, start_rendering) = &*self.renderer_controls;
                let mut guard = controls.lock();
                guard.model = Some(nodiff_actor.clone());
                drop(guard);

                start_rendering.notify_all();
            }

            // collect steps
            let (memory, mut metrics) = self.collector.run(nodiff_actor);
            let collect_elapsed = collect_start.elapsed().as_secs_f64();

            // train the model
            let train_start = Instant::now();
            let num_new_steps;
            (self.model, num_new_steps) = self.ppo.learn(
                self.model,
                memory,
                &mut self.rng,
                &mut metrics,
                &mut self.stats,
            );

            let learn_elapsed = train_start.elapsed().as_secs_f64();
            let overall_elapsed = collect_start.elapsed().as_secs_f64();

            metrics[".Episode length"] = num_new_steps.into();
            metrics[".Collection time"] = collect_elapsed.into();
            metrics[".Collected SPS"] = (num_new_steps as f64 / collect_elapsed).into();
            metrics[".Learning time"] = learn_elapsed.into();
            metrics[".Overall time"] = overall_elapsed.into();
            metrics[".Overall SPS"] =
                (num_new_steps as f64 / collect_start.elapsed().as_secs_f64()).into();
            metrics[".Cumulative steps"] = self.stats.cumulative_timesteps.into();
            metrics[".Cumulative epochs"] = self.stats.cumulative_epochs.into();
            metrics[".Cumulative updates"] = self.stats.cumulative_model_updates.into();

            #[cfg(feature = "wandb")]
            if let Some(wandb) = wandb.as_mut() {
                metrics.report_wandb(wandb);
            }

            println!("{metrics}");

            for input in r.try_iter() {
                if !self.handle_input(input) {
                    break 'train;
                }
            }

            if self.stats.cumulative_timesteps - self.last_save_timestep > self.timesteps_per_save {
                save_model(
                    self.model.valid(),
                    &self.stats,
                    &self.checkpoints_folder,
                    self.checkpoints_limit,
                );
                self.last_save_timestep = self.stats.cumulative_timesteps;
            }

            Self::print_controls_prompt();
        }

        {
            // Make render thread exit
            let (controls, start_renderer) = &*self.renderer_controls;
            let mut guard = controls.lock();
            guard.quit = true;
            drop(guard);

            // if render = false, this will wake the thread up to exit
            start_renderer.notify_all();
        }

        save_model(
            self.model,
            &self.stats,
            &self.checkpoints_folder,
            self.checkpoints_limit,
        );

        #[cfg(feature = "wandb")]
        if let Some(mut wandb) = wandb {
            wandb.finish();
        }

        println!("Waiting for threads to exit...");
        self.renderer.join().unwrap();
        self.collector.join();

        println!("Done.")
    }

    /// Only run the renderer. Useful for debugging.
    pub fn render(self) {
        let (s, r) = channel();

        thread::spawn(move || {
            stdin_reader(s);
        });

        Self::print_controls_prompt();

        let nodiff_actor = self.model.actor.valid();

        // update the model the renderer is using
        {
            let (controls, start_rendering) = &*self.renderer_controls;
            let mut guard = controls.lock();
            guard.model = Some(nodiff_actor.clone());
            drop(guard);

            start_rendering.notify_all();
        }

        for input in r.iter() {
            if !self.handle_input(input) {
                break;
            }
        }

        {
            // Make render thread exit
            let (controls, start_renderer) = &*self.renderer_controls;
            let mut guard = controls.lock();
            guard.quit = true;
            drop(guard);

            // if render = false, this will wake the thread up to exit
            start_renderer.notify_all();
        }

        save_model(
            self.model,
            &self.stats,
            &self.checkpoints_folder,
            self.checkpoints_limit,
        );

        println!("Waiting for threads to exit...");
        self.renderer.join().unwrap();
        self.collector.join();

        println!("Done.")
    }
}
