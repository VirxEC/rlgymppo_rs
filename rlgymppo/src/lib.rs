#![allow(clippy::type_complexity)]
mod agent;
mod base;
mod environment;

pub mod utils;

use std::io::{Read, stdin};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Sender, channel};
use std::thread;
use std::time::Instant;

use agent::Ppo;
pub use agent::config::PpoLearnerConfig;
use agent::model::Actic;
pub use agent::self_play::SelfPlayConfig;
use agent::self_play::VersionManager;
use agent::skill_tracker::SkillTracker;
pub use agent::skill_tracker::SkillTrackerConfig;
use base::TerminalState;
pub use burn::backend;
use burn::module::{AutodiffModule, Module, Quantizer};
use burn::nn::modules::norm::NormalizationConfig;
use burn::nn::{LayerNormConfig, RmsNormConfig};
use burn::tensor::backend::AutodiffBackend;
use environment::render::{Renderer, RendererControls};
use environment::sim::RewardSamplingConfig;
use environment::thread_sim::ThreadSim;
use parking_lot::{Condvar, Mutex};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
pub use rlgym::{self, rocketsim};
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};
use utils::running_stat::Stats;
use utils::serde::{latest_checkpoint_folder, load_latest_model, save_checkpoint};
use utils::shared_info::{SharedInfoReport, SharedInfoRng};

#[derive(Clone, Copy)]
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
                #[cfg(not(feature = "tui"))]
                println!("Finishing iteration, saving, then exiting...");
                s.send(HumanInput::Quit).unwrap();
                return;
            }
            's' => {
                #[cfg(not(feature = "tui"))]
                println!("Saving model after this iteration...");
                s.send(HumanInput::Save).unwrap();
            }
            'r' => {
                #[cfg(not(feature = "tui"))]
                println!("Renderer will be toggled after this iteration...");
                s.send(HumanInput::ToggleRender).unwrap();
            }
            'd' => {
                #[cfg(not(feature = "tui"))]
                println!(
                    "Toggling deterministic mode for the rendered model after this iteration..."
                );
                s.send(HumanInput::ToggleDeterministic).unwrap();
            }
            _ => {}
        }
    }
}

/// Which normalization layer to apply after each hidden linear layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NormSelection {
    /// No normalization.
    None,
    /// Layer Normalization (learnable affine with gamma & beta).
    LayerNorm,
    /// RMS Normalization (learnable scale only, no beta).
    RmsNorm,
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
    pub quantizer: Option<Quantizer>,
    /// The device to use for rendering.
    /// Will default to the default device from the given backend.
    pub render_device: B::Device,
    /// The layer sizes for the policy network.
    pub policy_layer_sizes: Vec<usize>,
    /// The layer sizes for the critic network.
    pub critic_layer_sizes: Vec<usize>,
    /// Normalization to apply after every hidden linear layer.
    pub norm: NormSelection,
    /// Layer sizes for the shared feature extractor (empty = no shared head).
    /// When set, the actor and critic take their input from this head's output.
    pub shared_head_layer_sizes: Vec<usize>,
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

    /// The number of additional iterations (episodes) to run training for,
    /// exiting after that.
    /// `None` means run indefinitely.
    pub num_additional_iterations: Option<u64>,
    /// If true, one extra instance will be launched to visualize training.
    /// RocketSim's built-in renderer is used for visualization.
    pub render: bool,

    // ── Self‑Play ─────────────────────────────────────────────────────
    /// Configuration for saving old policy versions and occasionally
    /// training against them (self-play).
    pub self_play: SelfPlayConfig,

    // ── Skill Tracker ────────────────────────────────────────────────
    /// Elo-based skill rating system that periodically evaluates the
    /// current policy against old versions.  Reports `"Rating/1v1"`,
    /// `"Rating/2v2"`, etc.
    pub skill_tracker: SkillTrackerConfig,

    // ── Weights & Biases ─────────────────────────────────────────────
    /// Project name for wandb (requires the `wandb` feature).
    /// When `None`, wandb logging is disabled.
    pub wandb_project_name: Option<String>,
    /// Group name for wandb (default: `"unnamed-runs"`).
    pub wandb_group_name: Option<String>,
    /// Run name for wandb (default: `"rlgymppo-run"`).
    pub wandb_run_name: Option<String>,
}

impl<B: AutodiffBackend> Default for LearnerConfig<B> {
    fn default() -> Self {
        Self {
            ppo: PpoLearnerConfig::default(),
            checkpoints_folder: PathBuf::from("checkpoints"),
            device: B::Device::default(),
            quantizer: None,
            render_device: B::Device::default(),
            policy_layer_sizes: vec![256; 3],
            critic_layer_sizes: vec![256; 3],
            norm: NormSelection::LayerNorm,
            shared_head_layer_sizes: vec![256],
            checkpoints_limit: None,
            timesteps_per_save: 1_000_000,
            num_threads: 4,
            num_games_per_thread: 64,

            num_additional_iterations: None,
            render: false,
            self_play: SelfPlayConfig::default(),
            skill_tracker: SkillTrackerConfig::default(),
            wandb_project_name: None,
            wandb_group_name: None,
            wandb_run_name: None,
        }
    }
}

impl<B: AutodiffBackend> LearnerConfig<B> {
    pub fn init<F, SS, OBS, ACT, REW, TERM, TRUNC, SI>(
        self,
        create_env: F,
    ) -> Learner<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Clone + Send + 'static,
        SS: StateSetter<SI>,
        SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
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
        let env = (create_env)(None);
        let obs_space = env.get_obs_space();
        let action_space = env.get_action_space();

        let norm_config = match self.norm {
            NormSelection::None => None,
            NormSelection::LayerNorm => Some(NormalizationConfig::Layer(LayerNormConfig::new(0))),
            NormSelection::RmsNorm => Some(NormalizationConfig::Rms(RmsNormConfig::new(0))),
        };

        let model = Actic::<B>::new(
            obs_space,
            action_space,
            self.policy_layer_sizes,
            self.critic_layer_sizes,
            &self.shared_head_layer_sizes,
            &self.device,
            norm_config,
        );

        if let Some(ref head) = model.shared_head {
            println!("# parameters in shared head: {}", head.num_params());
        }
        println!("# parameters in actor: {}", model.actor.num_params());
        println!("# parameters in critic: {}", model.critic.num_params());

        let renderer_controls = Arc::new((
            Mutex::new(RendererControls::new(self.render)),
            Condvar::new(),
        ));

        let renderer = {
            let create_env = create_env.clone();
            let renderer_controls = renderer_controls.clone();

            thread::spawn(move || {
                Renderer::new((create_env)(None), renderer_controls, self.render_device).run();
            })
        };

        let reward_sampling = RewardSamplingConfig {
            add_rewards_to_metrics: self.ppo.add_rewards_to_metrics,
            reward_sample_interval: self.ppo.reward_sample_interval,
            max_reward_samples: self.ppo.max_reward_samples,
        };

        let skill_tracker = if self.skill_tracker.enabled {
            let create_env_skill = create_env.clone();
            let create_arena = move |game_idx: usize| {
                let env = (create_env_skill)(Some(game_idx));

                (env.arena, env.observations, env.action, env.shared_info)
            };

            Some(SkillTracker::new(
                self.skill_tracker.clone(),
                create_arena,
                self.device.clone(),
            ))
        } else {
            None
        };

        let thread_sim = ThreadSim::new(
            create_env,
            self.ppo.batch_size,
            self.num_threads,
            self.num_games_per_thread,
            self.device.clone(),
            reward_sampling,
            self.ppo.max_episode_length,
        );

        let mut self_play_config = self.self_play;
        if self_play_config.train_against_old_versions || self.skill_tracker.enabled {
            self_play_config.save_policy_versions = true;
        }

        let version_mgr = VersionManager::new(
            self.checkpoints_folder.join("policy_versions"),
            self_play_config.clone(),
        );

        Learner {
            ppo: self.ppo.init(self.device.clone()),
            rng: SmallRng::from_rng(&mut rng()),
            stats: Stats::default(),
            device: self.device,
            quantizer: self.quantizer,
            model,
            wandb_project_name: self.wandb_project_name,
            wandb_group_name: self.wandb_group_name,
            wandb_run_name: self.wandb_run_name,
            checkpoints_folder: self.checkpoints_folder,
            checkpoints_limit: self.checkpoints_limit,
            timesteps_per_save: self.timesteps_per_save,
            last_save_timestep: 0,
            num_additional_iterations: self.num_additional_iterations,
            renderer_controls: renderer_controls.clone(),
            renderer,
            collector: thread_sim,
            self_play_config,
            version_mgr,
            skill_tracker,
        }
    }
}

pub struct Learner<B: AutodiffBackend, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    ppo: Ppo<B>,
    rng: SmallRng,
    stats: Stats,
    device: B::Device,
    quantizer: Option<Quantizer>,
    model: Actic<B>,
    wandb_project_name: Option<String>,
    #[cfg_attr(not(feature = "wandb"), allow(dead_code))]
    wandb_group_name: Option<String>,
    #[cfg_attr(not(feature = "wandb"), allow(dead_code))]
    wandb_run_name: Option<String>,
    checkpoints_folder: PathBuf,
    checkpoints_limit: Option<usize>,
    timesteps_per_save: u64,
    last_save_timestep: u64,
    num_additional_iterations: Option<u64>,
    renderer_controls: Arc<(Mutex<RendererControls<B::InnerBackend>>, Condvar)>,
    renderer: thread::JoinHandle<()>,
    // ── Self‑Play ──────────────────────────────────────────────────
    self_play_config: SelfPlayConfig,
    version_mgr: VersionManager<B::InnerBackend>,
    // ── Skill Tracker ────────────────────────────────────────────
    skill_tracker: Option<SkillTracker<B::InnerBackend, OBS, ACT, SI>>,

    collector: ThreadSim<B::InnerBackend, SS, OBS, ACT, REW, TERM, TRUNC, SI>,
}

impl<B: AutodiffBackend, SS, OBS, ACT, REW, TERM, TRUNC, SI>
    Learner<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    /// Load the previously saved model, training stats, and optimizer state.
    /// Does nothing if the model can't be loaded, this is safe to call unconditionally.
    pub fn load(&mut self) {
        (self.model, self.stats) =
            load_latest_model(self.model.clone(), &self.checkpoints_folder, &self.device);

        // Align save timer so we don't immediately re-save the loaded checkpoint.
        self.last_save_timestep = self.stats.cumulative_timesteps;

        if let Some(latest_folder) = latest_checkpoint_folder(&self.checkpoints_folder) {
            self.ppo.load_optimizers(&latest_folder);
        }

        // Load saved policy versions from disk.
        {
            let template = self.model.valid();
            self.version_mgr.load_versions(
                &template,
                &self.device,
                self.stats.cumulative_timesteps,
            );
        }

        // Restore skill tracker ratings from the checkpoint.
        if let (Some(st), Some(ratings)) = (&mut self.skill_tracker, &self.stats.skill_ratings) {
            st.cur_ratings.data = ratings.clone();
        }
    }

    fn print_controls_prompt() {
        #[cfg(not(feature = "tui"))]
        {
            println!("Press Q to quit, S to quick save, R to toggle rendering,");
            println!("and D to toggle deterministic mode for the renderer.");
            println!("!!! Must be confirmed by pressing enter. !!!\n");
        }
    }

    fn handle_input(&mut self, input: HumanInput) -> bool {
        match input {
            HumanInput::Quit => {
                return false;
            }
            HumanInput::Save => {
                // Serialise skill tracker ratings before saving.
                if let Some(ref st) = self.skill_tracker {
                    self.stats.skill_ratings = Some(st.cur_ratings.data.clone());
                }
                save_checkpoint(
                    self.model.valid(),
                    &self.ppo,
                    &self.stats,
                    &self.checkpoints_folder,
                    self.checkpoints_limit,
                );
                self.version_mgr.save_versions();
            }
            HumanInput::ToggleRender => {
                let (controls, start_renderer) = &*self.renderer_controls;
                let mut guard = controls.lock();
                let render = !guard.render;
                guard.render = render;
                drop(guard);

                #[cfg(not(feature = "tui"))]
                if render {
                    println!("Starting renderer...");
                } else {
                    println!("Stopping renderer...");
                }

                if render {
                    start_renderer.notify_all();
                }
            }
            HumanInput::ToggleDeterministic => {
                let (controls, _) = &*self.renderer_controls;
                let mut guard = controls.lock();
                guard.deterministic = !guard.deterministic;
                #[cfg(not(feature = "tui"))]
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

        // Initialise the wandb MetricSender via embedded Python before
        // the TUI so the "wandb run started" message goes to stdout
        // before the alternate screen takes over.
        #[cfg(feature = "wandb")]
        #[cfg_attr(not(all(feature = "tui", feature = "wandb")), allow(dead_code))]
        let (wandb_tx, wandb_handle, wandb_run_id) = if let Some(project_name) =
            self.wandb_project_name.as_ref()
        {
            let group = self.wandb_group_name.as_deref().unwrap_or("unnamed-runs");
            let name = self.wandb_run_name.as_deref().unwrap_or("rlgymppo-run");
            let run_id = self
                .stats
                .wandb_run
                .as_ref()
                .map(|r| r.run_id.as_str())
                .unwrap_or("");

            match rlgymppo_wandb::MetricSender::new(project_name, group, name, run_id) {
                Ok(sender) => {
                    // Persist the run ID for resume on restart.
                    let id = sender.run_id().to_owned();
                    self.stats.wandb_run =
                        Some(crate::utils::running_stat::WandbRun { run_id: id.clone() });
                    println!(" > wandb run started with ID: \"{id}\"");

                    let (tx, rx) =
                        std::sync::mpsc::sync_channel::<std::collections::HashMap<String, f64>>(1);
                    let handle = thread::Builder::new()
                        .name("wandb".into())
                        .spawn(move || {
                            // `sender` moves into this thread; dropped
                            // when the channel closes -> calls finish().
                            while let Ok(metrics) = rx.recv() {
                                if let Err(e) = sender.send(&metrics) {
                                    eprintln!("Warning: wandb send failed: {e}");
                                }
                            }
                        })
                        .expect("Failed to spawn wandb-sender thread");
                    (Some(tx), Some(handle), Some(id))
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to initialise wandb MetricSender: {e}\n\
                             Metrics will not be logged to wandb."
                    );
                    (None, None, None)
                }
            }
        } else {
            (None, None, None)
        };

        // Initialise the TUI display (ratatui-based terminal dashboard).
        #[cfg(feature = "tui")]
        let mut tui_display = match rlgymppo_tui::TuiDisplay::new() {
            Ok(tui) => Some(tui),
            Err(e) => {
                eprintln!("Warning: Failed to initialise TUI display: {e}",);
                eprintln!("Falling back to plain-text iteration logs.");
                None
            }
        };

        // Show the wandb run ID in the TUI now that it's active.
        #[cfg(all(feature = "tui", feature = "wandb"))]
        if let Some(ref mut tui) = tui_display
            && let Some(ref id) = wandb_run_id
        {
            tui.notify(format!("Wandb run started: {id}"));
        }

        let (s, r) = channel();

        thread::spawn(move || {
            stdin_reader(s);
        });

        #[cfg(not(feature = "tui"))]
        println!("Running for the first time. This might be slow at first...");
        Self::print_controls_prompt();

        let inital_cumulative_updates = self.stats.cumulative_model_updates;
        'train: while self
            .num_additional_iterations
            .is_none_or(|n| self.stats.cumulative_model_updates - inital_cumulative_updates < n)
        {
            let collect_start = Instant::now();

            let mut nodiff_model = self.model.valid();
            if let Some(quantizer) = &mut self.quantizer {
                nodiff_model = nodiff_model.quantize_weights(quantizer);
            }

            // update the model the renderer is using
            {
                let (controls, start_rendering) = &*self.renderer_controls;
                let mut guard = controls.lock();
                guard.model = Some(nodiff_model.clone());
                drop(guard);

                start_rendering.notify_all();
            }

            // ── Self‑play: stochastically decide to use an old version ──
            let self_play = if self.self_play_config.train_against_old_versions
                && !self.version_mgr.is_empty()
                && (self.rng.next_u32() as f64 / u32::MAX as f64)
                    < self.self_play_config.train_against_old_chance as f64
            {
                let idx = self.version_mgr.random_index(&mut self.rng);
                let old_team = if self.rng.next_u32().is_multiple_of(2) {
                    0
                } else {
                    1
                };
                #[cfg(not(feature = "tui"))]
                println!(
                    " > Training against old version {} (team {})",
                    self.version_mgr.versions[idx].timesteps, old_team
                );
                Some((self.version_mgr.versions[idx].model.clone(), old_team))
            } else {
                None
            };

            // collect steps
            let (memory, mut metrics) = self.collector.run(nodiff_model, self_play);
            let collect_elapsed = collect_start.elapsed().as_secs_f64();

            // train the model
            let is_first_iteration = self.stats.cumulative_model_updates == 0;
            let train_start = Instant::now();
            let num_new_steps;
            (self.model, num_new_steps) = self.ppo.learn(
                self.model,
                memory,
                &mut self.rng,
                &mut metrics,
                &mut self.stats,
                is_first_iteration,
            );

            let consumption_elapsed = train_start.elapsed().as_secs_f64();

            // ── Self‑play: save a policy version if we crossed a boundary ──
            let prev_timesteps = self.stats.cumulative_timesteps - num_new_steps as u64;
            let nodiff_model_snapshot = self.model.valid();

            // Skill tracker ratings (if enabled) are frozen into new versions.
            let cur_ratings = self.skill_tracker.as_ref().map(|st| &st.cur_ratings);
            self.version_mgr.on_iteration(
                &nodiff_model_snapshot,
                self.stats.cumulative_timesteps,
                prev_timesteps,
                cur_ratings,
            );

            // ── Skill tracker: run evaluation matches if due ──────
            if let Some(ref mut st) = self.skill_tracker {
                st.on_iteration(&nodiff_model_snapshot, &mut self.version_mgr.versions);
                // Always report current ratings so the TUI displays them.
                st.report_ratings(&mut metrics);
            }

            // Episode length = total_steps / number of NORMAL terminals.
            let n_term = (0..memory.len())
                .filter(|&i| memory.terminals()[i] == TerminalState::Normal)
                .count();
            let ep_len = if n_term > 0 {
                memory.len() as f64 / n_term as f64
            } else {
                memory.len() as f64
            };
            metrics["Collect/episode length"] = ep_len.into();
            metrics["Collect/timesteps"] = num_new_steps.into();
            metrics["Timing/collection"] = collect_elapsed.into();
            metrics["Timing/consumption"] = consumption_elapsed.into();
            metrics["Throughput/collected"] = (num_new_steps as f64 / collect_elapsed).into();
            metrics["Throughput/consumption"] =
                (num_new_steps as f64 / consumption_elapsed.max(1e-12)).into();
            metrics["Throughput/overall"] =
                (num_new_steps as f64 / collect_start.elapsed().as_secs_f64()).into();
            metrics["Cumulative/steps"] = self.stats.cumulative_timesteps.into();
            metrics["Cumulative/epochs"] = self.stats.cumulative_epochs.into();
            metrics["Cumulative/updates"] = self.stats.cumulative_model_updates.into();

            #[cfg(feature = "wandb")]
            if let Some(ref tx) = wandb_tx {
                let flat = metrics.to_flat_map();
                // Non-blocking send; silently drop if the background
                // wandb thread is still busy with the previous batch.
                let _ = tx.try_send(flat);
            }

            // ── Update TUI dashboard (or fall back to plain print) ─
            #[cfg(feature = "tui")]
            if let Some(ref mut tui) = tui_display {
                let flat = metrics.to_flat_map();
                if let Err(e) = tui.update(&flat) {
                    eprintln!("Warning: TUI display update failed: {e}");
                }
            }

            #[cfg(not(feature = "tui"))]
            println!("{metrics}");

            for input in r.try_iter() {
                #[cfg(feature = "tui")]
                let was_save = matches!(input, HumanInput::Save);
                #[cfg(feature = "tui")]
                let was_toggle_render = matches!(input, HumanInput::ToggleRender);
                #[cfg(feature = "tui")]
                let was_toggle_det = matches!(input, HumanInput::ToggleDeterministic);

                if !self.handle_input(input) {
                    break 'train;
                }

                #[cfg(feature = "tui")]
                if let Some(ref mut tui) = tui_display {
                    if was_save {
                        tui.notify("Saved model.");
                    } else if was_toggle_render {
                        tui.notify("Renderer toggled.");
                    } else if was_toggle_det {
                        tui.notify("Deterministic mode toggled.");
                    }
                }
            }

            if self.stats.cumulative_timesteps - self.last_save_timestep > self.timesteps_per_save {
                #[cfg(feature = "tui")]
                if let Some(ref mut tui) = tui_display {
                    tui.notify("Auto-saving model...");
                }

                // Serialise skill tracker ratings into the stats.
                if let Some(ref st) = self.skill_tracker {
                    self.stats.skill_ratings = Some(st.cur_ratings.data.clone());
                }

                save_checkpoint(
                    self.model.valid(),
                    &self.ppo,
                    &self.stats,
                    &self.checkpoints_folder,
                    self.checkpoints_limit,
                );
                self.version_mgr.save_versions();
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

        // Serialise skill tracker ratings before final save.
        if let Some(ref st) = self.skill_tracker {
            self.stats.skill_ratings = Some(st.cur_ratings.data.clone());
        }

        save_checkpoint(
            self.model.valid(),
            &self.ppo,
            &self.stats,
            &self.checkpoints_folder,
            self.checkpoints_limit,
        );
        self.version_mgr.save_versions();

        // Close the TUI display (restores normal terminal).
        #[cfg(feature = "tui")]
        if let Some(mut tui) = tui_display
            && let Err(e) = tui.close()
        {
            eprintln!("Warning: TUI display close failed: {e}");
        }

        // Close the wandb channel so the background thread exits, which
        // drops the MetricSender and calls `wandb.finish()`.
        #[cfg(feature = "wandb")]
        {
            drop(wandb_tx);
            if let Some(handle) = wandb_handle {
                handle.join().expect("wandb-sender thread panicked");
            }
        }

        println!("Waiting for threads to exit...");
        self.collector.join();
        self.renderer.join().unwrap();

        println!("Done.")
    }

    /// Only run the renderer. Useful for debugging.
    pub fn render(mut self) {
        let (s, r) = channel();

        thread::spawn(move || {
            stdin_reader(s);
        });

        Self::print_controls_prompt();

        let nodiff_model = self.model.valid();

        // update the model the renderer is using
        {
            let (controls, start_rendering) = &*self.renderer_controls;
            let mut guard = controls.lock();
            guard.model = Some(nodiff_model.clone());
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

        save_checkpoint(
            self.model.valid(),
            &self.ppo,
            &self.stats,
            &self.checkpoints_folder,
            self.checkpoints_limit,
        );
        self.version_mgr.save_versions();

        println!("Waiting for threads to exit...");
        self.renderer.join().unwrap();

        println!("Done.")
    }
}
