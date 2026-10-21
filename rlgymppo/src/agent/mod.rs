pub mod config;
mod gae;
pub mod model;
pub mod self_play;
pub mod skill_tracker;
pub mod transfer_learn;

use std::path::Path;
use std::time::{Duration, Instant};

use burn::module::Module;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::modules::norm::Normalization;
use burn::optim::{GradientsAccumulator, GradientsParams, ModuleOptimizer};
use burn::prelude::*;
use burn::tensor::{DType, FloatDType, Transaction};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rlgymppo_utils::Report;

use crate::OptimizerNetwork;
use crate::agent::config::{PpoLearnerConfig, RegenConfig};
use crate::agent::gae::{GAEOutput, get_gae};
use crate::agent::model::{Actic, Net, PPOOutput};
use crate::base::{Memory, TerminalState, get_batch_1d, get_states_batch_range};
use crate::utils::running_stat::Stats;

pub struct Ppo {
    config: PpoLearnerConfig,
    policy_optimizer: ModuleOptimizer,
    value_optimizer: ModuleOptimizer,
    shared_head_optimizer: ModuleOptimizer,
    /// The model-aware factory is retained so optimizers can be rebuilt after
    /// loading a checkpoint with freshly-created model parameters.
    make_optim: Option<Box<dyn Fn(OptimizerNetwork, &Net) -> ModuleOptimizer>>,
    device: Device,
    /// Fixed initialization anchor for regenerative regularization.
    /// Captured from the pre-update network on the first training call that
    /// enables `regen` (see `ensure_regen_snapshot`), then preserved across
    /// checkpoint save/load. `None` until needed.
    init_snapshot: Option<InitSnapshot>,
    /// Transfer-learning rounds completed; drives the LR schedule there.
    transfer_rounds: u64,
    /// EMA of the batch advantage spread for percentile scaling.
    adv_scale_ema: f32,
    /// Last computed plasticity diagnostics, re-emitted every iteration so
    /// metric keys (and their TUI section) stay present between computations.
    last_plasticity: Option<PlasticitySnapshot>,
    /// Last computed policy-only (actor + shared head) diagnostics for
    /// transfer learning, re-emitted every round. Separate from
    /// `last_plasticity` so transfer rounds never emit frozen-critic keys
    /// and PPO never emits head keys.
    last_transfer_plasticity: Option<PolicyPlasticitySnapshot>,
}

/// Parameter snapshot used by regenerative regularization. Cloned device
/// tensors, so enabling `regen` holds one extra copy of the parameters.
struct InitSnapshot {
    actor: Net,
    critic: Net,
    head: Option<Net>,
}

/// Plasticity diagnostics for one actor/critic pair. Cached on `Ppo` and
/// re-emitted every iteration so the keys stay present in every report.
#[derive(Clone, Copy, Debug, Default)]
struct PlasticitySnapshot {
    actor_norm: f64,
    critic_norm: f64,
    actor_rank: f64,
    critic_rank: f64,
    actor_dormant: f64,
    critic_dormant: f64,
}

impl PlasticitySnapshot {
    fn emit(self, metrics: &mut Report) {
        metrics["Update/actor weight norm"] = self.actor_norm.into();
        metrics["Update/critic weight norm"] = self.critic_norm.into();
        metrics["Update/actor stable rank"] = self.actor_rank.into();
        metrics["Update/critic stable rank"] = self.critic_rank.into();
        metrics["Update/actor dormant frac"] = self.actor_dormant.into();
        metrics["Update/critic dormant frac"] = self.critic_dormant.into();
    }
}

/// Policy-only plasticity diagnostics for transfer learning (actor + shared
/// head). Cached on `Ppo` separately from [`PlasticitySnapshot`] and
/// re-emitted every round so the keys stay present between recomputations.
/// Head fields are `None` when the student has no shared head; `emit` then
/// skips the head keys instead of reporting misleading zeros.
#[derive(Clone, Copy, Debug, Default)]
struct PolicyPlasticitySnapshot {
    actor_norm: f64,
    actor_rank: f64,
    actor_dormant: f64,
    head_norm: Option<f64>,
    head_rank: Option<f64>,
    head_dormant: Option<f64>,
}

impl PolicyPlasticitySnapshot {
    fn emit(self, metrics: &mut Report) {
        metrics["Update/actor weight norm"] = self.actor_norm.into();
        metrics["Update/actor stable rank"] = self.actor_rank.into();
        metrics["Update/actor dormant frac"] = self.actor_dormant.into();
        if let (Some(norm), Some(rank), Some(dormant)) =
            (self.head_norm, self.head_rank, self.head_dormant)
        {
            metrics["Update/head weight norm"] = norm.into();
            metrics["Update/head stable rank"] = rank.into();
            metrics["Update/head dormant frac"] = dormant.into();
        }
    }
}

impl Ppo {
    pub fn new(
        config: PpoLearnerConfig,
        device: Device,
        make_optim: impl Fn() -> ModuleOptimizer,
    ) -> Self {
        Self {
            policy_optimizer: make_optim(),
            value_optimizer: make_optim(),
            shared_head_optimizer: make_optim(),
            make_optim: None,
            config,
            device,
            init_snapshot: None,
            transfer_rounds: 0,
            adv_scale_ema: 1.0,
            last_plasticity: None,
            last_transfer_plasticity: None,
        }
    }

    pub fn new_with_model(
        config: PpoLearnerConfig,
        device: Device,
        model: &Actic,
        make_optim: impl Fn(OptimizerNetwork, &Net) -> ModuleOptimizer + 'static,
    ) -> Self {
        let make_optim: Box<dyn Fn(OptimizerNetwork, &Net) -> ModuleOptimizer> =
            Box::new(make_optim);
        let shared_head = model.shared_head.as_ref().unwrap_or(&model.actor);
        Self {
            policy_optimizer: make_optim(OptimizerNetwork::Policy, &model.actor),
            value_optimizer: make_optim(OptimizerNetwork::Value, &model.critic),
            shared_head_optimizer: make_optim(OptimizerNetwork::SharedHead, shared_head),
            make_optim: Some(make_optim),
            config,
            device,
            init_snapshot: None,
            transfer_rounds: 0,
            adv_scale_ema: 1.0,
            last_plasticity: None,
            last_transfer_plasticity: None,
        }
    }

    /// Recreate model-aware optimizers for a newly-loaded model.
    ///
    /// Clears the regen anchor: call `load_regen_snapshot` afterwards to
    /// restore the original anchor from the checkpoint folder (legacy
    /// checkpoints without anchor files fall back to the loaded weights with
    /// a warning when `regen` is enabled).
    pub fn reinit_optimizers(&mut self, model: &Actic) {
        let Some(make_optim) = self.make_optim.as_ref() else {
            return;
        };

        let shared_head = model.shared_head.as_ref().unwrap_or(&model.actor);
        self.policy_optimizer = make_optim(OptimizerNetwork::Policy, &model.actor);
        self.value_optimizer = make_optim(OptimizerNetwork::Value, &model.critic);
        self.shared_head_optimizer = make_optim(OptimizerNetwork::SharedHead, shared_head);
        // The loaded model replaces the parameters, so any initialization
        // snapshot is stale until `load_regen_snapshot` restores the saved
        // anchor (or `ensure_regen_snapshot` re-captures with a warning).
        self.init_snapshot = None;
    }

    /// Save the optimizer states (momentum/velocity buffers) to a checkpoint folder.
    pub fn save_optimizers(&self, folder: &Path) {
        #[cfg(not(feature = "tui"))]
        println!("Saving optimizer states...");

        self.policy_optimizer
            .save(folder.join("policy_optimizer.bpk"))
            .unwrap();
        self.value_optimizer
            .save(folder.join("value_optimizer.bpk"))
            .unwrap();
        self.shared_head_optimizer
            .save(folder.join("shared_head_optimizer.bpk"))
            .unwrap();

        #[cfg(not(feature = "tui"))]
        println!("Saved optimizer states to: {folder:?}");
    }

    /// Completed transfer-learning rounds (one per `transfer_learn` call).
    /// Drives the transfer LR schedule. Persisted via `Stats::transfer_rounds`
    /// (see `Learner::load`); use the accessor pair to keep them in sync.
    pub fn transfer_rounds(&self) -> u64 {
        self.transfer_rounds
    }

    /// Restore the transfer-round count from a checkpoint (see
    /// `Stats::transfer_rounds`). Legacy checkpoints without the field load
    /// as `0`; callers warn instead of silently assuming it is correct.
    pub fn set_transfer_rounds(&mut self, rounds: u64) {
        self.transfer_rounds = rounds;
    }

    /// Whether a regen anchor is currently held.
    pub fn has_regen_snapshot(&self) -> bool {
        self.init_snapshot.is_some()
    }

    /// Whether the PPO path has regenerative regularization enabled.
    /// Transfer-learning regen is per-call (`TransferLearnConfig::regen`),
    /// so this only reflects the PPO config.
    pub fn ppo_regen_enabled(&self) -> bool {
        self.config.regen.is_some()
    }

    /// Capture the regen anchor from the pre-update network when `regen` is
    /// enabled and no anchor exists yet. Must run before the first optimizer
    /// step so the anchor is the true initialization, not post-update weights.
    pub fn ensure_regen_snapshot(&mut self, net: &Actic, regen: Option<RegenConfig>) {
        if regen.is_none() || self.init_snapshot.is_some() {
            return;
        }
        self.init_snapshot = Some(InitSnapshot {
            actor: net.actor.valid(),
            critic: net.critic.valid(),
            head: net.shared_head.as_ref().map(|head| head.valid()),
        });
    }

    /// Save the regen anchor (`*_init` files) alongside a checkpoint. No-op
    /// when no anchor exists yet. Values only; live `ParamId`s are preserved
    /// by `regen_step` (see `blend_param`).
    pub fn save_regen_snapshot(&self, folder: &Path) {
        let Some(snapshot) = self.init_snapshot.as_ref() else {
            return;
        };
        snapshot
            .actor
            .clone()
            .save_file(folder.join("actor_init.bpk"))
            .unwrap();
        snapshot
            .critic
            .clone()
            .save_file(folder.join("critic_init.bpk"))
            .unwrap();
        if let Some(head) = snapshot.head.as_ref() {
            head.clone()
                .save_file(folder.join("shared_head_init.bpk"))
                .unwrap();
        }
    }

    /// Restore the regen anchor saved by `save_regen_snapshot`. No-op for
    /// legacy checkpoints without anchor files (caller warns and falls back
    /// to the loaded weights when `regen` is enabled). Never fails on
    /// missing files; mismatched architectures fall back with a warning.
    pub fn load_regen_snapshot(&mut self, folder: &Path, model: &Actic) {
        for component in ["actor_init", "critic_init", "shared_head_init"] {
            let legacy = folder.join(format!("{component}.mpk.gz"));
            if legacy.is_file() && !folder.join(format!("{component}.bpk")).is_file() {
                panic!(
                    "Found legacy regen checkpoint {}. Burn 0.22 cannot load .mpk.gz; convert it with Burn 0.21 before resuming.",
                    legacy.display()
                );
            }
        }
        if !folder.join("actor_init.bpk").is_file() {
            return;
        }
        let actor = model
            .actor
            .clone()
            .try_load_file(folder.join("actor_init.bpk"))
            .unwrap_or_else(|e| {
                eprintln!(
                    "Warning: failed to load regen actor anchor ({e}); using loaded weights."
                );
                model.actor.clone()
            });
        let critic = model
            .critic
            .clone()
            .try_load_file(folder.join("critic_init.bpk"))
            .unwrap_or_else(|e| {
                eprintln!(
                    "Warning: failed to load regen critic anchor ({e}); using loaded weights."
                );
                model.critic.clone()
            });
        let head = match (
            model.shared_head.clone(),
            folder.join("shared_head_init.bpk").is_file(),
        ) {
            (Some(template), true) => Some(
                template
                    .try_load_file(folder.join("shared_head_init.bpk"))
                    .unwrap_or_else(|e| {
                        eprintln!(
                            "Warning: failed to load regen head anchor ({e}); using loaded weights."
                        );
                        model.shared_head.clone().expect("head just matched")
                    }),
            ),
            (Some(_), false) => {
                eprintln!(
                    "Warning: regen anchor is partial (actor anchor present but shared-head anchor \
                     missing); using loaded head weights for the head."
                );
                model.shared_head.clone()
            }
            (None, _) => None,
        };
        self.init_snapshot = Some(InitSnapshot {
            actor,
            critic,
            head,
        });
    }

    /// Load the optimizer states from a checkpoint folder.
    pub fn load_optimizers(&mut self, folder: &Path) {
        #[cfg(not(feature = "tui"))]
        println!("Loading optimizer states...");

        for (name, optimizer) in [
            ("policy_optimizer", &mut self.policy_optimizer),
            ("value_optimizer", &mut self.value_optimizer),
            ("shared_head_optimizer", &mut self.shared_head_optimizer),
        ] {
            let path = folder.join(format!("{name}.bpk"));
            let legacy = folder.join(format!("{name}.mpk.gz"));
            if legacy.is_file() && !path.is_file() {
                panic!(
                    "Found legacy optimizer checkpoint at {}. Burn 0.22 cannot load MessagePack optimizer state. Move this file aside to resume from model weights with fresh optimizer state, or use a separate custom optimizer-state migration tool; Burn provides no built-in conversion.",
                    legacy.display()
                );
            }
            if path.is_file() {
                *optimizer = optimizer.clone().load(&path).unwrap_or_else(|error| {
                    panic!(
                        "Failed to load optimizer checkpoint {}: {error}",
                        path.display()
                    )
                });
            }
        }

        #[cfg(not(feature = "tui"))]
        println!("Loaded optimizer states from: {folder:?}");
    }
}

impl Ppo {
    pub fn learn<R: Rng>(
        &mut self,
        mut net: Actic,
        memory: &Memory,
        rng: &mut R,
        metrics: &mut Report,
        stats: &mut Stats,
        is_first_iteration: bool,
    ) -> (Actic, usize) {
        // Overbatching uses the complete bounded rollout, including the final
        // trajectory prefix that extends beyond `timesteps_per_iteration`.
        let rollout_size =
            if memory.len() > self.config.timesteps_per_iteration && self.config.overbatching {
                // Train on the complete collected rollout instead of cutting it back
                // to the nominal budget.
                memory.len()
            } else {
                self.config.timesteps_per_iteration
            };

        let memory_indices = (0..rollout_size).collect::<Vec<_>>();

        // Fail loudly with row counts when collection returns a short
        // rollout. Without this, the first batching helper below panics with
        // a bare index-out-of-bounds that names no buffer.
        assert!(
            memory.len() >= rollout_size,
            "Learner memory holds {} rows but the rollout needs {rollout_size}",
            memory.len()
        );
        memory
            .validate()
            .unwrap_or_else(|error| panic!("Invalid learner memory: {error}"));

        // Fix the regen anchor to the pre-update weights (not post-step).
        // Preserved across checkpoints via save/load; legacy checkpoints warn
        // and fall back to the loaded weights.
        {
            let had_snapshot = self.has_regen_snapshot();
            self.ensure_regen_snapshot(&net, self.config.regen);
            if self.config.regen.is_some() && !had_snapshot && stats.cumulative_model_updates > 0 {
                eprintln!(
                    "Warning: regen anchor missing (legacy checkpoint); anchoring to the loaded \
                     weights, not the run's original initialization."
                );
            }
        }

        // Snapshot parameters before training for update-magnitude computation.
        // This downloads all weights to the CPU, so it is timed separately.
        let snapshot_start = Instant::now();
        let actor_params_before = flatten_net(&net.actor);
        let critic_params_before = flatten_net(&net.critic);
        metrics["PPO/param snapshot time"] = snapshot_start.elapsed().as_secs_f64().into();

        // Upload ordered states once as `f16` and keep them resident on the
        // GPU. Value inference slices them with `narrow`, training gathers
        // batches with `select`; each slice is upcast to `f32` on the GPU
        // before running the model. States cross the bus a single time
        // instead of once per phase, at half the bytes. Timed as staging
        // work (including the parallel `f32` to `f16` conversion).
        let resident_start = Instant::now();
        let resident_states = get_states_batch_range(
            memory.states(),
            memory.state_width(),
            0,
            rollout_size,
            &self.device,
        );
        let resident_secs = resident_start.elapsed().as_secs_f64();

        // Compute old critic values for GAE in chunks sliced from the
        // resident states: no upload per chunk, and no second shared-head
        // pass anywhere else. Chunking bounds forward activation memory.
        let value_inference_start = Instant::now();
        let old_values = {
            let nodiff_net = net.valid();
            let mb = self.config.gpu_timestep_buffer_size;
            let n = rollout_size;
            // Queue every chunk forward, then a single transaction
            // readback: one CPU/GPU sync for the whole rollout instead of
            // one per chunk, with no concat kernel. Chunk activations stay
            // bounded (handles drop each iteration) while forwards pipeline.
            let mut transaction = Transaction::default();
            for start in (0..n).step_by(mb) {
                let end = (start + mb).min(n);
                let states = resident_states
                    .clone()
                    .narrow(0, start, end - start)
                    .cast(DType::F32);
                let features = nodiff_net.apply_shared_head(states);
                transaction = transaction.register(nodiff_net.critic.forward(features));
            }

            let mut values = Vec::with_capacity(n);
            for data in transaction.execute() {
                values.extend_from_slice(data.as_slice().unwrap());
            }

            values
        };
        metrics["PPO/value inference time"] = value_inference_start.elapsed().as_secs_f64().into();

        let return_std = if self.config.standardize_returns {
            stats.return_stat.get_std()
        } else {
            1.0
        };

        let terminals = get_batch_1d(memory.terminals(), &memory_indices);

        // Run the critic on truncation next-state observations for the
        // bootstrap in configurable batches on a non-autodiff model clone.
        let trunc_start = Instant::now();
        let trunc_val_preds = {
            // Only truncated rows inside the training window receive a
            // bootstrap prediction. Complete-trajectories mode can push a
            // final trajectory past `timesteps_per_iteration`; its truncated
            // boundary row is excluded from training, so its prediction must
            // be excluded too. Predictions are stored in forward terminal
            // order, so the windowed rows are the first `window_truncations`.
            let window_truncations = terminals
                .iter()
                .filter(|&&terminal| terminal == TerminalState::Truncated)
                .count();
            if window_truncations == 0 {
                Vec::new()
            } else {
                let nodiff_net = net.valid();
                let mb = self.config.truncation_value_batch_size;
                let width = memory.state_width();
                let states = &memory.trunc_next_states()[..window_truncations * width];

                // Same single-sync shape as value inference above.
                let mut transaction = Transaction::default();
                for start in (0..window_truncations).step_by(mb) {
                    let end = (start + mb).min(window_truncations);
                    let batch = get_states_batch_range(states, width, start, end, &self.device)
                        .cast(DType::F32);

                    let features = nodiff_net.apply_shared_head(batch);
                    transaction = transaction.register(nodiff_net.critic.forward(features));
                }

                let mut values = Vec::with_capacity(window_truncations);
                for data in transaction.execute() {
                    values.extend_from_slice(data.as_slice().unwrap());
                }

                values
            }
        };

        metrics["PPO/trunc bootstrap time"] = trunc_start.elapsed().as_secs_f64().into();

        let gae_start = Instant::now();
        let GAEOutput {
            returns,
            target_vals,
            mut advantages,
            rew_clip_portion,
            deltas,
        } = get_gae(
            old_values,
            get_batch_1d(memory.rewards(), &memory_indices),
            &terminals,
            &trunc_val_preds,
            self.config.gamma,
            self.config.lambda,
            return_std,
            self.config.reward_clip_range,
            self.config.gae_estimator,
        );
        metrics["GAE/time"] = gae_start.elapsed().as_secs_f64().into();
        metrics["GAE/reward clip portion"] = rew_clip_portion.into();

        // GAE distribution metrics.
        let n = returns.len().max(1) as f32;
        metrics["GAE/avg return"] =
            ((returns.iter().map(|x| x.abs()).sum::<f32>() / n) as f64).into();
        metrics["GAE/avg advantage"] =
            ((advantages.iter().map(|x| x.abs()).sum::<f32>() / n) as f64).into();
        metrics["GAE/avg val target"] =
            ((target_vals.iter().map(|x| x.abs()).sum::<f32>() / n) as f64).into();

        if self.config.standardize_returns {
            // Randomly sample returns for the running stat.
            let n_to_sample = self
                .config
                .max_returns_per_stats_increment
                .min(returns.len());
            if n_to_sample > 0 {
                for _ in 0..n_to_sample {
                    let idx = rng.next_u32() as usize % returns.len();
                    stats.return_stat.increment(vec![returns[idx]]);
                }
            }
        }

        metrics["GAE/returns STD"] = (stats.return_stat.get_std() as f64).into();

        // Optionally standardize advantages to zero mean and unit variance.
        // Percentile scaling takes precedence when both are configured: the
        // two normalizations compose badly (standardized advantages always
        // have a spread above 1, which would pin the percentile scale).
        if let Some(percentile) = self.config.percentile_adv_scaling {
            let n = advantages.len();
            let mut sorted = advantages.clone();
            sorted.sort_by(f32::total_cmp);
            let quantile = |p: f32| sorted[((p * (n - 1) as f32) as usize).min(n - 1)];
            let spread = quantile(0.95) - quantile(0.05);
            self.adv_scale_ema =
                percentile.ema_decay * self.adv_scale_ema + (1.0 - percentile.ema_decay) * spread;
            metrics["GAE/adv scale"] = f64::from(self.adv_scale_ema).into();
            if self.adv_scale_ema > 1.0 {
                for advantage in &mut advantages {
                    *advantage /= self.adv_scale_ema;
                }
            }
        } else if self.config.standardize_advantages {
            let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
            let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>()
                / advantages.len() as f32;
            let std = var.sqrt().max(f32::EPSILON);

            for a in &mut advantages {
                *a = (*a - mean) / std;
            }
        }

        // Per-row PPO clip bounds from TD-residual coherence. Disabled
        // (`None`) means every row uses the fixed scalar `clip_range`.
        let clip_eps = match self.config.adaptive_clip {
            Some(clip) => {
                assert!(clip.window > 0, "adaptive clip window must be positive");
                let scales = coherence_scales(&deltas, &terminals, clip.window);
                let mean_scale = scales.iter().sum::<f32>() / scales.len().max(1) as f32;
                metrics["PPO/mean clip scale"] = f64::from(mean_scale).into();
                scales
                    .iter()
                    .map(|scale| {
                        self.config.clip_range * clip.min_scale.max(clip.max_scale.min(*scale))
                    })
                    .collect::<Vec<_>>()
            }
            None => vec![self.config.clip_range; advantages.len()],
        };

        // Learning-rate schedule factor for this update (1.0 when Constant).
        // Computed once per `learn` call: every mini-batch in this rollout
        // shares the same factor.
        let lr_scale = self
            .config
            .lr_schedule
            .scale(stats.cumulative_model_updates);

        let mut metric_totals = MetricTotals::new(&self.device);
        let training_start = Instant::now();
        let mut batch_staging_time = Duration::from_secs_f64(resident_secs);
        let mut batch_training_time = Duration::ZERO;

        // Wrap the resident states for the autodiff backend once (no copy:
        // the training graph references the same device storage). Batches
        // gather their states from it on the GPU.
        let resident_b = resident_states;

        // Shuffle the full rollout up front so the RNG stream matches the
        // serial implementation exactly.
        let mut epoch_batches: Vec<Vec<usize>> = Vec::with_capacity(
            rollout_size.div_ceil(self.config.batch_size) * self.config.epochs.max(1),
        );
        for _ in 0..self.config.epochs {
            let mut rollout_order: Vec<usize> = (0..rollout_size).collect();
            rollout_order.shuffle(rng);
            for batch_indices in rollout_order.chunks(self.config.batch_size) {
                epoch_batches.push(batch_indices.to_vec());
            }
        }

        // Double-buffered pipeline: a staging thread fills the small CPU
        // batch buffers (actions, log-probs, advantages, targets, masks)
        // while the learner thread uploads them, gathers states from the
        // resident tensor, and trains. VRAM holds the resident states plus
        // one batch instead of the full rollout plus one batch.
        let epoch_batches_ref = &epoch_batches;
        std::thread::scope(|scope| {
            // Bounded so at most two filled batches queue up: caps queued
            // CPU RAM and keeps the producer from racing pointlessly ahead.
            let (staged_tx, staged_rx) = std::sync::mpsc::sync_channel::<StagedBatch>(2);

            scope.spawn(move || {
                for batch_indices in epoch_batches_ref {
                    let staged = StagedBatch::stage(
                        memory,
                        &advantages,
                        &target_vals,
                        &clip_eps,
                        batch_indices,
                    );
                    if staged_tx.send(staged).is_err() {
                        break;
                    }
                }
            });

            let mut upload_time = Duration::ZERO;
            let mut optim_time = Duration::ZERO;
            let mut primed = false;
            for batch_indices in &epoch_batches {
                let recv_start = Instant::now();
                let staged = staged_rx.recv().expect("staging thread is live");
                if !primed {
                    // Time to the first staged batch: pipeline fill latency.
                    metrics["PPO/pipeline prime time"] = recv_start.elapsed().as_secs_f64().into();
                    primed = true;
                }
                batch_staging_time += Duration::from_secs_f64(staged.staged_secs);
                // Upload and training both run on the learner thread, so both
                // count as training time. Staging work overlaps with them.
                // Upload is timed separately for attribution.
                let training_start = Instant::now();
                let upload_start = Instant::now();
                // One index build plus upload per batch, shared by the state
                // gather; the small fields arrive already gathered.
                let index = Tensor::<1, Int>::from_data(
                    TensorData::new(
                        batch_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                        [batch_indices.len()],
                    ),
                    &self.device,
                );
                let states = resident_b.clone().select(0, index).cast(DType::F32);
                let batch = GpuBatch::from_staged(states, staged, &self.device);
                upload_time += upload_start.elapsed();
                self.train_gpu_batch(
                    &mut net,
                    &batch,
                    &mut metric_totals,
                    &mut optim_time,
                    lr_scale,
                );
                batch_training_time += training_start.elapsed();
            }
            metrics["PPO/upload time"] = upload_time.as_secs_f64().into();
            metrics["PPO/optimizer step time"] = optim_time.as_secs_f64().into();
        });

        metrics["PPO/training time"] = training_start.elapsed().as_secs_f64().into();
        metrics["PPO/batch staging time"] = batch_staging_time.as_secs_f64().into();
        metrics["PPO/batch training time"] = batch_training_time.as_secs_f64().into();

        // Plasticity diagnostics, re-emitted every iteration from the cache
        // so the keys stay present between recomputations. Gated by
        // `plasticity_log_interval` (`None` pays nothing).
        {
            let sample_len = rollout_size.min(2048);
            let sample = resident_b.clone().narrow(0, 0, sample_len).cast(DType::F32);
            self.maybe_log_plasticity(
                stats.cumulative_model_updates,
                &net.actor,
                &net.critic,
                net.shared_head.as_ref(),
                sample,
                metrics,
            );
        }

        stats.cumulative_timesteps += rollout_size as u64;
        stats.cumulative_epochs += self.config.epochs as u64;
        stats.cumulative_model_updates += 1;

        let batch_iters =
            1.max(rollout_size.div_ceil(self.config.batch_size) * self.config.epochs) as f32;

        // Synchronize the accumulated scalar metrics only once after all epochs.
        let [entropy_data, kl_data, clip_data, policy_data, critic_data] = Transaction::default()
            .register(metric_totals.entropy)
            .register(metric_totals.divergence)
            .register(metric_totals.clip_fraction)
            .register(metric_totals.policy_loss)
            .register(metric_totals.value_loss)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let mean_entropy = entropy_data.try_to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_divergence = kl_data.try_to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_clip_fraction = clip_data.try_to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_policy_loss = policy_data.try_to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_val_loss = critic_data.try_to_vec::<f32>().unwrap()[0] / batch_iters;
        assert!(
            !mean_val_loss.is_nan(),
            "Value loss is NaN: {mean_val_loss}"
        );
        let mean_rel_entropy_loss =
            (mean_entropy * self.config.entropy_scale) / mean_policy_loss.abs().max(f32::EPSILON);

        // Compute parameter-update magnitudes (L2 norm of param diff).
        let actor_params_after = flatten_net(&net.actor);
        let critic_params_after = flatten_net(&net.critic);
        let policy_magnitude = l2_diff(&actor_params_before, &actor_params_after);
        let critic_magnitude = l2_diff(&critic_params_before, &critic_params_after);

        metrics["Loss/entropy"] = mean_entropy.into();
        metrics["Loss/KL divergence"] = mean_divergence.into();

        // Loss/magnitude metrics produce bad graph scales on the first iteration
        // because the model is freshly initialised, so we skip them.
        if !is_first_iteration {
            metrics["Loss/policy"] = mean_policy_loss.into();
            metrics["Loss/value"] = mean_val_loss.into();
            metrics["Loss/clip fraction"] = mean_clip_fraction.into();
            metrics["Loss/relative entropy"] = mean_rel_entropy_loss.into();
            metrics["Update/policy magnitude"] = policy_magnitude.into();
            metrics["Update/critic magnitude"] = critic_magnitude.into();
        } else {
            // Always report these even on first iteration.
            metrics["Loss/value"] = mean_val_loss.into();
        }

        (net, rollout_size)
    }

    /// Train one GPU-resident batch, sliced into contiguous mini-batches.
    /// The batch already holds shuffled rows, so each mini-batch is a
    /// zero-copy `narrow` view: no index build, no upload, no gather.
    fn train_gpu_batch(
        &mut self,
        net: &mut Actic,
        batch: &GpuBatch,
        metric_totals: &mut MetricTotals,
        optim_time: &mut Duration,
        lr_scale: f64,
    ) {
        let mut actor_gradients = GradientsAccumulator::new();
        let mut critic_gradients = GradientsAccumulator::new();
        let mut shared_head_gradients = GradientsAccumulator::new();
        let batch_len = batch.len();

        for offset in (0..batch_len).step_by(self.config.mini_batch_size) {
            let mini_batch_len = (offset + self.config.mini_batch_size).min(batch_len) - offset;
            let mini_batch = batch.slice(offset, mini_batch_len);

            let state_batch = mini_batch.states;
            let action_batch = mini_batch.actions;
            let old_log_probs_batch = mini_batch.old_log_probs;
            let advantage_batch = mini_batch.advantages;
            let target_vals_batch = mini_batch.target_vals;
            let mask_batch = mini_batch.action_masks;
            let clip_eps_batch = mini_batch.clip_eps;

            self.train_mini_batch(
                net,
                state_batch,
                action_batch,
                old_log_probs_batch,
                advantage_batch,
                target_vals_batch,
                mask_batch,
                clip_eps_batch,
                mini_batch_len as f32 / batch_len as f32,
                &mut actor_gradients,
                &mut critic_gradients,
                &mut shared_head_gradients,
                metric_totals,
            );
        }

        let optim_start = Instant::now();
        self.step_optimizers(
            net,
            actor_gradients,
            critic_gradients,
            shared_head_gradients,
            lr_scale,
        );
        *optim_time += optim_start.elapsed();
    }

    #[allow(clippy::too_many_arguments)]
    fn train_mini_batch(
        &self,
        net: &Actic,
        state_batch: Tensor<2>,
        action_batch: Tensor<2, Int>,
        old_log_probs_batch: Tensor<2>,
        advantage_batch: Tensor<2>,
        target_vals_batch: Tensor<2>,
        mask_batch: Option<Tensor<2>>,
        clip_eps_batch: Tensor<2>,
        mini_batch_weight: f32,
        actor_gradients: &mut GradientsAccumulator<Net>,
        critic_gradients: &mut GradientsAccumulator<Net>,
        shared_head_gradients: &mut GradientsAccumulator<Net>,
        metric_totals: &mut MetricTotals,
    ) {
        let PPOOutput {
            log_probs: log_prob,
            values: value_batch,
        } = net.forward(state_batch, mask_batch);

        let num_actions = log_prob.shape().dims::<2>()[1];
        let entropy_sum = -(log_prob.clone() * log_prob.clone().exp()).sum_dim(1);
        let entropy_per_sample = if self.config.normalize_entropy {
            entropy_sum / (num_actions as f32).ln()
        } else {
            entropy_sum
        };
        let entropy = entropy_per_sample.mean();

        let action_log_prob = log_prob.gather(1, action_batch.clone());
        let log_prob_diff = action_log_prob - old_log_probs_batch;
        let ratios = log_prob_diff.clone().exp();
        // Per-sample clip bounds (`clip_eps` is the fixed `clip_range` for
        // every row unless adaptive clipping is enabled).
        let lower = clip_eps_batch.clone().mul_scalar(-1.0).add_scalar(1.0);
        let upper = clip_eps_batch.add_scalar(1.0);
        let clipped_ratios = ratios.clone().min_pair(upper).max_pair(lower);
        let kl_mean = ((ratios.clone() - 1.0) - log_prob_diff).mean();
        let clip_fraction = clipped_ratios
            .clone()
            .sub(ratios.clone())
            .abs()
            .greater_elem(0.0)
            .float()
            .mean();
        let actor_loss = -(ratios * advantage_batch.clone())
            .min_pair(clipped_ratios * advantage_batch)
            .mean();

        let ppo_loss = actor_loss.clone() - entropy.clone() * self.config.entropy_scale;
        let critic_loss = MseLoss.forward(value_batch, target_vals_batch, Reduction::Mean);

        let metric_entropy = entropy.detach();
        let metric_kl = kl_mean.detach();
        let metric_clip_fraction = clip_fraction.detach();
        let metric_actor_loss = actor_loss.detach();
        let metric_critic_loss = critic_loss.clone().detach();

        let mut grads = ((ppo_loss + critic_loss) * mini_batch_weight).backward();
        actor_gradients.accumulate(
            &net.actor,
            GradientsParams::from_module(&mut grads, &net.actor),
        );
        critic_gradients.accumulate(
            &net.critic,
            GradientsParams::from_module(&mut grads, &net.critic),
        );
        if let Some(head) = net.shared_head.as_ref() {
            shared_head_gradients.accumulate(head, GradientsParams::from_module(&mut grads, head));
        }

        metric_totals.add_weighted(
            metric_entropy,
            metric_kl,
            metric_clip_fraction,
            metric_actor_loss,
            metric_critic_loss,
            mini_batch_weight,
        );
    }

    fn step_optimizers(
        &mut self,
        net: &mut Actic,
        mut actor_gradients: GradientsAccumulator<Net>,
        mut critic_gradients: GradientsAccumulator<Net>,
        mut shared_head_gradients: GradientsAccumulator<Net>,
        lr_scale: f64,
    ) {
        let base_actor_lr = self
            .config
            .actor_learning_rate
            .unwrap_or(self.config.learning_rate);
        let base_critic_lr = self
            .config
            .critic_learning_rate
            .unwrap_or(self.config.learning_rate);
        let actor_lr = (f64::from(base_actor_lr) * lr_scale) as f32;
        let critic_lr = (f64::from(base_critic_lr) * lr_scale) as f32;
        let shared_head_lr = actor_lr.min(critic_lr);

        net.actor = self.policy_optimizer.step(
            f64::from(actor_lr),
            net.actor.clone(),
            actor_gradients.grads(),
        );
        net.critic = self.value_optimizer.step(
            f64::from(critic_lr),
            net.critic.clone(),
            critic_gradients.grads(),
        );
        if let Some(head) = net.shared_head.take() {
            net.shared_head = Some(self.shared_head_optimizer.step(
                f64::from(shared_head_lr),
                head,
                shared_head_gradients.grads(),
            ));
        }

        self.apply_regen(net, self.config.regen, true);
    }

    /// Compute fresh plasticity diagnostics when `update_index` hits the
    /// configured interval, cache them, and emit the cached values into
    /// `metrics` every call. Re-emitting keeps the keys (and their TUI
    /// section) present on iterations that skip recomputation. Does nothing
    /// unless `plasticity_log_interval` is set.
    fn maybe_log_plasticity(
        &mut self,
        update_index: u64,
        actor: &Net,
        critic: &Net,
        head: Option<&Net>,
        sample_states: Tensor<2>,
        metrics: &mut Report,
    ) {
        let interval = match self.config.plasticity_log_interval {
            Some(interval) if interval > 0 => interval as u64,
            _ => return,
        };
        if update_index.is_multiple_of(interval) {
            self.last_plasticity = Some(plasticity_snapshot(actor, critic, head, sample_states));
        }
        if let Some(snapshot) = self.last_plasticity {
            snapshot.emit(metrics);
        }
    }

    /// Compute fresh policy-only (actor + shared head) diagnostics when `round`
    /// hits the configured interval, cache them, and emit the cached values
    /// into `metrics` every call. Mirrors [`Self::maybe_log_plasticity`] but
    /// never touches the frozen critic: distillation leaves it untrained, so
    /// reporting it would show misleading dormancy. Does nothing unless
    /// `plasticity_log_interval` is set.
    fn maybe_log_transfer_plasticity(
        &mut self,
        round: u64,
        actor: &Net,
        head: Option<&Net>,
        sample_states: Tensor<2>,
        metrics: &mut Report,
    ) {
        let interval = match self.config.plasticity_log_interval {
            Some(interval) if interval > 0 => interval as u64,
            _ => return,
        };
        if round.is_multiple_of(interval) {
            self.last_transfer_plasticity =
                Some(policy_plasticity_snapshot(actor, head, sample_states));
        }
        if let Some(snapshot) = self.last_transfer_plasticity {
            snapshot.emit(metrics);
        }
    }

    /// Pull the configured networks toward the fixed initialization anchor.
    /// No-op when `regen` is `None`. The anchor must already exist via
    /// `ensure_regen_snapshot` (called before the first update); the fallback
    /// below only covers regen enabled mid-run after updates. Distillation
    /// passes its own config and `include_critic: false` to preserve its
    /// critic-untouched invariant.
    fn apply_regen(&mut self, net: &mut Actic, regen: Option<RegenConfig>, include_critic: bool) {
        let Some(regen) = regen else {
            return;
        };
        assert!(
            regen.strength >= 0.0 && regen.strength < 1.0,
            "regen strength must be in [0, 1)"
        );
        assert!(
            regen.noise_std >= 0.0,
            "regen noise_std must be non-negative"
        );
        if self.init_snapshot.is_none() {
            self.init_snapshot = Some(InitSnapshot {
                actor: net.actor.clone(),
                critic: net.critic.clone(),
                head: net.shared_head.clone(),
            });
        }
        let snapshot = self.init_snapshot.as_ref().expect("snapshot just taken");
        let strength = f64::from(regen.strength);
        let noise = f64::from(regen.noise_std);
        net.actor
            .regen_step(&snapshot.actor, strength, noise, &self.device);
        net.actor = net.actor.clone().valid().unfreeze().train();
        if include_critic {
            net.critic
                .regen_step(&snapshot.critic, strength, noise, &self.device);
            net.critic = net.critic.clone().valid().unfreeze().train();
        }
        if let (Some(head), Some(init_head)) = (net.shared_head.as_mut(), snapshot.head.as_ref()) {
            head.regen_step(init_head, strength, noise, &self.device);
            *head = head.clone().valid().unfreeze().train();
        }
    }
}

struct GpuBatch {
    states: Tensor<2>,
    actions: Tensor<2, Int>,
    old_log_probs: Tensor<2>,
    advantages: Tensor<2>,
    target_vals: Tensor<2>,
    action_masks: Option<Tensor<2>>,
    /// Per-row PPO clip bound (fixed `clip_range` unless adaptive clipping).
    clip_eps: Tensor<2>,
}

/// One batch of rollout rows gathered on the CPU staging thread, ready to
/// upload. States are excluded: they stay resident on the GPU and batches
/// gather them with `select`. Every buffer here is allocated once at exact
/// size: the `Vec`s move straight into `TensorData` on upload, so there are
/// no growth reallocs.
struct StagedBatch {
    rows: usize,
    mask_width: usize,
    /// CPU fill time, attributed to batch staging time by the learner thread.
    staged_secs: f64,
    actions: Vec<u32>,
    log_probs: Vec<f32>,
    advantages: Vec<f32>,
    target_vals: Vec<f32>,
    /// Per-row PPO clip bound (fixed `clip_range` unless adaptive clipping).
    clip_eps: Vec<f32>,
    /// `u8` masks (1 = valid), uploaded as-is and upcast on the GPU.
    /// Empty when the env provides no masks.
    masks: Vec<u8>,
}

impl StagedBatch {
    /// Gather one shuffled batch across all CPU cores. Row copies run as
    /// parallel chunk writes; scalar columns use parallel collect. All
    /// reads are disjoint, so no synchronization is needed.
    fn stage(
        memory: &Memory,
        advantages: &[f32],
        target_vals: &[f32],
        clip_eps: &[f32],
        indices: &[usize],
    ) -> Self {
        let fill_start = Instant::now();
        let rows = indices.len();
        let mask_width = memory.action_mask_width();
        let actions = memory.actions();
        let log_probs = memory.log_probs();
        let stored_masks = memory.action_masks();

        let batch_actions = indices
            .par_iter()
            .map(|&index| actions[index] as u32)
            .collect::<Vec<_>>();

        let batch_log_probs = indices
            .par_iter()
            .map(|&index| log_probs[index])
            .collect::<Vec<_>>();

        let batch_advantages = indices
            .par_iter()
            .map(|&index| advantages[index])
            .collect::<Vec<_>>();

        let batch_target_vals = indices
            .par_iter()
            .map(|&index| target_vals[index])
            .collect::<Vec<_>>();

        let batch_clip_eps = indices
            .par_iter()
            .map(|&index| clip_eps[index])
            .collect::<Vec<_>>();

        let mut batch_masks = Vec::new();
        if !stored_masks.is_empty() {
            batch_masks = vec![0u8; rows * mask_width];
            batch_masks
                .par_chunks_mut(mask_width)
                .enumerate()
                .for_each(|(out_row, out)| {
                    let start = indices[out_row] * mask_width;
                    out.copy_from_slice(&stored_masks[start..start + mask_width]);
                });
        }

        Self {
            rows,
            mask_width,
            staged_secs: fill_start.elapsed().as_secs_f64(),
            actions: batch_actions,
            log_probs: batch_log_probs,
            advantages: batch_advantages,
            target_vals: batch_target_vals,
            clip_eps: batch_clip_eps,
            masks: batch_masks,
        }
    }
}

impl GpuBatch {
    /// Combine GPU-gathered states with one staged small-field batch. The
    /// staging `Vec`s move into `TensorData` with no further copy on host.
    fn from_staged(states: Tensor<2>, staged: StagedBatch, device: &Device) -> Self {
        let rows = staged.rows;
        Self {
            states,
            actions: Tensor::from_data(TensorData::new(staged.actions, [rows, 1]), device),
            old_log_probs: Tensor::from_data(TensorData::new(staged.log_probs, [rows, 1]), device),
            advantages: Tensor::from_data(TensorData::new(staged.advantages, [rows, 1]), device),
            target_vals: Tensor::from_data(TensorData::new(staged.target_vals, [rows, 1]), device),
            clip_eps: Tensor::from_data(TensorData::new(staged.clip_eps, [rows, 1]), device),
            action_masks: (!staged.masks.is_empty()).then(|| {
                Tensor::<2, Int>::from_data(
                    TensorData::new(staged.masks, [rows, staged.mask_width]),
                    device,
                )
                .cast(FloatDType::F32)
            }),
        }
    }

    /// Contiguous view of `[offset, offset + len)`: no index tensor, no copy.
    fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            states: self.states.clone().narrow(0, offset, len),
            actions: self.actions.clone().narrow(0, offset, len),
            old_log_probs: self.old_log_probs.clone().narrow(0, offset, len),
            advantages: self.advantages.clone().narrow(0, offset, len),
            target_vals: self.target_vals.clone().narrow(0, offset, len),
            clip_eps: self.clip_eps.clone().narrow(0, offset, len),
            action_masks: self
                .action_masks
                .as_ref()
                .map(|masks| masks.clone().narrow(0, offset, len)),
        }
    }

    fn len(&self) -> usize {
        self.states.shape().dims::<2>()[0]
    }
}

struct MetricTotals {
    entropy: Tensor<1>,
    value_loss: Tensor<1>,
    clip_fraction: Tensor<1>,
    policy_loss: Tensor<1>,
    divergence: Tensor<1>,
}

impl MetricTotals {
    fn new(device: &Device) -> Self {
        Self {
            entropy: Tensor::zeros([1], device),
            value_loss: Tensor::zeros([1], device),
            clip_fraction: Tensor::zeros([1], device),
            policy_loss: Tensor::zeros([1], device),
            divergence: Tensor::zeros([1], device),
        }
    }

    fn add_weighted(
        &mut self,
        entropy: Tensor<1>,
        divergence: Tensor<1>,
        clip_fraction: Tensor<1>,
        policy_loss: Tensor<1>,
        value_loss: Tensor<1>,
        weight: f32,
    ) {
        self.entropy = self.entropy.clone() + entropy * weight;
        self.divergence = self.divergence.clone() + divergence * weight;
        self.clip_fraction = self.clip_fraction.clone() + clip_fraction * weight;
        self.policy_loss = self.policy_loss.clone() + policy_loss * weight;
        self.value_loss = self.value_loss.clone() + value_loss * weight;
    }
}

/// Flatten all trainable parameters of a `Net` into a single `Vec<f32>`.
/// Used to compute the L2 norm of parameter updates across a training iteration.
fn flatten_net(net: &Net) -> Vec<f32> {
    let mut data = Vec::new();
    for layer in net.linear_layers() {
        data.extend(
            layer
                .weight
                .val()
                .clone()
                .into_data()
                .try_into_vec::<f32>()
                .unwrap(),
        );
        if let Some(bias) = &layer.bias {
            data.extend(
                bias.val()
                    .clone()
                    .into_data()
                    .try_into_vec::<f32>()
                    .unwrap(),
            );
        }
    }
    for norm in net.layer_norms() {
        match norm {
            Normalization::Layer(ln) => {
                data.extend(
                    ln.gamma
                        .val()
                        .clone()
                        .into_data()
                        .try_into_vec::<f32>()
                        .unwrap(),
                );
                if let Some(beta) = &ln.beta {
                    data.extend(
                        beta.val()
                            .clone()
                            .into_data()
                            .try_into_vec::<f32>()
                            .unwrap(),
                    );
                }
            }
            Normalization::Rms(rms) => {
                data.extend(
                    rms.gamma
                        .val()
                        .clone()
                        .into_data()
                        .try_into_vec::<f32>()
                        .unwrap(),
                );
                // RmsNorm has no beta parameter.
            }
            _ => {
                // Other normalization variants (Batch, Group, Instance) are not
                // currently used in this codebase, but we skip them gracefully.
            }
        }
    }
    data
}

/// L2 norm of `a - b` (element-wise Euclidean distance).
fn l2_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// L2 norm of a flat parameter vector.
fn l2_norm(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// Per-row TD-residual coherence in `[0, 1]`: the fraction of residuals in a
/// forward window (including the row itself) sharing the row's sign. Windows
/// stop after a trajectory boundary row, so coherence never leaks across
/// episodes. Zero residuals count as agreeing; rows with no nonzero residuals
/// score `0.5`.
fn coherence_scales(deltas: &[f32], terminals: &[TerminalState], window: usize) -> Vec<f32> {
    assert_eq!(
        deltas.len(),
        terminals.len(),
        "deltas and terminals must have the same length"
    );
    let window = window.max(1);
    let n = deltas.len();
    let mut scales = vec![0.5; n];
    for i in 0..n {
        let sign = deltas[i].signum();
        let mut agree = 0usize;
        let mut total = 0usize;
        let mut j = i;
        while j < n && total < window {
            let delta = deltas[j];
            if delta != 0.0 {
                total += 1;
                if sign == 0.0 || delta.signum() == sign {
                    agree += 1;
                }
            }
            let boundary = terminals[j] != TerminalState::None;
            j += 1;
            if boundary {
                break;
            }
        }
        if total > 0 {
            scales[i] = agree as f32 / total as f32;
        }
    }
    scales
}

/// Largest singular value of a row-major `rows x cols` matrix via power
/// iteration (deterministic uniform start, no dependencies).
fn top_singular_value(matrix: &[f32], rows: usize, cols: usize) -> f32 {
    assert_eq!(matrix.len(), rows * cols);
    let mut vec = vec![1.0 / (cols as f32).sqrt(); cols];
    let mut next = vec![0.0; cols.max(rows)];
    for _ in 0..40 {
        // u = W v, normalized.
        for (row, out) in matrix.chunks_exact(cols).take(rows).zip(next.iter_mut()) {
            *out = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        }
        let norm = next[..rows]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for out in next.iter_mut().take(rows) {
            *out /= norm;
        }
        // v = W^T u, normalized.
        for (c, out) in vec.iter_mut().enumerate() {
            *out = matrix
                .iter()
                .skip(c)
                .step_by(cols)
                .take(rows)
                .zip(next.iter())
                .map(|(a, u)| a * u)
                .sum();
        }
        let norm = vec
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            .max(f32::EPSILON);
        for out in vec.iter_mut() {
            *out /= norm;
        }
    }
    // sigma = ||W v|| with the converged v.
    let sigma_sq: f32 = matrix
        .chunks_exact(cols)
        .take(rows)
        .map(|row| {
            let dot: f32 = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
            dot * dot
        })
        .sum();
    sigma_sq.sqrt().max(f32::EPSILON)
}

/// Mean stable rank (`||W||_F^2 / sigma_max^2`) over a `Net`'s hidden linear
/// layers. Low rank (near 1) with many units signals representation collapse.
fn mean_stable_rank(net: &Net) -> f64 {
    let hidden = if net.layer_norms().is_empty() {
        net.linear_layers().len()
    } else {
        net.layer_norms().len()
    };
    let mut rank_sum = 0.0;
    let mut layers = 0usize;
    for layer in net.linear_layers().iter().take(hidden) {
        let data = layer
            .weight
            .val()
            .clone()
            .into_data()
            .try_into_vec::<f32>()
            .unwrap();
        let [rows, cols] = layer.weight.val().shape().dims::<2>();
        let frob_sq: f32 = data.iter().map(|x| x * x).sum();
        let sigma = top_singular_value(&data, rows, cols);
        rank_sum += f64::from(frob_sq / (sigma * sigma).max(f32::EPSILON));
        layers += 1;
    }
    rank_sum / layers.max(1) as f64
}

/// Compute plasticity diagnostics for one actor/critic pair: parameter L2
/// norms, mean hidden-layer stable rank, and the fraction of hidden units
/// whose mean absolute post-ReLU activation falls below `1e-3` (dormant).
/// When a shared feature head is present, `sample_states` are raw
/// observations and are passed through it first so dormancy is measured on
/// each net's true inputs.
fn plasticity_snapshot(
    actor: &Net,
    critic: &Net,
    head: Option<&Net>,
    sample_states: Tensor<2>,
) -> PlasticitySnapshot {
    const DORMANT_THRESHOLD: f32 = 1e-3;

    let actor_flat = flatten_net(actor);
    let critic_flat = flatten_net(critic);
    let actor_norm = l2_norm(&actor_flat);
    let critic_norm = l2_norm(&critic_flat);
    let actor_rank = mean_stable_rank(actor);
    let critic_rank = mean_stable_rank(critic);

    let sample_states = match head {
        Some(head) => head.forward(sample_states),
        None => sample_states,
    };
    let mut dormant = [0.0; 2];
    for (i, net) in [actor, critic].iter().enumerate() {
        let means = net.hidden_dormancy(sample_states.clone());
        dormant[i] = means
            .iter()
            .filter(|mean| **mean < DORMANT_THRESHOLD)
            .count() as f64
            / means.len().max(1) as f64;
    }

    PlasticitySnapshot {
        actor_norm,
        critic_norm,
        actor_rank,
        critic_rank,
        actor_dormant: dormant[0],
        critic_dormant: dormant[1],
    }
}

/// Compute policy-only plasticity diagnostics for transfer learning: actor and
/// shared-head L2 norms, mean hidden-layer stable rank, and dormant-unit
/// fraction. The critic is never read: it stays frozen during distillation,
/// so its dormancy would mislead. `sample_states` are raw observations; the
/// actor is measured on head-transformed features while the head itself is
/// measured on the raw inputs. A missing head yields `None` head fields so
/// [`PolicyPlasticitySnapshot::emit`] skips those keys.
fn policy_plasticity_snapshot(
    actor: &Net,
    head: Option<&Net>,
    sample_states: Tensor<2>,
) -> PolicyPlasticitySnapshot {
    const DORMANT_THRESHOLD: f32 = 1e-3;

    let actor_norm = l2_norm(&flatten_net(actor));
    let actor_rank = mean_stable_rank(actor);

    let (head_norm, head_rank, head_dormant) = match head {
        Some(head) => {
            let norm = l2_norm(&flatten_net(head));
            let rank = mean_stable_rank(head);
            let means = head.hidden_dormancy(sample_states.clone());
            let dormant = means
                .iter()
                .filter(|mean| **mean < DORMANT_THRESHOLD)
                .count() as f64
                / means.len().max(1) as f64;
            (Some(norm), Some(rank), Some(dormant))
        }
        None => (None, None, None),
    };

    let actor_inputs = match head {
        Some(head) => head.forward(sample_states),
        None => sample_states,
    };
    let actor_means = actor.hidden_dormancy(actor_inputs);
    let actor_dormant = actor_means
        .iter()
        .filter(|mean| **mean < DORMANT_THRESHOLD)
        .count() as f64
        / actor_means.len().max(1) as f64;

    PolicyPlasticitySnapshot {
        actor_norm,
        actor_rank,
        actor_dormant,
        head_norm,
        head_rank,
        head_dormant,
    }
}

#[cfg(all(test, any(feature = "flex", feature = "torch")))]
mod runtime_device_tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use burn::optim::AdamWConfig;

    use super::*;

    fn gradients(
        net: &Actic,
        device: &Device,
    ) -> (
        GradientsAccumulator<Net>,
        GradientsAccumulator<Net>,
        GradientsAccumulator<Net>,
    ) {
        let output = net.forward(Tensor::<2>::ones([2, 3], device), None);
        let mut gradients = (output.log_probs.sum() + output.values.sum()).backward();
        let mut actor_gradients = GradientsAccumulator::new();
        let mut critic_gradients = GradientsAccumulator::new();
        let mut shared_head_gradients = GradientsAccumulator::new();
        actor_gradients.accumulate(
            &net.actor,
            GradientsParams::from_module(&mut gradients, &net.actor),
        );
        critic_gradients.accumulate(
            &net.critic,
            GradientsParams::from_module(&mut gradients, &net.critic),
        );
        if let Some(head) = net.shared_head.as_ref() {
            shared_head_gradients
                .accumulate(head, GradientsParams::from_module(&mut gradients, head));
        }
        (actor_gradients, critic_gradients, shared_head_gradients)
    }

    fn exercise_full_learn(device: Device) {
        let config = PpoLearnerConfig {
            epochs: 1,
            timesteps_per_iteration: 4,
            batch_size: 4,
            mini_batch_size: 2,
            gpu_timestep_buffer_size: 4,
            truncation_value_batch_size: 1,
            standardize_returns: false,
            standardize_advantages: false,
            ..Default::default()
        };
        let mut net = Actic::new(3, 2, vec![4], vec![4], &[4], &device, None);
        let actor_before = flatten_net(&net.actor);
        let critic_before = flatten_net(&net.critic);
        let mut memory = Memory::with_capacity(4);
        memory.push_player(
            vec![
                0.1, 0.2, 0.3, 0.2, 0.1, -0.1, -0.2, 0.3, 0.1, 0.4, -0.1, 0.2,
            ],
            3,
            vec![0, 0, 0, 1],
            vec![-0.69; 4],
            vec![0.2, -0.1, 0.4, 1.0],
            vec![
                TerminalState::None,
                TerminalState::Truncated,
                TerminalState::None,
                TerminalState::Normal,
            ],
            vec![1, 1, 1, 0, 1, 1, 1, 1],
            2,
            Vec::new(),
            0,
            Some(vec![0.3, 0.2, -0.1]),
        );

        let mut ppo = Ppo::new(config, device, || AdamWConfig::new().init());
        let mut rng = rand::rng();
        let mut metrics = Report::default();
        let mut stats = Stats::default();
        for iteration in 0..2 {
            (net, _) = ppo.learn(
                net,
                &memory,
                &mut rng,
                &mut metrics,
                &mut stats,
                iteration == 0,
            );
            let flat_metrics = metrics.to_flat_map();
            assert!(
                flat_metrics.values().all(|value| value.is_finite()),
                "iteration {iteration} emitted non-finite metrics: {flat_metrics:?}"
            );
        }
        assert_ne!(
            flatten_net(&net.actor),
            actor_before,
            "actor did not update"
        );
        assert_ne!(
            flatten_net(&net.critic),
            critic_before,
            "critic did not update"
        );
        assert_eq!(stats.cumulative_model_updates, 2);
        assert_eq!(stats.cumulative_epochs, 2);
        assert_eq!(stats.cumulative_timesteps, 8);
    }

    fn exercise_regen_and_optimizer_state_round_trip(device: Device) {
        let mut net = Actic::new(3, 2, vec![4], vec![4], &[4], &device, None);
        let actor_before = flatten_net(&net.actor);
        let param_ids = net
            .actor
            .linear_layers()
            .iter()
            .map(|layer| layer.weight.id)
            .collect::<Vec<_>>();
        let regen = RegenConfig {
            strength: 0.05,
            noise_std: 0.0,
        };
        let mut ppo = Ppo::new(
            PpoLearnerConfig {
                regen: Some(regen),
                ..Default::default()
            },
            device.clone(),
            || AdamWConfig::new().init(),
        );
        ppo.ensure_regen_snapshot(&net, Some(regen));

        let mut previous_weights = actor_before;
        for update in 1..=2 {
            let (actor_gradients, critic_gradients, shared_head_gradients) =
                gradients(&net, &device);
            ppo.step_optimizers(
                &mut net,
                actor_gradients,
                critic_gradients,
                shared_head_gradients,
                1.0,
            );

            let actor_after = flatten_net(&net.actor);
            assert_ne!(
                actor_after, previous_weights,
                "update {update} changed no actor weights"
            );
            assert!(
                net.actor
                    .linear_layers()
                    .iter()
                    .all(|layer| layer.weight.val().is_require_grad()),
                "regen left actor parameters without gradient tracking after update {update}"
            );
            assert_eq!(
                net.actor
                    .linear_layers()
                    .iter()
                    .map(|layer| layer.weight.id)
                    .collect::<Vec<_>>(),
                param_ids,
                "regen changed actor parameter IDs"
            );
            previous_weights = actor_after;
        }

        let folder = std::env::temp_dir().join(format!(
            "rlgymppo-optimizer-bpk-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&folder).unwrap();
        net.actor
            .clone()
            .save_file(folder.join("actor.bpk"))
            .unwrap();
        net.critic
            .clone()
            .save_file(folder.join("critic.bpk"))
            .unwrap();
        net.shared_head
            .as_ref()
            .unwrap()
            .clone()
            .save_file(folder.join("shared_head.bpk"))
            .unwrap();
        ppo.save_optimizers(&folder);
        ppo.save_regen_snapshot(&folder);

        let mut resumed_net = Actic::new(3, 2, vec![4], vec![4], &[4], &device, None);
        resumed_net.actor = resumed_net
            .actor
            .clone()
            .try_load_file(folder.join("actor.bpk"))
            .unwrap();
        resumed_net.critic = resumed_net
            .critic
            .clone()
            .try_load_file(folder.join("critic.bpk"))
            .unwrap();
        resumed_net.shared_head = Some(
            resumed_net
                .shared_head
                .as_ref()
                .unwrap()
                .clone()
                .try_load_file(folder.join("shared_head.bpk"))
                .unwrap(),
        );
        let mut resumed = Ppo::new(
            PpoLearnerConfig {
                regen: Some(regen),
                ..Default::default()
            },
            device.clone(),
            || AdamWConfig::new().init(),
        );
        resumed.reinit_optimizers(&resumed_net);
        resumed.load_optimizers(&folder);
        resumed.load_regen_snapshot(&folder, &resumed_net);
        assert!(
            resumed.has_regen_snapshot(),
            "regen anchor was not restored"
        );

        let (actor_gradients, critic_gradients, shared_head_gradients) = gradients(&net, &device);
        ppo.step_optimizers(
            &mut net,
            actor_gradients,
            critic_gradients,
            shared_head_gradients,
            1.0,
        );
        let (actor_gradients, critic_gradients, shared_head_gradients) =
            gradients(&resumed_net, &device);
        resumed.step_optimizers(
            &mut resumed_net,
            actor_gradients,
            critic_gradients,
            shared_head_gradients,
            1.0,
        );

        for (component, original, restored) in [
            (
                "actor",
                flatten_net(&net.actor),
                flatten_net(&resumed_net.actor),
            ),
            (
                "critic",
                flatten_net(&net.critic),
                flatten_net(&resumed_net.critic),
            ),
            (
                "shared head",
                flatten_net(net.shared_head.as_ref().unwrap()),
                flatten_net(resumed_net.shared_head.as_ref().unwrap()),
            ),
        ] {
            assert_eq!(original.len(), restored.len());
            for (index, (original, restored)) in original.iter().zip(&restored).enumerate() {
                assert!(
                    (original - restored).abs() <= 1e-6,
                    "{component} weight {index} differs after resumed update: {original} vs {restored}"
                );
            }
        }
        std::fs::remove_dir_all(folder).unwrap();
    }

    #[cfg(feature = "flex")]
    #[test]
    fn flex_regen_keeps_parameters_trainable_across_updates_and_optimizer_state_round_trips() {
        let device = Device::flex().autodiff();
        exercise_regen_and_optimizer_state_round_trip(device.clone());
        exercise_full_learn(device);
    }

    #[cfg(feature = "torch")]
    #[test]
    #[allow(deprecated)]
    fn torch_regen_keeps_parameters_trainable_across_updates_and_optimizer_state_round_trips() {
        let device = Device::libtorch().autodiff();
        exercise_regen_and_optimizer_state_round_trip(device.clone());
        exercise_full_learn(device);
    }
}
