pub mod config;
mod gae;
pub mod model;
pub mod self_play;
pub mod skill_tracker;
pub mod transfer_learn;

use std::path::Path;
use std::time::{Duration, Instant};

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::modules::norm::Normalization;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, GradientsAccumulator, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder, Recorder, RecorderError};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{DType, FloatDType, Transaction};
use rand::Rng;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use rlgymppo_utils::Report;

use crate::OptimizerNetwork;
use crate::agent::config::PpoLearnerConfig;
use crate::agent::gae::{GAEOutput, get_gae};
use crate::agent::model::{Actic, Net, PPOOutput};
use crate::base::{Memory, TerminalState, get_batch_1d, get_states_batch_range};
use crate::utils::running_stat::Stats;

pub struct Ppo<B: AutodiffBackend, O: Optimizer<Net<B>, B> = OptimizerAdaptor<AdamW, Net<B>, B>> {
    config: PpoLearnerConfig,
    policy_optimizer: O,
    value_optimizer: O,
    shared_head_optimizer: O,
    /// The model-aware factory is retained so optimizers can be rebuilt after
    /// loading a checkpoint with freshly-created model parameters.
    make_optim: Option<Box<dyn Fn(OptimizerNetwork, &Net<B>) -> O>>,
    device: B::Device,
}

impl<B: AutodiffBackend, O: Optimizer<Net<B>, B>> Ppo<B, O> {
    pub fn new(config: PpoLearnerConfig, device: B::Device, make_optim: impl Fn() -> O) -> Self {
        Self {
            policy_optimizer: make_optim(),
            value_optimizer: make_optim(),
            shared_head_optimizer: make_optim(),
            make_optim: None,
            config,
            device,
        }
    }

    pub fn new_with_model(
        config: PpoLearnerConfig,
        device: B::Device,
        model: &Actic<B>,
        make_optim: impl Fn(OptimizerNetwork, &Net<B>) -> O + 'static,
    ) -> Self {
        let make_optim: Box<dyn Fn(OptimizerNetwork, &Net<B>) -> O> = Box::new(make_optim);
        let shared_head = model.shared_head.as_ref().unwrap_or(&model.actor);
        Self {
            policy_optimizer: make_optim(OptimizerNetwork::Policy, &model.actor),
            value_optimizer: make_optim(OptimizerNetwork::Value, &model.critic),
            shared_head_optimizer: make_optim(OptimizerNetwork::SharedHead, shared_head),
            make_optim: Some(make_optim),
            config,
            device,
        }
    }

    /// Recreate model-aware optimizers for a newly-loaded model.
    pub fn reinit_optimizers(&mut self, model: &Actic<B>) {
        let Some(make_optim) = self.make_optim.as_ref() else {
            return;
        };

        let shared_head = model.shared_head.as_ref().unwrap_or(&model.actor);
        self.policy_optimizer = make_optim(OptimizerNetwork::Policy, &model.actor);
        self.value_optimizer = make_optim(OptimizerNetwork::Value, &model.critic);
        self.shared_head_optimizer = make_optim(OptimizerNetwork::SharedHead, shared_head);
    }

    /// Save the optimizer states (momentum/velocity buffers) to a checkpoint folder.
    pub fn save_optimizers(&self, folder: &Path) {
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        #[cfg(not(feature = "tui"))]
        println!("Saving optimizer states...");

        recorder
            .record(
                self.policy_optimizer.to_record(),
                folder.join("policy_optimizer"),
            )
            .unwrap();
        recorder
            .record(
                self.value_optimizer.to_record(),
                folder.join("value_optimizer"),
            )
            .unwrap();
        recorder
            .record(
                self.shared_head_optimizer.to_record(),
                folder.join("shared_head_optimizer"),
            )
            .unwrap();

        #[cfg(not(feature = "tui"))]
        println!("Saved optimizer states to: {folder:?}");
    }

    /// Load the optimizer states from a checkpoint folder.
    pub fn load_optimizers(&mut self, folder: &Path) {
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        #[cfg(not(feature = "tui"))]
        println!("Loading optimizer states...");

        let try_load_optim = |name: &str, target: &mut O| -> Result<(), RecorderError> {
            let record = recorder.load(folder.join(name), &self.device)?;
            *target = target.clone().load_record(record);
            Ok(())
        };

        let _ =
            try_load_optim("policy_optimizer", &mut self.policy_optimizer).inspect_err(
                |e| match e {
                    RecorderError::FileNotFound(_) => {}
                    e => panic!("Failed to load policy optimizer: {e}"),
                },
            );
        let _ =
            try_load_optim("value_optimizer", &mut self.value_optimizer).inspect_err(|e| match e {
                RecorderError::FileNotFound(_) => {}
                e => panic!("Failed to load value optimizer: {e}"),
            });
        let _ = try_load_optim("shared_head_optimizer", &mut self.shared_head_optimizer)
            .inspect_err(|e| match e {
                RecorderError::FileNotFound(_) => {}
                e => panic!("Failed to load shared-head optimizer: {e}"),
            });

        #[cfg(not(feature = "tui"))]
        println!("Loaded optimizer states from: {folder:?}");
    }
}

impl<B: AutodiffBackend, O: Optimizer<Net<B>, B>> Ppo<B, O> {
    pub fn learn<R: Rng>(
        &mut self,
        mut net: Actic<B>,
        memory: &Memory,
        rng: &mut R,
        metrics: &mut Report,
        stats: &mut Stats,
        is_first_iteration: bool,
    ) -> (Actic<B>, usize) {
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
        let resident_states = get_states_batch_range::<B::InnerBackend>(
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
                    let batch = get_states_batch_range::<B::InnerBackend>(
                        states,
                        width,
                        start,
                        end,
                        &self.device,
                    )
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
        } = get_gae(
            old_values,
            get_batch_1d(memory.rewards(), &memory_indices),
            terminals,
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
        if self.config.standardize_advantages {
            let mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
            let var = advantages.iter().map(|a| (a - mean).powi(2)).sum::<f32>()
                / advantages.len() as f32;
            let std = var.sqrt().max(f32::EPSILON);

            for a in &mut advantages {
                *a = (*a - mean) / std;
            }
        }

        let mut metric_totals = MetricTotals::new(&self.device);
        let training_start = Instant::now();
        let mut batch_staging_time = Duration::from_secs_f64(resident_secs);
        let mut batch_training_time = Duration::ZERO;

        // Wrap the resident states for the autodiff backend once (no copy:
        // the training graph references the same device storage). Batches
        // gather their states from it on the GPU.
        let resident_b = Tensor::<B, 2>::from_inner(resident_states);

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
                    let staged =
                        StagedBatch::stage(memory, &advantages, &target_vals, batch_indices);
                    if staged_tx.send(staged).is_err() {
                        break;
                    }
                }
            });

            let mut upload_time = Duration::ZERO;
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
                let index = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(
                        batch_indices.iter().map(|&i| i as i64).collect::<Vec<_>>(),
                        [batch_indices.len()],
                    ),
                    &self.device,
                );
                let states = resident_b.clone().select(0, index).cast(DType::F32);
                let batch = GpuBatch::from_staged(states, staged, &self.device);
                upload_time += upload_start.elapsed();
                self.train_gpu_batch(&mut net, &batch, &mut metric_totals);
                batch_training_time += training_start.elapsed();
            }
            metrics["PPO/upload time"] = upload_time.as_secs_f64().into();
        });

        metrics["PPO/training time"] = training_start.elapsed().as_secs_f64().into();
        metrics["PPO/batch staging time"] = batch_staging_time.as_secs_f64().into();
        metrics["PPO/batch training time"] = batch_training_time.as_secs_f64().into();

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

        let mean_entropy = entropy_data.to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_divergence = kl_data.to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_clip_fraction = clip_data.to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_policy_loss = policy_data.to_vec::<f32>().unwrap()[0] / batch_iters;
        let mean_val_loss = critic_data.to_vec::<f32>().unwrap()[0] / batch_iters;
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
        net: &mut Actic<B>,
        batch: &GpuBatch<B>,
        metric_totals: &mut MetricTotals<B>,
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

            self.train_mini_batch(
                net,
                state_batch,
                action_batch,
                old_log_probs_batch,
                advantage_batch,
                target_vals_batch,
                mask_batch,
                mini_batch_len as f32 / batch_len as f32,
                &mut actor_gradients,
                &mut critic_gradients,
                &mut shared_head_gradients,
                metric_totals,
            );
        }

        self.step_optimizers(
            net,
            actor_gradients,
            critic_gradients,
            shared_head_gradients,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn train_mini_batch(
        &self,
        net: &Actic<B>,
        state_batch: Tensor<B, 2>,
        action_batch: Tensor<B, 2, Int>,
        old_log_probs_batch: Tensor<B, 2>,
        advantage_batch: Tensor<B, 2>,
        target_vals_batch: Tensor<B, 2>,
        mask_batch: Option<Tensor<B, 2>>,
        mini_batch_weight: f32,
        actor_gradients: &mut GradientsAccumulator<Net<B>>,
        critic_gradients: &mut GradientsAccumulator<Net<B>>,
        shared_head_gradients: &mut GradientsAccumulator<Net<B>>,
        metric_totals: &mut MetricTotals<B>,
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
        let clipped_ratios = ratios
            .clone()
            .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);
        let kl_mean = ((ratios.clone() - 1.0) - log_prob_diff).mean();
        let clip_fraction = (ratios.clone() - 1.0)
            .abs()
            .greater_elem(self.config.clip_range)
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
        net: &mut Actic<B>,
        mut actor_gradients: GradientsAccumulator<Net<B>>,
        mut critic_gradients: GradientsAccumulator<Net<B>>,
        mut shared_head_gradients: GradientsAccumulator<Net<B>>,
    ) {
        let actor_lr = self
            .config
            .actor_learning_rate
            .unwrap_or(self.config.learning_rate);
        let critic_lr = self
            .config
            .critic_learning_rate
            .unwrap_or(self.config.learning_rate);
        let shared_head_lr = actor_lr.min(critic_lr);

        net.actor =
            self.policy_optimizer
                .step(actor_lr.into(), net.actor.clone(), actor_gradients.grads());
        net.critic = self.value_optimizer.step(
            critic_lr.into(),
            net.critic.clone(),
            critic_gradients.grads(),
        );
        if let Some(head) = net.shared_head.take() {
            net.shared_head = Some(self.shared_head_optimizer.step(
                shared_head_lr.into(),
                head,
                shared_head_gradients.grads(),
            ));
        }
    }
}

struct GpuBatch<B: Backend> {
    states: Tensor<B, 2>,
    actions: Tensor<B, 2, Int>,
    old_log_probs: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
    target_vals: Tensor<B, 2>,
    action_masks: Option<Tensor<B, 2>>,
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
    /// `u8` masks (1 = valid), uploaded as-is and upcast on the GPU.
    /// Empty when the env provides no masks.
    masks: Vec<u8>,
}

impl StagedBatch {
    /// Gather one shuffled batch across all CPU cores. Row copies run as
    /// parallel chunk writes; scalar columns use parallel collect. All
    /// reads are disjoint, so no synchronization is needed.
    fn stage(memory: &Memory, advantages: &[f32], target_vals: &[f32], indices: &[usize]) -> Self {
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
            masks: batch_masks,
        }
    }
}

impl<B: Backend> GpuBatch<B> {
    /// Combine GPU-gathered states with one staged small-field batch. The
    /// staging `Vec`s move into `TensorData` with no further copy on host.
    fn from_staged(states: Tensor<B, 2>, staged: StagedBatch, device: &B::Device) -> Self {
        let rows = staged.rows;
        Self {
            states,
            actions: Tensor::from_data(TensorData::new(staged.actions, [rows, 1]), device),
            old_log_probs: Tensor::from_data(TensorData::new(staged.log_probs, [rows, 1]), device),
            advantages: Tensor::from_data(TensorData::new(staged.advantages, [rows, 1]), device),
            target_vals: Tensor::from_data(TensorData::new(staged.target_vals, [rows, 1]), device),
            action_masks: (!staged.masks.is_empty()).then(|| {
                Tensor::<B, 2, Int>::from_data(
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

struct MetricTotals<B: Backend> {
    entropy: Tensor<B, 1>,
    value_loss: Tensor<B, 1>,
    clip_fraction: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    divergence: Tensor<B, 1>,
}

impl<B: Backend> MetricTotals<B> {
    fn new(device: &B::Device) -> Self {
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
        entropy: Tensor<B, 1>,
        divergence: Tensor<B, 1>,
        clip_fraction: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        value_loss: Tensor<B, 1>,
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
fn flatten_net<B: Backend>(net: &Net<B>) -> Vec<f32> {
    let mut data = Vec::new();
    for layer in net.linear_layers() {
        data.extend(
            layer
                .weight
                .val()
                .clone()
                .into_data()
                .into_vec::<f32>()
                .unwrap(),
        );
        if let Some(bias) = &layer.bias {
            data.extend(bias.val().clone().into_data().into_vec::<f32>().unwrap());
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
                        .into_vec::<f32>()
                        .unwrap(),
                );
                if let Some(beta) = &ln.beta {
                    data.extend(beta.val().clone().into_data().into_vec::<f32>().unwrap());
                }
            }
            Normalization::Rms(rms) => {
                data.extend(
                    rms.gamma
                        .val()
                        .clone()
                        .into_data()
                        .into_vec::<f32>()
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
