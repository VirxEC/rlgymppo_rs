pub mod config;
pub mod model;
pub mod self_play;
pub mod skill_tracker;

use std::path::Path;
use std::time::Instant;

use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::nn::modules::norm::Normalization;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, GradientsParams, Optimizer, SimpleOptimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder, Recorder, RecorderError};
use burn::tensor::Transaction;
use burn::tensor::backend::AutodiffBackend;
use rand::Rng;
use rand::seq::SliceRandom;
use ringbuffer::RingBuffer;

use crate::agent::config::PpoLearnerConfig;
use crate::agent::model::{Actic, Net, PPOOutput};
use crate::base::{
    Memory, TerminalState, get_action_batch, get_action_masks_batch, get_batch_1d,
    get_generic_batch, get_log_probs_batch, get_states_batch,
};
use crate::utils::Report;
use crate::utils::running_stat::Stats;

pub struct Ppo<B: AutodiffBackend, O: SimpleOptimizer<B::InnerBackend> = Adam> {
    config: PpoLearnerConfig,
    policy_optimizer: OptimizerAdaptor<O, Net<B>, B>,
    value_optimizer: OptimizerAdaptor<O, Net<B>, B>,
    shared_head_optimizer: OptimizerAdaptor<O, Net<B>, B>,
    device: B::Device,
}

impl<B: AutodiffBackend, O: SimpleOptimizer<B::InnerBackend>> Ppo<B, O> {
    pub fn new(
        config: PpoLearnerConfig,
        device: B::Device,
        make_optim: impl Fn() -> OptimizerAdaptor<O, Net<B>, B>,
    ) -> Self {
        Self {
            policy_optimizer: make_optim(),
            value_optimizer: make_optim(),
            shared_head_optimizer: make_optim(),
            config,
            device,
        }
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

        let try_load_optim = |name: &str,
                              target: &mut OptimizerAdaptor<O, Net<B>, B>|
         -> Result<(), RecorderError> {
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

impl<B: AutodiffBackend, O: SimpleOptimizer<B::InnerBackend>> Ppo<B, O> {
    pub fn learn<R: Rng>(
        &mut self,
        mut net: Actic<B>,
        memory: &Memory,
        rng: &mut R,
        metrics: &mut Report,
        stats: &mut Stats,
        is_first_iteration: bool,
    ) -> (Actic<B>, usize) {
        // When overbatching, use ALL collected experience (the memory may have
        // slightly more entries than config.batch_size due to the ceil division
        // in thread_sim, or due to truncation next-states).
        let effective_batch_size =
            if self.config.overbatching && memory.len() > self.config.batch_size {
                memory.len()
            } else {
                self.config.batch_size
            };

        let mut memory_indices = (0..effective_batch_size).collect::<Vec<_>>();

        // Snapshot parameters before training for update-magnitude computation.
        let actor_params_before = flatten_net(&net.actor);
        let critic_params_before = flatten_net(&net.critic);

        // Compute old critic values for GAE in mini-batches using a
        // non-autodiff model clone so no gradient graph accumulates.
        let old_values = {
            let nodiff_net = net.valid();
            let mb = self.config.mini_batch_size;
            let n = effective_batch_size;
            let mut values = Vec::with_capacity(n);
            for start in (0..n).step_by(mb) {
                let end = (start + mb).min(n);
                let slice = &memory_indices[start..end];
                let states =
                    get_states_batch::<B::InnerBackend>(memory.states(), slice, &self.device);
                let features = nodiff_net.apply_shared_head(states);
                let batch_vals = nodiff_net
                    .critic
                    .forward(features)
                    .into_data()
                    .into_vec::<f32>()
                    .unwrap();
                values.extend(batch_vals);
            }
            values
        };

        let return_std = if self.config.standardize_returns {
            stats.return_stat.get_std()
        } else {
            1.0
        };

        let terminals = get_batch_1d(memory.terminals(), &memory_indices);

        // Run the critic on truncation next-state observations for the
        // bootstrap, mini-batched on a non-autodiff model clone.
        let trunc_val_preds = {
            let nodiff_net = net.valid();
            if memory.trunc_next_states().is_empty() {
                Vec::new()
            } else {
                let mb = self.config.mini_batch_size;
                let n = memory.trunc_next_states().len();
                let mut values = Vec::with_capacity(n);
                for start in (0..n).step_by(mb) {
                    let end = (start + mb).min(n);
                    let indices: Vec<usize> = (start..end).collect();
                    let batch = get_states_batch::<B::InnerBackend>(
                        memory.trunc_next_states(),
                        &indices,
                        &self.device,
                    );
                    let features = nodiff_net.apply_shared_head(batch);
                    let batch_vals = nodiff_net
                        .critic
                        .forward(features)
                        .into_data()
                        .into_vec::<f32>()
                        .unwrap();
                    values.extend(batch_vals);
                }
                values
            }
        };

        let gae_start = Instant::now();
        let GAEOutput {
            returns,
            target_vals,
            advantages,
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

        let mut mean_entropy = 0.0;
        let mut mean_val_loss = 0.0;
        let mut mean_clip_fraction = 0.0;
        let mut mean_rel_entropy_loss = 0.0;
        let mut mean_policy_loss = 0.0;
        let mut mean_divergence = 0.0;

        for _ in 0..self.config.epochs {
            memory_indices.shuffle(rng);

            for start_idx in (0..effective_batch_size).step_by(self.config.mini_batch_size) {
                let end_idx = (start_idx + self.config.mini_batch_size).min(effective_batch_size);
                let sample_indices = &memory_indices[start_idx..end_idx];

                let state_batch = get_states_batch(memory.states(), sample_indices, &self.device);
                let action_batch = get_action_batch(memory.actions(), sample_indices, &self.device);
                let old_log_probs_batch =
                    get_log_probs_batch(memory.log_probs(), sample_indices, &self.device);
                let advantage_batch = get_generic_batch(&advantages, sample_indices, &self.device);
                let target_vals_batch =
                    get_generic_batch(&target_vals, sample_indices, &self.device);

                // Build action mask tensor for this mini-batch.
                let mask_batch = (!memory.action_masks().is_empty()).then(|| {
                    get_action_masks_batch(memory.action_masks(), sample_indices, &self.device)
                });

                let PPOOutput {
                    policies: policy_batch,
                    values: value_batch,
                } = net.forward(state_batch, mask_batch);

                let log_prob = policy_batch.clone().log();

                // Per-sample entropy normalised by ln(n_actions).
                let num_actions = policy_batch.shape().dims::<2>()[1];
                let entropy_per_sample =
                    -(log_prob.clone() * policy_batch).sum_dim(1) / (num_actions as f32).ln();
                let entropy = entropy_per_sample.mean();

                let action_log_prob = log_prob.gather(1, action_batch.clone());
                let log_prob_diff = action_log_prob - old_log_probs_batch;
                let ratios = log_prob_diff.clone().exp();

                let clipped_ratios = ratios
                    .clone()
                    .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                let kl_tensor = (ratios.clone() - 1.0) - log_prob_diff;
                let kl_mean = kl_tensor.mean();

                let clip_fraction = (ratios.clone() - 1.0)
                    .abs()
                    .greater_elem(self.config.clip_range)
                    .float()
                    .mean();

                let actor_loss = -(ratios * advantage_batch.clone())
                    .min_pair(clipped_ratios * advantage_batch)
                    .mean();

                let batch_size_f = self.config.batch_size as f32;
                let mb_size_f = (end_idx - start_idx) as f32;
                let ratio = mb_size_f / batch_size_f;
                let lr = self.config.learning_rate.into();

                let ppo_loss = actor_loss.clone() - entropy.clone() * self.config.entropy_scale;
                let critic_loss = MseLoss.forward(value_batch, target_vals_batch, Reduction::Mean);

                // Batch all metric syncs into a single device transaction.
                let [entropy_data, kl_data, clip_data, policy_data, critic_data] =
                    Transaction::default()
                        .register(entropy.detach())
                        .register(kl_mean.detach())
                        .register(clip_fraction)
                        .register(actor_loss.detach())
                        .register(critic_loss.clone().detach())
                        .execute()
                        .try_into()
                        .expect("Correct amount of tensor data");

                let cur_entropy = entropy_data.to_vec::<f32>().unwrap()[0];
                mean_entropy += cur_entropy;
                mean_divergence += kl_data.to_vec::<f32>().unwrap()[0];
                mean_clip_fraction += clip_data.to_vec::<f32>().unwrap()[0];

                let cur_policy_loss = policy_data.to_vec::<f32>().unwrap()[0];
                mean_policy_loss += cur_policy_loss;

                mean_rel_entropy_loss += (cur_entropy * self.config.entropy_scale)
                    / cur_policy_loss.abs().max(f32::EPSILON);

                let loss = critic_data.to_vec::<f32>().unwrap()[0];
                assert!(!loss.is_nan(), "Value loss is NaN: {loss}");
                mean_val_loss += loss;

                let combined = (ppo_loss + critic_loss) * ratio;
                let mut grads = combined.backward();

                let actor_grads = GradientsParams::from_module(&mut grads, &net.actor);
                net.actor = self.policy_optimizer.step(lr, net.actor, actor_grads);

                let critic_grads = GradientsParams::from_module(&mut grads, &net.critic);
                net.critic = self.value_optimizer.step(lr, net.critic, critic_grads);

                if let Some(head) = net.shared_head.take() {
                    let head_grads = GradientsParams::from_module(&mut grads, &head);
                    net.shared_head = Some(self.shared_head_optimizer.step(lr, head, head_grads));
                }
            }
        }

        stats.cumulative_timesteps += effective_batch_size as u64;
        stats.cumulative_epochs += self.config.epochs as u64;
        stats.cumulative_model_updates += 1;

        let mini_batch_iters = 1
            .max(effective_batch_size.div_ceil(self.config.mini_batch_size) * self.config.epochs)
            as f32;

        mean_policy_loss /= mini_batch_iters;
        mean_val_loss /= mini_batch_iters;
        mean_entropy /= mini_batch_iters;
        mean_clip_fraction /= mini_batch_iters;
        mean_rel_entropy_loss /= mini_batch_iters;
        mean_divergence /= mini_batch_iters;

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

        (net, effective_batch_size)
    }
}

struct GAEOutput {
    returns: Vec<f32>,
    target_vals: Vec<f32>,
    advantages: Vec<f32>,
    rew_clip_portion: f32,
}

#[allow(clippy::too_many_arguments)]
fn get_gae(
    values: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<TerminalState>,
    trunc_val_preds: &[f32],
    gamma: f32,
    lambda: f32,
    return_std: f32,
    clip_range: f32,
) -> GAEOutput {
    let n_returns = values.len();
    let mut returns = vec![0.0; n_returns];
    let mut target_vals = returns.clone();
    let mut advantages = returns.clone();

    let mut return_scale = 1.0 / return_std;
    if return_scale.is_nan() || return_scale == 0.0 {
        return_scale = 1.0;
    }

    let mut last_return = 0.0;
    let mut running_advantage = 0.0;
    let mut trunc_count = 0_usize;

    let mut total_reward = 0.0_f32;
    let mut total_clipped_reward = 0.0_f32;

    for i in (0..n_returns).rev() {
        let term = terminals[i];
        let is_done = term == TerminalState::Normal;
        let is_trunc = term == TerminalState::Truncated;

        let not_done = f32::from(!is_done);
        let not_trunc = f32::from(!is_trunc);

        // Normalize reward.
        let mut norm_reward = rewards[i] * return_scale;
        total_reward += norm_reward.abs();

        if clip_range > 0.0 {
            norm_reward = norm_reward.clamp(-clip_range, clip_range);
        }
        total_clipped_reward += norm_reward.abs();

        // Determine next-value prediction:
        //   - TRUNCATED  → pull from `trunc_val_preds`
        //   - NOT_TERMINAL → pull from `values[i + 1]`
        //   - NORMAL / end-of-buffer → 0.0 (the `(1-done)` multiplier zeros it anyway)
        let next_val = if is_trunc {
            let v = trunc_val_preds[trunc_count];
            trunc_count += 1;
            v
        } else if !is_done && i + 1 < n_returns {
            values[i + 1]
        } else {
            0.0
        };

        let pred_ret = norm_reward + gamma * next_val * not_done;
        let delta = pred_ret - values[i];

        last_return = rewards[i] + last_return * gamma * not_done * not_trunc;
        returns[i] = last_return;

        running_advantage = delta + gamma * lambda * not_done * not_trunc * running_advantage;
        advantages[i] = running_advantage;

        target_vals[i] = values[i] + running_advantage;
    }

    let rew_clip_portion = (total_reward - total_clipped_reward) / total_reward.max(f32::EPSILON);
    GAEOutput {
        returns,
        target_vals,
        advantages,
        rew_clip_portion,
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
