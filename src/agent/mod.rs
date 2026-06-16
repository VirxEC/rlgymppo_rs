pub mod config;
pub mod model;

use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};
use rand::{Rng, seq::SliceRandom};
use ringbuffer::RingBuffer;

use crate::{
    agent::{
        config::PpoLearnerConfig,
        model::{Actic, Net, PPOOutput},
    },
    base::{
        Memory, TERMINAL_NORMAL, TERMINAL_TRUNCATED, TerminalState, get_action_batch,
        get_action_masks_batch, get_batch_1d, get_generic_batch, get_log_probs_batch,
        get_states_batch,
    },
    utils::{Report, elementwise_min, running_stat::Stats},
};

pub struct Ppo<B: AutodiffBackend> {
    config: PpoLearnerConfig,
    policy_optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    value_optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    shared_head_optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    device: B::Device,
}

impl<B: AutodiffBackend> Ppo<B> {
    pub fn new(config: PpoLearnerConfig, device: B::Device) -> Self {
        let make_optim = || {
            AdamConfig::new()
                .with_epsilon(1e-8)
                .with_grad_clipping(config.clip_grad.clone())
                .init()
        };
        Self {
            policy_optimizer: make_optim(),
            value_optimizer: make_optim(),
            shared_head_optimizer: make_optim(),
            config,
            device,
        }
    }
}

impl<B: AutodiffBackend> Ppo<B> {
    pub fn learn<R: Rng>(
        &mut self,
        mut net: Actic<B>,
        memory: &Memory,
        rng: &mut R,
        metrics: &mut Report,
        stats: &mut Stats,
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

        let states_batch: Tensor<B, 2> =
            get_states_batch(memory.states(), &memory_indices, &self.device);
        let features = net.apply_shared_head(states_batch);
        let old_values_batch = net.critic.forward(features).detach();
        let old_values = old_values_batch.into_data().into_vec::<f32>().unwrap();

        let return_std = if self.config.standardize_returns {
            stats.return_stat.get_std()
        } else {
            1.0
        };

        let terminals = get_batch_1d(memory.terminals(), &memory_indices);

        // Run the critic on truncation next-state observations for the bootstrap.
        let trunc_val_preds = if memory.trunc_next_states().is_empty() {
            Vec::new()
        } else {
            let indices: Vec<usize> = (0..memory.trunc_next_states().len()).collect();
            let batch = get_states_batch(memory.trunc_next_states(), &indices, &self.device);
            let features = net.apply_shared_head(batch);
            net.critic
                .forward(features)
                .detach()
                .into_data()
                .into_vec::<f32>()
                .unwrap()
        };

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

        metrics[".Reward clip portion"] = rew_clip_portion.into();

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

        let mut mean_entropy = 0.0;
        let mut mean_val_loss = 0.0;
        let mut mean_clip_fraction = 0.0;
        // let mut mean_divergence = 0.0;
        // let mut mean_ratio = 0.0;

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
                mean_entropy += entropy.clone().into_scalar().to_f32();

                let action_log_prob = log_prob.gather(1, action_batch.clone());
                let log_prob_diff = action_log_prob - old_log_probs_batch;
                let ratios = log_prob_diff.clone().exp();
                // mean_ratio += ratios.clone().mean().detach().into_scalar().to_f32();

                let clipped_ratios = ratios
                    .clone()
                    .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                {
                    // let kl_tensor = (ratios.clone() - 1.0) - log_prob_diff;
                    // mean_divergence += kl_tensor.mean().detach().into_scalar().to_f32();

                    let clip_fraction = (ratios.clone() - 1.0)
                        .abs()
                        .greater_elem(self.config.clip_range)
                        .float()
                        .mean();
                    mean_clip_fraction += clip_fraction.detach().into_scalar().to_f32();
                }

                let actor_loss = -elementwise_min(
                    ratios * advantage_batch.clone(),
                    clipped_ratios * advantage_batch,
                )
                .mean();

                let batch_size_f = self.config.batch_size as f32;
                let mb_size_f = (end_idx - start_idx) as f32;
                let ratio = mb_size_f / batch_size_f;
                let lr = self.config.learning_rate.into();

                let ppo_loss = actor_loss - entropy * self.config.entropy_scale;
                let critic_loss = MseLoss.forward(value_batch, target_vals_batch, Reduction::Mean);
                let loss = critic_loss.clone().detach().into_scalar().to_f32();
                assert!(!loss.is_nan(), "Value loss is NaN: {loss}");
                mean_val_loss += loss;

                // Combined backward pass (like C++): accumulate gradients for the
                // actor, critic, AND shared head — all through the shared features.
                let combined = (ppo_loss + critic_loss) * ratio;
                let mut grads = combined.backward();

                let actor_grads = GradientsParams::from_module(&mut grads, &net.actor);
                net.actor = self.policy_optimizer.step(lr, net.actor, actor_grads);

                let critic_grads = GradientsParams::from_module(&mut grads, &net.critic);
                net.critic = self.value_optimizer.step(lr, net.critic, critic_grads);

                if net.shared_head.is_some() {
                    let head = net.shared_head.take().unwrap();
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

        mean_val_loss /= mini_batch_iters;
        mean_entropy /= mini_batch_iters;
        mean_clip_fraction /= mini_batch_iters;
        // mean_divergence /= mini_batch_iters;
        // mean_ratio /= mini_batch_iters;

        metrics[".Value loss"] = mean_val_loss.into();
        metrics[".Entropy"] = mean_entropy.into();
        metrics[".Clip fraction"] = mean_clip_fraction.into();
        // metrics[".Divergence"] = mean_divergence.into();
        // metrics[".Ratio"] = mean_ratio.into();

        (net, self.config.batch_size)
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
        let is_done = term == TERMINAL_NORMAL;
        let is_trunc = term == TERMINAL_TRUNCATED;

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
