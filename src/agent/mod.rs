pub mod config;
pub mod model;

use crate::{
    agent::{
        config::PpoLearnerConfig,
        model::{Actic, Net, PPOOutput},
    },
    base::{
        Memory, get_action_batch, get_batch_1d, get_generic_batch, get_log_probs_batch,
        get_states_batch,
    },
    utils::{Report, elementwise_min, running_stat::Stats, update_parameters},
};
use burn::{
    nn::loss::{MseLoss, Reduction},
    optim::{Adam, AdamConfig, adaptor::OptimizerAdaptor},
    prelude::*,
    tensor::{backend::AutodiffBackend, cast::ToElement},
};
use rand::{Rng, seq::SliceRandom};

pub struct Ppo<B: AutodiffBackend> {
    config: PpoLearnerConfig,
    policy_optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    value_optimizer: OptimizerAdaptor<Adam, Net<B>, B>,
    device: B::Device,
}

impl<B: AutodiffBackend> Ppo<B> {
    pub fn new(config: PpoLearnerConfig, device: B::Device) -> Self {
        Self {
            policy_optimizer: AdamConfig::new()
                .with_epsilon(1e-8)
                .with_grad_clipping(config.clip_grad.clone())
                .init(),
            value_optimizer: AdamConfig::new()
                .with_epsilon(1e-8)
                .with_grad_clipping(config.clip_grad.clone())
                .init(),
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
        let mut memory_indices = (0..self.config.batch_size).collect::<Vec<_>>();

        let states_batch: Tensor<B, 2> =
            get_states_batch(memory.states(), &memory_indices, &self.device);
        let old_values_batch = net.critic.infer(states_batch).detach();
        let old_values = old_values_batch.into_data().into_vec::<f32>().unwrap();

        let return_std = if self.config.standardize_returns {
            stats.return_stat.get_std()
        } else {
            1.0
        };

        let GAEOutput {
            returns,
            target_vals,
            advantages,
        } = get_gae(
            old_values,
            get_batch_1d(memory.rewards(), &memory_indices),
            get_batch_1d(memory.dones(), &memory_indices),
            get_batch_1d(memory.truncateds(), &memory_indices),
            self.config.gamma,
            self.config.lambda,
            return_std,
            self.config.reward_clip_range,
            self.config.max_returns_per_stats_increment,
        );

        if self.config.standardize_returns {
            stats.return_stat.increment(returns);
        }

        let mut mean_entropy = 0.0;
        let mut mean_val_loss = 0.0;
        let mut mean_clip_fraction = 0.0;
        // let mut mean_divergence = 0.0;
        // let mut mean_ratio = 0.0;

        let minibatch_ratio = self.config.mini_batch_size as f32 / self.config.batch_size as f32;

        for _ in 0..self.config.epochs {
            memory_indices.shuffle(rng);

            for start_idx in (0..self.config.batch_size).step_by(self.config.mini_batch_size) {
                let end_idx = start_idx + self.config.mini_batch_size;
                let sample_indices = &memory_indices[start_idx..end_idx];

                let state_batch = get_states_batch(memory.states(), sample_indices, &self.device);
                let action_batch = get_action_batch(memory.actions(), sample_indices, &self.device);
                let old_log_probs_batch =
                    get_log_probs_batch(memory.log_probs(), sample_indices, &self.device);
                let advantage_batch = get_generic_batch(&advantages, sample_indices, &self.device);
                let target_vals_batch =
                    get_generic_batch(&target_vals, sample_indices, &self.device);

                let PPOOutput {
                    policies: policy_batch,
                    values: value_batch,
                } = net.forward(state_batch);

                let log_prob = policy_batch.clone().log();
                let entropy = -(log_prob.clone() * policy_batch).sum_dim(1).mean();
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

                let ppo_loss = actor_loss - entropy * self.config.entropy_coeff;
                net.actor = update_parameters(
                    ppo_loss * minibatch_ratio,
                    net.actor,
                    &mut self.policy_optimizer,
                    self.config.learning_rate.into(),
                );

                let critic_loss = MseLoss.forward(value_batch, target_vals_batch, Reduction::Mean);
                let loss = critic_loss.clone().detach().into_scalar().to_f32();
                assert!(!loss.is_nan(), "Value loss is NaN: {loss}");
                mean_val_loss += loss;

                net.critic = update_parameters(
                    critic_loss * minibatch_ratio,
                    net.critic,
                    &mut self.value_optimizer,
                    self.config.learning_rate.into(),
                );
            }
        }

        stats.cumulative_timesteps += self.config.batch_size as u64;
        stats.cumulative_epochs += self.config.epochs as u64;
        stats.cumulative_model_updates += 1;

        let mini_batch_iters =
            1.max(self.config.batch_size / self.config.mini_batch_size * self.config.epochs) as f32;

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
}

#[allow(clippy::too_many_arguments)]
fn get_gae(
    values: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    truncateds: Vec<bool>,
    gamma: f32,
    lambda: f32,
    return_std: f32,
    clip_range: f32,
    max_n_returns: usize,
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
    let mut last_val = 0.0;
    let mut running_advantage = 0.0;

    for i in (0..n_returns).rev() {
        let not_done = f32::from(!dones[i]);
        let not_truncated = f32::from(!truncateds[i]);

        let mut norm_reward = rewards[i] * return_scale;
        if clip_range > 0.0 {
            norm_reward = norm_reward.clamp(-clip_range, clip_range);
        }

        let pred_ret = norm_reward + gamma * last_val * not_done;
        let delta = pred_ret - values[i];
        last_val = values[i];

        last_return = rewards[i] + last_return * gamma * not_done * not_truncated;
        returns[i] = last_return;

        running_advantage = delta + gamma * lambda * not_done * not_truncated * running_advantage;
        advantages[i] = running_advantage;

        target_vals[i] = last_val + running_advantage;
    }

    returns.truncate(max_n_returns);
    GAEOutput {
        returns,
        target_vals,
        advantages,
    }
}
