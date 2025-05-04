pub mod config;
pub mod model;

use crate::{
    agent::{
        config::PpoLearnerConfig,
        model::{Actic, Net, PPOOutput},
    },
    base::{Memory, get_action_batch, get_batch_1d, get_states_batch},
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
                .with_grad_clipping(config.clip_grad.clone())
                .init(),
            value_optimizer: AdamConfig::new()
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
    ) -> Actic<B> {
        let mut memory_indices = (0..memory.len()).collect::<Vec<_>>();
        let PPOOutput {
            policies: mut old_log_probs,
            values: mut old_values,
        } = net.forward(get_states_batch(
            memory.states(),
            &memory_indices,
            &self.device,
        ));
        old_log_probs = old_log_probs.log().detach();
        old_values = old_values.detach();

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
            old_values.into_data().into_vec().unwrap(),
            get_batch_1d(memory.rewards(), &memory_indices),
            get_batch_1d(memory.dones(), &memory_indices),
            get_batch_1d(memory.truncateds(), &memory_indices),
            self.config.gamma,
            self.config.lambda,
            return_std,
            self.config.clip_range,
            self.config.max_returns_per_stats_increment,
            &self.device,
        );

        if self.config.standardize_returns {
            stats.return_stat.increment(returns);
        }

        let mut mean_entropy = 0.0;
        let mut mean_val_loss = 0.0;
        let mut mean_divergence = 0.0;
        let mut mean_clip_fraction = 0.0;

        for _ in 0..self.config.epochs {
            memory_indices.shuffle(rng);

            for start_idx in (0..memory.len()).step_by(self.config.mini_batch_size) {
                let end_idx = memory.len().min(start_idx + self.config.mini_batch_size);
                let sample_indices = &memory_indices[start_idx..end_idx];
                let batch_size_ratio = sample_indices.len() as f32 / memory.len() as f32;

                let sample_indices_tensor = Tensor::from_ints(sample_indices, &self.device);
                let state_batch = get_states_batch(memory.states(), sample_indices, &self.device);
                let action_batch = get_action_batch(memory.actions(), sample_indices, &self.device);
                let old_log_probs_batch = old_log_probs
                    .clone()
                    .select(0, sample_indices_tensor.clone());
                let advantage_batch = advantages.clone().select(0, sample_indices_tensor.clone());
                let target_vals_batch = target_vals
                    .clone()
                    .select(0, sample_indices_tensor)
                    .detach();

                let PPOOutput {
                    policies: policy_batch,
                    values: value_batch,
                } = net.forward(state_batch);

                let log_prob = policy_batch.clone().log();
                let entropy = (-(log_prob.clone() * policy_batch)).sum_dim(1).mean();
                mean_entropy += entropy.clone().into_scalar().to_f32();

                let action_log_prob = log_prob.gather(1, action_batch.clone());
                let old_log_prob = old_log_probs_batch.gather(1, action_batch);
                let log_prob_diff = action_log_prob - old_log_prob;
                let ratios = log_prob_diff.clone().exp();
                let clipped_ratios = ratios
                    .clone()
                    .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                {
                    let kl_tensor = (ratios.clone() - 1.0) - log_prob_diff;
                    mean_divergence += kl_tensor.mean().detach().into_scalar().to_f32();

                    let clip_fraction = (ratios.clone() - 1.0)
                        .abs()
                        .greater_elem(self.config.clip_range)
                        .float()
                        .mean();
                    mean_clip_fraction += clip_fraction.into_scalar().to_f32();
                }

                let actor_loss = -elementwise_min(
                    ratios * advantage_batch.clone(),
                    clipped_ratios * advantage_batch,
                )
                .sum();

                let ppo_loss = actor_loss - entropy.mul_scalar(self.config.entropy_coeff);
                net.actor = update_parameters(
                    ppo_loss * batch_size_ratio,
                    net.actor,
                    &mut self.policy_optimizer,
                    self.config.learning_rate.into(),
                );

                let critic_loss = MseLoss.forward(value_batch, target_vals_batch, Reduction::Sum)
                    * batch_size_ratio;
                mean_val_loss += critic_loss.clone().into_scalar().to_f32();

                net.critic = update_parameters(
                    critic_loss,
                    net.critic,
                    &mut self.value_optimizer,
                    self.config.learning_rate.into(),
                );
            }
        }

        stats.cumulative_epochs += self.config.epochs as u64;
        stats.cumulative_model_updates += 1;

        let mini_batch_iters =
            1.max(memory.len() / self.config.mini_batch_size * self.config.epochs);

        mean_val_loss /= mini_batch_iters as f32;
        mean_entropy /= mini_batch_iters as f32;
        mean_divergence /= mini_batch_iters as f32;
        mean_clip_fraction /= mini_batch_iters as f32;

        metrics[".Value Loss"] = mean_val_loss.into();
        metrics[".Entropy"] = mean_entropy.into();
        metrics[".Divergence"] = mean_divergence.into();
        metrics[".Clip fraction"] = mean_clip_fraction.into();

        net
    }
}

struct GAEOutput<B: Backend> {
    returns: Vec<f32>,
    target_vals: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
}

#[allow(clippy::too_many_arguments)]
fn get_gae<B: Backend>(
    values: Vec<f32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    truncateds: Vec<bool>,
    gamma: f32,
    lambda: f32,
    return_std: f32,
    clip_range: f32,
    num_returns: usize,
    device: &B::Device,
) -> GAEOutput<B> {
    let num_samples = rewards.len();
    let mut target_vals = vec![0.0; num_samples];
    let mut returns = vec![0.0; num_samples];
    let mut advantages = returns.clone();

    let mut return_scale = 1.0 / return_std;
    if return_scale.is_nan() || return_scale == 0.0 {
        return_scale = 1.0;
    }

    let mut last_return = 0.0;
    let mut last_val = 0.0;
    let mut running_advantage = 0.0;

    for i in (0..num_samples).rev() {
        let not_done = f32::from(u8::from(!dones[i]));
        let not_truncated = f32::from(u8::from(!truncateds[i]));

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

    returns.truncate(num_returns);
    GAEOutput {
        returns,
        target_vals: Tensor::<B, 2>::from_data(
            TensorData::new(target_vals, [num_samples, 1]),
            device,
        ),
        advantages: Tensor::<B, 2>::from_data(
            TensorData::new(advantages, [num_samples, 1]),
            device,
        ),
    }
}
