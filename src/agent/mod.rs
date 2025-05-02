pub mod config;
pub mod model;

use crate::{
    agent::{
        config::PpoLearnerConfig,
        model::{Actic, Net, PPOOutput},
    },
    base::{Memory, MemoryIndices, get_action_batch, get_batch_1d, get_states_batch},
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
    pub fn train<R: Rng>(
        &mut self,
        mut net: Actic<B>,
        memory: &Memory,
        rng: &mut R,
        metrics: &mut Report,
        stats: &mut Stats,
    ) -> Actic<B> {
        let mut memory_indices = (0..memory.len()).collect::<MemoryIndices>();
        let PPOOutput {
            policies: mut old_polices,
            values: mut old_values,
        } = net.forward(get_states_batch(
            memory.states(),
            &memory_indices,
            &self.device,
        ));
        old_polices = old_polices.detach();
        old_values = old_values.detach();

        let GAEOutput {
            expected_returns,
            advantages,
        } = get_gae(
            old_values.into_data().into_vec().unwrap(),
            get_batch_1d(memory.rewards(), &memory_indices),
            get_batch_1d(memory.dones(), &memory_indices),
            get_batch_1d(memory.truncateds(), &memory_indices),
            self.config.gamma,
            self.config.lambda,
            &self.device,
        );

        let mut mean_entropy = 0.0;
        let mut mean_val_loss = 0.0;

        for _ in 0..self.config.epochs {
            memory_indices.shuffle(rng);

            for start_idx in (0..memory.len()).step_by(self.config.mini_batch_size) {
                let end_idx = memory.len().min(start_idx + self.config.mini_batch_size);
                let sample_indices = &memory_indices[start_idx..end_idx];
                let batch_size_ratio = sample_indices.len() as f32 / memory.len() as f32;

                let sample_indices_tensor = Tensor::from_ints(sample_indices, &self.device);
                let state_batch = get_states_batch(memory.states(), sample_indices, &self.device);
                let action_batch = get_action_batch(memory.actions(), sample_indices, &self.device);
                let old_policy_batch = old_polices.clone().select(0, sample_indices_tensor.clone());
                let advantage_batch = advantages.clone().select(0, sample_indices_tensor.clone());
                let expected_return_batch = expected_returns
                    .clone()
                    .select(0, sample_indices_tensor)
                    .detach();

                let PPOOutput {
                    policies: policy_batch,
                    values: value_batch,
                } = net.forward(state_batch);

                let ratios = policy_batch
                    .clone()
                    .div(old_policy_batch)
                    .gather(1, action_batch);
                let clipped_ratios = ratios
                    .clone()
                    .clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);

                let actor_loss = -elementwise_min(
                    ratios * advantage_batch.clone(),
                    clipped_ratios * advantage_batch,
                )
                .sum();

                let entropy = -(policy_batch.clone().log() * policy_batch)
                    .sum_dim(1)
                    .mean();
                mean_entropy += entropy.clone().into_scalar().to_f32();

                let ppo_loss = actor_loss + entropy.mul_scalar(self.config.entropy_coeff);
                net.actor = update_parameters(
                    ppo_loss * batch_size_ratio,
                    net.actor,
                    &mut self.policy_optimizer,
                    self.config.learning_rate.into(),
                );

                let critic_loss =
                    MseLoss.forward(expected_return_batch, value_batch, Reduction::Sum)
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

        metrics[".Value Loss"] = mean_val_loss.into();
        metrics[".Entropy"] = mean_entropy.into();

        net
    }
}

struct GAEOutput<B: Backend> {
    expected_returns: Tensor<B, 2>,
    advantages: Tensor<B, 2>,
}

impl<B: Backend> GAEOutput<B> {
    fn new(expected_returns: Tensor<B, 2>, advantages: Tensor<B, 2>) -> Self {
        Self {
            expected_returns,
            advantages,
        }
    }
}

fn get_gae<B: Backend>(
    values: Vec<f32>,
    rewards: Vec<f32>,
    not_dones: Vec<bool>,
    not_truncateds: Vec<bool>,
    gamma: f32,
    lambda: f32,
    device: &B::Device,
) -> GAEOutput<B> {
    let num_samples = rewards.len();
    let mut returns = vec![0.0; num_samples];
    let mut advantages = returns.clone();

    let mut last_val = 0.0;
    let mut running_return = 0.0;
    let mut running_advantage = 0.0;

    for i in (0..num_samples).rev() {
        let reward = rewards[i];
        let not_done = f32::from(u8::from(not_dones[i]));
        let not_truncated = f32::from(u8::from(not_truncateds[i]));

        running_return = reward + gamma * running_return * not_done * not_truncated;
        running_advantage = reward - values[i]
            + gamma * not_done * not_truncated * (last_val + lambda * running_advantage);

        last_val = values[i];
        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    GAEOutput::new(
        Tensor::<B, 2>::from_data(TensorData::new(returns, [num_samples, 1]), device),
        Tensor::<B, 2>::from_data(TensorData::new(advantages, [num_samples, 1]), device),
    )
}
