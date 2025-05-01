pub mod config;
pub mod model;
pub mod net;

use crate::agent::config::PPOTrainingConfig;
use crate::agent::model::PPOOutput;
use crate::base::{Memory, MemoryIndices, Model, get_action_batch, get_batch_1d, get_states_batch};
use crate::utils::{
    elementwise_min, sample_actions_from_tensor, to_state_tensor_2d, update_parameters,
};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use net::{Actic, Net};
use rand::Rng;
use rand::seq::SliceRandom;
use std::marker::PhantomData;

pub struct PPO<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> PPO<B> {
    pub fn react<R: Rng>(
        state: &[Vec<f32>],
        model: &Actic<B>,
        rng: &mut R,
        device: &B::Device,
    ) -> Vec<usize> {
        sample_actions_from_tensor(model.infer(to_state_tensor_2d(state, device)), rng)
    }
}

impl<B: AutodiffBackend> PPO<B> {
    pub fn train<R: Rng>(
        mut net: Actic<B>,
        memory: &Memory,
        policy_optimizer: &mut (impl Optimizer<Net<B>, B> + Sized),
        value_optimizer: &mut (impl Optimizer<Net<B>, B> + Sized),
        config: &PPOTrainingConfig,
        rng: &mut R,
        device: &B::Device,
    ) -> Actic<B> {
        let mut memory_indices = (0..memory.len()).collect::<MemoryIndices>();
        let PPOOutput {
            policies: mut old_polices,
            values: mut old_values,
        } = net.forward(get_states_batch(memory.states(), &memory_indices, device));
        old_polices = old_polices.detach();
        old_values = old_values.detach();

        let GAEOutput {
            expected_returns,
            advantages,
        } = get_gae(
            old_values.reshape([-1]).into_data().into_vec().unwrap(),
            get_batch_1d(memory.rewards(), &memory_indices),
            get_batch_1d(memory.dones(), &memory_indices),
            get_batch_1d(memory.truncateds(), &memory_indices),
            config.gamma,
            config.lambda,
            device,
        );

        for _ in 0..config.epochs {
            memory_indices.shuffle(rng);

            for start_idx in (0..memory.len()).step_by(config.mini_batch_size) {
                let end_idx = memory.len().min(start_idx + config.mini_batch_size);
                let sample_indices = &memory_indices[start_idx..end_idx];
                let batch_size_ratio = sample_indices.len() as f32 / memory.len() as f32;

                let sample_indices_tensor = Tensor::from_ints(sample_indices, device);

                let state_batch = get_states_batch(memory.states(), sample_indices, device);
                let action_batch = get_action_batch(memory.actions(), sample_indices, device);
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
                    .clamp(1.0 - config.clip_range, 1.0 + config.clip_range);

                let actor_loss = -elementwise_min(
                    ratios * advantage_batch.clone(),
                    clipped_ratios * advantage_batch,
                )
                .sum();

                let policy_negative_entropy = -(policy_batch.clone().log() * policy_batch)
                    .sum_dim(1)
                    .mean();

                let ppo_loss =
                    actor_loss + policy_negative_entropy.mul_scalar(config.entropy_coeff);
                net.actor = update_parameters(
                    ppo_loss * batch_size_ratio,
                    net.actor,
                    policy_optimizer,
                    config.learning_rate.into(),
                );

                let critic_loss =
                    MseLoss.forward(expected_return_batch, value_batch, Reduction::Sum);
                net.critic = update_parameters(
                    critic_loss * batch_size_ratio,
                    net.critic,
                    value_optimizer,
                    config.learning_rate.into(),
                );
            }
        }

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
    let mut returns = vec![0.0; rewards.len()];
    let mut advantages = returns.clone();

    let mut last_val = 0.0;
    let mut running_return = 0.0;
    let mut running_advantage = 0.0;

    for i in (0..rewards.len()).rev() {
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
        Tensor::<B, 1>::from_floats(returns.as_slice(), device).reshape([returns.len(), 1]),
        Tensor::<B, 1>::from_floats(advantages.as_slice(), device).reshape([advantages.len(), 1]),
    )
}
