pub mod config;
pub mod model;

use crate::agent::config::PPOTrainingConfig;
use crate::agent::model::{PPOModel, PPOOutput};
use crate::base::{Agent, Memory, MemoryIndices, get_batch, sample_indices};
use crate::utils::{
    elementwise_min, get_elem, sample_action_from_tensor, sample_actions_from_tensor, to_action_tensor, to_not_done_tensor, to_reward_tensor, to_state_tensor, to_state_tensor_2d, update_parameters
};
use burn::module::AutodiffModule;
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::Optimizer;
use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
use rand::rngs::ThreadRng;
use std::marker::PhantomData;

pub struct PPO<B: Backend, M: PPOModel<B>> {
    model: M,
    backend: PhantomData<B>,
}

impl<B: Backend, M: PPOModel<B>> Agent for PPO<B, M> {
    fn react(&self, state: &[f32]) -> usize {
        sample_action_from_tensor::<B>(self.model.infer(to_state_tensor(state).unsqueeze()))
    }
}

impl<B: Backend, M: PPOModel<B>> PPO<B, M> {
    pub fn react_with_model(state: &[Vec<f32>], model: &M, rng: &mut ThreadRng) -> Vec<usize> {
        sample_actions_from_tensor(model.forward(to_state_tensor_2d(state)).policies, rng)
    }
}

impl<B: AutodiffBackend, M: PPOModel<B> + AutodiffModule<B>> PPO<B, M> {
    pub fn train<const CAP: usize>(
        mut policy_net: M,
        memory: &Memory<B, CAP>,
        optimizer: &mut (impl Optimizer<M, B> + Sized),
        config: &PPOTrainingConfig,
    ) -> M {
        let memory_indices = (0..memory.len()).collect::<MemoryIndices>();
        let PPOOutput {
            policies: mut old_polices,
            values: mut old_values,
        } = policy_net.forward(get_batch(memory.states(), &memory_indices, |v| {
            to_state_tensor(v.as_slice())
        }));
        old_polices = old_polices.detach();
        old_values = old_values.detach();

        if let Some(GAEOutput {
            expected_returns,
            advantages,
        }) = get_gae(
            old_values,
            get_batch(memory.rewards(), &memory_indices, |v| to_reward_tensor(*v)),
            get_batch(memory.dones(), &memory_indices, |v| to_not_done_tensor(*v)),
            config.gamma,
            config.lambda,
        ) {
            for _ in 0..config.epochs {
                for _ in 0..(memory.len() / config.batch_size) {
                    let sample_indices = sample_indices(memory_indices.clone(), config.batch_size);

                    let sample_indices_tensor = Tensor::from_ints(
                        sample_indices
                            .iter()
                            .map(|x| *x as i32)
                            .collect::<Vec<_>>()
                            .as_slice(),
                        &Default::default(),
                    );

                    let state_batch = get_batch(memory.states(), &sample_indices, |v| {
                        to_state_tensor(v.as_slice())
                    });
                    let action_batch =
                        get_batch(memory.actions(), &sample_indices, |v| to_action_tensor(*v));
                    let old_policy_batch =
                        old_polices.clone().select(0, sample_indices_tensor.clone());
                    let advantage_batch =
                        advantages.clone().select(0, sample_indices_tensor.clone());
                    let expected_return_batch = expected_returns
                        .clone()
                        .select(0, sample_indices_tensor)
                        .detach();

                    let PPOOutput {
                        policies: policy_batch,
                        values: value_batch,
                    } = policy_net.forward(state_batch);

                    let ratios = policy_batch
                        .clone()
                        .div(old_policy_batch)
                        .gather(1, action_batch);
                    let clipped_ratios = ratios
                        .clone()
                        .clamp(1.0 - config.epsilon_clip, 1.0 + config.epsilon_clip);

                    let actor_loss = -elementwise_min(
                        ratios * advantage_batch.clone(),
                        clipped_ratios * advantage_batch,
                    )
                    .sum();
                    let critic_loss =
                        MseLoss.forward(expected_return_batch, value_batch, Reduction::Sum);
                    let policy_negative_entropy = -(policy_batch.clone().log() * policy_batch)
                        .sum_dim(1)
                        .mean();

                    let loss = actor_loss
                        + critic_loss.mul_scalar(config.critic_weight)
                        + policy_negative_entropy.mul_scalar(config.entropy_weight);
                    policy_net =
                        update_parameters(loss, policy_net, optimizer, config.learning_rate.into());
                }
            }
        }
        policy_net
    }

    pub fn valid(model: M) -> PPO<B::InnerBackend, M::InnerModule>
    where
        <M as AutodiffModule<B>>::InnerModule: PPOModel<<B as AutodiffBackend>::InnerBackend>,
    {
        PPO::<B::InnerBackend, M::InnerModule> {
            model: model.valid(),
            backend: PhantomData,
        }
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
    values: Tensor<B, 2>,
    rewards: Tensor<B, 2>,
    not_dones: Tensor<B, 2>,
    gamma: f32,
    lambda: f32,
) -> Option<GAEOutput<B>> {
    let mut returns = vec![0.0; rewards.shape().num_elements()];
    let mut advantages = returns.clone();

    let mut running_return = 0.0;
    let mut running_advantage = 0.0;

    for i in (0..rewards.shape().num_elements()).rev() {
        let reward = get_elem(i, &rewards).unwrap();
        let not_done = get_elem(i, &not_dones).unwrap();

        running_return = reward + gamma * running_return * not_done;
        running_advantage = reward - get_elem(i, &values).unwrap()
            + gamma
                * not_done
                * (get_elem(i + 1, &values).unwrap_or(0.0) + lambda * running_advantage);

        returns[i] = running_return;
        advantages[i] = running_advantage;
    }

    Some(GAEOutput::new(
        Tensor::<B, 1>::from_floats(returns.as_slice(), &Default::default())
            .reshape([returns.len(), 1]),
        Tensor::<B, 1>::from_floats(advantages.as_slice(), &Default::default())
            .reshape([advantages.len(), 1]),
    ))
}
