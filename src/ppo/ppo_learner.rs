use std::sync::Arc;

use super::{discrete::DiscretePolicy, value_est::ValueEstimator};
use tch::{
    nn::{self, OptimizerConfig},
    Device, Reduction,
};

#[derive(Debug, Clone)]
pub struct PPOLearnerConfig {
    pub policy_layer_sizes: Vec<i64>,
    pub critic_layer_sizes: Vec<i64>,
    pub batch_size: u64,
    pub epochs: u8,
    /// Policy learning rate
    pub policy_lr: f64,
    /// Critic learning rate
    pub critic_lr: f64,
    /// Entropy coefficient
    pub ent_coef: f32,
    pub clip_range: f32,
    /// Set to 0 to just use batchSize
    pub mini_batch_size: u64,
    // /// https://openai.com/index/how-ai-training-scales/
    // /// Measures the noise of both policy and critic gradients every epoch
    // pub measure_gradient_noise: bool,
    // pub gradient_noise_update_interval: i64,
    // pub gradient_noise_avg_decay: f32,
}

impl Default for PPOLearnerConfig {
    fn default() -> Self {
        Self {
            policy_layer_sizes: vec![256, 256, 256],
            critic_layer_sizes: vec![256, 256, 256],
            batch_size: 50 * 1000,
            epochs: 10,
            policy_lr: 3e-4,
            critic_lr: 3e-4,
            ent_coef: 0.005,
            clip_range: 0.2,
            mini_batch_size: 0,
            // measure_gradient_noise: false,
            // gradient_noise_update_interval: 10,
            // gradient_noise_avg_decay: 0.9925,
        }
    }
}

pub struct PPOLearner {
    policy: Arc<DiscretePolicy>,
    value_net: ValueEstimator,
    policy_optimizer: nn::Optimizer,
    value_optimizer: nn::Optimizer,
    policy_store: nn::VarStore,
    value_store: nn::VarStore,
    // noiseTrackerPolicy: GradNoiseTracker,
    // noiseTrackerValueNet: GradNoiseTracker,
    value_loss_fn: Reduction,
    config: PPOLearnerConfig,
    cumulative_model_updates: u64,
}

impl PPOLearner {
    pub fn new(
        obs_space_size: usize,
        action_space_size: usize,
        mut config: PPOLearnerConfig,
        device: Device,
    ) -> Self {
        if config.mini_batch_size == 0 {
            config.mini_batch_size = config.batch_size;
        }

        if config.batch_size % config.mini_batch_size != 0 {
            panic!("Batch size must be divisible by mini batch size");
        }

        let policy_store = nn::VarStore::new(device);
        let policy = DiscretePolicy::new(
            obs_space_size as i64,
            action_space_size as i64,
            &config.policy_layer_sizes,
            policy_store.root(),
            device,
        );
        let policy_optimizer = nn::Adam::default()
            .build(&policy_store, config.policy_lr)
            .unwrap();

        let value_store = nn::VarStore::new(device);
        let value_net = ValueEstimator::new(
            obs_space_size as i64,
            &config.critic_layer_sizes,
            value_store.root(),
            device,
        );
        let value_optimizer = nn::Adam::default()
            .build(&value_store, config.critic_lr)
            .unwrap();

        // grad noise stuff?

        Self {
            policy: Arc::new(policy),
            value_net,
            policy_optimizer,
            value_optimizer,
            policy_store,
            value_store,
            value_loss_fn: Reduction::Mean,
            config,
            cumulative_model_updates: 0,
        }
    }

    pub fn get_policy(&self) -> Arc<DiscretePolicy> {
        self.policy.clone()
    }
}
