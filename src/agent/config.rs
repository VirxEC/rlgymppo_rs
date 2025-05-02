use burn::{grad_clipping::GradientClippingConfig, tensor::backend::AutodiffBackend};

use super::PPO;

pub struct PPOTrainingConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip_range: f32,
    pub entropy_coeff: f32,
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub mini_batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for PPOTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_range: 0.2,
            entropy_coeff: 0.01,
            learning_rate: 3e-4,
            epochs: 2,
            batch_size: 50_000,
            mini_batch_size: 20_000,
            clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        }
    }
}

impl PPOTrainingConfig {
    pub fn init<B: AutodiffBackend>(self, device: B::Device) -> PPO<B> {
        PPO::new(self, device)
    }
}
