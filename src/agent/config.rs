use burn::{grad_clipping::GradientClippingConfig, tensor::backend::AutodiffBackend};

use super::Ppo;

pub struct PpoLearnerConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub clip_range: f32,
    pub reward_clip_range: f32,
    /// Entropy scale (applied after dividing by ln(n_actions)).
    pub entropy_scale: f32,
    pub standardize_returns: bool,
    pub max_returns_per_stats_increment: usize,
    pub clip_grad: Option<GradientClippingConfig>,
    /// Learning rate for the optimizer.
    pub learning_rate: f32,
    /// Number of epochs to train for with the same batch.
    pub epochs: usize,
    /// Size of the batch to train on.
    pub batch_size: usize,
    /// Size to split the batch into mini batches for training.
    /// This must be a divisor of the batch size.
    pub mini_batch_size: usize,
    /// Extend the last batch to use all remaining experience when it's less than
    /// 2x the batch size.
    pub overbatching: bool,
}

impl Default for PpoLearnerConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            clip_range: 0.2,
            reward_clip_range: 10.,
            entropy_scale: 0.018,
            standardize_returns: true,
            max_returns_per_stats_increment: 150,
            learning_rate: 3e-4,
            epochs: 4,
            batch_size: 60_000,
            mini_batch_size: 20_000,
            overbatching: true,
            clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        }
    }
}

impl PpoLearnerConfig {
    pub fn init<B: AutodiffBackend>(self, device: B::Device) -> Ppo<B> {
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "Batch size must be divisible by mini batch size"
        );

        Ppo::new(self, device)
    }
}
