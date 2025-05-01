use burn::grad_clipping::GradientClippingConfig;

pub struct PPOTrainingConfig {
    pub gamma: f32,
    pub lambda: f32,
    pub epsilon_clip: f32,
    pub critic_weight: f32,
    pub entropy_weight: f32,
    pub learning_rate: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub clip_grad: Option<GradientClippingConfig>,
}

impl Default for PPOTrainingConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            epsilon_clip: 0.2,
            critic_weight: 0.5,
            entropy_weight: 0.005,
            learning_rate: 3e-4,
            epochs: 10,
            batch_size: 50_000,
            clip_grad: Some(GradientClippingConfig::Norm(0.5)),
        }
    }
}
