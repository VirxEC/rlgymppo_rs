use burn::grad_clipping::GradientClippingConfig;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamConfig, SimpleOptimizer};
use burn::tensor::backend::AutodiffBackend;

use super::Ppo;
use super::model::Net;

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

    // ── Reward metrics sampling ─────────────────────────────────────
    /// Whether to include per-component reward metrics in the report.
    /// Reward components whose name starts with `"Reward/"` are tracked.
    pub add_rewards_to_metrics: bool,
    /// Sample reward metrics once every N steps (1 = every step).
    /// Set to 0 to disable sampling (track every step).
    pub reward_sample_interval: usize,
    /// Maximum number of random reward-component samples to include
    /// per metric report step (unused in the current Rust impl, kept
    /// for API compatibility with GigaLearn).
    pub max_reward_samples: usize,

    /// Maximum number of steps per episode before the trajectory is
    /// force-truncated (None = no limit).  When a player's trajectory
    /// reaches this length the game is terminated with a TRUNCATED
    /// terminal, providing the next-state observation for the critic
    /// bootstrap.  Defaults to 1800 (120 seconds at 15 steps/second,
    /// matching the GGL default).
    pub max_episode_length: Option<usize>,
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
            add_rewards_to_metrics: true,
            reward_sample_interval: 8,
            max_reward_samples: 50,
            max_episode_length: Some(1800),
        }
    }
}

impl PpoLearnerConfig {
    /// Initialize with the default Adam optimizer.
    pub fn init<B: AutodiffBackend>(self, device: B::Device) -> Ppo<B> {
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "Batch size must be divisible by mini batch size"
        );

        let clip_grad = self.clip_grad.clone();
        Ppo::new(self, device, || {
            let mut cfg = AdamConfig::new().with_epsilon(1e-8);
            if let Some(ref clip) = clip_grad {
                cfg = cfg.with_grad_clipping(Some(clip.clone()));
            }
            cfg.init()
        })
    }

    /// Initialize with a custom optimizer.
    ///
    /// The `make_optim` closure is called three times (once per sub-network)
    /// and must return a freshly created [`OptimizerAdaptor`] each time.
    pub fn init_with<B: AutodiffBackend, O: SimpleOptimizer<B::InnerBackend>>(
        self,
        device: B::Device,
        make_optim: impl Fn() -> OptimizerAdaptor<O, Net<B>, B>,
    ) -> Ppo<B, O> {
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "Batch size must be divisible by mini batch size"
        );

        Ppo::new(self, device, make_optim)
    }
}
