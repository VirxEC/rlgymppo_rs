use burn::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, Optimizer};
use burn::tensor::backend::AutodiffBackend;

use super::Ppo;
use super::model::{Actic, Net};
use crate::OptimizerNetwork;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GaeEstimator {
    /// Ordinary GAE truncated at explicit rollout boundaries.
    #[default]
    Truncated,
    /// Finite-horizon GAE computed over complete trajectories, ending at
    /// actual environment termination or the configured maximum episode length.
    TerminationTime,
}

/// Learning-rate schedule applied per optimizer update.
///
/// Iteration counting starts at zero and uses `Stats::cumulative_model_updates`
/// on the PPO path (one increment per `learn` call) and a separate
/// transfer-round counter on the transfer-learning path (one increment per
/// `transfer_learn` call, persisted via `Stats::transfer_rounds` so restarts
/// continue rather than restart the schedule). `Constant` preserves the
/// historical flat-learning-rate behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum LrSchedule {
    /// No decay: the configured learning rate is used unchanged.
    #[default]
    Constant,
    /// Multiply the learning rate by `decay_per_iteration` every update.
    /// `1.0` disables decay. For example, halving over 100,000 updates needs
    /// `0.5f32.powf(1.0 / 100_000.0) ~= 0.999993`.
    Exponential { decay_per_iteration: f32 },
    /// Cosine decay from `1.0` to `min_scale` over `total_iterations`
    /// updates, then flat at `min_scale`.
    Cosine {
        total_iterations: u64,
        min_scale: f32,
    },
    /// Exponential decay with a floor: multiplies by `decay_per_iteration`
    /// every update but never drops below `min_scale`. Pins an early
    /// milestone (e.g. 3e-3 -> 1e-3 at 200B steps) while guaranteeing a
    /// long convergence tail instead of decaying into the dirt.
    ExponentialWithFloor {
        decay_per_iteration: f32,
        min_scale: f32,
    },
}

impl LrSchedule {
    /// Multiplicative learning-rate factor for the given completed-update count.
    pub fn scale(self, completed_updates: u64) -> f64 {
        match self {
            Self::Constant => 1.0,
            Self::Exponential {
                decay_per_iteration,
            } => f64::from(decay_per_iteration).powi(completed_updates.min(1_000_000_000) as i32),
            Self::Cosine {
                total_iterations,
                min_scale,
            } => {
                let total = total_iterations.max(1) as f64;
                let t = (completed_updates as f64 / total).min(1.0);
                let cosine = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
                f64::from(min_scale) + (1.0 - f64::from(min_scale)) * cosine
            }
            Self::ExponentialWithFloor {
                decay_per_iteration,
                min_scale,
            } => f64::from(decay_per_iteration)
                .powi(completed_updates.min(1_000_000_000) as i32)
                .max(f64::from(min_scale)),
        }
    }
}

/// Regenerative regularization (soft shrink-and-perturb toward initialization).
///
/// After every optimizer step each parameter is pulled toward its value at the
/// start of the run: `theta <- theta - strength * (theta - theta_init)`, plus
/// optional isotropic Gaussian noise (`noise_std`). The anchor is captured
/// from the pre-update network on the first training call that enables it
/// and preserved across checkpoint save/load (`*_init` files), so enabling
/// this adds one extra copy of the parameters in memory. Legacy checkpoints
/// without anchor files re-anchor to the loaded weights with a warning.
/// `None` (the default) disables it entirely.
///
/// This is the "continuous regularizer" family that mitigates plasticity loss
/// in on-policy RL; the noise term makes it behave like soft
/// shrink-and-perturb, except blending is toward the run's own initialization
/// rather than freshly sampled weights.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RegenConfig {
    /// Pull toward initialization per optimizer step. `0.0` disables the pull
    /// (noise still applies when `noise_std > 0.0`). Must be in `[0, 1)`.
    pub strength: f32,
    /// Standard deviation of Gaussian noise added per optimizer step.
    pub noise_std: f32,
}

/// CAPO-style coherence-adaptive PPO clipping.
///
/// Per rollout row, the directional consistency (sign agreement) of TD
/// residuals inside a forward window is measured; rows whose local signal is
/// consistent get a wider clip bound (up to `max_scale * clip_range`) and
/// rows with contradictory signal get a narrower one (down to
/// `min_scale * clip_range`). Windows stop at trajectory boundaries, so no
/// coherence leaks across episodes. `None` (the default) keeps the fixed
/// scalar `clip_range` behavior with zero extra cost.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AdaptiveClipConfig {
    /// Forward window size (including the row itself) for the consistency vote.
    pub window: usize,
    /// Clip multiplier for fully incoherent rows (e.g. `0.5`).
    pub min_scale: f32,
    /// Clip multiplier for fully coherent rows (e.g. `1.5`).
    pub max_scale: f32,
}

/// DreamerV3-style percentile advantage scaling.
///
/// Tracks an EMA of the batch `p95 - p5` advantage spread and divides
/// advantages by it whenever it exceeds `1.0`, keeping policy-gradient scales
/// stable as returns grow. Takes precedence over `standardize_advantages`
/// when both are set. `None` (the default) disables it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PercentileAdvConfig {
    /// EMA decay for the spread estimate (e.g. `0.99`).
    pub ema_decay: f32,
}

pub struct PpoLearnerConfig {
    pub gamma: f32,
    pub lambda: f32,
    /// Advantage estimator used for each learner update.
    pub gae_estimator: GaeEstimator,
    pub clip_range: f32,
    pub reward_clip_range: f32,
    /// Entropy scale (applied after dividing by ln(n_actions) when
    /// `normalize_entropy` is true).
    pub entropy_scale: f32,
    /// Divide entropy by ln(n_actions) so it lies in [0, 1].
    /// When false, the raw entropy value is used.
    pub normalize_entropy: bool,
    pub standardize_returns: bool,
    /// Standardize advantages to zero mean and unit variance before each
    /// training batch.
    pub standardize_advantages: bool,
    pub max_returns_per_stats_increment: usize,
    pub clip_grad: Option<GradientClippingConfig>,
    /// Default learning rate for the optimizers.
    pub learning_rate: f32,
    /// Optional learning rate override for the actor (policy) optimizer.
    pub actor_learning_rate: Option<f32>,
    /// Optional learning rate override for the critic (value) optimizer.
    pub critic_learning_rate: Option<f32>,
    /// Number of epochs to train for with the same rollout.
    pub epochs: usize,
    /// Number of environment timesteps to collect before each training iteration.
    pub timesteps_per_iteration: usize,
    /// Number of samples contributing to each optimizer update.
    /// Gradients are accumulated across its mini-batches before one optimizer step.
    /// This must divide `timesteps_per_iteration`.
    pub batch_size: usize,
    /// Number of GPU-resident samples used per forward and backward pass.
    /// This must divide `batch_size`.
    pub mini_batch_size: usize,
    /// Chunk size for critic-only inference passes (old values, truncation
    /// bootstraps). Larger chunks mean fewer launches and can usually exceed
    /// `mini_batch_size` because they hold no gradients.
    /// This must be at least `batch_size` and divide `timesteps_per_iteration`.
    pub gpu_timestep_buffer_size: usize,
    /// Number of truncation next-state observations evaluated by the critic at once
    /// when bootstrapping GAE. Set this to `gpu_timestep_buffer_size` to use one
    /// contiguous CPU-to-GPU upload whenever all truncation observations fit in one batch.
    pub truncation_value_batch_size: usize,
    /// Allow the final completed trajectory to extend the rollout while keeping
    /// the collected size strictly below 2x `timesteps_per_iteration`.
    pub overbatching: bool,

    // ── Reward metrics sampling ─────────────────────────────────────
    /// Whether to include per-component reward metrics in the report.
    /// Reward components whose name starts with `"Reward/"` are tracked.
    pub add_rewards_to_metrics: bool,
    /// Sample reward metrics once every N steps (1 = every step).
    /// Set to 0 to disable sampling (track every step).
    pub reward_sample_interval: usize,

    /// Maximum number of steps per episode before the trajectory is
    /// force-truncated (None = no limit).  When a player's trajectory
    /// reaches this length the game is terminated with a TRUNCATED
    /// terminal, providing the next-state observation for the critic
    /// bootstrap.  Defaults to 1800 (120 seconds at 15 steps/second,
    /// matching the GGL default).
    pub max_episode_length: Option<usize>,
    /// Retain completed episodes that exceed an iteration's collection budget
    /// and return them before collecting new experience next iteration.
    ///
    /// Disabled by default to preserve the historical behavior of discarding
    /// overflow episodes.
    pub retain_overflow_episodes: bool,

    // ── Long-run training extensions ──────────────────────────────
    /// Learning-rate schedule applied per optimizer update.
    /// `Constant` (the default) preserves the historical flat rate.
    pub lr_schedule: LrSchedule,
    /// Regenerative regularization toward initialization (soft
    /// shrink-and-perturb family). `None` (the default) disables it.
    pub regen: Option<RegenConfig>,
    /// Coherence-adaptive per-sample PPO clipping. `None` (the default)
    /// keeps the fixed scalar `clip_range`.
    pub adaptive_clip: Option<AdaptiveClipConfig>,
    /// DreamerV3-style percentile advantage scaling. Takes precedence over
    /// `standardize_advantages` when both are set. `None` disables it.
    pub percentile_adv_scaling: Option<PercentileAdvConfig>,
    /// Report plasticity diagnostics (`Update/* weight norm`, `stable rank`,
    /// `dormant frac`) every N updates. `None` (the default) disables them.
    pub plasticity_log_interval: Option<usize>,
}

impl Default for PpoLearnerConfig {
    fn default() -> Self {
        Self {
            gamma: 0.99,
            lambda: 0.95,
            gae_estimator: GaeEstimator::default(),
            clip_range: 0.2,
            reward_clip_range: 10.,
            entropy_scale: 0.018,
            normalize_entropy: true,
            standardize_returns: true,
            standardize_advantages: true,
            max_returns_per_stats_increment: 150,
            learning_rate: 3e-4,
            actor_learning_rate: None,
            critic_learning_rate: None,
            epochs: 3,
            timesteps_per_iteration: 60_000,
            batch_size: 60_000,
            mini_batch_size: 20_000,
            gpu_timestep_buffer_size: 60_000,
            truncation_value_batch_size: 20_000,
            overbatching: false,
            clip_grad: Some(GradientClippingConfig::Norm(0.5)),
            add_rewards_to_metrics: true,
            reward_sample_interval: 8,
            max_episode_length: Some(1800),
            retain_overflow_episodes: false,
            lr_schedule: LrSchedule::Constant,
            regen: None,
            adaptive_clip: None,
            percentile_adv_scaling: None,
            plasticity_log_interval: None,
        }
    }
}

impl PpoLearnerConfig {
    /// Initialize with the default AdamW optimizer.
    pub fn init<B: AutodiffBackend>(self, device: B::Device) -> Ppo<B> {
        self.validate_batching();

        let clip_grad = self.clip_grad.clone();
        Ppo::new(self, device, || {
            let mut cfg = AdamWConfig::new().with_epsilon(1e-8);
            if let Some(ref clip) = clip_grad {
                cfg = cfg.with_grad_clipping(Some(clip.clone()));
            }
            cfg.init()
        })
    }

    /// Initialize with a custom optimizer.
    ///
    /// The `make_optim` closure is called three times (once per sub-network)
    /// and must return a freshly created optimizer each time.
    pub fn init_with<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
        self,
        device: B::Device,
        make_optim: impl Fn() -> O,
    ) -> Ppo<B, O> {
        self.validate_batching();

        Ppo::new(self, device, make_optim)
    }

    pub(crate) fn validate_batching(&self) {
        if self.gae_estimator == GaeEstimator::TerminationTime {
            assert!(
                self.lambda > 0.0 && self.lambda < 1.0,
                "Termination-time GAE requires lambda in the open interval (0, 1)"
            );
            assert!(
                self.max_episode_length.is_some_and(|length| length > 0),
                "Termination-time GAE requires a finite, positive max_episode_length"
            );
        }

        assert!(
            self.timesteps_per_iteration > 0,
            "Timesteps per iteration must be greater than zero"
        );
        assert!(self.batch_size > 0, "Batch size must be greater than zero");
        assert!(
            self.mini_batch_size > 0,
            "Mini batch size must be greater than zero"
        );
        assert!(
            self.gpu_timestep_buffer_size > 0,
            "GPU timestep buffer size must be greater than zero"
        );
        assert!(
            self.truncation_value_batch_size > 0,
            "Truncation value batch size must be greater than zero"
        );
        assert_eq!(
            self.timesteps_per_iteration % self.batch_size,
            0,
            "Timesteps per iteration must be divisible by batch size"
        );
        assert_eq!(
            self.batch_size % self.mini_batch_size,
            0,
            "Batch size must be divisible by mini batch size"
        );
        assert_eq!(
            self.timesteps_per_iteration % self.gpu_timestep_buffer_size,
            0,
            "Timesteps per iteration must be divisible by GPU timestep buffer size"
        );
        assert!(
            self.gpu_timestep_buffer_size >= self.batch_size,
            "GPU timestep buffer size must be at least batch size"
        );
    }

    /// Initialize with a custom optimizer factory that can inspect each network.
    ///
    /// The factory is called once for the actor, critic, and shared head. This
    /// allows optimizers such as Muon/AdamW hybrids to select parameters by ID.
    pub fn init_with_model<B: AutodiffBackend, O: Optimizer<Net<B>, B>>(
        self,
        device: B::Device,
        model: &Actic<B>,
        make_optim: impl Fn(OptimizerNetwork, &Net<B>) -> O + 'static,
    ) -> Ppo<B, O> {
        self.validate_batching();

        Ppo::new_with_model(self, device, model, make_optim)
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn regression_default_uses_truncated_gae() {
        let config = PpoLearnerConfig::default();
        assert_eq!(config.gae_estimator, GaeEstimator::Truncated);
        config.validate_batching();
    }

    #[test]
    #[should_panic(expected = "finite, positive max_episode_length")]
    fn regression_termination_time_requires_a_finite_horizon() {
        let config = PpoLearnerConfig {
            gae_estimator: GaeEstimator::TerminationTime,
            max_episode_length: None,
            ..PpoLearnerConfig::default()
        };
        config.validate_batching();
    }

    #[test]
    #[should_panic(expected = "lambda in the open interval")]
    fn regression_termination_time_rejects_lambda_endpoints() {
        let config = PpoLearnerConfig {
            gae_estimator: GaeEstimator::TerminationTime,
            lambda: 1.0,
            ..PpoLearnerConfig::default()
        };
        config.validate_batching();
    }

    #[test]
    #[should_panic(expected = "Truncation value batch size must be greater than zero")]
    fn regression_zero_truncation_value_batch_size_is_rejected() {
        let config = PpoLearnerConfig {
            truncation_value_batch_size: 0,
            ..PpoLearnerConfig::default()
        };
        config.validate_batching();
    }
}
