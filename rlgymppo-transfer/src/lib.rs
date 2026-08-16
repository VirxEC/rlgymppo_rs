//! Transfer learning for the `rlgymppo-trainer` parent model.
//!
//! This crate pretends `rlgymppo-trainer` is the parent model: the teacher is
//! exactly what `rlgymppo_trainer::run` trains. The parent's architecture,
//! checkpoint folder, and obs builder (`DefaultObs<3>`) are copied here to
//! match the trainer's setup, and the parent's obs builder runs alongside the
//! student's in the collector. The crate distills that parent into a smaller
//! student policy, then you continue with normal PPO training via
//! [`rlgymppo_trainer::run`].
//!
//! Run the ready-made example:
//!
//! ```sh
//! cargo run -p rlgymppo-transfer --example transfer_learn --features torch
//! ```

use std::path::PathBuf;
use std::thread::available_parallelism;

use burn::tensor::backend::AutodiffBackend;
use rlgymppo::rlgym::{Env, Obs};
use rlgymppo::rocketsim::{Arena, CarBodyConfig, GameMode, Team, init_from_default};
use rlgymppo::utils::actions::DefaultAction;
use rlgymppo::utils::obs::DefaultObs;
use rlgymppo::utils::rewards;
use rlgymppo::utils::state_setters::{KickoffState, RandomState, WeightedState};
use rlgymppo::utils::terminal::{
    AnyTerminal, NoTouchCondition, OnGoalCondition, RandomGameEndedCondition,
};
use rlgymppo::{
    GaeEstimator, LearnerConfig, NormSelection, PpoLearnerConfig, SelfPlayConfig,
    SkillTrackerConfig, TeacherConfig, TransferLearnConfig, any_terminal, combined_rewards,
    default_adamw_optimizer, weighted_state,
};

const MIN_GAME_DURATION: u64 = 60 * 120;
const MAX_GAME_DURATION: u64 = 3 * 60 * 120;
type GameEndCond = RandomGameEndedCondition<MIN_GAME_DURATION, MAX_GAME_DURATION>;

const MAX_NO_TOUCH_DURATION: u64 = 10 * 120;

/// The student's environment: a copy of `rlgymppo_trainer::create_env`, but
/// observing with the smaller `DefaultObs<1>` (53 floats) instead of the
/// parent's `DefaultObs<3>` (141 floats). Everything else — state setters,
/// actions, rewards, terminals, shared info — is identical.
#[allow(clippy::type_complexity)]
pub fn create_env(
    game_id: Option<usize>,
) -> Env<
    WeightedState<rlgymppo_trainer::SharedInfo>,
    DefaultObs<1>,
    DefaultAction<6, 8, 0>,
    rewards::CombinedRewards<rlgymppo_trainer::SharedInfo>,
    AnyTerminal<rlgymppo_trainer::SharedInfo>,
    NoTouchCondition<MAX_NO_TOUCH_DURATION>,
    rlgymppo_trainer::SharedInfo,
> {
    let game_id = game_id.unwrap_or(0);

    let mut arena = Arena::new(GameMode::Soccar);

    for _ in 0..=game_id % 3 {
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    }

    Env::new(
        arena,
        weighted_state![
            KickoffState, 0.1;
            RandomState<true, false, true>, 0.4;
            RandomState<true, true, true>, 0.2;
            RandomState<true, true, false>, 0.3;
        ],
        DefaultObs::<1>,
        DefaultAction::default(),
        combined_rewards![
            "Reward/In Air", rewards::AirReward => 0.25;
            "Reward/Face ball", rewards::FaceBallReward => 0.25;
            "Reward/Velocity to ball", rewards::VelocityToBallReward => 4.0;
        ],
        any_terminal![OnGoalCondition, GameEndCond],
        NoTouchCondition::default(),
        rlgymppo_trainer::SharedInfo::default(),
    )
}

/// The parent model's architecture and checkpoint folder, copied from
/// `rlgymppo-trainer` (if you change the trainer's setup, update these to
/// match). The parent's obs builder is `DefaultObs<3>` (141 floats), which
/// [`transfer_learn`] runs alongside the student's.
pub fn teacher_config() -> TeacherConfig {
    TeacherConfig {
        // The parent model's checkpoints (`rlgymppo_trainer` saves to
        // `./checkpoints`).
        models_path: PathBuf::from("checkpoints"),
        // The parent's architecture.
        policy_layer_sizes: vec![256; 3],
        shared_head_layer_sizes: vec![256; 2],
        norm: NormSelection::RmsNorm,
    }
}

/// The student's `LearnerConfig`: a copy of `rlgymppo_trainer::default_config`
/// with a smaller network and its own checkpoint folder (if you change the
/// trainer's setup, update these to match).
fn default_config<B: AutodiffBackend>(
    device: B::Device,
    render_device: B::Device,
    skill_tracker_device: Option<B::Device>,
    async_skill_tracker: bool,
) -> LearnerConfig<B> {
    // Samples collected before each PPO update. Larger values reduce per-update
    // overhead but use more CPU memory and make policy updates less frequent.
    let timesteps_per_iteration = 100_000;
    // Effective samples per optimizer update. Gradients from its mini-batches are
    // accumulated before one update.
    let batch_size = timesteps_per_iteration;
    // Samples per forward/backward pass. Decrease when training runs out of VRAM.
    let mini_batch_size = 20_000;
    // CPU-to-GPU staging capacity. It must be at least `batch_size`.
    let gpu_timestep_buffer_size = batch_size;
    // Inference-only batch for critic bootstrapping at truncated trajectories.
    // It can usually be larger than `mini_batch_size` because it holds no gradients.
    let truncation_value_batch_size = batch_size;
    let lr = 1e-3;
    let num_pools = 2;

    LearnerConfig {
        render: false,
        num_pools,
        num_threads_per_pool: available_parallelism().unwrap().get() / num_pools,
        num_games_per_pool: 512 / num_pools,
        timesteps_per_save: 10_000_000,
        checkpoints_limit: None,
        ppo: PpoLearnerConfig {
            timesteps_per_iteration,
            batch_size,
            mini_batch_size,
            gpu_timestep_buffer_size,
            truncation_value_batch_size,
            epochs: 2,
            learning_rate: lr,
            entropy_scale: 0.024,
            gae_estimator: GaeEstimator::TerminationTime,
            max_episode_length: None,
            ..Default::default()
        },
        self_play: SelfPlayConfig {
            save_policy_versions: true,
            ts_per_version: 500_000_000,
            max_old_versions: 10,
            max_saved_versions: None,
            train_against_old_versions: false,
            train_against_old_chance: 0.15,
        },
        skill_tracker: SkillTrackerConfig {
            enabled: true,
            num_arenas: 12,
            update_interval: 10,
            async_eval: async_skill_tracker,
            ..Default::default()
        },
        // The new, smaller student network (fewer parameters than the parent).
        shared_head_layer_sizes: vec![128],
        policy_layer_sizes: vec![128; 2],
        critic_layer_sizes: vec![256; 3],
        // Keep student checkpoints out of the parent's checkpoint directory.
        checkpoints_folder: PathBuf::from("checkpoints_transfer"),
        device,
        render_device,
        skill_tracker_device,
        #[cfg(feature = "wandb")]
        wandb_project_name: Some("rlgym-ppo".into()),
        #[cfg(feature = "wandb")]
        wandb_run_name: Some("ppo-transfer-v1".into()),
        ..Default::default()
    }
}

/// Distill the `rlgymppo-trainer` parent model into a smaller student policy.
///
/// The teacher is [`teacher_config`] (the parent) and the distillation
/// hyperparameters are [`TransferLearnConfig::default`]. The student network
/// is smaller than the parent's: one shared-head layer of 128 and two
/// 128-unit actor/critic layers, versus the parent's 256-unit layers. It also
/// observes a *different*, smaller space: the parent was trained on
/// `DefaultObs<3>` (141 floats) while the student uses `DefaultObs<1>` (53
/// floats). The parent's obs builder runs alongside the student's in the
/// collector so both networks score the same states; everything else
/// (actions, rewards, terminals, shared info) is identical.
///
/// Student checkpoints are saved to `./checkpoints_transfer` so they never
/// shadow the parent's `./checkpoints`. Once distillation has converged
/// (watch `Transfer/loss` and `Transfer/accuracy`), stop with `Q`, then run
/// `rlgymppo_trainer::run` with `checkpoints_folder` pointed at
/// `checkpoints_transfer` to continue with normal PPO training.
pub fn transfer_learn<B: AutodiffBackend>(
    device: B::Device,
    render_device: B::Device,
    skill_tracker_device: Option<B::Device>,
    async_skill_tracker: bool,
) {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    // The parent model's obs builder, run in lockstep with the student's.
    let make_old_obs =
        || Box::new(DefaultObs::<3>) as Box<dyn Obs<rlgymppo_trainer::SharedInfo> + Send>;

    let mut learner = default_config::<B>(
        device,
        render_device,
        skill_tracker_device,
        async_skill_tracker,
    )
    .init_with_old_obs(create_env, default_adamw_optimizer::<B>(), make_old_obs);
    learner.load(); // resume an earlier distillation run if one exists
    learner.transfer_learn(teacher_config(), TransferLearnConfig::default());
}
