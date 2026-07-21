#![recursion_limit = "256"]

use burn::tensor::backend::AutodiffBackend;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgymppo::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo::rocketsim::{Arena, ArenaEvent, CarBodyConfig, GameMode, Team, init_from_default};
use rlgymppo::utils::actions::DefaultAction;
use rlgymppo::utils::obs::DefaultObs;
use rlgymppo::utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo::utils::state_setters::{KickoffState, RandomState, WeightedState};
use rlgymppo::utils::terminal::{
    AnyTerminal, NoTouchCondition, OnGoalCondition, RandomGameEndedCondition,
};
use rlgymppo::utils::{AvgTracker, Report, rewards};
use rlgymppo::{
    LearnerConfig, PpoLearnerConfig, SelfPlayConfig, SkillTrackerConfig, any_terminal,
    combined_rewards, default_adamw_optimizer, weighted_state,
};

pub struct SharedInfo {
    rng: SmallRng,
    metrics: Report,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(rng().next_u64()),
            metrics: Report::default(),
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, _initial_state: &GameState) {}

    fn update(&mut self, game_state: &GameState) {
        for (info, state) in &game_state.cars {
            let dist_to_ball = state.pos.distance(game_state.ball.pos);
            self.metrics["Player/Distance to ball"] += AvgTracker::from(dist_to_ball);

            self.metrics["Player/In Air Ratio"] += AvgTracker::from(!state.is_on_ground);
            self.metrics["Player/Demoed Ratio"] += AvgTracker::from(state.is_demoed);

            self.metrics["Player/Speed"] += AvgTracker::from(state.vel.length());

            let dir_to_ball = (game_state.ball.pos - state.pos).normalize_or_zero();
            let speed_towards_ball = state.vel.dot(dir_to_ball).max(0.0);
            self.metrics["Player/Speed Towards Ball"] += AvgTracker::from(speed_towards_ball);

            self.metrics["Player/Boost"] += AvgTracker::from(state.boost);

            let has_touched = game_state.events.iter().any(
                |event| matches!(event, ArenaEvent::CarHitBall(hit) if hit.car_idx == info.idx),
            );
            self.metrics["Player/Ball Touch Ratio"] += AvgTracker::from(has_touched);
        }

        for event in &game_state.events {
            if let ArenaEvent::CarHitBall(car_hit_ball) = event {
                self.metrics["Player/Touch Height"] +=
                    AvgTracker::from(car_hit_ball.contact_point.z);
            }
        }
    }
}

impl SharedInfoRng for SharedInfo {
    type Rng = SmallRng;

    fn rng(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl SharedInfoReport for SharedInfo {
    fn report(&mut self) -> &mut Report {
        &mut self.metrics
    }
}

const MIN_GAME_DURATION: u64 = 60 * 120;
const MAX_GAME_DURATION: u64 = 3 * 60 * 120;
type GameEndCond = RandomGameEndedCondition<MIN_GAME_DURATION, MAX_GAME_DURATION>;

const MAX_NO_TOUCH_DURATION: u64 = 10 * 120;

#[allow(clippy::type_complexity)]
pub fn create_env(
    game_id: Option<usize>,
) -> Env<
    WeightedState<SharedInfo>,
    DefaultObs<3>,
    DefaultAction<6, 8, 1>,
    rewards::CombinedRewards<SharedInfo>,
    AnyTerminal<SharedInfo>,
    NoTouchCondition<MAX_NO_TOUCH_DURATION>,
    SharedInfo,
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
        DefaultObs,
        DefaultAction::default(),
        combined_rewards![
            "Reward/In Air", rewards::AirReward => 0.25;
            "Reward/Face ball", rewards::FaceBallReward => 0.25;
            "Reward/Velocity to ball", rewards::VelocityToBallReward => 4.0;
        ],
        any_terminal![OnGoalCondition, GameEndCond],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}

pub fn default_config<B: AutodiffBackend>(
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
    let lr = 2e-4;

    LearnerConfig {
        render: false,
        num_threads: 4,
        num_games_per_thread: 64,
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
            max_episode_length: None,
            overbatching: false,
            retain_overflow_episodes: false,
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
        shared_head_layer_sizes: vec![256; 2],
        policy_layer_sizes: vec![256; 3],
        critic_layer_sizes: vec![256; 3],
        device,
        render_device,
        skill_tracker_device,
        #[cfg(feature = "wandb")]
        wandb_project_name: Some("rlgym-ppo".into()),
        #[cfg(feature = "wandb")]
        wandb_run_name: Some("ppo-bot-v1".into()),
        ..Default::default()
    }
}

pub fn run<B: AutodiffBackend>(
    device: B::Device,
    render_device: B::Device,
    skill_tracker_device: Option<B::Device>,
    async_skill_tracker: bool,
) {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    let mut learner = default_config::<B>(
        device,
        render_device,
        skill_tracker_device,
        async_skill_tracker,
    )
    .init(create_env, default_adamw_optimizer::<B>());
    learner.load();
    learner.learn();
}
