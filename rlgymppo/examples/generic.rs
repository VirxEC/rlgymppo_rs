#![recursion_limit = "256"]

use burn::backend::LibTorch;
use burn::backend::libtorch::LibTorchDevice;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng, rng};
use rlgym::rocketsim::{ArenaEvent, init_from_default};
use rlgymppo::backend::Autodiff;
use rlgymppo::rlgym::{Env, GameState, SharedInfoProvider};
use rlgymppo::rocketsim::{Arena, CarBodyConfig, GameMode, Team};
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
    combined_rewards, weighted_state,
};

struct SharedInfo {
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

        // Track touch height from car-ball contact events
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

const MIN_GAME_DURATION: u64 = 60 * 120; // 1 minute in ticks
const MAX_GAME_DURATION: u64 = 3 * 60 * 120; // 3 minutes in ticks
type GameEndCond = RandomGameEndedCondition<MIN_GAME_DURATION, MAX_GAME_DURATION>;

const MAX_NO_TOUCH_DURATION: u64 = 10 * 120; // 10 seconds in ticks

#[allow(clippy::type_complexity)]
fn create_env(
    game_id: Option<usize>,
) -> Env<
    WeightedState<SharedInfo>,
    DefaultObs<3>,
    DefaultAction<6, 8>,
    rewards::CombinedRewards<SharedInfo>,
    AnyTerminal<SharedInfo>,
    NoTouchCondition<MAX_NO_TOUCH_DURATION>,
    SharedInfo,
> {
    // `game_id` is None for the game used to calculate the policy obs/action space,
    // as well as for the renderer.
    // Otherwise, every env gets a unique id starting from 0 and incrementing by 1.
    let game_id = game_id.unwrap_or(0);

    let mut arena = Arena::new(GameMode::Soccar);

    // pseudo-random game mode: 1v1, 2v2, 3v3
    // using game id ensures an equal, predictable distribution of game modes
    // do not change the number of players between episodes
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
            // "Reward/Velocity ball to goal", rewards::ZeroSumReward::new(
            //     rewards::VelocityBallToGoalReward, 1.0, 1.0
            // ) => 2.0;
        ],
        any_terminal![OnGoalCondition, GameEndCond],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}

fn main() {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    let mini_batch_size = 40_000;
    let batch_size = mini_batch_size * 2;
    let lr = 1.5e-4;

    // Router will fallback to NdArray if Wgpu is not available
    // Realistically more useful for using CUDA and falling back to NdArray
    let config = LearnerConfig::<Autodiff<LibTorch>> {
        // if the renderer is on by default or not (can be toggled at runtime)
        render: false,
        // Only increase the number of threads if your cpu is having trouble
        // saturating your gpu. Each thread is faster than you think!
        num_threads: 4,
        // 4 (threads) * 64 (games per thread) = 256 (total games)
        // 256 total games is a good number to target.
        //
        // If you increase `num_threads`,
        // you can decrease `num_games_per_thread` to keep the total games around 256.
        num_games_per_thread: 64,
        timesteps_per_save: 10_000_000,
        checkpoints_limit: Some(10),
        ppo: PpoLearnerConfig {
            batch_size,
            mini_batch_size,
            epochs: 1,
            learning_rate: lr,
            // This scales differently than "ent_coef" in other frameworks;
            // This is the scale for normalized entropy,
            // which means you won't have to change it if you add more actions
            entropy_scale: 0.036,
            ..Default::default()
        },
        self_play: SelfPlayConfig {
            save_policy_versions: true,
            ts_per_version: 100_000_000,
            max_old_versions: 10,
            train_against_old_versions: false,
            train_against_old_chance: 0.15,
        },
        skill_tracker: SkillTrackerConfig {
            enabled: true,
            ..Default::default()
        },
        shared_head_layer_sizes: vec![256; 2],
        policy_layer_sizes: vec![256; 3],
        critic_layer_sizes: vec![256; 3],
        device: LibTorchDevice::Cuda(0),
        render_device: LibTorchDevice::Cpu,
        #[cfg(feature = "wandb")]
        wandb_project_name: Some("rlgym-ppo".into()),
        #[cfg(feature = "wandb")]
        wandb_run_name: Some("ppo-bot-v1".into()),
        ..Default::default()
    };

    let mut learner = config.init(create_env);
    learner.load();
    learner.learn();
}
