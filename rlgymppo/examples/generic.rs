#![recursion_limit = "256"]

use burn::backend::{Rocm, rocm::RocmDevice};
use itertools::repeat_n;
use rand::{Rng, SeedableRng, rng, rngs::SmallRng};
use rlgym::rocketsim::{ArenaEvent, Vec3A, consts, init_from_default};
use rlgymppo::{
    LearnerConfig, PpoLearnerConfig, SelfPlayConfig, any_terminal,
    backend::Autodiff,
    combined_rewards,
    rlgym::{Env, FullObs, GameState, Obs, SharedInfoProvider},
    rocketsim::{Arena, BallState, CarBodyConfig, CarInfo, CarState, GameMode, Team},
    utils::{
        AvgTracker, Report,
        actions::DefaultAction,
        rewards,
        shared_info::{SharedInfoReport, SharedInfoRng},
        state_setters::{KickoffState, RandomState, WeightedState},
        terminal::{AnyTerminal, NoTouchCondition, OnGoalCondition, RandomGameEndedCondition},
    },
    weighted_state,
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

struct MyObs;

impl MyObs {
    const ZERO_PADDING: usize = 3;
    const BALL_OBS: usize = 9;
    const CAR_OBS: usize = 19;

    const OBS_SPACE: usize = Self::BALL_OBS + Self::CAR_OBS * Self::ZERO_PADDING * 2;

    const POS_COEF: Vec3A = Vec3A::new(1.0 / 4096.0, 1.0 / 5120.0, 1.0 / 2044.0);

    const VEL_COEF: f32 = 1.0 / consts::ball::MAX_SPEED;
    const ANG_VEL_COEF: f32 = 1.0 / consts::ball::MAX_ANG_SPEED;

    const BOOST_COEF: f32 = 1.0 / consts::car::boost::MAX;
    const DEMO_COEF: f32 = 1.0 / consts::car::spawn::RESPAWN_TIME;

    fn get_ball_obs(ball: &BallState) -> Vec<f32> {
        let mut obs_vec = Vec::with_capacity(Self::BALL_OBS);
        obs_vec.extend((ball.pos * Self::POS_COEF).to_array());
        obs_vec.extend((ball.vel * Self::VEL_COEF).to_array());
        obs_vec.extend((ball.ang_vel * Self::ANG_VEL_COEF).to_array());

        assert_eq!(obs_vec.len(), Self::BALL_OBS);
        obs_vec
    }

    fn get_all_car_obs(cars: &[(CarInfo, CarState)]) -> Vec<(usize, Team, Vec<f32>)> {
        cars.iter()
            .map(|(info, state)| {
                if state.is_demoed {
                    let mut obs_vec = vec![0.0; Self::CAR_OBS];
                    obs_vec[Self::CAR_OBS - 1] = state.demo_respawn_timer * Self::DEMO_COEF;

                    return (info.idx, info.team, obs_vec);
                }

                let mut obs_vec = Vec::with_capacity(Self::CAR_OBS);
                obs_vec.extend((state.pos * Self::POS_COEF).to_array());
                obs_vec.extend((state.vel * Self::VEL_COEF).to_array());
                obs_vec.extend((state.ang_vel * Self::ANG_VEL_COEF).to_array());
                obs_vec.extend(state.rot_mat.x_axis.to_array());
                obs_vec.extend(state.rot_mat.z_axis.to_array());
                obs_vec.push(state.boost * Self::BOOST_COEF);
                obs_vec.push(f32::from(state.is_on_ground));
                obs_vec.push(f32::from(state.has_flip_or_jump()));
                obs_vec.push(state.demo_respawn_timer * Self::DEMO_COEF);

                assert_eq!(obs_vec.len(), Self::CAR_OBS);
                (info.idx, info.team, obs_vec)
            })
            .collect()
    }
}

impl Obs<SharedInfo> for MyObs {
    fn get_obs_space(&self, _shared_info: &SharedInfo) -> usize {
        Self::OBS_SPACE
    }

    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SharedInfo) {}

    fn build_obs(&mut self, state: &GameState, _shared_info: &mut SharedInfo) -> FullObs {
        let mut obs = Vec::with_capacity(state.cars.len());

        let ball_obs = Self::get_ball_obs(&state.ball);
        let cars = Self::get_all_car_obs(&state.cars);

        for (current_car, _) in &state.cars {
            let mut obs_vec: Vec<f32> = Vec::with_capacity(Self::OBS_SPACE);
            obs_vec.extend(&ball_obs);

            // current car's obs
            obs_vec.extend(
                &cars
                    .iter()
                    .find(|(car_id, _, _)| *car_id == current_car.idx)
                    .unwrap()
                    .2,
            );

            // teammate's obs
            let mut num_teammates = 0;
            for (car_id, team, obs) in &cars {
                if *team == current_car.team && *car_id != current_car.idx {
                    obs_vec.extend(obs);
                    num_teammates += 1;
                }
            }

            // zero padding
            for _ in 0..Self::ZERO_PADDING - num_teammates - 1 {
                obs_vec.extend(repeat_n(0.0, Self::CAR_OBS));
            }

            // opponent's obs
            let mut num_opponents = 0;
            for (_, team, obs) in &cars {
                if *team != current_car.team {
                    obs_vec.extend(obs);
                    num_opponents += 1;
                }
            }

            // zero padding
            for _ in 0..Self::ZERO_PADDING - num_opponents {
                obs_vec.extend(repeat_n(0.0, Self::CAR_OBS));
            }

            assert_eq!(obs_vec.len(), Self::OBS_SPACE);
            obs.push(obs_vec);
        }

        assert!(obs.len() <= Self::ZERO_PADDING * 2);
        obs
    }
}

const MIN_GAME_DURATION: u64 = 60 * 120; // 1 minute in ticks
const MAX_GAME_DURATION: u64 = 3 * 60 * 120; // 3 minutes in ticks
type GameEndCond = RandomGameEndedCondition<MIN_GAME_DURATION, MAX_GAME_DURATION, SharedInfo>;

const MAX_NO_TOUCH_DURATION: u64 = 30 * 120; // 30 seconds in ticks

#[allow(clippy::type_complexity)]
fn create_env(
    game_id: Option<usize>,
) -> Env<
    WeightedState<SharedInfo>,
    MyObs,
    DefaultAction<6>,
    rewards::CombinedRewards<SharedInfo>,
    AnyTerminal<SharedInfo>,
    NoTouchCondition<MAX_NO_TOUCH_DURATION, SharedInfo>,
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
            RandomState<true, false, true, SharedInfo>, 0.4;
            RandomState<true, true, true, SharedInfo>, 0.2;
            RandomState<true, true, false, SharedInfo>, 0.3;
        ],
        MyObs,
        DefaultAction::default(),
        combined_rewards![
            "Reward/In Air", rewards::AirReward => 0.05;
            "Reward/Touch ball", rewards::BallTouchReward => 10.0;
            "Reward/Face ball", rewards::FaceBallReward => 0.2;
            "Reward/Velocity to ball", rewards::VelocityToBallReward => 1.0;
            // "Reward/Velocity ball to goal", rewards::ZeroSumReward::new(
            //     rewards::VelocityBallToGoalReward, 1.0, 1.0
            // ) => 2.0;
        ],
        any_terminal![OnGoalCondition<SharedInfo>, GameEndCond],
        NoTouchCondition::default(),
        SharedInfo::default(),
    )
}

fn main() {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    let mini_batch_size = 40_000;
    let batch_size = mini_batch_size * 2;
    let lr = 2e-4;

    // Router will fallback to NdArray if Wgpu is not available
    // Realistically more useful for using CUDA and falling back to NdArray
    let config = LearnerConfig::<Autodiff<Rocm>> {
        // if the renderer is on by default or not (can be toggled at runtime)
        render: false,
        // !!! WATCH OUT !!!
        //
        // 6 (cars in a 3v3) * 180 (max seconds per episode) * 15 (steps per second, 120 tps / 8 tick skip = 8)
        // = 16_200 (!!!)
        // Each thread may run over by almost 11k ticks! this is minimized due to randomness,
        // but if your total batch size is small, keep the number of threads low.
        //
        // Each thread is faster than you think!
        //
        // Only increase the number of threads if your cpu is having trouble saturating your gpu!
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
            epochs: 2,
            learning_rate: lr,
            // This scales differently than "ent_coef" in other frameworks;
            // This is the scale for normalized entropy,
            // which means you won't have to change it if you add more actions
            entropy_scale: 0.018,
            ..Default::default()
        },
        self_play: SelfPlayConfig {
            save_policy_versions: true,
            ts_per_version: 100_000_000,
            max_old_versions: 10,
            train_against_old_versions: false,
            train_against_old_chance: 0.15,
        },
        shared_head_layer_sizes: vec![256],
        policy_layer_sizes: vec![512; 2],
        critic_layer_sizes: vec![512; 4],
        device: RocmDevice::new(0),
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
