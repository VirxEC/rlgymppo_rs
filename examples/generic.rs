#![recursion_limit = "256"]

use burn::backend::{LibTorch, libtorch::LibTorchDevice};
use itertools::repeat_n;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use rlgym::rocketsim::init_from_default;
use rlgymppo::{
    LearnerConfig, PpoLearnerConfig,
    backend::Autodiff,
    rlgym::{Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate},
    rocketsim::{
        Arena, ArenaState, BallState, CarBodyConfig, CarInfo, CarState, GameMode, Team, consts,
    },
    utils::{AvgTracker, Report, actions::DefaultAction},
};

struct SharedInfo {
    rng: SmallRng,
    start_tick: u64,
    metrics: Report,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: SmallRng::seed_from_u64(0),
            start_tick: 0,
            metrics: Report::default(),
        }
    }
}

impl SharedInfoProvider for SharedInfo {
    fn reset(&mut self, initial_state: &ArenaState) {
        self.start_tick = initial_state.tick_count;
    }

    fn update(&mut self, game_state: &ArenaState) {
        for (_, state) in &game_state.cars {
            let dist_to_ball = state.pos.distance(game_state.ball.pos);
            self.metrics["Avg. dist to ball"] += AvgTracker::new(dist_to_ball as f64, 1).into();
            self.metrics["Avg. velocity"] += AvgTracker::new(state.vel.length() as f64, 1).into();
        }
    }
}

struct MyStateSetter;

impl StateSetter<SharedInfo> for MyStateSetter {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SharedInfo) {
        arena.reset_to_random_kickoff(Some(shared_info.rng.random()));

        let mut state = arena.get_arena_state();
        for (info, state) in &mut state.cars {
            state.vel.x = shared_info.rng.random_range(-1300.0..1300.0);
            state.vel.y = shared_info.rng.random_range(-1300.0..1300.0);
            arena.set_car_state(info.idx, *state);
        }
    }
}

struct MyObs;

impl MyObs {
    const ZERO_PADDING: usize = 3;
    const BALL_OBS: usize = 9;
    const CAR_OBS: usize = 16;

    const OBS_SPACE: usize = Self::BALL_OBS + Self::CAR_OBS * Self::ZERO_PADDING * 2;

    fn get_ball_obs(ball: &BallState) -> Vec<f32> {
        let mut obs_vec = Vec::with_capacity(Self::BALL_OBS);
        obs_vec.extend(ball.pos.to_array());
        obs_vec.extend(ball.vel.to_array());
        obs_vec.extend(ball.ang_vel.to_array());

        assert_eq!(obs_vec.len(), Self::BALL_OBS);
        obs_vec
    }

    fn get_all_car_obs(cars: &[(CarInfo, CarState)]) -> Vec<(usize, Team, Vec<f32>)> {
        cars.iter()
            .map(|(info, state)| {
                if state.is_demoed {
                    let obs_vec = vec![0.0; Self::CAR_OBS];
                    return (info.idx, info.team, obs_vec);
                }

                let mut obs_vec = Vec::with_capacity(Self::CAR_OBS);
                obs_vec.extend(state.pos.to_array());
                obs_vec.extend(state.vel.to_array());
                obs_vec.extend(state.ang_vel.to_array());
                obs_vec.extend(state.rot_mat.x_axis.to_array());
                obs_vec.extend(state.rot_mat.z_axis.to_array());
                obs_vec.push(state.boost);

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

    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn build_obs(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> FullObs {
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

struct WeightedReward {
    func: Box<dyn Reward<SharedInfo>>,
    weight: f32,
}

struct CombinedWeightedRewards {
    rewards: Box<[WeightedReward]>,
}

macro_rules! new_rewards {
    ($(($reward:expr, $weight:expr)),* $(,)?) => {
        CombinedWeightedRewards {
            rewards: vec![$(WeightedReward {
                func: Box::new($reward) as Box<dyn Reward<SharedInfo>>,
                weight: $weight
            }),*].into_boxed_slice(),
        }
    };
}

impl Reward<SharedInfo> for CombinedWeightedRewards {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &ArenaState, shared_info: &mut SharedInfo) -> Vec<f32> {
        let mut rewards: Vec<f32> = vec![0.0; state.cars.len()];

        for reward in &mut self.rewards {
            let fn_rewards = reward.func.get_rewards(state, shared_info);

            for (total, extra) in rewards.iter_mut().zip(fn_rewards) {
                *total += extra * reward.weight;
            }
        }

        rewards
    }
}

pub struct FaceBallReward;

impl FaceBallReward {
    fn get_reward(car: &CarState, ball: &BallState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let car_to_ball = (ball.pos - car.pos).normalize();
        car.rot_mat.x_axis.dot(car_to_ball)
    }
}

impl Reward<SharedInfo> for FaceBallReward {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car, &state.ball))
            .collect()
    }
}

pub struct VelocityToBallReward;

impl VelocityToBallReward {
    fn get_reward(car: &CarState, ball: &BallState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let car_to_ball_norm = (ball.pos - car.pos).normalize_or_zero();
        let car_to_ball_dot = car.vel.dot(car_to_ball_norm);

        car_to_ball_dot / 2300.0
    }
}

impl Reward<SharedInfo> for VelocityToBallReward {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car, &state.ball))
            .collect()
    }
}

struct OnGoal;

fn ball_within_hoops_goal_xy_margin_eq(x: f32, y: f32) -> f32 {
    const SCALE_Y: f32 = 0.9;
    const OFFSET_Y: f32 = 2770.0;
    const RADIUS_SQ: f32 = 716.0 * 716.0;

    let dy = y.abs() * SCALE_Y - OFFSET_Y;
    let dist_sq = x * x + dy * dy;
    dist_sq - RADIUS_SQ
}

impl Terminal<SharedInfo> for OnGoal {
    fn reset(&mut self, _initial_state: &ArenaState, _shared_info: &mut SharedInfo) {}

    fn is_terminal(&mut self, state: &ArenaState, _shared_info: &mut SharedInfo) -> bool {
        match state.game_mode() {
            GameMode::Soccar | GameMode::Heatseeker | GameMode::Snowday => {
                state.ball.pos.y.abs()
                    > consts::goal::SOCCAR_GOAL_SCORE_BASE_THRESHOLD_Y
                        + consts::ball::get_radius(state.game_mode())
            }
            GameMode::Hoops => {
                if state.ball.pos.z < consts::goal::HOOPS_GOAL_SCORE_THRESHOLD_Z {
                    ball_within_hoops_goal_xy_margin_eq(state.ball.pos.x, state.ball.pos.y) < 0.0
                } else {
                    false
                }
            }
            GameMode::Dropshot => {
                state.ball.pos.z < -consts::ball::get_radius(state.game_mode()) * 1.75
            }
            GameMode::TheVoid => false,
        }
    }
}

#[derive(Default)]
struct EpisodeDurationMax {
    episode_duration: f32,
}

impl Truncate<SharedInfo> for EpisodeDurationMax {
    fn reset(&mut self, _initial_state: &ArenaState, shared_info: &mut SharedInfo) {
        self.episode_duration = shared_info.rng.random_range(2.0..5.0);
    }

    fn should_truncate(&mut self, state: &ArenaState, shared_info: &mut SharedInfo) -> bool {
        const SECS_TO_MIN: f32 = 1.0 / 60.0;

        let elapsed =
            (state.tick_count - shared_info.start_tick) as f32 * consts::TICK_TIME * SECS_TO_MIN;

        // reset after some minutes
        if elapsed < self.episode_duration {
            return false;
        }

        state.ball.pos.z < 94.5
    }
}

fn create_env(
    game_id: Option<usize>,
) -> Env<
    MyStateSetter,
    MyObs,
    DefaultAction<6>,
    CombinedWeightedRewards,
    OnGoal,
    EpisodeDurationMax,
    SharedInfo,
> {
    // `game_id` is None for the game used to calculate the policy obs/action space,
    // as well as for the renderer.
    // Otherwise, every env gets a unique id starting from 0 and incrementing by 1.
    let game_id = game_id.unwrap_or(0);

    let mut arena = Arena::new(GameMode::Soccar);

    // pseudo-random game mode: 1v1, 2v2, 3v3
    // using game id ensures an equal, predictable distribution of game modes
    // it's not the best idea to change the number of players between episodes
    for _ in 0..=game_id % 3 {
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    }

    Env::new(
        arena,
        MyStateSetter,
        MyObs,
        DefaultAction::default(),
        new_rewards![(VelocityToBallReward, 1.0), (FaceBallReward, 0.2)],
        OnGoal,
        EpisodeDurationMax::default(),
        SharedInfo::default(),
    )
}

fn step_callback(report: &mut Report, shared_info: &mut SharedInfo, _game_state: &ArenaState) {
    *report += &shared_info.metrics;
    shared_info.metrics.clear();
}

fn main() {
    init_from_default(cfg!(not(debug_assertions))).unwrap();

    let mini_batch_size = 20_000;
    let batch_size = mini_batch_size * 2;
    let lr = 2e-4;

    // Router will fallback to NdArray if Wgpu is not available
    // Realistically more useful for using CUDA and falling back to NdArray
    let config = LearnerConfig::<Autodiff<LibTorch>> {
        render: false,
        num_threads: 8,
        num_games_per_thread: 256,
        exp_buffer_size: batch_size,
        timesteps_per_save: 10_000_000,
        checkpoints_limit: Some(3),
        ppo: PpoLearnerConfig {
            batch_size,
            mini_batch_size,
            epochs: 1,
            learning_rate: lr,
            ..Default::default()
        },
        policy_layer_sizes: vec![256; 4],
        critic_layer_sizes: vec![512; 4],
        device: LibTorchDevice::Cuda(0),
        ..Default::default()
    };

    let mut learner = config.init(create_env, step_callback);
    learner.load();
    learner.learn();
}
