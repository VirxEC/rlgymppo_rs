#![recursion_limit = "256"]

use rand::{Rng, SeedableRng, rngs::SmallRng};
use rlgymppo::{
    LearnerConfig, PpoLearnerConfig,
    backend::{Autodiff, NdArray, Rocm, Router, Wgpu},
    rlgym::{
        Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    },
    rocketsim_rs::{
        cxx::UniquePtr,
        glam_ext::{BallA, CarInfoA, GameStateA},
        init,
        sim::{Arena, CarConfig, CarControls, Team},
    },
    utils::{AvgTracker, Report},
};

struct SharedInfo {
    rng: SmallRng,
    metrics: Report,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: SmallRng::from_os_rng(),
            metrics: Report::default(),
        }
    }
}

struct MySharedInfoProvider;

impl SharedInfoProvider<SharedInfo> for MySharedInfoProvider {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn apply(&mut self, game_state: &GameStateA, shared_info: &mut SharedInfo) {
        for car in &game_state.cars {
            let dist_to_ball = car.state.pos.distance(game_state.ball.pos);
            shared_info.metrics["Avg. dist to ball"] +=
                AvgTracker::new(dist_to_ball as f64, 1).into();
            shared_info.metrics["Avg. velocity"] +=
                AvgTracker::new(car.state.vel.length() as f64, 1).into();
        }
    }
}

struct MyStateSetter;

impl StateSetter<SharedInfo> for MyStateSetter {
    fn apply(&mut self, arena: &mut UniquePtr<Arena>, shared_info: &mut SharedInfo) {
        arena.pin_mut().reset_tick_count();

        // remove previous cars
        for car_id in arena.pin_mut().get_cars() {
            arena.pin_mut().remove_car(car_id).unwrap();
        }

        // random game mode, 1v1, 2v2, or 3v3
        let octane = CarConfig::octane();
        for _ in 0..shared_info.rng.random_range(1..4) {
            let _ = arena.pin_mut().add_car(Team::Blue, octane);
            let _ = arena.pin_mut().add_car(Team::Orange, octane);
        }

        arena
            .pin_mut()
            .reset_to_random_kickoff(Some(shared_info.rng.random()));

        let mut state = arena.pin_mut().get_game_state();
        for car in &mut state.cars {
            car.state.vel.x = shared_info.rng.random_range(-1300.0..1300.0);
            car.state.vel.y = shared_info.rng.random_range(-1300.0..1300.0);
        }
        arena.pin_mut().set_game_state(&state).unwrap();

        arena.pin_mut().set_goal_scored_callback(
            |arena, _, _| {
                arena.reset_to_random_kickoff(None);
            },
            0,
        );
    }
}

struct MyObs;

impl MyObs {
    const ZERO_PADDING: usize = 3;
    const BALL_OBS: usize = 9;
    const CAR_OBS: usize = 9;

    const OBS_SPACE: usize = Self::BALL_OBS + Self::CAR_OBS * Self::ZERO_PADDING * 2;

    fn get_ball_obs(ball: &BallA) -> Vec<f32> {
        let mut obs_vec = Vec::with_capacity(Self::BALL_OBS);
        obs_vec.extend(ball.pos.to_array());
        obs_vec.extend(ball.vel.to_array());
        obs_vec.extend(ball.ang_vel.to_array());

        obs_vec
    }

    fn get_all_car_obs(cars: &[CarInfoA]) -> Vec<(u32, Team, Vec<f32>)> {
        cars.iter()
            .map(|car| {
                let mut obs_vec = Vec::with_capacity(Self::CAR_OBS);
                obs_vec.extend(car.state.pos.to_array());
                obs_vec.extend(car.state.vel.to_array());
                obs_vec.extend(car.state.ang_vel.to_array());

                (car.id, car.team, obs_vec)
            })
            .collect()
    }
}

impl Obs<SharedInfo> for MyObs {
    fn get_obs_space(&self, _shared_info: &SharedInfo) -> usize {
        Self::OBS_SPACE
    }

    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn build_obs(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> FullObs {
        let mut obs = Vec::with_capacity(state.cars.len());

        let ball_obs = Self::get_ball_obs(&state.ball);
        let cars = Self::get_all_car_obs(&state.cars);

        for current_car in &state.cars {
            let mut obs_vec: Vec<f32> = Vec::with_capacity(Self::OBS_SPACE);
            obs_vec.extend(&ball_obs);

            // current car's obs
            obs_vec.extend(
                &cars
                    .iter()
                    .find(|(car_id, _, _)| *car_id == current_car.id)
                    .unwrap()
                    .2,
            );

            // teammate's obs
            let mut num_teammates = 0;
            for (car_id, team, obs) in &cars {
                if *team == current_car.team && *car_id != current_car.id {
                    obs_vec.extend(obs);
                    num_teammates += 1;
                }
            }

            // zero padding
            for _ in 0..Self::ZERO_PADDING - num_teammates - 1 {
                obs_vec.extend(vec![0.0; Self::CAR_OBS]);
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
                obs_vec.extend(vec![0.0; Self::CAR_OBS]);
            }

            assert_eq!(obs_vec.len(), Self::OBS_SPACE);
            obs.push(obs_vec);
        }

        assert!(obs.len() <= Self::ZERO_PADDING * 2);
        obs
    }
}

struct MyAction {
    actions_table: Vec<CarControls>,
    action_buffer: [(u32, CarControls); MyObs::ZERO_PADDING * 2],
}

impl Default for MyAction {
    fn default() -> Self {
        let mut actions_table = Vec::new();

        // ground
        for throttle in [-1.0, 1.0] {
            for steer in [-1.0, 0.0, 1.0] {
                for boost in [false, true] {
                    for handbrake in [false, true] {
                        if boost && throttle != 1.0 {
                            continue;
                        }

                        actions_table.push(CarControls {
                            throttle,
                            steer,
                            boost,
                            handbrake,
                            jump: false,
                            pitch: 0.0,
                            yaw: 0.0,
                            roll: 0.0,
                        });
                    }
                }
            }
        }

        // aerial
        // for pitch in [-1.0, 0.0, 1.0] {
        //     for yaw in [-1.0, 0.0, 1.0] {
        //         for roll in [-1.0, 0.0, 1.0] {
        //             for jump in [false, true] {
        //                 for boost in [false, true] {
        //                     // Only need roll for sideflip
        //                     if jump && yaw != 0.0 {
        //                         continue;
        //                     }

        //                     // Duplicate with ground
        //                     if pitch == 0.0 && roll == 0.0 && !jump {
        //                         continue;
        //                     }

        //                     // Enable handbrake for potential wavedashes
        //                     let handbrake = jump && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);
        //                     actions_table.push(CarControls {
        //                         throttle: 0.0,
        //                         steer: yaw,
        //                         boost,
        //                         handbrake,
        //                         jump,
        //                         pitch,
        //                         yaw,
        //                         roll,
        //                     });
        //                 }
        //             }
        //         }
        //     }
        // }

        Self {
            actions_table,
            action_buffer: Default::default(),
        }
    }
}

impl Action<SharedInfo> for MyAction {
    type Input = usize;

    fn get_tick_skip() -> u32 {
        8
    }

    fn get_action_space(&self, _shared_info: &SharedInfo) -> usize {
        self.actions_table.len()
    }

    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn parse_actions(
        &mut self,
        actions: &[usize],
        state: &GameStateA,
        _shared_info: &mut SharedInfo,
    ) -> &[(u32, CarControls)] {
        for ((buf, car), action) in self.action_buffer.iter_mut().zip(&state.cars).zip(actions) {
            *buf = (car.id, self.actions_table[*action]);
        }

        &self.action_buffer[..state.cars.len()]
    }
}

struct CombinedReward {
    rewards: Vec<Box<dyn Reward<SharedInfo>>>,
}

impl CombinedReward {
    fn new(rewards: Vec<Box<dyn Reward<SharedInfo>>>) -> Self {
        Self { rewards }
    }
}

impl Reward<SharedInfo> for CombinedReward {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> Vec<f32> {
        let mut rewards: Vec<f32> = vec![0.0; state.cars.len()];

        for reward_fn in &mut self.rewards {
            let mut fn_rewards = reward_fn.get_rewards(state, _shared_info);

            for (i, reward) in fn_rewards.drain(..).enumerate() {
                rewards[i] += reward;
            }
        }

        rewards
    }
}

struct DistanceToBall;

impl Reward<SharedInfo> for DistanceToBall {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|car| -car.state.pos.distance(state.ball.pos) / 12000.)
            .collect()
    }
}

struct VelocityReward;

impl Reward<SharedInfo> for VelocityReward {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|car| car.state.vel.length() / 2300.)
            .collect()
    }
}

#[derive(Default)]
struct MyTerminal {
    episode_duration: f32,
}

impl Terminal<SharedInfo> for MyTerminal {
    fn reset(&mut self, _initial_state: &GameStateA, shared_info: &mut SharedInfo) {
        self.episode_duration = shared_info.rng.random_range(2.0..5.0);
    }

    fn is_terminal(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> bool {
        let elapsed = state.tick_count as f32 / state.tick_rate / 60.0;

        // reset after some minutes
        if elapsed < self.episode_duration {
            return false;
        }

        state.ball.pos.z < 94.5
    }
}

struct MyTruncate;

impl Truncate<SharedInfo> for MyTruncate {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn should_truncate(&mut self, _state: &GameStateA, _shared_info: &mut SharedInfo) -> bool {
        false
    }
}

fn create_env() -> Env<
    MyStateSetter,
    MySharedInfoProvider,
    MyObs,
    MyAction,
    CombinedReward,
    MyTerminal,
    MyTruncate,
    SharedInfo,
> {
    let mut arena = Arena::default_standard();
    arena
        .pin_mut()
        .set_goal_scored_callback(|arena, _, _| arena.reset_to_random_kickoff(None), 0);

    // set the initial environment to the max number of cars
    // this will help avoid errors during training
    let octane = CarConfig::octane();
    for _ in 0..MyObs::ZERO_PADDING {
        let _ = arena.pin_mut().add_car(Team::Blue, octane);
        let _ = arena.pin_mut().add_car(Team::Orange, octane);
    }

    Env::new(
        arena,
        MyStateSetter,
        MySharedInfoProvider,
        MyObs,
        MyAction::default(),
        CombinedReward::new(vec![Box::new(VelocityReward), Box::new(DistanceToBall)]),
        MyTerminal::default(),
        MyTruncate,
        SharedInfo::default(),
    )
}

fn step_callback(report: &mut Report, shared_info: &mut SharedInfo, _game_state: &GameStateA) {
    *report += &shared_info.metrics;
    shared_info.metrics.clear();
}

fn main() {
    init(None, true);

    // let num_threads = if cfg!(debug_assertions) {
    //     NonZeroUsize::new(1).unwrap()
    // } else {
    //     available_parallelism().unwrap()
    // };
    let mini_batch_size = 20000;
    let batch_size = mini_batch_size * 3;
    let lr = 3e-4;

    let config = LearnerConfig::<Autodiff<Router<(Rocm, Wgpu, NdArray)>>> {
        // render: true,
        // num_threads,
        num_games_per_thread: 5,
        exp_buffer_size: batch_size,
        timesteps_per_save: 10_000_000,
        checkpoints_limit: Some(3),
        ppo: PpoLearnerConfig {
            batch_size,
            mini_batch_size,
            epochs: 2,
            learning_rate: lr,
            ..Default::default()
        },
        policy_layer_sizes: vec![512; 4],
        critic_layer_sizes: vec![512; 4],
        ..Default::default()
    };

    let mut learner = config.init(create_env, step_callback);
    learner.load();
    learner.learn();
}
