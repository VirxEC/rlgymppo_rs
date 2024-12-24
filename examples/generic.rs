use rlgymppo_rs::{
    rlgym_rs::{
        Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    },
    rocketsim_rs::{
        cxx::UniquePtr,
        glam_ext::{BallA, CarInfoA, GameStateA},
        init,
        sim::{Arena, CarConfig, CarControls, Team},
    },
    AvgTracker, Learner, LearnerConfig, PPOLearnerConfig, Report,
};
use std::{num::NonZeroUsize, thread::available_parallelism};

struct SharedInfo {
    rng: fastrand::Rng,
    avg_dist_to_ball: AvgTracker,
    avg_vel: AvgTracker,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: fastrand::Rng::new(),
            avg_dist_to_ball: AvgTracker::default(),
            avg_vel: AvgTracker::default(),
        }
    }
}

struct MySharedInfoProvider;

impl SharedInfoProvider<SharedInfo> for MySharedInfoProvider {
    fn reset(&mut self, _initial_state: &GameStateA, shared_info: &mut SharedInfo) {
        shared_info.avg_dist_to_ball.reset();
    }

    fn apply(&mut self, game_state: &GameStateA, shared_info: &mut SharedInfo) {
        for car in &game_state.cars {
            let dist_to_ball = car.state.pos.distance(game_state.ball.pos);
            shared_info.avg_dist_to_ball += dist_to_ball as f64;

            shared_info.avg_vel += car.state.vel.length() as f64;
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
        for _ in 0..shared_info.rng.u8(1..4) {
            let _ = arena.pin_mut().add_car(Team::Blue, octane);
            let _ = arena.pin_mut().add_car(Team::Orange, octane);
        }

        arena
            .pin_mut()
            .reset_to_random_kickoff(Some(shared_info.rng.i32(-1000..1000)));

        let mut state = arena.pin_mut().get_game_state();
        for car in &mut state.cars {
            car.state.vel.x = (shared_info.rng.f32() * 2. - 1.) * 1300.;
            car.state.vel.y = (shared_info.rng.f32() * 2. - 1.) * 1300.;
        }
        arena.pin_mut().set_game_state(&state).unwrap();
    }
}

struct MyObs;

impl MyObs {
    const ZERO_PADDING: usize = 3;
    const BALL_OBS: usize = 9;
    const CAR_OBS: usize = 9;

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
    fn get_obs_space(&self, _agent_id: u32, _shared_info: &SharedInfo) -> usize {
        Self::BALL_OBS + Self::CAR_OBS * Self::ZERO_PADDING * 2
    }

    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn build_obs(&mut self, state: &GameStateA, shared_info: &mut SharedInfo) -> FullObs {
        let mut obs = Vec::with_capacity(state.cars.len());

        let ball_obs = Self::get_ball_obs(&state.ball);
        let cars = Self::get_all_car_obs(&state.cars);

        let full_obs = self.get_obs_space(0, shared_info);
        for current_car in &state.cars {
            let mut obs_vec: Vec<f32> = Vec::with_capacity(full_obs);
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

            assert_eq!(obs_vec.len(), full_obs);
            obs.push(obs_vec);
        }

        assert!(obs.len() <= Self::ZERO_PADDING * 2);
        obs
    }
}

struct MyAction {
    actions_table: Vec<CarControls>,
}

impl Default for MyAction {
    fn default() -> Self {
        let mut actions_table = Vec::new();

        // ground
        for throttle in [-1.0, 0.0, 1.0] {
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
        for pitch in [-1.0, 0.0, 1.0] {
            for yaw in [-1.0, 0.0, 1.0] {
                for roll in [-1.0, 0.0, 1.0] {
                    for jump in [false, true] {
                        for boost in [false, true] {
                            // Only need roll for sideflip
                            if jump && yaw != 0.0 {
                                continue;
                            }

                            // Duplicate with ground
                            if pitch == 0.0 && roll == 0.0 && !jump {
                                continue;
                            }

                            // Enable handbrake for potential wavedashes
                            let handbrake = jump && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);
                            actions_table.push(CarControls {
                                throttle: 0.0,
                                steer: yaw,
                                boost,
                                handbrake,
                                jump,
                                pitch,
                                yaw,
                                roll,
                            });
                        }
                    }
                }
            }
        }

        Self { actions_table }
    }
}

impl Action<SharedInfo> for MyAction {
    type Input = Vec<i32>;

    fn get_tick_skip() -> u32 {
        8
    }

    fn get_action_space(&self, _agent_id: u32, _shared_info: &SharedInfo) -> usize {
        self.actions_table.len()
    }

    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn parse_actions(
        &mut self,
        actions: Self::Input,
        _state: &GameStateA,
        _shared_info: &mut SharedInfo,
    ) -> Vec<CarControls> {
        actions
            .into_iter()
            // .map(|action| dbg!(self.actions_table[dbg!(action) as usize]))
            .map(|action| self.actions_table[action as usize])
            .collect()
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

struct MyTerminal;

impl Terminal<SharedInfo> for MyTerminal {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn is_terminal(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> bool {
        let elapsed = state.tick_count as f32 / state.tick_rate / 60.0;

        // reset after some minutes
        if elapsed < 2.0 {
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
        CombinedReward::new(vec![Box::new(VelocityReward)]),
        MyTerminal,
        MyTruncate,
        SharedInfo::default(),
    )
}

fn step_callback(report: &mut Report, shared_info: &SharedInfo, _game_state: &GameStateA) {
    report["Avg. distance to ball"] = shared_info.avg_dist_to_ball.into();
    report["Avg. velocity"] = shared_info.avg_vel.into();
}

fn main() {
    init(None, true);

    // let num_threads = NonZeroUsize::new(1).unwrap();
    let num_threads = available_parallelism().unwrap();
    let num_games_per_thread = NonZeroUsize::new(12).unwrap();
    let mini_batch_size = 15_000;
    let batch_size = mini_batch_size * 5;
    let lr = 3e-4;

    let config = LearnerConfig {
        num_threads,
        num_games_per_thread,
        render: false,
        exp_buffer_size: batch_size,
        timesteps_per_save: 10_000_000,
        collection_timesteps_overflow: 0,
        check_update_frequency: 15,
        collection_during_learn: false,
        device: tch::Device::cuda_if_available(),
        ppo: PPOLearnerConfig {
            batch_size,
            mini_batch_size,
            epochs: 3,
            ent_coef: 0.01,
            policy_lr: lr,
            critic_lr: lr,
            policy_layer_sizes: vec![512, 512, 512, 512, 512],
            critic_layer_sizes: vec![512, 512, 512, 512, 512, 512],
            // policy_layer_sizes: vec![16, 16, 16],
            // critic_layer_sizes: vec![16, 16, 16],
            ..Default::default()
        },
        ..Default::default()
    };

    let mut learner = Learner::new(create_env, step_callback, config);
    learner.load();
    learner.learn();
    learner.save();
}
