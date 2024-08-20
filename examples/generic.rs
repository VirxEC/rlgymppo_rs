use rlgym_rs::{FullObs, SharedInfoProvider, Truncate};
use rlgymppo_rs::{
    rlgym_rs::{Action, Env, Obs, Reward, StateSetter, Terminal},
    rocketsim_rs::{
        cxx::UniquePtr,
        glam_ext::{BallA, CarInfoA, GameStateA},
        init,
        sim::{Arena, CarConfig, CarControls, Team},
    },
    Learner, LearnerConfig, PPOLearnerConfig, Report,
};
use std::{cmp::Ordering, num::NonZeroUsize, thread::available_parallelism};
use tch::Cuda;

struct SharedInfo {
    rng: fastrand::Rng,
}

impl Default for SharedInfo {
    fn default() -> Self {
        Self {
            rng: fastrand::Rng::new(),
        }
    }
}

struct MySharedInfoProvider;

impl SharedInfoProvider<SharedInfo> for MySharedInfoProvider {
    fn reset(&mut self, initial_state: &GameStateA, shared_info: &mut SharedInfo) {}
    fn apply(&mut self, game_state: &GameStateA, shared_info: &mut SharedInfo) {}
}

struct MyStateSetter;

impl StateSetter<SharedInfo> for MyStateSetter {
    fn apply(&mut self, arena: &mut UniquePtr<Arena>, shared_info: &mut SharedInfo) {
        arena.pin_mut().reset_tick_count();

        let car_ids = arena.pin_mut().get_cars();
        let mode = shared_info.rng.u8(1..4);

        match (mode as usize).cmp(&(car_ids.len() / 2)) {
            Ordering::Less => {
                let to_remove = car_ids.len() / 2 - mode as usize;
                for car_id in car_ids.into_iter().take(to_remove) {
                    arena.pin_mut().remove_car(car_id).unwrap();
                }
            }
            Ordering::Greater => {
                let to_add = mode as usize - car_ids.len() / 2;
                let car_config = CarConfig::octane();
                for _ in 0..to_add {
                    let _ = arena.pin_mut().add_car(Team::Blue, car_config);
                    let _ = arena.pin_mut().add_car(Team::Orange, car_config);
                }
            }
            Ordering::Equal => {}
        }

        arena
            .pin_mut()
            .reset_to_random_kickoff(Some(shared_info.rng.i32(-1000..1000)));
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

        for throttle in [1.0, 0.0, -1.0] {
            for steer in [1.0, 0.0, -1.0] {
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

struct DistanceToBallReward;

impl Reward<SharedInfo> for DistanceToBallReward {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn get_rewards(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|car| {
                let car_ball_dist = car.state.pos.distance(state.ball.pos);

                -car_ball_dist / 12_000.
            })
            .collect()
    }
}

struct MyTerminal;

impl Terminal<SharedInfo> for MyTerminal {
    fn reset(&mut self, _initial_state: &GameStateA, _shared_info: &mut SharedInfo) {}

    fn is_terminal(&mut self, state: &GameStateA, _shared_info: &mut SharedInfo) -> bool {
        // reset after 5 minutes
        let elapsed = state.tick_count as f32 / state.tick_rate / 60.0;

        if elapsed < 5.0 {
            return false;
        }

        println!("Episode lasted {:.2} minutes", elapsed);
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

    Env::new(
        arena,
        MyStateSetter,
        MySharedInfoProvider,
        MyObs,
        MyAction::default(),
        CombinedReward::new(vec![Box::new(DistanceToBallReward)]),
        MyTerminal,
        MyTruncate,
        SharedInfo::default(),
    )
}

fn step_callback(report: &mut Report, shared_info: &SharedInfo, game_state: &GameStateA) {}

fn main() {
    init(None, true);

    // let num_threads = NonZeroUsize::new(3).unwrap();
    let num_threads = available_parallelism().unwrap();
    let num_games_per_thread = NonZeroUsize::new(16).unwrap();
    let mini_batch_size = 20_000;
    let batch_size = mini_batch_size * 3;

    let config = LearnerConfig {
        num_threads,
        num_games_per_thread,
        render: false,
        exp_buffer_size: batch_size,
        timestep_limit: 1_000_000,
        timesteps_per_save: 10_000_000,
        collection_timesteps_overflow: 0,
        controls_update_frequency: 15,
        collection_during_learn: false,
        ppo: PPOLearnerConfig {
            batch_size,
            mini_batch_size,
            ..Default::default()
        },
        device: if Cuda::is_available() {
            tch::Device::Cuda(0)
        } else {
            tch::Device::Cpu
        },
        ..Default::default()
    };

    let mut learner = Learner::new(create_env, step_callback, config);
    learner.learn();
}
