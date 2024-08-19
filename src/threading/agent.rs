use super::{game_inst::GameInstance, trajectory::Trajectory};
use crate::{
    ppo::discrete::DiscretePolicy, threading::trajectory::TrajectoryTensors, util::report::Report,
};
use rlgym_rs::{Action, Env, FullObs, Obs, Reward, StateSetter, Terminal, Truncate};
use std::{
    iter::repeat_with,
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, MutexGuard, RwLock,
    },
    thread::{self, sleep},
    time::{Duration, Instant},
};
use tch::{kind::FLOAT_CPU, Device, IndexOp, Kind, Tensor};

fn obs_to_tensor(obs: Rc<FullObs>) -> Tensor {
    let inner_size = obs[0].len();
    let tensor = Tensor::zeros([obs.len() as i64, inner_size as i64], FLOAT_CPU);

    for (i, obs) in obs.iter().enumerate() {
        tensor
            .slice(0, i as i64, (i + 1) as i64, 1)
            .copy_(&Tensor::from_slice(&obs[..inner_size]));
    }

    tensor
}

fn make_games_obs_tensor<SS, OBS, ACT, REW, TERM, TRUNC, SI>(
    games: &[GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>],
) -> Tensor
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    assert!(!games.is_empty());

    let obs_tensors = games
        .iter()
        .map(GameInstance::get_obs)
        .map(obs_to_tensor)
        .collect::<Vec<_>>();

    Tensor::concat(&obs_tensors, 0)
}

fn tensor_to_i32_vec(tensor: &Tensor) -> Vec<i32> {
    assert_eq!(tensor.dim(), 1);
    let tensor = tensor.to(Device::Cpu).detach().to_kind(Kind::Int);
    Vec::<i32>::try_from(tensor).unwrap()
}

#[derive(Clone)]
pub struct AgentConfig {
    pub deterministic: bool,
    pub num_games: usize,
    pub max_steps: u64,
    pub controls_update_frequency: u64,
    pub device: Device,
}

#[derive(Default)]
pub struct AgentControls {
    pub should_run: AtomicBool,
    pub paused: AtomicBool,
}

#[derive(Default)]
struct AgentData {
    trajectories: Mutex<Vec<Vec<Trajectory>>>,
    steps_collected: AtomicU64,
}

struct Agent<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    index: usize,
    game_instances: Vec<GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>>,
    times: Report,
    config: AgentConfig,
    data: Arc<AgentData>,
    controls: Arc<AgentControls>,
    policy: Arc<RwLock<Arc<DiscretePolicy>>>,
}

impl<SS, OBS, ACT, REW, TERM, TRUNC, SI> Agent<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new<F>(
        config: AgentConfig,
        create_env_fn: F,
        index: usize,
        data: Arc<AgentData>,
        controls: Arc<AgentControls>,
        policy: Arc<RwLock<Arc<DiscretePolicy>>>,
    ) -> Self
    where
        F: Fn() -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut game_instances = Vec::with_capacity(config.num_games);

        let mut traj = data.trajectories.lock().unwrap();
        traj.reserve(config.num_games);

        for _ in 0..config.num_games {
            let env = create_env_fn();
            let num_cars = env.num_cars();
            let game_instance = GameInstance::new(env);
            game_instances.push(game_instance);

            traj.push(Vec::from_iter(
                repeat_with(Trajectory::default).take(num_cars),
            ));
        }

        drop(traj);

        Self {
            index,
            game_instances,
            config,
            data,
            controls,
            times: Report::default(),
            policy,
        }
    }

    pub fn run(&mut self) {
        while !self.controls.should_run.load(Ordering::Relaxed) {
            sleep(Duration::from_millis(100));
        }

        self.game_instances.iter_mut().for_each(GameInstance::start);

        // Will stores our current observations for all our games
        let mut cur_obs_tensor = make_games_obs_tensor(&self.game_instances);

        let mut policy = self.policy.read().unwrap().clone();

        let mut steps_since_update = 0;

        'outer: loop {
            if steps_since_update == self.config.controls_update_frequency {
                while self.controls.paused.load(Ordering::Relaxed)
                    || self.data.steps_collected.load(Ordering::Relaxed) >= self.config.max_steps
                {
                    if !self.controls.should_run.load(Ordering::Relaxed) {
                        break 'outer;
                    }

                    sleep(Duration::from_millis(1));
                }

                policy = self.policy.read().unwrap().clone();

                steps_since_update = 0;
            } else {
                steps_since_update += 1;
            }

            // Infer the policy to get actions for all our agents in all our games
            let cur_obs_on_device =
                cur_obs_tensor.to_device_(self.config.device, Kind::Float, true, false);

            let action_results = {
                let policy_infer_start = Instant::now();
                let results = policy.get_action(&cur_obs_on_device, self.config.deterministic);

                let policy_infer_elapsed = policy_infer_start.elapsed();
                self.times["policy_infer_time"] += policy_infer_elapsed.as_secs_f64();
                results
            };

            let gym_step_start = Instant::now();

            let mut step_results = Vec::with_capacity(self.config.num_games);
            let mut action_offset = 0;

            for game in &mut self.game_instances {
                let num_cars = game.num_cars();

                let action_slice = action_results.action.slice(
                    0,
                    action_offset as i64,
                    (action_offset + num_cars) as i64,
                    1,
                );
                step_results.push(game.step(tensor_to_i32_vec(&action_slice)));

                action_offset += num_cars;
            }

            assert!(action_offset == action_results.action.size1().unwrap() as usize);

            let gym_step_elapsed = gym_step_start.elapsed();
            self.times["gym_step_time"] += gym_step_elapsed.as_secs_f64();

            let next_obs_tensor = make_games_obs_tensor(&self.game_instances);

            {
                let trajectory_append_start = Instant::now();
                let mut trajectories = self.data.trajectories.lock().unwrap();
                let mut player_offset = 0;

                for i in 0..self.config.num_games {
                    let num_cars = self.game_instances[i].num_cars();
                    let step_result = &step_results[i];

                    let done = if step_result.is_terminal { 1. } else { 0. };

                    let truncated = if step_result.truncated { 1. } else { 0. };

                    let t_done = Tensor::from_slice(&[done]);
                    let t_truncated = Tensor::from_slice(&[truncated]);

                    while num_cars > trajectories[i].len() {
                        trajectories[i].push(Trajectory::default());
                    }

                    for j in 0..num_cars {
                        trajectories[i][j].append_single_step(TrajectoryTensors::new(
                            cur_obs_tensor.i((player_offset + j) as i64),
                            action_results
                                .action
                                .i(((player_offset + j) as i64,))
                                .view([1]),
                            action_results
                                .log_prob
                                .i((player_offset + j) as i64)
                                .view([1]),
                            Tensor::from_slice(&[step_result.rewards[j]]),
                            next_obs_tensor.i((player_offset + j) as i64),
                            t_done.shallow_clone(),
                            t_truncated.shallow_clone(),
                        ));
                    }

                    self.data
                        .steps_collected
                        .fetch_add(num_cars as u64, Ordering::Relaxed);
                    player_offset += num_cars;
                }

                let trajectory_append_elapsed = trajectory_append_start.elapsed();
                self.times["trajectory_append_time"] += trajectory_append_elapsed.as_secs_f64();
            }

            cur_obs_tensor = next_obs_tensor;
        }
    }
}

pub struct AgentController {
    data: Arc<AgentData>,
    policy: Arc<RwLock<Arc<DiscretePolicy>>>,
    thread_handle: thread::JoinHandle<()>,
}

impl AgentController {
    pub fn new<F, SS, OBS, ACT, REW, TERM, TRUNC, SI>(
        config: AgentConfig,
        controls: Arc<AgentControls>,
        policy: Arc<DiscretePolicy>,
        create_env_fn: F,
        index: usize,
    ) -> Self
    where
        F: Fn() -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Send + 'static,
        SS: StateSetter<SI>,
        OBS: Obs<SI>,
        ACT: Action<SI, Input = Vec<i32>>,
        REW: Reward<SI>,
        TERM: Terminal<SI>,
        TRUNC: Truncate<SI>,
    {
        let data: Arc<AgentData> = Arc::default();
        let policy = Arc::new(RwLock::new(policy));

        let thread_data = data.clone();
        let thread_policy = policy.clone();
        let thread_handle = thread::spawn(move || {
            Agent::new(
                config,
                create_env_fn,
                index,
                thread_data,
                controls,
                thread_policy,
            )
            .run();
        });

        Self {
            data,
            policy,
            thread_handle,
        }
    }

    pub fn update_policy(&self, new_policy: Arc<DiscretePolicy>) {
        let mut active_policy = self.policy.write().unwrap();
        *active_policy = new_policy;
    }

    pub fn get_num_timesteps(&self) -> u64 {
        self.data.steps_collected.load(Ordering::Relaxed)
    }

    pub fn reset_timesteps(&self) {
        self.data.steps_collected.store(0, Ordering::Relaxed);
    }

    pub fn get_trajectories(&self) -> MutexGuard<Vec<Vec<Trajectory>>> {
        self.data.trajectories.lock().unwrap()
    }

    pub fn wait_for_close(self) {
        self.thread_handle.join().unwrap();
    }
}
