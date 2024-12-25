use super::{
    game_inst::{GameInstance, GameMetrics},
    trajectory::Trajectory,
};
use crate::{
    ppo::discrete::DiscretePolicy,
    threading::trajectory::TrajectoryTensors,
    util::{compute::NonBlockingTransfer, report::Report},
};
use rlgym_rs::{
    rocketsim_rs::glam_ext::GameStateA, Action, Env, FullObs, Obs, Reward, SharedInfoProvider,
    StateSetter, Terminal, Truncate,
};
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

fn make_games_obs_tensor<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
    games: &[GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>],
    get_next: bool,
) -> Tensor
where
    C: Fn(&mut Report, &SI, &GameStateA) + Clone,
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    assert!(!games.is_empty());

    let obs_tensors = if get_next {
        games
            .iter()
            .map(GameInstance::get_next_obs)
            .map(obs_to_tensor)
            .collect::<Vec<_>>()
    } else {
        games
            .iter()
            .map(GameInstance::get_obs)
            .map(obs_to_tensor)
            .collect::<Vec<_>>()
    };

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
    metrics: Mutex<GameMetrics>,
    steps_collected: AtomicU64,
}

struct Agent<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &SI, &GameStateA),
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    game_instances: Vec<GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>>,
    config: AgentConfig,
    data: Arc<AgentData>,
    controls: Arc<AgentControls>,
    policy: Arc<RwLock<Arc<DiscretePolicy>>>,
}

impl<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI> Agent<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &SI, &GameStateA) + Clone,
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new<F>(
        config: AgentConfig,
        create_env_fn: F,
        step_callback: C,
        data: Arc<AgentData>,
        controls: Arc<AgentControls>,
        policy: Arc<RwLock<Arc<DiscretePolicy>>>,
    ) -> Self
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut game_instances = Vec::with_capacity(config.num_games);

        let mut traj = data.trajectories.lock().unwrap();
        traj.reserve_exact(config.num_games);

        for _ in 0..config.num_games {
            let env = create_env_fn();
            let num_cars = env.num_cars();
            let game_instance = GameInstance::new(env, step_callback.clone());
            game_instances.push(game_instance);

            traj.push(Vec::from_iter(
                repeat_with(Trajectory::default).take(num_cars),
            ));
        }

        drop(traj);

        Self {
            game_instances,
            config,
            data,
            controls,
            policy,
        }
    }

    pub fn run(&mut self) {
        while !self.controls.should_run.load(Ordering::Relaxed) {
            sleep(Duration::from_millis(100));
        }

        self.game_instances.iter_mut().for_each(GameInstance::start);

        // Will store our current observations for all our games
        let mut cur_obs_tensor = make_games_obs_tensor(&self.game_instances, false);

        let mut policy = self.policy.read().unwrap().clone();

        let mut since_controls_update = 0;
        let mut since_metrics_update = 0;

        'outer: loop {
            if since_controls_update == self.config.controls_update_frequency {
                while self.controls.paused.load(Ordering::Relaxed)
                    || self.data.steps_collected.load(Ordering::Relaxed) >= self.config.max_steps
                {
                    if !self.controls.should_run.load(Ordering::Relaxed) {
                        break 'outer;
                    }

                    sleep(Duration::from_millis(1));
                }

                policy = self.policy.read().unwrap().clone();

                since_controls_update = 0;
            } else {
                since_controls_update += 1;
            }

            if since_metrics_update == self.config.controls_update_frequency {
                let mut metrics = self.data.metrics.lock().unwrap();

                for game in &mut self.game_instances {
                    *metrics += game.get_metrics();
                    game.reset_metrics();
                }

                since_metrics_update = 0;
            } else {
                since_metrics_update += 1;
            }

            // Infer the policy to get actions for all our agents in all our games
            let cur_obs_on_device = cur_obs_tensor.no_block_to(self.config.device);

            let (action_results, policy_infer_elapsed) = {
                let policy_infer_start = Instant::now();
                let results = policy.get_action(&cur_obs_on_device, self.config.deterministic);

                (results, policy_infer_start.elapsed())
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

                step_results.push(game.step(tensor_to_i32_vec(&action_slice), false));

                action_offset += num_cars;
            }

            assert!(action_offset == action_results.action.size1().unwrap() as usize);
            let env_step_elapsed = gym_step_start.elapsed();
            let next_obs_tensor = make_games_obs_tensor(&self.game_instances, true);
            let mut can_be_reused = true;

            let trajectory_append_elapsed = {
                let trajectory_append_start = Instant::now();
                let mut trajectories = self.data.trajectories.lock().unwrap();
                let mut player_offset = 0;

                for i in 0..self.config.num_games {
                    let step_result = &step_results[i];
                    let num_cars = step_result.state.cars.len();

                    if num_cars != self.game_instances[i].num_cars() {
                        can_be_reused = false;
                    }

                    let done = if step_result.is_terminal { 1. } else { 0. };

                    let truncated = if step_result.truncated { 1. } else { 0. };

                    let t_done = Tensor::from_slice(&[done]);
                    let t_truncated = Tensor::from_slice(&[truncated]);

                    for j in 0..num_cars {
                        trajectories[i][j].append_single_step(TrajectoryTensors {
                            states: cur_obs_tensor.i((player_offset + j) as i64),
                            actions: action_results
                                .action
                                .i(((player_offset + j) as i64,))
                                .view([1]),
                            log_probs: action_results
                                .log_prob
                                .i((player_offset + j) as i64)
                                .view([1]),
                            rewards: Tensor::from_slice(&[step_result.rewards[j]]),
                            next_states: next_obs_tensor.i((player_offset + j) as i64),
                            dones: t_done.shallow_clone(),
                            truncateds: t_truncated.shallow_clone(),
                        });
                    }

                    self.data
                        .steps_collected
                        .fetch_add(num_cars as u64, Ordering::Relaxed);
                    player_offset += num_cars;
                }

                trajectory_append_start.elapsed()
            };

            {
                let mut metrics = self.data.metrics.lock().unwrap();

                metrics.report["Policy infer time"] += policy_infer_elapsed.as_secs_f64().into();
                metrics.report["Env step time"] += env_step_elapsed.as_secs_f64().into();
                metrics.report["Trajectory append time"] +=
                    trajectory_append_elapsed.as_secs_f64().into();
            }

            if can_be_reused {
                cur_obs_tensor = next_obs_tensor;
            } else {
                drop(next_obs_tensor);
                cur_obs_tensor = make_games_obs_tensor(&self.game_instances, false);
            }
        }
    }

    pub fn run_render(&mut self, try_launch_exe: bool) {
        while !self.controls.should_run.load(Ordering::Relaxed) {
            sleep(Duration::from_millis(100));
        }

        self.game_instances[0].start();
        self.game_instances[0].open_rlviser(try_launch_exe);

        'outer: loop {
            if !self.controls.should_run.load(Ordering::Relaxed) {
                break 'outer;
            }

            let mut was_paused = false;
            while self.controls.paused.load(Ordering::Relaxed) {
                if !self.controls.should_run.load(Ordering::Relaxed) {
                    break 'outer;
                }

                if !was_paused {
                    self.game_instances[0].close_rlviser();
                }
                was_paused = true;
                sleep(Duration::from_millis(50));
            }

            if was_paused {
                self.game_instances[0].open_rlviser(false);
            }

            let policy = self.policy.read().unwrap().clone();

            let cur_obs_on_device = make_games_obs_tensor(&self.game_instances[..1], false)
                .no_block_to(self.config.device);
            let action_results = policy.get_action(&cur_obs_on_device, self.config.deterministic);
            let action_slice =
                action_results
                    .action
                    .slice(0, 0, self.game_instances[0].num_cars() as i64, 1);
            self.game_instances[0].step(tensor_to_i32_vec(&action_slice), true);
        }

        self.game_instances[0].close_rlviser();
    }
}

pub struct AgentController {
    data: Arc<AgentData>,
    policy: Arc<RwLock<Arc<DiscretePolicy>>>,
    thread_handle: thread::JoinHandle<()>,
}

impl AgentController {
    pub fn new<F, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
        config: AgentConfig,
        controls: Arc<AgentControls>,
        policy: Arc<DiscretePolicy>,
        create_env_fn: F,
        step_callback: C,
        render: bool,
        try_launch_exe: bool,
    ) -> Self
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI> + Send + 'static,
        C: Fn(&mut Report, &SI, &GameStateA) + Clone + Send + 'static,
        SS: StateSetter<SI>,
        SIP: SharedInfoProvider<SI>,
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
            let mut agent = Agent::new(
                config,
                create_env_fn,
                step_callback,
                thread_data,
                controls,
                thread_policy,
            );

            if render {
                agent.run_render(try_launch_exe);
            } else {
                agent.run();
            }
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

    pub fn get_metrics(&self) -> MutexGuard<GameMetrics> {
        self.data.metrics.lock().unwrap()
    }

    pub fn wait_for_close(self) {
        self.thread_handle.join().unwrap();
    }
}
