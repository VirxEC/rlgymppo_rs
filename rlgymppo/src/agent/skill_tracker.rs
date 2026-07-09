use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::thread;
use std::time::Instant;

use burn::prelude::*;
use rand::Rng;
use rlgym::rocketsim::Arena;
use rlgym::{Action, Env, GameState, Obs, Reward, SharedInfoProvider, Truncate};
use serde::{Deserialize, Serialize};

use super::model::Actic;
use super::self_play::PolicyVersion;
use crate::environment::sim::{GameInstance, RewardSamplingConfig};
use crate::utils::Report;
use crate::utils::shared_info::{SharedInfoReport, SharedInfoRng};
use crate::utils::state_setters::KickoffState;
use crate::utils::terminal::OnGoalCondition;

/// Per-mode Elo ratings (e.g. `"1v1"`, `"2v2"`, `"3v3"`).
///
/// Serializes as a TOML table, e.g.:
/// ```toml
/// [data]
/// "1v1" = 12.3
/// "2v2" = -5.1
/// ```
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SkillRating {
    pub data: HashMap<String, f32>,
}

impl SkillRating {
    /// Get-or-insert a rating for a named mode.
    pub fn get_or_default(&mut self, mode: &str, default: f32) -> &mut f32 {
        self.data.entry(mode.to_string()).or_insert(default)
    }

    /// Get-or-insert a rating for the team-size mode.
    pub fn get_for_teams(&mut self, teams: &[usize], default: f32) -> &mut f32 {
        let blue = teams.iter().filter(|&&t| t == 0).count();
        let orange = teams.len() - blue;
        let min = blue.min(orange) as u32;
        let max = blue.max(orange) as u32;
        let mode = format!("{min}v{max}");
        self.get_or_default(&mode, default)
    }
}

/// Configuration for the skill-tracking Elo rating system.
///
/// When `enabled`, periodic evaluation matches are run between the
/// current policy and a randomly chosen old version.  Each goal scored
/// updates per-mode Elo ratings and is reported as `"Rating/1v1"`,
/// `"Rating/2v2"`, etc.
#[derive(Clone, Debug)]
pub struct SkillTrackerConfig {
    /// Master on/off switch.
    pub enabled: bool,
    /// Number of parallel evaluation arenas.
    pub num_arenas: usize,
    /// Target simulation time per arena batch (seconds).
    pub sim_time_secs: f32,
    /// Hard limit on total simulation time before a continuation
    /// (seconds).
    pub max_sim_time_secs: f32,
    /// Training iterations between skill-rating runs.
    pub update_interval: usize,
    /// Elo K-factor — rating increment scale per goal.
    pub rating_inc: f32,
    /// Initial rating of the first policy version.
    pub initial_rating: f32,
    /// Run evaluations on a background worker. When false, evaluations run
    /// synchronously during the training iteration that triggers them.
    pub async_eval: bool,
    /// Use deterministic (argmax) inference during evaluation.
    pub deterministic: bool,
}

impl Default for SkillTrackerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_arenas: 16,
            sim_time_secs: 45.0,
            max_sim_time_secs: 240.0,
            update_interval: 16,
            rating_inc: 5.0,
            initial_rating: 0.0,
            async_eval: false,
            deterministic: false,
        }
    }
}

/// Reward function that always returns zero.
#[derive(Clone, Default)]
struct ZeroReward;

impl<SI> Reward<SI> for ZeroReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}
    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        vec![0.0; state.cars.len()]
    }
}

/// Truncation condition that never triggers.
pub struct NeverTruncate;

impl<SI: SharedInfoProvider> Truncate<SI> for NeverTruncate {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}
    fn should_truncate(&mut self, _state: &GameState, _shared_info: &mut SI) -> bool {
        false
    }
}

enum SkillWorkerCmd {
    Step(Vec<usize>),
    Reset,
    Shutdown,
}

struct SkillArenaResult {
    arena_idx: usize,
    obs: Vec<Vec<f32>>,
    masks: Vec<Vec<bool>>,
    teams: Vec<usize>,
    goal: Option<SkillGoalEvent>,
}

struct SkillGoalEvent {
    teams: Vec<usize>,
    blue_scored: bool,
}

#[derive(Clone)]
pub(crate) struct SkillTrackerUpdate {
    pub eval_id: u64,
    pub cur_ratings: SkillRating,
    pub elapsed_secs: f64,
}

#[allow(clippy::large_enum_variant)]
enum AsyncSkillTrackerJob<B: Backend> {
    Run {
        eval_id: u64,
        current_model: Actic<B>,
        old_version: PolicyVersion<B>,
        cur_ratings: SkillRating,
    },
    Shutdown,
}

struct AsyncSkillTrackerResult {
    update: SkillTrackerUpdate,
    version_ratings: Vec<(u64, SkillRating)>,
    continuation_old_timesteps: Option<u64>,
}

pub struct AsyncSkillTracker<B: Backend, OBS, ACT, SI>
where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    config: SkillTrackerConfig,
    job_tx: SyncSender<AsyncSkillTrackerJob<B>>,
    result_rx: Receiver<AsyncSkillTrackerResult>,
    worker: Option<thread::JoinHandle<()>>,
    device: B::Device,
    pub cur_ratings: SkillRating,
    last_elapsed_secs: Option<f64>,
    iterations_since_ran: usize,
    next_eval_id: u64,
    running_eval_id: Option<u64>,
    continuation_old_timesteps: Option<u64>,
    _phantom: PhantomData<(OBS, ACT, SI)>,
}

/// Runs skill-rating evaluation matches.
pub struct SkillTracker<B: Backend, OBS, ACT, SI>
where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    config: SkillTrackerConfig,

    worker_txs: Vec<Sender<SkillWorkerCmd>>,
    worker_rx: Receiver<SkillArenaResult>,
    workers: Vec<thread::JoinHandle<()>>,

    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    np: Vec<usize>,
    player_teams: Vec<usize>,

    pub cur_ratings: SkillRating,
    cur_goals: usize,

    do_continuation: bool,
    prev_old_version_timesteps: u64,
    prev_new_team: usize,
    prev_total_ticks: u64,

    device: B::Device,
    tick_skip: u8,

    _phantom: PhantomData<(OBS, ACT, SI)>,
}

fn send_skill_reset<OBS, ACT, SI>(
    arena_idx: usize,
    game: &mut GameInstance<KickoffState, OBS, ACT, ZeroReward, OnGoalCondition, NeverTruncate, SI>,
    result_tx: &Sender<SkillArenaResult>,
) where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    let (obs, masks) = game.reset();
    let teams = game.player_teams();
    result_tx
        .send(SkillArenaResult {
            arena_idx,
            obs,
            masks,
            teams,
            goal: None,
        })
        .unwrap();
}

fn send_skill_step<OBS, ACT, SI>(
    arena_idx: usize,
    game: &mut GameInstance<KickoffState, OBS, ACT, ZeroReward, OnGoalCondition, NeverTruncate, SI>,
    actions: &[usize],
    result_tx: &Sender<SkillArenaResult>,
) where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    let result = game.step(actions);

    let (teams, obs, masks, goal) = if result.is_terminal {
        let ball_y = game.last_game_state().ball.pos.y;
        let goal_teams = game.player_teams();
        let blue_scored = ball_y.is_sign_positive();
        let (obs, masks) = game.reset();
        let teams = game.player_teams();
        (
            teams,
            obs,
            masks,
            Some(SkillGoalEvent {
                teams: goal_teams,
                blue_scored,
            }),
        )
    } else {
        let teams = game.player_teams();
        (teams, result.obs, result.action_masks, None)
    };

    result_tx
        .send(SkillArenaResult {
            arena_idx,
            obs,
            masks,
            teams,
            goal,
        })
        .unwrap();
}

impl<B, OBS, ACT, SI> SkillTracker<B, OBS, ACT, SI>
where
    B: Backend,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    /// Construct a skill tracker with one worker thread per arena.
    pub fn new<F>(config: SkillTrackerConfig, create_arena: F, device: B::Device) -> Self
    where
        F: Fn(usize) -> (Arena, OBS, ACT, SI) + Clone + Send + 'static,
    {
        let tick_skip = ACT::get_tick_skip();
        let num_arenas = config.num_arenas;

        let (result_tx, worker_rx) = channel();
        let mut worker_txs = Vec::with_capacity(num_arenas);
        let mut workers = Vec::with_capacity(num_arenas);

        let reward_sampling = RewardSamplingConfig {
            add_rewards_to_metrics: false,
            ..Default::default()
        };

        for game_idx in 0..num_arenas {
            let (cmd_tx, cmd_rx) = channel();
            let result_tx = result_tx.clone();
            let create_arena = create_arena.clone();
            let reward_sampling = reward_sampling.clone();

            let worker = thread::spawn(move || {
                let (arena, obs, action, shared_info) = (create_arena)(game_idx);

                let env = Env::new(
                    arena,
                    KickoffState,
                    obs,
                    action,
                    ZeroReward,
                    OnGoalCondition,
                    NeverTruncate,
                    shared_info,
                );

                let mut game = GameInstance::new(env, reward_sampling);
                send_skill_reset(game_idx, &mut game, &result_tx);

                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        SkillWorkerCmd::Step(actions) => {
                            send_skill_step(game_idx, &mut game, &actions, &result_tx);
                        }
                        SkillWorkerCmd::Reset => {
                            send_skill_reset(game_idx, &mut game, &result_tx);
                        }
                        SkillWorkerCmd::Shutdown => break,
                    }
                }
            });

            worker_txs.push(cmd_tx);
            workers.push(worker);
        }
        drop(result_tx);

        let mut initial_results = Vec::with_capacity(num_arenas);
        for _ in 0..num_arenas {
            initial_results.push(worker_rx.recv().unwrap());
        }
        initial_results.sort_by_key(|result| result.arena_idx);

        let mut next_obs = Vec::new();
        let mut next_masks = Vec::new();
        let mut np = Vec::with_capacity(num_arenas);
        let mut player_teams = Vec::new();
        for result in initial_results {
            let n = result.teams.len();
            np.push(n);
            player_teams.extend(result.teams);
            next_obs.extend(result.obs);
            next_masks.extend(result.masks);
        }

        Self {
            config,
            worker_txs,
            worker_rx,
            workers,
            next_obs,
            next_masks,
            np,
            player_teams,
            cur_ratings: SkillRating::default(),
            cur_goals: 0,
            do_continuation: false,
            prev_old_version_timesteps: 0,
            prev_new_team: 0,
            prev_total_ticks: 0,
            device,
            tick_skip,
            _phantom: PhantomData,
        }
    }

    fn run_matches(&mut self, current_model: &Actic<B>, old_version: &mut PolicyVersion<B>) {
        let num_arenas = self.config.num_arenas;

        let continuing =
            self.do_continuation && self.prev_old_version_timesteps == old_version.timesteps;
        let (new_team, mut total_ticks) = if continuing {
            (self.prev_new_team, self.prev_total_ticks)
        } else {
            let mut rng = rand::rng();
            let team = (rng.next_u32() as usize) % 2;
            self.reset_all_arenas();
            self.cur_goals = 0;
            (team, 0u64)
        };
        self.do_continuation = false;

        let mut new_players = Vec::new();
        let mut old_players = Vec::new();
        for (i, &t) in self.player_teams.iter().enumerate() {
            if t == new_team {
                new_players.push(i);
            } else {
                old_players.push(i);
            }
        }

        let sim_ticks = ((self.config.sim_time_secs * 120.0) as u64).max(self.tick_skip as u64);
        let max_ticks = (self.config.max_sim_time_secs * 120.0) as u64;
        let slice_end = total_ticks.saturating_add(sim_ticks).min(max_ticks);

        #[cfg(not(feature = "tui"))]
        let prev_ratings = self.cur_ratings.clone();

        #[cfg(not(feature = "tui"))]
        println!(
            " > Running skill matches (sim_time={:.1}s, new_team={}, old_version_ts={})...",
            self.config.sim_time_secs, new_team, old_version.timesteps,
        );

        while total_ticks < slice_end && self.cur_goals < num_arenas {
            let new_actions = if new_players.is_empty() {
                Vec::new()
            } else if self.config.deterministic {
                current_model.react_deterministic_indexed(
                    &self.next_obs,
                    &self.next_masks,
                    &new_players,
                    &self.device,
                )
            } else {
                current_model
                    .react_indexed(&self.next_obs, &self.next_masks, &new_players, &self.device)
                    .0
            };

            let old_actions = if old_players.is_empty() {
                Vec::new()
            } else if self.config.deterministic {
                old_version.model.react_deterministic_indexed(
                    &self.next_obs,
                    &self.next_masks,
                    &old_players,
                    &self.device,
                )
            } else {
                old_version
                    .model
                    .react_indexed(&self.next_obs, &self.next_masks, &old_players, &self.device)
                    .0
            };

            let total_players = self.next_obs.len();
            let mut combined = vec![0usize; total_players];
            for (k, &pi) in new_players.iter().enumerate() {
                combined[pi] = new_actions[k];
            }
            for (k, &pi) in old_players.iter().enumerate() {
                combined[pi] = old_actions[k];
            }

            self.next_obs.clear();
            self.next_masks.clear();

            let mut player_start = 0;
            for game_idx in 0..num_arenas {
                let n = self.np[game_idx];
                let actions = combined[player_start..player_start + n].to_vec();
                self.worker_txs[game_idx]
                    .send(SkillWorkerCmd::Step(actions))
                    .unwrap();
                player_start += n;
            }

            let mut arena_results = Vec::with_capacity(num_arenas);
            for _ in 0..num_arenas {
                arena_results.push(self.worker_rx.recv().unwrap());
            }
            arena_results.sort_by_key(|result| result.arena_idx);

            for result in arena_results {
                let game_idx = result.arena_idx;
                let n = result.teams.len();
                let player_start: usize = self.np[..game_idx].iter().sum();

                if let Some(goal) = result.goal {
                    let scorer_was_new = (new_team == 0) == goal.blue_scored;

                    let update = |winner: &mut SkillRating, loser: &mut SkillRating| {
                        let w = winner.get_for_teams(&goal.teams, self.config.initial_rating);
                        let l = loser.get_for_teams(&goal.teams, self.config.initial_rating);
                        let exp_delta = (*l - *w) / 400.0;
                        let expected = 1.0 / (10.0_f32.powf(exp_delta) + 1.0);
                        *w += self.config.rating_inc * (1.0 - expected);
                        *l += self.config.rating_inc * (expected - 1.0);
                    };

                    if scorer_was_new {
                        update(&mut self.cur_ratings, &mut old_version.ratings);
                    } else {
                        update(&mut old_version.ratings, &mut self.cur_ratings);
                    }

                    self.cur_goals += 1;
                }

                self.np[game_idx] = n;
                self.player_teams[player_start..player_start + n].copy_from_slice(&result.teams);

                self.next_obs.extend(result.obs);
                self.next_masks.extend(result.masks);
            }

            total_ticks += self.tick_skip as u64;
        }

        if self.cur_goals < num_arenas && total_ticks < max_ticks {
            #[cfg(not(feature = "tui"))]
            println!(
                " > Forcing continuation ({}/{})",
                self.cur_goals, num_arenas
            );
            self.do_continuation = true;
            self.prev_old_version_timesteps = old_version.timesteps;
            self.prev_new_team = new_team;
            self.prev_total_ticks = total_ticks;
        } else {
            self.cur_goals = 0;
        }

        #[cfg(not(feature = "tui"))]
        for (mode, &rating) in &self.cur_ratings.data {
            let prev = prev_ratings
                .data
                .get(mode)
                .copied()
                .unwrap_or(self.config.initial_rating);
            let delta = rating - prev;
            if delta != 0.0 {
                println!(
                    " > {mode} = {prev:.1} ({}{delta:.1})",
                    if delta >= 0.0 { '+' } else { '-' }
                );
            } else {
                println!(" > {mode} = {prev:.1}");
            }
        }
    }

    /// Reset all arenas.
    fn reset_all_arenas(&mut self) {
        self.next_obs.clear();
        self.next_masks.clear();
        self.player_teams.clear();

        for worker_tx in &self.worker_txs {
            worker_tx.send(SkillWorkerCmd::Reset).unwrap();
        }

        let mut arena_results = Vec::with_capacity(self.config.num_arenas);
        for _ in 0..self.config.num_arenas {
            arena_results.push(self.worker_rx.recv().unwrap());
        }
        arena_results.sort_by_key(|result| result.arena_idx);

        for result in arena_results {
            let n = result.teams.len();
            self.np[result.arena_idx] = n;
            self.next_obs.extend(result.obs);
            self.next_masks.extend(result.masks);
            self.player_teams.extend(result.teams);
        }
    }
}

impl<B, OBS, ACT, SI> Drop for SkillTracker<B, OBS, ACT, SI>
where
    B: Backend,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    fn drop(&mut self) {
        for worker_tx in &self.worker_txs {
            let _ = worker_tx.send(SkillWorkerCmd::Shutdown);
        }

        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

impl<B, OBS, ACT, SI> AsyncSkillTracker<B, OBS, ACT, SI>
where
    B: Backend + Send + 'static,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    pub fn new<F>(
        config: SkillTrackerConfig,
        create_arena: F,
        device: B::Device,
        metric_tx: Sender<SkillTrackerUpdate>,
    ) -> Self
    where
        F: Fn(usize) -> (Arena, OBS, ACT, SI) + Clone + Send + 'static,
        B::Device: Send,
    {
        let (job_tx, job_rx) = sync_channel(1);
        let (result_tx, result_rx) = channel();
        let tracker_config = config.clone();
        let tracker_device = device.clone();

        let worker = thread::spawn(move || {
            let mut tracker = SkillTracker::new(tracker_config, create_arena, tracker_device);

            while let Ok(job) = job_rx.recv() {
                match job {
                    AsyncSkillTrackerJob::Run {
                        eval_id,
                        current_model,
                        mut old_version,
                        cur_ratings,
                    } => {
                        tracker.cur_ratings = cur_ratings;
                        let start = Instant::now();
                        tracker.run_matches(&current_model, &mut old_version);
                        let elapsed_secs = start.elapsed().as_secs_f64();

                        let version_ratings = vec![(old_version.timesteps, old_version.ratings)];
                        let continuation_old_timesteps = tracker
                            .do_continuation
                            .then_some(tracker.prev_old_version_timesteps);

                        let update = SkillTrackerUpdate {
                            eval_id,
                            cur_ratings: tracker.cur_ratings.clone(),
                            elapsed_secs,
                        };

                        let _ = metric_tx.send(update.clone());
                        let _ = result_tx.send(AsyncSkillTrackerResult {
                            update,
                            version_ratings,
                            continuation_old_timesteps,
                        });
                    }
                    AsyncSkillTrackerJob::Shutdown => break,
                }
            }
        });

        Self {
            config,
            job_tx,
            result_rx,
            worker: Some(worker),
            device,
            cur_ratings: SkillRating::default(),
            last_elapsed_secs: None,
            iterations_since_ran: 0,
            next_eval_id: 0,
            running_eval_id: None,
            continuation_old_timesteps: None,
            _phantom: PhantomData,
        }
    }

    pub fn on_iteration(
        &mut self,
        current_model: &Actic<B>,
        versions: &mut [PolicyVersion<B>],
    ) -> Option<u64> {
        if !self.config.enabled || versions.is_empty() || self.running_eval_id.is_some() {
            return None;
        }

        self.iterations_since_ran += 1;
        if self.iterations_since_ran < self.config.update_interval {
            return None;
        }

        let mut old_version = self
            .continuation_old_timesteps
            .and_then(|timesteps| versions.iter().find(|v| v.timesteps == timesteps))
            .unwrap_or_else(|| {
                let mut rng = rand::rng();
                &versions[(rng.next_u32() as usize) % versions.len()]
            })
            .clone();
        old_version.model = old_version.model.to_device(&self.device);

        let eval_id = self.next_eval_id;
        let job = AsyncSkillTrackerJob::Run {
            eval_id,
            current_model: current_model.clone().to_device(&self.device),
            old_version,
            cur_ratings: self.cur_ratings.clone(),
        };

        if self.config.async_eval {
            match self.job_tx.try_send(job) {
                Ok(()) => {
                    self.iterations_since_ran = 0;
                    self.next_eval_id += 1;
                    self.running_eval_id = Some(eval_id);
                    Some(eval_id)
                }
                Err(TrySendError::Full(_)) => None,
                Err(TrySendError::Disconnected(_)) => None,
            }
        } else {
            if self.job_tx.send(job).is_err() {
                return None;
            }
            if let Ok(result) = self.result_rx.recv() {
                self.apply_result(result, versions);
                self.iterations_since_ran = 0;
                self.next_eval_id += 1;
                Some(eval_id)
            } else {
                None
            }
        }
    }

    pub fn poll_updates(&mut self, versions: &mut [PolicyVersion<B>]) -> Vec<SkillTrackerUpdate> {
        let mut updates = Vec::new();

        while let Ok(result) = self.result_rx.try_recv() {
            updates.push(self.apply_result(result, versions));
        }

        updates
    }

    fn apply_result(
        &mut self,
        result: AsyncSkillTrackerResult,
        versions: &mut [PolicyVersion<B>],
    ) -> SkillTrackerUpdate {
        self.cur_ratings = result.update.cur_ratings.clone();
        self.last_elapsed_secs = Some(result.update.elapsed_secs);
        self.continuation_old_timesteps = result.continuation_old_timesteps;
        if self.running_eval_id == Some(result.update.eval_id) {
            self.running_eval_id = None;
        }

        for (timesteps, ratings) in result.version_ratings {
            if let Some(version) = versions.iter_mut().find(|v| v.timesteps == timesteps) {
                version.ratings = ratings;
            }
        }

        result.update
    }

    pub fn report_ratings(&self, report: &mut Report) {
        for (mode, &rating) in &self.cur_ratings.data {
            let key = format!("Rating/{mode}");
            report[key.as_str()] = rating.into();
        }
        if let Some(elapsed_secs) = self.last_elapsed_secs {
            report["Timing/skill tracker"] = elapsed_secs.into();
        }
    }

    pub fn join(mut self, versions: &mut [PolicyVersion<B>]) -> SkillRating {
        let _ = self.job_tx.send(AsyncSkillTrackerJob::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.poll_updates(versions);
        self.cur_ratings.clone()
    }
}

impl<B, OBS, ACT, SI> Drop for AsyncSkillTracker<B, OBS, ACT, SI>
where
    B: Backend,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    fn drop(&mut self) {
        let _ = self.job_tx.send(AsyncSkillTrackerJob::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}
