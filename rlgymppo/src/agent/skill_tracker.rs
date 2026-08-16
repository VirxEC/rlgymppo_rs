use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::thread;
use std::time::Instant;

use burn::prelude::*;
use rand::Rng;
use rlgym::rocketsim::{Arena, CarControls, consts};
use rlgym::{Action, Env, GameState, Obs, Reward, SharedInfoProvider, Truncate};
use rlgymppo_nexto::{NextoAction, NextoModel, NextoObs, NextoObsBuilder};
use rlgymppo_utils::Report;
use rlgymppo_utils::shared_info::{SharedInfoReport, SharedInfoRng};
use rlgymppo_utils::state_setters::KickoffState;
use rlgymppo_utils::terminal::OnGoalCondition;
use serde::{Deserialize, Serialize};

use super::model::Actic;
use super::self_play::PolicyVersion;
use crate::environment::sim::{GameInstance, RewardSamplingConfig};

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

pub(crate) fn report_skill_ratings(
    report: &mut Report,
    ratings: &SkillRating,
    nexto_mmr: Option<f32>,
) {
    for (mode, &rating) in &ratings.data {
        let key = format!("Rating/{mode}");
        report[key.as_str()] = rating.into();
    }
    if let Some(nexto_mmr) = nexto_mmr {
        report["Rating/Nexto"] = nexto_mmr.into();
    }
}

/// Configuration for the Elo rating system that periodically evaluates the
/// current policy against saved previous versions and the fixed Nexto
/// policy.
///
/// Each scheduled evaluation plays matches against a randomly chosen saved
/// version first, when versions exist. When `nexto_mmr` is `Some`, the
/// evaluation then plays matches against Nexto. Each goal updates the
/// per-mode Elo rating of the current policy and is reported as
/// `"Rating/{mode}"`. When Nexto is enabled, its fixed rating is also
/// reported as `"Rating/Nexto"`.
///
/// The value `nexto_mmr.unwrap_or(0.0)` seeds the current and saved-version
/// ratings. `Some(1500.0)` puts both opponents on one scale. `None` keeps
/// the historical 0.0 scale. `None` also means the tracker never loads
/// Nexto, never builds Nexto observations, and never plays Nexto matches.
#[derive(Clone, Debug)]
pub struct SkillTrackerConfig {
    /// Master switch for all skill-tracker evaluations. When false, the
    /// learner does not construct or run the tracker.
    pub enabled: bool,
    /// Nexto rating and Nexto switch. `Some(mmr)` enables Nexto
    /// evaluation matches with the fixed rating `mmr`. The same value seeds
    /// the current and saved-version ratings when they are first created.
    /// `None` disables Nexto entirely (no model, no observations, no
    /// matches) while keeping saved-version evaluation available.
    pub nexto_mmr: Option<f32>,
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
    /// Run evaluations on a background worker. When false, evaluations run
    /// synchronously during the training iteration that triggers them.
    pub async_eval: bool,
    /// Use deterministic (argmax) inference for the current policy during
    /// evaluation. Nexto always uses deterministic (argmax) inference.
    pub deterministic: bool,
}

impl Default for SkillTrackerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            nexto_mmr: Some(1500.0),
            num_arenas: 16,
            sim_time_secs: 45.0,
            max_sim_time_secs: 240.0,
            update_interval: 16,
            rating_inc: 5.0,
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

/// One-sided Elo delta for a goal against the fixed Nexto rating.
///
/// Only the current policy's rating moves; Nexto keeps the fixed rating.
fn nexto_rating_delta(current: f32, nexto_mmr: f32, rating_inc: f32, current_won: bool) -> f32 {
    let expected = 1.0 / (10.0_f32.powf((nexto_mmr - current) / 400.0) + 1.0);
    if current_won {
        rating_inc * (1.0 - expected)
    } else {
        -rating_inc * expected
    }
}

/// Two-sided Elo deltas for a goal: `(winner_delta, loser_delta)`.
///
/// Both the current policy and the saved version move when they play each
/// other.
fn two_sided_rating_delta(winner: f32, loser: f32, rating_inc: f32) -> (f32, f32) {
    let expected = 1.0 / (10.0_f32.powf((loser - winner) / 400.0) + 1.0);
    (rating_inc * (1.0 - expected), rating_inc * (expected - 1.0))
}

/// Identifies which opponent a skill-evaluation phase is played against.
///
/// The worker stores the phase to resume on continuation. The async side
/// uses the same identity to re-select the matching saved version. The two
/// sides stay synchronized through the eval result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SkillOpponentId {
    /// A saved previous version of the policy, identified by timesteps.
    Previous(u64),
    /// The fixed Nexto policy.
    Nexto,
}

/// The opponents a scheduled evaluation plays against, in order.
///
/// A saved previous version (identified by its timesteps) is played first
/// when one is available. Nexto is played second when `nexto_mmr` is `Some`.
fn eval_opponents(
    old_version_timesteps: Option<u64>,
    nexto_mmr: Option<f32>,
) -> Vec<SkillOpponentId> {
    let mut opponents = Vec::with_capacity(2);
    if let Some(timesteps) = old_version_timesteps {
        opponents.push(SkillOpponentId::Previous(timesteps));
    }
    if nexto_mmr.is_some() {
        opponents.push(SkillOpponentId::Nexto);
    }
    opponents
}

/// Uniformly random saved version, when any exist.
fn random_version<B: Backend>(versions: &[PolicyVersion<B>]) -> Option<&PolicyVersion<B>> {
    if versions.is_empty() {
        return None;
    }
    let mut rng = rand::rng();
    versions.get((rng.next_u32() as usize) % versions.len())
}

impl<SI: SharedInfoProvider> Truncate<SI> for NeverTruncate {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}
    fn should_truncate(&mut self, _state: &GameState, _shared_info: &mut SI) -> bool {
        false
    }
}

enum SkillWorkerCmd {
    Step {
        /// Generic policy actions, one per car.
        actions: Vec<usize>,
        /// Nexto actions as (arena-local car index, action).
        nexto_actions: Vec<(usize, NextoAction)>,
    },
    Reset,
    Shutdown,
}

struct SkillArenaResult {
    arena_idx: usize,
    obs: Vec<Vec<f32>>,
    masks: Vec<Vec<bool>>,
    nexto_obs: Vec<NextoObs>,
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
    pub nexto_mmr: Option<f32>,
}

#[allow(clippy::large_enum_variant)]
enum AsyncSkillTrackerJob<B: Backend> {
    Run {
        eval_id: u64,
        current_model: Actic<B>,
        /// The randomly selected saved version to play, when one exists or a
        /// Previous-phase continuation requires it.
        old_version: Option<PolicyVersion<B>>,
        cur_ratings: SkillRating,
    },
    Shutdown,
}

struct AsyncSkillTrackerResult {
    update: SkillTrackerUpdate,
    /// Updated per-mode ratings per saved version, keyed by timesteps.
    version_ratings: Vec<(u64, SkillRating)>,
    /// The phase that still needs more jobs, when the eval is not complete.
    continuation: Option<SkillOpponentId>,
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
    /// Phase that the running eval must resume on its next job.
    continuation: Option<SkillOpponentId>,
    /// Clone of the saved version dispatched for the running eval.
    ///
    /// Kept so a Previous-phase continuation can still re-dispatch the same
    /// model/ratings snapshot after `VersionManager` prunes the version from
    /// the live `versions` slice.
    pending_old_version: Option<PolicyVersion<B>>,
    _phantom: PhantomData<(OBS, ACT, SI)>,
}

/// Runs skill-rating evaluation matches between the current policy, saved
/// previous versions of the policy, and the fixed Nexto policy.
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
    /// Nexto observations, aligned with `next_obs` (one per car).
    nexto_obs: Vec<NextoObs>,
    players_per_arena: Vec<usize>,
    player_teams: Vec<usize>,

    pub cur_ratings: SkillRating,
    cur_goals: usize,

    do_continuation: bool,
    /// The phase to resume when `do_continuation` is set.
    prev_opponent: SkillOpponentId,
    prev_new_team: usize,
    prev_total_ticks: u64,

    device: B::Device,
    tick_skip: u8,
    /// Loaded on the skill tracker thread when `nexto_mmr` is set.
    nexto_model: Option<NextoModel<B>>,

    _phantom: PhantomData<(OBS, ACT, SI)>,
}

fn send_skill_reset<OBS, ACT, SI>(
    arena_idx: usize,
    game: &mut GameInstance<KickoffState, OBS, ACT, ZeroReward, OnGoalCondition, NeverTruncate, SI>,
    nexto_builder: Option<&NextoObsBuilder>,
    prev_nexto_actions: &mut Vec<NextoAction>,
    result_tx: &Sender<SkillArenaResult>,
) where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    let (obs, _old_obs, masks) = game.reset();
    let teams = game.player_teams();

    // A fresh episode has no previous action yet.
    prev_nexto_actions.clear();
    let nexto_obs = if let Some(builder) = nexto_builder {
        prev_nexto_actions.resize(teams.len(), NextoAction::ZERO);
        builder.build(game.last_game_state(), prev_nexto_actions)
    } else {
        Vec::new()
    };

    result_tx
        .send(SkillArenaResult {
            arena_idx,
            obs,
            masks,
            nexto_obs,
            teams,
            goal: None,
        })
        .unwrap();
}

fn send_skill_step<OBS, ACT, SI>(
    arena_idx: usize,
    game: &mut GameInstance<KickoffState, OBS, ACT, ZeroReward, OnGoalCondition, NeverTruncate, SI>,
    nexto_builder: Option<&NextoObsBuilder>,
    prev_nexto_actions: &mut Vec<NextoAction>,
    actions: &[usize],
    nexto_actions: &[(usize, NextoAction)],
    result_tx: &Sender<SkillArenaResult>,
) where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    let nexto_controls: Vec<(usize, CarControls)> = if nexto_builder.is_some() {
        let current_state = game.last_game_state().clone();
        nexto_actions
            .iter()
            .map(|&(car_index, action)| {
                (
                    current_state.cars[car_index].0.idx,
                    action.to_car_controls(),
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let result = game.step_mixed(actions, &nexto_controls);

    // The next observation must see the action that was just applied.
    for &(car_index, action) in nexto_actions {
        if nexto_builder.is_some() {
            prev_nexto_actions[car_index] = action;
        }
    }

    let (teams, obs, masks, nexto_obs, goal) = if result.is_terminal {
        let ball_y = game.last_game_state().ball.pos.y;
        let goal_teams = game.player_teams();
        let blue_scored = ball_y.is_sign_positive();
        let (obs, _old_obs, masks) = game.reset();
        let teams = game.player_teams();

        prev_nexto_actions.clear();
        let nexto_obs = if let Some(builder) = nexto_builder {
            prev_nexto_actions.resize(teams.len(), NextoAction::ZERO);
            builder.build(game.last_game_state(), prev_nexto_actions)
        } else {
            Vec::new()
        };

        (
            teams,
            obs,
            masks,
            nexto_obs,
            Some(SkillGoalEvent {
                teams: goal_teams,
                blue_scored,
            }),
        )
    } else {
        let teams = game.player_teams();
        let nexto_obs = nexto_builder
            .map(|builder| builder.build(game.last_game_state(), prev_nexto_actions))
            .unwrap_or_default();
        (teams, result.obs, result.action_masks, nexto_obs, None)
    };

    result_tx
        .send(SkillArenaResult {
            arena_idx,
            obs,
            masks,
            nexto_obs,
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
        assert!(
            config.num_arenas > 0,
            "skill tracker requires at least one evaluation arena"
        );
        let tick_skip = ACT::get_tick_skip();
        let num_arenas = config.num_arenas;
        let nexto_enabled = config.enabled && config.nexto_mmr.is_some();

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

                let mut game = GameInstance::new(env, None, reward_sampling);
                let nexto_builder = nexto_enabled.then(NextoObsBuilder::default);
                let mut prev_nexto_actions = Vec::new();
                send_skill_reset(
                    game_idx,
                    &mut game,
                    nexto_builder.as_ref(),
                    &mut prev_nexto_actions,
                    &result_tx,
                );

                while let Ok(cmd) = cmd_rx.recv() {
                    match cmd {
                        SkillWorkerCmd::Step {
                            actions,
                            nexto_actions,
                        } => {
                            send_skill_step(
                                game_idx,
                                &mut game,
                                nexto_builder.as_ref(),
                                &mut prev_nexto_actions,
                                &actions,
                                &nexto_actions,
                                &result_tx,
                            );
                        }
                        SkillWorkerCmd::Reset => {
                            send_skill_reset(
                                game_idx,
                                &mut game,
                                nexto_builder.as_ref(),
                                &mut prev_nexto_actions,
                                &result_tx,
                            );
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
        let mut nexto_obs = Vec::new();
        let mut players_per_arena = Vec::with_capacity(num_arenas);
        let mut player_teams = Vec::new();
        for result in initial_results {
            let n = result.teams.len();
            players_per_arena.push(n);
            player_teams.extend(result.teams);
            next_obs.extend(result.obs);
            next_masks.extend(result.masks);
            nexto_obs.extend(result.nexto_obs);
        }

        let nexto_model =
            (config.enabled && config.nexto_mmr.is_some()).then(|| NextoModel::new(&device));

        Self {
            config,
            worker_txs,
            worker_rx,
            workers,
            next_obs,
            next_masks,
            nexto_obs,
            players_per_arena,
            player_teams,
            cur_ratings: SkillRating::default(),
            cur_goals: 0,
            do_continuation: false,
            prev_opponent: SkillOpponentId::Nexto,
            prev_new_team: 0,
            prev_total_ticks: 0,
            device,
            tick_skip,
            nexto_model,
            _phantom: PhantomData,
        }
    }

    /// Deterministic (argmax) Nexto actions for the given players.
    fn infer_nexto_actions(&self, nexto_players: &[usize]) -> Vec<usize> {
        let model = self
            .nexto_model
            .as_ref()
            .expect("Nexto model is loaded when nexto_mmr is Some");
        let observations: Vec<&NextoObs> = nexto_players
            .iter()
            .map(|&player_idx| &self.nexto_obs[player_idx])
            .collect();
        model.actions_from_observations(&observations)
    }

    fn run_matches(
        &mut self,
        current_model: &Actic<B>,
        mut old_version: Option<&mut PolicyVersion<B>>,
    ) {
        let num_arenas = self.config.num_arenas;
        let nexto_mmr = self.config.nexto_mmr;
        let rating_default = nexto_mmr.unwrap_or(0.0);

        // Phases run in order: the Previous phase (when a saved version is
        // available) and then the Nexto phase (when `nexto_mmr` is set).
        let opponents = eval_opponents(
            old_version.as_ref().map(|version| version.timesteps),
            nexto_mmr,
        );
        if opponents.is_empty() {
            // Nothing to evaluate; the async side should not have sent a job.
            self.do_continuation = false;
            self.cur_goals = 0;
            return;
        }

        // Resume the phase the previous job left incomplete. A
        // Previous-phase continuation is only valid when the job carries the
        // same saved version again.
        let continuing = self.do_continuation && opponents.contains(&self.prev_opponent);
        self.do_continuation = false;

        let mut opponent = if continuing {
            self.prev_opponent
        } else {
            opponents[0]
        };

        let (mut new_team, mut total_ticks) = if continuing {
            (self.prev_new_team, self.prev_total_ticks)
        } else {
            let mut rng = rand::rng();
            let team = (rng.next_u32() as usize) % 2;
            self.reset_all_arenas();
            self.cur_goals = 0;
            (team, 0u64)
        };

        #[cfg(not(feature = "tui"))]
        let prev_ratings = self.cur_ratings.clone();

        loop {
            let sim_ticks =
                ((self.config.sim_time_secs * consts::TICK_RATE) as u64).max(self.tick_skip as u64);
            let max_ticks = (self.config.max_sim_time_secs * consts::TICK_RATE) as u64;
            let slice_end = total_ticks.saturating_add(sim_ticks).min(max_ticks);

            #[cfg(not(feature = "tui"))]
            match opponent {
                SkillOpponentId::Previous(timesteps) => println!(
                    " > Running skill matches vs saved version (sim_time={:.1}s, new_team={}, \
                     old_version_ts={})...",
                    self.config.sim_time_secs, new_team, timesteps,
                ),
                SkillOpponentId::Nexto => println!(
                    " > Running skill matches vs Nexto (sim_time={:.1}s, new_team={}, \
                     nexto_mmr={:.1})...",
                    self.config.sim_time_secs, new_team, rating_default,
                ),
            }

            let mut new_players = Vec::new();
            let mut opponent_players = Vec::new();
            for (i, &t) in self.player_teams.iter().enumerate() {
                if t == new_team {
                    new_players.push(i);
                } else {
                    opponent_players.push(i);
                }
            }

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

                let opponent_actions = match opponent {
                    SkillOpponentId::Previous(_) => {
                        let version = old_version
                            .as_ref()
                            .expect("the Previous phase always carries the selected saved version");
                        if opponent_players.is_empty() {
                            Vec::new()
                        } else if self.config.deterministic {
                            version.model.react_deterministic_indexed(
                                &self.next_obs,
                                &self.next_masks,
                                &opponent_players,
                                &self.device,
                            )
                        } else {
                            version
                                .model
                                .react_indexed(
                                    &self.next_obs,
                                    &self.next_masks,
                                    &opponent_players,
                                    &self.device,
                                )
                                .0
                        }
                    }
                    SkillOpponentId::Nexto => {
                        if opponent_players.is_empty() {
                            Vec::new()
                        } else {
                            self.infer_nexto_actions(&opponent_players)
                        }
                    }
                };

                let total_players = self.next_obs.len();
                let mut combined = vec![0usize; total_players];
                for (k, &pi) in new_players.iter().enumerate() {
                    combined[pi] = new_actions[k];
                }
                if !matches!(opponent, SkillOpponentId::Nexto) {
                    for (k, &pi) in opponent_players.iter().enumerate() {
                        combined[pi] = opponent_actions[k];
                    }
                }
                // Nexto indices belong to its own action table. Keep the
                // generic action parser on a neutral index; raw Nexto
                // controls overwrite these cars in `step_mixed`.

                self.next_obs.clear();
                self.next_masks.clear();
                self.nexto_obs.clear();

                let mut player_start = 0;
                for game_idx in 0..num_arenas {
                    let n = self.players_per_arena[game_idx];
                    let actions = combined[player_start..player_start + n].to_vec();
                    // Nexto players are driven through the mixed-step path.
                    // Every other phase sends no `nexto_actions`, so all cars
                    // use generic policy actions.
                    let nexto_actions: Vec<(usize, NextoAction)> = match opponent {
                        SkillOpponentId::Nexto => opponent_players
                            .iter()
                            .copied()
                            .zip(opponent_actions.iter().copied())
                            .filter(|&(pi, _)| (player_start..player_start + n).contains(&pi))
                            .map(|(pi, action)| {
                                (pi - player_start, NextoAction::from_index(action))
                            })
                            .collect(),
                        SkillOpponentId::Previous(_) => Vec::new(),
                    };
                    self.worker_txs[game_idx]
                        .send(SkillWorkerCmd::Step {
                            actions,
                            nexto_actions,
                        })
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
                    let player_start: usize = self.players_per_arena[..game_idx].iter().sum();

                    if let Some(goal) = result.goal {
                        let scorer_was_new = (new_team == 0) == goal.blue_scored;

                        match opponent {
                            SkillOpponentId::Previous(_) => {
                                let version = old_version.as_mut().expect(
                                    "the Previous phase always carries the selected saved version",
                                );
                                // Two-sided Elo: the current policy and the
                                // saved version both move.
                                let (winner, loser) = if scorer_was_new {
                                    (&mut self.cur_ratings, &mut version.ratings)
                                } else {
                                    (&mut version.ratings, &mut self.cur_ratings)
                                };
                                let w = winner.get_for_teams(&goal.teams, rating_default);
                                let l = loser.get_for_teams(&goal.teams, rating_default);
                                let (w_delta, l_delta) =
                                    two_sided_rating_delta(*w, *l, self.config.rating_inc);
                                *w += w_delta;
                                *l += l_delta;
                            }
                            SkillOpponentId::Nexto => {
                                // Only the current policy's rating moves;
                                // Nexto keeps the fixed `nexto_mmr` rating.
                                let cur =
                                    self.cur_ratings.get_for_teams(&goal.teams, rating_default);
                                *cur += nexto_rating_delta(
                                    *cur,
                                    rating_default,
                                    self.config.rating_inc,
                                    scorer_was_new,
                                );
                            }
                        }

                        self.cur_goals += 1;
                    }

                    self.players_per_arena[game_idx] = n;
                    self.player_teams[player_start..player_start + n]
                        .copy_from_slice(&result.teams);

                    self.next_obs.extend(result.obs);
                    self.next_masks.extend(result.masks);
                    self.nexto_obs.extend(result.nexto_obs);
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
                self.prev_opponent = opponent;
                self.prev_new_team = new_team;
                self.prev_total_ticks = total_ticks;
                break;
            }

            // This phase finished (goals reached or the sim-time cap hit).
            // Move on to the next phase, if any.
            self.cur_goals = 0;
            let current_idx = opponents
                .iter()
                .position(|&candidate| candidate == opponent)
                .unwrap();
            let Some(&next_opponent) = opponents.get(current_idx + 1) else {
                break;
            };
            opponent = next_opponent;
            let mut rng = rand::rng();
            new_team = (rng.next_u32() as usize) % 2;
            total_ticks = 0;
            self.reset_all_arenas();
        }

        #[cfg(not(feature = "tui"))]
        for (mode, &rating) in &self.cur_ratings.data {
            let prev = prev_ratings
                .data
                .get(mode)
                .copied()
                .unwrap_or(rating_default);
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
        self.nexto_obs.clear();
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
            self.players_per_arena[result.arena_idx] = n;
            self.next_obs.extend(result.obs);
            self.next_masks.extend(result.masks);
            self.nexto_obs.extend(result.nexto_obs);
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
                        tracker.run_matches(&current_model, old_version.as_mut());
                        let elapsed_secs = start.elapsed().as_secs_f64();

                        let version_ratings = old_version
                            .map(|version| (version.timesteps, version.ratings))
                            .into_iter()
                            .collect();
                        let continuation = tracker.do_continuation.then_some(tracker.prev_opponent);

                        let update = SkillTrackerUpdate {
                            eval_id,
                            cur_ratings: tracker.cur_ratings.clone(),
                            elapsed_secs,
                            nexto_mmr: tracker.config.nexto_mmr,
                        };

                        let _ = metric_tx.send(update.clone());
                        let _ = result_tx.send(AsyncSkillTrackerResult {
                            update,
                            version_ratings,
                            continuation,
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
            continuation: None,
            pending_old_version: None,
            _phantom: PhantomData,
        }
    }

    pub fn on_iteration(
        &mut self,
        current_model: &Actic<B>,
        versions: &mut [PolicyVersion<B>],
    ) -> Option<u64> {
        if !self.config.enabled || self.running_eval_id.is_some() {
            return None;
        }

        // Nothing to evaluate against when no saved versions exist and Nexto
        // is disabled.
        if versions.is_empty() && self.config.nexto_mmr.is_none() {
            return None;
        }

        self.iterations_since_ran += 1;
        if self.iterations_since_ran < self.config.update_interval {
            return None;
        }

        // Re-select the saved version for this job. A continuation of the
        // Previous phase re-uses the same version: first the live copy in
        // `versions`, then the retained snapshot (in case `VersionManager`
        // pruned the version since the last job), then any remaining
        // version. A fresh eval picks a random version whenever any exist.
        // A Nexto continuation (or an eval without versions) sends no
        // version.
        let mut old_version = match self.continuation {
            Some(SkillOpponentId::Previous(timesteps)) => versions
                .iter()
                .find(|version| version.timesteps == timesteps)
                .or_else(|| {
                    self.pending_old_version
                        .as_ref()
                        .filter(|version| version.timesteps == timesteps)
                })
                .or_else(|| random_version(versions))
                .cloned(),
            Some(SkillOpponentId::Nexto) => None,
            None if !versions.is_empty() => random_version(versions).cloned(),
            None => None,
        };
        if let Some(mut version) = old_version.take() {
            version.model = version.model.to_device(&self.device);
            old_version = Some(version);
        }

        // Retain a clone of the dispatched version so a later Previous-phase
        // continuation can still be dispatched after `VersionManager` prunes
        // it from `versions`.
        self.pending_old_version = old_version.clone();

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
        self.continuation = result.continuation;
        if self.running_eval_id == Some(result.update.eval_id) {
            self.running_eval_id = None;
        }

        // Write updated ratings back to the live versions.
        for (timesteps, ratings) in &result.version_ratings {
            if let Some(version) = versions.iter_mut().find(|v| v.timesteps == *timesteps) {
                version.ratings = ratings.clone();
            }
        }

        // Keep the retained snapshot in sync so a Previous-phase
        // continuation can still be dispatched after `VersionManager` prunes
        // the version.
        match self.continuation {
            Some(SkillOpponentId::Previous(timesteps)) => {
                let snapshot_matches = self
                    .pending_old_version
                    .as_ref()
                    .is_some_and(|version| version.timesteps == timesteps);
                if !snapshot_matches {
                    // The continued version is not the retained snapshot.
                    self.pending_old_version = None;
                } else if let Some((_, ratings)) = result
                    .version_ratings
                    .iter()
                    .find(|(ts, _)| *ts == timesteps)
                {
                    // Keep the retained snapshot's ratings in sync.
                    self.pending_old_version
                        .as_mut()
                        .expect("snapshot_matches guarantees a retained snapshot")
                        .ratings = ratings.clone();
                }
            }
            // The eval is complete or continues against Nexto, so the
            // snapshot is no longer needed.
            _ => self.pending_old_version = None,
        }

        result.update
    }

    pub fn report_ratings(&self, report: &mut Report) {
        report_skill_ratings(report, &self.cur_ratings, self.config.nexto_mmr);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uses_nexto_at_1500_mmr() {
        let config = SkillTrackerConfig::default();
        assert_eq!(config.nexto_mmr, Some(1500.0));
        assert!(!config.enabled);
    }

    #[test]
    fn disabled_is_the_default_tracker_state() {
        assert!(!SkillTrackerConfig::default().enabled);
    }

    #[test]
    fn none_disables_the_nexto_baseline() {
        let config = SkillTrackerConfig {
            nexto_mmr: None,
            ..Default::default()
        };
        assert_eq!(config.nexto_mmr, None);
    }

    #[test]
    fn first_rating_for_a_mode_uses_nexto_mmr() {
        let mut ratings = SkillRating::default();
        assert_eq!(*ratings.get_or_default("1v1", 1500.0), 1500.0);
        assert_eq!(ratings.data.get("1v1"), Some(&1500.0));
    }

    #[test]
    fn nexto_rating_delta_uses_fixed_opponent_rating() {
        let win_delta = nexto_rating_delta(1500.0, 1500.0, 5.0, true);
        let loss_delta = nexto_rating_delta(1500.0, 1500.0, 5.0, false);
        assert!((win_delta - 2.5).abs() < f32::EPSILON);
        assert!((loss_delta + 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn two_sided_delta_moves_both_ratings() {
        // Equal ratings split the goal evenly.
        let (win, loss) = two_sided_rating_delta(1500.0, 1500.0, 5.0);
        assert!((win - 2.5).abs() < f32::EPSILON);
        assert!((loss + 2.5).abs() < f32::EPSILON);
        // The underdog gains more than the favourite when they score.
        let (win, loss) = two_sided_rating_delta(1400.0, 1500.0, 5.0);
        assert!(win > 2.5);
        assert!(loss < -2.5);
    }

    #[test]
    fn none_keeps_only_the_previous_phase() {
        // Nexto is disabled: a saved version is still a valid opponent, so
        // old-version evaluation remains schedulable.
        let opponents = eval_opponents(Some(1_000), None);
        assert_eq!(opponents, vec![SkillOpponentId::Previous(1_000)]);
    }

    #[test]
    fn some_orders_previous_before_nexto() {
        let opponents = eval_opponents(Some(1_000), Some(1500.0));
        assert_eq!(
            opponents,
            vec![SkillOpponentId::Previous(1_000), SkillOpponentId::Nexto]
        );
        // Without versions only Nexto remains.
        assert_eq!(
            eval_opponents(None, Some(1500.0)),
            vec![SkillOpponentId::Nexto]
        );
        // Without versions and with Nexto disabled there is nothing to play.
        assert!(eval_opponents(None, None).is_empty());
    }

    #[test]
    fn team_mode_derivation() {
        let mut ratings = SkillRating::default();
        assert_eq!(*ratings.get_for_teams(&[0, 0, 1, 1], 0.0), 0.0);
        assert_eq!(ratings.data.get("2v2"), Some(&0.0));
        assert_eq!(*ratings.get_for_teams(&[0, 1, 1], 100.0), 100.0);
        assert_eq!(ratings.data.get("1v2"), Some(&100.0));
    }

    #[test]
    fn skill_rating_serde_roundtrip() {
        let mut ratings = SkillRating::default();
        *ratings.get_or_default("1v1", 1500.0) = 1234.5;
        let toml_str = toml::to_string(&ratings).unwrap();
        let parsed: SkillRating = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.data.get("1v1"), Some(&1234.5));
    }
}
