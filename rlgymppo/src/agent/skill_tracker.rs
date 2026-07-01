use std::collections::HashMap;
use std::marker::PhantomData;

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

    /// Get-or-insert a rating for a mode derived from per-player team
    /// indices (0=Blue, 1=Orange).  Used when no [`GameState`] is
    /// available (worker threads don't send it).
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

/// Manages the skill-rating evaluation pool and Elo-ratings lifecycle.
///
/// Owns a set of in-process [`GameInstance`]s (one per arena) and steps
/// them synchronously during skill-evaluation matches.
pub struct SkillTracker<B: Backend, OBS, ACT, SI>
where
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    config: SkillTrackerConfig,

    /// In-process game instances, one per arena.
    games:
        Vec<GameInstance<KickoffState, OBS, ACT, ZeroReward, OnGoalCondition, NeverTruncate, SI>>,

    // ── Buffered observation state ──────────────────────────────
    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    np: Vec<usize>,
    player_teams: Vec<usize>,

    // ── Rating state ────────────────────────────────────────────
    pub cur_ratings: SkillRating,
    cur_goals: usize,
    iterations_since_ran: usize,

    // ── Continuation state ──────────────────────────────────────
    do_continuation: bool,
    prev_old_version_idx: usize,
    prev_new_team: usize,
    prev_total_ticks: u64,

    device: B::Device,
    tick_skip: u8,

    _phantom: PhantomData<(OBS, ACT, SI)>,
}

impl<B, OBS, ACT, SI> SkillTracker<B, OBS, ACT, SI>
where
    B: Backend,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    SI: SharedInfoProvider + SharedInfoReport + SharedInfoRng,
{
    /// Construct a new skill tracker.  Creates one in-process
    /// [`GameInstance`] per arena.
    ///
    /// `create_arena` is called with the arena index; it should return
    /// an [`Arena`](rlgym::rocketsim::Arena) configured identically to
    /// the training env (matching car bodies, etc.).
    pub fn new<F>(config: SkillTrackerConfig, create_arena: F, device: B::Device) -> Self
    where
        F: Fn(usize) -> (Arena, OBS, ACT, SI),
    {
        let tick_skip = ACT::get_tick_skip();
        let num_arenas = config.num_arenas;

        let mut games = Vec::with_capacity(num_arenas);
        let mut next_obs = Vec::new();
        let mut next_masks = Vec::new();
        let mut np = Vec::with_capacity(num_arenas);
        let mut player_teams = Vec::new();

        let reward_sampling = RewardSamplingConfig {
            add_rewards_to_metrics: false,
            ..Default::default()
        };

        for game_idx in 0..num_arenas {
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

            let mut game = GameInstance::new(env, reward_sampling.clone());
            let (obs, masks) = game.reset();
            let teams = game.player_teams();
            next_obs.extend(obs);
            next_masks.extend(masks);
            let n = teams.len();
            np.push(n);
            player_teams.extend(teams);

            games.push(game);
        }

        Self {
            config,
            games,
            next_obs,
            next_masks,
            np,
            player_teams,
            cur_ratings: SkillRating::default(),
            cur_goals: 0,
            iterations_since_ran: 0,
            do_continuation: false,
            prev_old_version_idx: 0,
            prev_new_team: 0,
            prev_total_ticks: 0,
            device,
            tick_skip,
            _phantom: PhantomData,
        }
    }

    /// Call once per training iteration.  Returns `true` when matches
    /// were actually executed.
    pub fn on_iteration(
        &mut self,
        current_model: &Actic<B>,
        versions: &mut [PolicyVersion<B>],
    ) -> bool {
        if !self.config.enabled || versions.is_empty() {
            return false;
        }

        self.iterations_since_ran += 1;
        if self.iterations_since_ran >= self.config.update_interval {
            self.iterations_since_ran = 0;
            self.run_matches(current_model, versions);
            true
        } else {
            false
        }
    }

    /// Always write the current ratings into `report` so the TUI can
    /// display them every iteration.
    pub fn report_ratings(&self, report: &mut Report) {
        for (mode, &rating) in &self.cur_ratings.data {
            let key = format!("Rating/{mode}");
            report[key.as_str()] = rating.into();
        }
    }

    fn run_matches(&mut self, current_model: &Actic<B>, versions: &mut [PolicyVersion<B>]) {
        let num_arenas = self.config.num_arenas;

        // ── Pick opponent & team ──────────────────────────────────
        let (old_idx, new_team, mut total_ticks) = if self.do_continuation {
            (
                self.prev_old_version_idx,
                self.prev_new_team,
                self.prev_total_ticks,
            )
        } else {
            let mut rng = rand::rng();
            let idx = (rng.next_u32() as usize) % versions.len();
            let team = (rng.next_u32() as usize) % 2;
            self.reset_all_arenas();
            self.cur_goals = 0;
            (idx, team, 0u64)
        };
        self.do_continuation = false;

        let old_version = &mut versions[old_idx];

        let mut new_players = Vec::new();
        let mut old_players = Vec::new();
        for (i, &t) in self.player_teams.iter().enumerate() {
            if t == new_team {
                new_players.push(i);
            } else {
                old_players.push(i);
            }
        }

        let sim_ticks = (self.config.sim_time_secs * 120.0) as u64;
        let max_ticks = (self.config.max_sim_time_secs * 120.0) as u64;

        #[cfg(not(feature = "tui"))]
        let prev_ratings = self.cur_ratings.clone();

        #[cfg(not(feature = "tui"))]
        println!(
            " > Running skill matches (sim_time={:.1}s, new_team={}, old_version_ts={})...",
            self.config.sim_time_secs, new_team, old_version.timesteps,
        );

        let mut slice_start = total_ticks;

        while total_ticks < max_ticks && self.cur_goals < num_arenas {
            // ── Split observations by model group ─────────────────
            let (no, nm): (Vec<_>, Vec<_>) = new_players
                .iter()
                .map(|&i| (self.next_obs[i].clone(), self.next_masks[i].clone()))
                .unzip();
            let (oo, om): (Vec<_>, Vec<_>) = old_players
                .iter()
                .map(|&i| (self.next_obs[i].clone(), self.next_masks[i].clone()))
                .unzip();

            let new_actions = if no.is_empty() {
                Vec::new()
            } else if self.config.deterministic {
                current_model.react_deterministic(&no, &nm, &self.device)
            } else {
                current_model.react(&no, &nm, &self.device).0
            };

            let old_actions = if oo.is_empty() {
                Vec::new()
            } else if self.config.deterministic {
                old_version
                    .model
                    .react_deterministic(&oo, &om, &self.device)
            } else {
                old_version.model.react(&oo, &om, &self.device).0
            };

            // Interleave actions into full player order.
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

            // ── Step games directly ───────────────────────────────
            for game_idx in 0..num_arenas {
                let n = self.np[game_idx];
                let player_start: usize = self.np[..game_idx].iter().sum();
                let actions = combined[player_start..player_start + n].to_vec();

                let result = self.games[game_idx].step(&actions);

                let (teams, obs, masks, is_terminal, ball_y) = if result.is_terminal {
                    let ball_y = self.games[game_idx].last_game_state().ball.pos.y;
                    let (obs, masks) = self.games[game_idx].reset();
                    let teams = self.games[game_idx].player_teams();
                    (teams, obs, masks, true, ball_y)
                } else {
                    let teams = self.games[game_idx].player_teams();
                    (teams, result.obs, result.action_masks, false, 0.0)
                };

                if is_terminal {
                    //   ball_y < 0 → Blue scored
                    let blue_scored = ball_y.is_sign_negative();
                    let scorer_was_new = (new_team == 0) == blue_scored;

                    let update = |winner: &mut SkillRating, loser: &mut SkillRating| {
                        let w = winner.get_for_teams(&teams, self.config.initial_rating);
                        let l = loser.get_for_teams(&teams, self.config.initial_rating);
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

                // Refresh team info (may have changed after a reset).
                self.player_teams[player_start..player_start + n].copy_from_slice(&teams);

                self.next_obs.extend(obs);
                self.next_masks.extend(masks);
            }

            total_ticks += self.tick_skip as u64;

            if total_ticks.saturating_sub(slice_start) >= sim_ticks {
                slice_start = total_ticks;
            }
        }

        // ── Continuation / finalisation ───────────────────────────
        if self.cur_goals < num_arenas && total_ticks < max_ticks {
            #[cfg(not(feature = "tui"))]
            println!(
                " > Forcing continuation ({}/{})",
                self.cur_goals, num_arenas
            );
            self.do_continuation = true;
            self.prev_old_version_idx = old_idx;
            self.prev_new_team = new_team;
            self.prev_total_ticks = total_ticks;
        } else {
            self.cur_goals = 0;
        }

        // ── Console report ───────────────────────────────────────
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

    /// Reset all arenas to a fresh kickoff.
    fn reset_all_arenas(&mut self) {
        self.next_obs.clear();
        self.next_masks.clear();
        self.player_teams.clear();

        for game_idx in 0..self.config.num_arenas {
            let (obs, masks) = self.games[game_idx].reset();
            let teams = self.games[game_idx].player_teams();
            let n = teams.len();
            self.np[game_idx] = n;
            self.next_obs.extend(obs);
            self.next_masks.extend(masks);
            self.player_teams.extend(teams);
        }
    }
}
