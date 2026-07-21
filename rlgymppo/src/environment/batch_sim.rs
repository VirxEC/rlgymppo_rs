use std::collections::VecDeque;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use burn::prelude::*;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::agent::model::Actic;
use crate::base::{Memory, TerminalState};
use crate::utils::Report;
use crate::utils::shared_info::SharedInfoReport;

/// Per-player trajectory buffer that persists across collection calls
/// so incomplete episodes carry over to the next iteration.
#[derive(Default)]
struct PlayerTraj {
    states: Vec<Vec<f32>>,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<TerminalState>,
    action_masks: Vec<Vec<bool>>,
}

pub struct BatchSim<B: Backend, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    games: Vec<GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>>,
    np: Vec<usize>,
    player_offsets: Vec<usize>,
    held_actions: Vec<Vec<usize>>,
    action_delay_primed: Vec<bool>,
    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    player_trajs: Vec<PlayerTraj>,
    overflow_trajs: VecDeque<(PlayerTraj, Option<Vec<f32>>)>,
    retain_overflow_episodes: bool,
    metrics: Report,
    device: B::Device,
    max_episode_length: Option<usize>,

    // ── Self‑play state ──────────────────────────────────────────
    /// Per-player team index (0 = Blue, 1 = Orange), cached at
    /// construction / game reset.
    player_teams: Vec<usize>,
}

impl<B, SS, OBS, ACT, REW, TERM, TRUNC, SI> BatchSim<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend,
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new<F>(
        create_env_fn: F,
        thread_num: usize,
        num_games: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
        max_episode_length: Option<usize>,
        retain_overflow_episodes: bool,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut games = Vec::with_capacity(num_games);
        let mut np = Vec::with_capacity(num_games);
        let mut player_offsets = Vec::with_capacity(num_games);
        let mut next_obs = Vec::with_capacity(num_games);
        let mut next_masks = Vec::with_capacity(num_games);
        let mut held_actions = Vec::with_capacity(num_games);
        let mut player_teams = Vec::new();

        let mut player_offset = 0;
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * (i + 1)));
            let mut game = GameInstance::new(env, reward_sampling.clone());
            let (obs, masks) = game.reset();
            let n = game.num_players();
            next_obs.extend(obs);
            next_masks.extend(masks);
            np.push(n);
            player_offsets.push(player_offset);
            player_offset += n;
            held_actions.push(vec![0; n]);
            player_teams.extend(game.player_teams());
            games.push(game);
        }

        let total_players: usize = np.iter().sum();
        let player_trajs = (0..total_players).map(|_| PlayerTraj::default()).collect();

        Self {
            metrics: Report::default(),
            next_obs,
            next_masks,
            games,
            np,
            player_offsets,
            held_actions,
            action_delay_primed: vec![false; num_games],
            player_trajs,
            overflow_trajs: VecDeque::new(),
            retain_overflow_episodes,
            device,
            player_teams,
            max_episode_length,
        }
    }

    /// Collect complete episodes until the shared iteration budget is exhausted.
    ///
    /// When `self_play` is `Some((old_model, old_team))`, the players on
    /// `old_team` (0 = Blue, 1 = Orange) use `old_model` for inference
    /// while the rest use the current `model`.  Only trajectories from
    /// current-policy players are recorded in the returned [`Memory`].
    pub fn run_with_budget(
        &mut self,
        model: &Actic<B>,
        remaining_steps: &AtomicUsize,
        memory_capacity_hint: usize,
        self_play: Option<(&Actic<B>, usize)>,
    ) -> (Memory, Report) {
        let (old_model, old_team) = self_play.unzip();

        // Build per-player tracking mask: `true` when player uses
        // the current policy.
        let player_is_tracked: Vec<bool> = if let Some(ot) = old_team {
            self.player_teams.iter().map(|&t| t != ot).collect()
        } else {
            vec![true; self.player_teams.len()]
        };

        let mut memory = Memory::with_capacity(memory_capacity_hint);

        // Completed episodes retained from the previous collection call were
        // generated by its policy snapshot. Consume them before advancing the
        // environments under the newly published policy.
        while remaining_steps.load(Ordering::Relaxed) > 0 {
            let Some((traj, trunc_next_state)) = self.overflow_trajs.pop_front() else {
                break;
            };
            let traj_len = traj.states.len();
            if Self::claim_steps(remaining_steps, traj_len) {
                Self::push_traj(&mut memory, traj, trunc_next_state);
            } else {
                self.overflow_trajs.push_front((traj, trunc_next_state));
                break;
            }
        }

        let mut total_infer_time = 0.0_f64;
        let mut total_env_step_time = 0.0_f64;

        while remaining_steps.load(Ordering::Relaxed) > 0 {
            let infer_start = Instant::now();
            let action_delay = ACT::get_action_delay();

            let (actions, log_probs) = if let (Some(old_model), Some(_ot)) = (old_model, old_team) {
                // ── Self-play: submit each policy's indexed batch ─────────
                let mut current_indices = Vec::new();
                let mut old_indices = Vec::new();
                for (index, &tracked) in player_is_tracked.iter().enumerate() {
                    if tracked {
                        current_indices.push(index);
                    } else {
                        old_indices.push(index);
                    }
                }

                let current_pending = (!current_indices.is_empty()).then(|| {
                    model.submit_react_indexed(
                        &self.next_obs,
                        &self.next_masks,
                        &current_indices,
                        &self.device,
                    )
                });
                let old_pending = (!old_indices.is_empty()).then(|| {
                    old_model.submit_react_indexed(
                        &self.next_obs,
                        &self.next_masks,
                        &old_indices,
                        &self.device,
                    )
                });

                if action_delay > 0 {
                    let delay_start = Instant::now();
                    for (game_idx, game) in self.games.iter_mut().enumerate() {
                        if self.action_delay_primed[game_idx] {
                            game.begin_delayed_step(&self.held_actions[game_idx]);
                        } else {
                            game.begin_neutral_delayed_step();
                        }
                    }
                    total_env_step_time += delay_start.elapsed().as_secs_f64();
                }

                let (current_actions, current_log_probs) = current_pending
                    .map(|pending| pending.wait())
                    .unwrap_or_default();
                let (old_actions, _) = old_pending
                    .map(|pending| pending.wait())
                    .unwrap_or_default();

                // Interleave results back into the original player order.
                let mut actions = vec![0usize; self.next_obs.len()];
                let mut log_probs = vec![0.0f32; self.next_obs.len()];
                for (offset, &index) in current_indices.iter().enumerate() {
                    actions[index] = current_actions[offset];
                    log_probs[index] = current_log_probs[offset];
                }
                for (offset, &index) in old_indices.iter().enumerate() {
                    actions[index] = old_actions[offset];
                }

                (actions, log_probs)
            } else if action_delay == 0 {
                model.react(&self.next_obs, &self.next_masks, &self.device)
            } else {
                // Dispatch inference before simulating the delayed action window.
                // `wait` below is the first host synchronization point.
                let pending = model.submit_react(&self.next_obs, &self.next_masks, &self.device);

                let delay_start = Instant::now();
                for (game_idx, game) in self.games.iter_mut().enumerate() {
                    if self.action_delay_primed[game_idx] {
                        game.begin_delayed_step(&self.held_actions[game_idx]);
                    } else {
                        game.begin_neutral_delayed_step();
                    }
                }
                total_env_step_time += delay_start.elapsed().as_secs_f64();

                pending.wait()
            };

            total_infer_time += infer_start.elapsed().as_secs_f64();

            let env_start = Instant::now();

            // Record pre‑step observations (current-policy only).
            for (i, (obs, mask)) in self.next_obs.iter().zip(self.next_masks.iter()).enumerate() {
                if player_is_tracked[i] {
                    self.player_trajs[i].states.push(obs.clone());
                    self.player_trajs[i].action_masks.push(mask.clone());
                }
            }

            self.next_obs.clear();
            self.next_masks.clear();

            // Step games in forward order.
            let mut action_offset = 0;
            for (game_idx, game) in self.games.iter_mut().enumerate() {
                let n = self.np[game_idx];

                let game_actions = &actions[action_offset..action_offset + n];
                let result = if action_delay == 0 {
                    game.step(game_actions)
                } else {
                    // The delay segment ran while the GPU evaluated this action.
                    game.finish_delayed_step(game_actions)
                };
                self.held_actions[game_idx].clear();
                self.held_actions[game_idx].extend_from_slice(game_actions);
                self.action_delay_primed[game_idx] = true;

                let mut terminal_type = if result.truncated {
                    TerminalState::Truncated
                } else if result.is_terminal {
                    TerminalState::Normal
                } else {
                    TerminalState::None
                };

                let player_start = self.player_offsets[game_idx];

                // Force-truncate if any tracked player in this game exceeds
                // the maximum episode length (matches GGL behaviour).
                if terminal_type == TerminalState::None
                    && let Some(max_len) = self.max_episode_length
                    && (player_start..player_start + n).any(|ti| {
                        player_is_tracked[ti] && self.player_trajs[ti].states.len() >= max_len
                    })
                {
                    terminal_type = TerminalState::Truncated;
                }

                for p in 0..n {
                    let ti = player_start + p;
                    if player_is_tracked[ti] {
                        self.player_trajs[ti].rewards.push(result.rewards[p]);
                        self.player_trajs[ti]
                            .actions
                            .push(actions[action_offset + p]);
                        self.player_trajs[ti]
                            .log_probs
                            .push(log_probs[action_offset + p]);
                        self.player_trajs[ti].terminals.push(terminal_type);
                    }
                }

                let is_terminal = terminal_type != TerminalState::None;
                if is_terminal {
                    // Episode ended — flush trajectories.
                    for p in 0..n {
                        let ti = player_start + p;
                        if player_is_tracked[ti] {
                            let traj_len = self.player_trajs[ti].states.len();
                            let trunc_next = (terminal_type == TerminalState::Truncated)
                                .then(|| result.obs[p].clone());
                            if Self::claim_steps(remaining_steps, traj_len) {
                                Self::push_traj(
                                    &mut memory,
                                    mem::take(&mut self.player_trajs[ti]),
                                    trunc_next,
                                );
                            } else {
                                let traj = mem::take(&mut self.player_trajs[ti]);
                                if self.retain_overflow_episodes {
                                    self.overflow_trajs.push_back((traj, trunc_next));
                                }
                            }
                        } else {
                            // Discard untracked player's buffers.
                            let _ = mem::take(&mut self.player_trajs[ti]);
                        }
                    }

                    if result.is_terminal || result.truncated {
                        // Env-level terminal — reset the game.
                        let (obs, masks) = game.reset();
                        self.action_delay_primed[game_idx] = false;
                        self.next_obs.extend(obs);
                        self.next_masks.extend(masks);

                        // Refresh cached team info.
                        let teams = game.player_teams();
                        self.player_teams[player_start..(player_start + n)]
                            .copy_from_slice(&teams[..n]);
                    } else {
                        // Force truncation: game continues from current state
                        // (matches GGL behaviour).
                        self.next_obs.extend(result.obs);
                        self.next_masks.extend(result.action_masks);
                    }
                } else {
                    self.next_obs.extend(result.obs);
                    self.next_masks.extend(result.action_masks);
                }

                action_offset += n;
            }

            total_env_step_time += env_start.elapsed().as_secs_f64();
        }

        let mut report = self.get_metrics();
        report["Collect/inference time"] = total_infer_time.into();
        report["Collect/env step time"] = total_env_step_time.into();

        (memory, report)
    }

    fn claim_steps(remaining_steps: &AtomicUsize, steps: usize) -> bool {
        remaining_steps
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                (remaining > 0).then_some(remaining.saturating_sub(steps))
            })
            .is_ok()
    }

    fn push_traj(memory: &mut Memory, traj: PlayerTraj, trunc_next_state: Option<Vec<f32>>) {
        memory.push_player(
            traj.states,
            traj.actions,
            traj.log_probs,
            traj.rewards,
            traj.terminals,
            traj.action_masks,
            trunc_next_state,
        );
    }

    fn get_metrics(&mut self) -> Report {
        for game in &mut self.games {
            self.metrics += game.get_metrics();
            game.clear_metrics();
        }

        let metrics = self.metrics.clone();
        self.metrics.clear();

        metrics
    }
}
