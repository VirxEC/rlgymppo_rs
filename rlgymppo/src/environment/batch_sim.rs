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
    /// Per-step observations stored row-major as `[step * state_width..]`.
    states: Vec<f32>,
    state_width: usize,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<TerminalState>,
    /// Per-step action masks stored row-major.
    action_masks: Vec<bool>,
    action_mask_width: usize,
}

impl PlayerTraj {
    fn with_width(state_width: usize, action_mask_width: usize) -> Self {
        Self {
            state_width,
            action_mask_width,
            ..Self::default()
        }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    /// Move the accumulated samples out while keeping this slot ready for the
    /// next episode with the same row widths.
    fn take(&mut self) -> Self {
        let state_width = self.state_width;
        let action_mask_width = self.action_mask_width;
        mem::replace(self, Self::with_width(state_width, action_mask_width))
    }

    fn split_off(&mut self, at: usize) -> Self {
        Self {
            states: self.states.split_off(at * self.state_width),
            state_width: self.state_width,
            actions: self.actions.split_off(at),
            log_probs: self.log_probs.split_off(at),
            rewards: self.rewards.split_off(at),
            terminals: self.terminals.split_off(at),
            action_masks: self.action_masks.split_off(at * self.action_mask_width),
            action_mask_width: self.action_mask_width,
        }
    }

    fn state_at(&self, index: usize) -> Vec<f32> {
        let start = index * self.state_width;
        self.states[start..start + self.state_width].to_vec()
    }

    fn truncate(&mut self, len: usize) {
        self.states.truncate(len * self.state_width);
        self.actions.truncate(len);
        self.log_probs.truncate(len);
        self.rewards.truncate(len);
        self.terminals.truncate(len);
        self.action_masks.truncate(len * self.action_mask_width);
    }
}

type OverflowTraj = (PlayerTraj, Option<Vec<f32>>);

struct ClaimedTrajectory {
    trajectory: PlayerTraj,
    len: usize,
    next_state: Option<Vec<f32>>,
}

struct TrajectoryPartition {
    claimed: Option<ClaimedTrajectory>,
    overflow: Option<OverflowTraj>,
}

fn partition_claimed_trajectory(
    mut traj: PlayerTraj,
    trunc_next_state: Option<Vec<f32>>,
    claimed: usize,
    retain_overflow: bool,
) -> TrajectoryPartition {
    debug_assert!(claimed <= traj.len());

    if claimed == traj.len() {
        return TrajectoryPartition {
            claimed: Some(ClaimedTrajectory {
                len: traj.len(),
                trajectory: traj,
                next_state: trunc_next_state,
            }),
            overflow: None,
        };
    }
    if claimed == 0 {
        return TrajectoryPartition {
            claimed: None,
            overflow: retain_overflow.then_some((traj, trunc_next_state)),
        };
    }

    let boundary_next_state = Some(traj.state_at(claimed));
    let overflow = retain_overflow.then(|| (traj.split_off(claimed), trunc_next_state));
    TrajectoryPartition {
        claimed: Some(ClaimedTrajectory {
            trajectory: traj,
            len: claimed,
            next_state: boundary_next_state,
        }),
        overflow,
    }
}

fn claim_overbatch_steps(
    remaining_steps: &AtomicUsize,
    steps: usize,
    rollout_budget: usize,
) -> usize {
    // Once this claim exceeds `remaining`, the rollout is over budget. Limit
    // that excess to `rollout_budget - 1`, keeping the total strictly below 2x.
    let claim = |remaining: usize| {
        let max_claim = remaining.saturating_add(rollout_budget.saturating_sub(1));
        steps.min(max_claim)
    };

    remaining_steps
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
            (remaining > 0).then(|| remaining.saturating_sub(claim(remaining)))
        })
        .map(claim)
        .unwrap_or(0)
}

fn claim_available_steps(remaining_steps: &AtomicUsize, steps: usize) -> usize {
    remaining_steps
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
            Some(remaining.saturating_sub(steps))
        })
        .map(|remaining| remaining.min(steps))
        .unwrap_or(0)
}

fn push_claimed_trajectory(memory: &mut Memory, claimed: Option<ClaimedTrajectory>) {
    if let Some(claimed) = claimed {
        push_traj_prefix(memory, claimed.trajectory, claimed.len, claimed.next_state);
    }
}

fn push_traj_prefix(
    memory: &mut Memory,
    mut traj: PlayerTraj,
    len: usize,
    trunc_next_state: Option<Vec<f32>>,
) {
    let len = len.min(traj.len());
    traj.truncate(len);
    if trunc_next_state.is_some()
        && let Some(terminal) = traj.terminals.last_mut()
    {
        *terminal = TerminalState::Truncated;
    }

    memory.push_player(
        traj.states,
        traj.state_width,
        traj.actions,
        traj.log_probs,
        traj.rewards,
        traj.terminals,
        traj.action_masks,
        traj.action_mask_width,
        trunc_next_state,
    );
}

fn reached_max_episode_length(
    player_trajs: &[PlayerTraj],
    player_is_tracked: &[bool],
    player_start: usize,
    player_count: usize,
    max_episode_length: Option<usize>,
) -> bool {
    max_episode_length.is_some_and(|max_len| {
        (player_start..player_start + player_count)
            .any(|ti| player_is_tracked[ti] && player_trajs[ti].len() + 1 >= max_len)
    })
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
    self_play_current_indices: Vec<usize>,
    self_play_old_indices: Vec<usize>,
    self_play_actions: Vec<usize>,
    self_play_log_probs: Vec<f32>,
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
        let state_width = next_obs.first().map_or(0, Vec::len);
        let action_mask_width = next_masks.first().map_or(0, Vec::len);
        debug_assert!(state_width > 0);
        let player_trajs = (0..total_players)
            .map(|_| PlayerTraj::with_width(state_width, action_mask_width))
            .collect();

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
            self_play_current_indices: Vec::new(),
            self_play_old_indices: Vec::new(),
            self_play_actions: Vec::new(),
            self_play_log_probs: Vec::new(),
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
        mut memory: Memory,
        rollout_budget: usize,
        self_play: Option<(&Actic<B>, usize)>,
        overbatching: bool,
    ) -> (Memory, Report) {
        let (old_model, old_team) = self_play.unzip();

        // Build per-player tracking mask: `true` when player uses
        // the current policy.
        let player_is_tracked: Vec<bool> = if let Some(ot) = old_team {
            self.player_teams.iter().map(|&t| t != ot).collect()
        } else {
            vec![true; self.player_teams.len()]
        };

        // Completed episodes retained from the previous collection call were
        // generated by its policy snapshot. Consume them before advancing the
        // environments under the newly published policy.
        while remaining_steps.load(Ordering::Relaxed) > 0 {
            let Some((traj, trunc_next_state)) = self.overflow_trajs.pop_front() else {
                break;
            };
            if overbatching {
                let claimed = claim_overbatch_steps(remaining_steps, traj.len(), rollout_budget);
                let partition = partition_claimed_trajectory(traj, trunc_next_state, claimed, true);
                push_claimed_trajectory(&mut memory, partition.claimed);
                if let Some(overflow) = partition.overflow {
                    self.overflow_trajs.push_front(overflow);
                }
                if claimed == 0 {
                    break;
                }
            } else {
                let claimed = claim_available_steps(remaining_steps, traj.len());
                let partition = partition_claimed_trajectory(traj, trunc_next_state, claimed, true);
                push_claimed_trajectory(&mut memory, partition.claimed);
                if let Some(overflow) = partition.overflow {
                    self.overflow_trajs.push_front(overflow);
                }
                if claimed == 0 {
                    break;
                }
            }
        }

        let mut total_infer_time = 0.0_f64;
        let mut total_env_step_time = 0.0_f64;

        while remaining_steps.load(Ordering::Relaxed) > 0 {
            let infer_start = Instant::now();
            let action_delay = ACT::get_action_delay();

            let (actions, log_probs) = if let (Some(old_model), Some(_ot)) = (old_model, old_team) {
                // ── Self-play: submit each policy's indexed batch ─────────
                self.self_play_current_indices.clear();
                self.self_play_old_indices.clear();
                for (index, &tracked) in player_is_tracked.iter().enumerate() {
                    if tracked {
                        self.self_play_current_indices.push(index);
                    } else {
                        self.self_play_old_indices.push(index);
                    }
                }

                let current_pending = (!self.self_play_current_indices.is_empty()).then(|| {
                    model.submit_react_indexed(
                        &self.next_obs,
                        &self.next_masks,
                        &self.self_play_current_indices,
                        &self.device,
                    )
                });
                let old_pending = (!self.self_play_old_indices.is_empty()).then(|| {
                    old_model.submit_react_indexed(
                        &self.next_obs,
                        &self.next_masks,
                        &self.self_play_old_indices,
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

                // Interleave results back into reusable buffers in the original
                // player order.
                let player_count = self.next_obs.len();
                self.self_play_actions.clear();
                self.self_play_actions.resize(player_count, 0);
                self.self_play_log_probs.clear();
                self.self_play_log_probs.resize(player_count, 0.0);
                for (offset, &index) in self.self_play_current_indices.iter().enumerate() {
                    self.self_play_actions[index] = current_actions[offset];
                    self.self_play_log_probs[index] = current_log_probs[offset];
                }
                for (offset, &index) in self.self_play_old_indices.iter().enumerate() {
                    self.self_play_actions[index] = old_actions[offset];
                }

                (
                    mem::take(&mut self.self_play_actions),
                    mem::take(&mut self.self_play_log_probs),
                )
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
                    self.player_trajs[i].states.extend_from_slice(obs);
                    self.player_trajs[i].action_masks.extend_from_slice(mask);
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

                // Force-truncate if the current step would bring any tracked
                // player's trajectory to the maximum episode length (matches
                // GGL behaviour).
                if terminal_type == TerminalState::None
                    && reached_max_episode_length(
                        &self.player_trajs,
                        &player_is_tracked,
                        player_start,
                        n,
                        self.max_episode_length,
                    )
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
                            let traj_len = self.player_trajs[ti].len();
                            let trunc_next = (terminal_type == TerminalState::Truncated)
                                .then(|| result.obs[p].clone());
                            let traj = self.player_trajs[ti].take();
                            if overbatching {
                                let claimed = claim_overbatch_steps(
                                    remaining_steps,
                                    traj_len,
                                    rollout_budget,
                                );
                                let partition = partition_claimed_trajectory(
                                    traj,
                                    trunc_next,
                                    claimed,
                                    self.retain_overflow_episodes,
                                );
                                push_claimed_trajectory(&mut memory, partition.claimed);
                                if let Some(overflow) = partition.overflow {
                                    self.overflow_trajs.push_back(overflow);
                                }
                            } else {
                                let claimed = claim_available_steps(remaining_steps, traj_len);
                                let partition = partition_claimed_trajectory(
                                    traj,
                                    trunc_next,
                                    claimed,
                                    self.retain_overflow_episodes,
                                );
                                push_claimed_trajectory(&mut memory, partition.claimed);
                                if let Some(overflow) = partition.overflow {
                                    self.overflow_trajs.push_back(overflow);
                                }
                            }
                        } else {
                            // Discard untracked player's buffers.
                            let _ = self.player_trajs[ti].take();
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

            if self_play.is_some() {
                self.self_play_actions = actions;
                self.self_play_log_probs = log_probs;
            }

            total_env_step_time += env_start.elapsed().as_secs_f64();
        }

        let mut report = self.get_metrics();
        report["Collect/inference time"] = total_infer_time.into();
        report["Collect/env step time"] = total_env_step_time.into();

        (memory, report)
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

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn trajectory(len: usize) -> PlayerTraj {
        let mut terminals = vec![TerminalState::None; len];
        *terminals.last_mut().unwrap() = TerminalState::Normal;
        PlayerTraj {
            states: (0..len).map(|i| i as f32).collect(),
            state_width: 1,
            actions: (0..len).collect(),
            log_probs: vec![0.0; len],
            rewards: vec![1.0; len],
            terminals,
            action_masks: vec![true; len],
            action_mask_width: 1,
        }
    }

    #[test]
    fn regression_episode_reuse_preserves_flat_row_widths() {
        let mut trajectory = PlayerTraj::with_width(2, 1);
        trajectory.states.extend_from_slice(&[1.0, 2.0]);
        trajectory.actions.push(0);
        trajectory.log_probs.push(0.0);
        trajectory.rewards.push(1.0);
        trajectory.terminals.push(TerminalState::Normal);
        trajectory.action_masks.push(true);

        let _completed = trajectory.take();
        trajectory.states.extend_from_slice(&[3.0, 4.0]);
        trajectory.actions.push(1);
        trajectory.log_probs.push(0.0);
        trajectory.rewards.push(1.0);
        trajectory.terminals.push(TerminalState::Normal);
        trajectory.action_masks.push(false);
        trajectory.truncate(1);

        assert_eq!(trajectory.states, &[3.0, 4.0]);
        assert_eq!(trajectory.actions, &[1]);
        assert_eq!(trajectory.action_masks, &[false]);
    }

    #[test]
    fn regression_retention_controls_unclaimed_trajectory_suffix() {
        let partition = partition_claimed_trajectory(trajectory(5), Some(vec![5.0]), 2, true);
        let claimed = partition.claimed.unwrap();
        let (suffix, final_next_state) = partition.overflow.unwrap();

        assert_eq!(claimed.len, 2);
        assert_eq!(claimed.trajectory.actions, &[0, 1]);
        assert_eq!(claimed.next_state, Some(vec![2.0]));
        assert_eq!(suffix.actions, &[2, 3, 4]);
        assert_eq!(suffix.terminals.last(), Some(&TerminalState::Normal));
        assert_eq!(final_next_state, Some(vec![5.0]));

        let partition = partition_claimed_trajectory(trajectory(5), Some(vec![5.0]), 2, false);
        let claimed = partition.claimed.unwrap();
        assert_eq!(claimed.len, 2);
        assert_eq!(claimed.trajectory.states.len(), 5);
        assert_eq!(claimed.next_state, Some(vec![2.0]));
        assert!(partition.overflow.is_none());
    }

    #[test]
    fn regression_overbatch_claim_is_strictly_below_twice_the_budget() {
        let remaining = AtomicUsize::new(100);
        assert_eq!(claim_overbatch_steps(&remaining, 90, 100), 90);
        assert_eq!(remaining.load(Ordering::Relaxed), 10);
        assert_eq!(claim_overbatch_steps(&remaining, 500, 100), 109);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);
        assert_eq!(claim_overbatch_steps(&remaining, 1, 100), 0);
    }

    #[test]
    fn regression_partial_claim_flushes_a_truncated_bootstrap_boundary() {
        let partition = partition_claimed_trajectory(trajectory(5), Some(vec![5.0]), 2, false);
        let mut memory = Memory::with_capacity(2);
        push_claimed_trajectory(&mut memory, partition.claimed);

        assert!(partition.overflow.is_none());
        assert_eq!(memory.len(), 2);
        assert_eq!(memory.terminals().last(), Some(&TerminalState::Truncated));
        assert_eq!(memory.trunc_next_states(), &[2.0]);
    }

    #[test]
    fn regression_zero_claim_is_retained_only_when_enabled() {
        let partition = partition_claimed_trajectory(trajectory(3), None, 0, false);
        assert!(partition.claimed.is_none());
        assert!(partition.overflow.is_none());

        let partition = partition_claimed_trajectory(trajectory(3), None, 0, true);
        assert!(partition.claimed.is_none());
        assert_eq!(partition.overflow.unwrap().0.len(), 3);
    }

    #[test]
    fn regression_max_episode_length_only_counts_tracked_players() {
        let player_trajs = vec![trajectory(2), trajectory(4)];

        assert!(!reached_max_episode_length(
            &player_trajs,
            &[true, false],
            0,
            2,
            Some(5),
        ));
        assert!(reached_max_episode_length(
            &player_trajs,
            &[true, true],
            0,
            2,
            Some(5),
        ));
        assert!(!reached_max_episode_length(
            &player_trajs,
            &[true, true],
            0,
            2,
            Some(6),
        ));
        assert!(!reached_max_episode_length(
            &player_trajs,
            &[true, true],
            0,
            2,
            None,
        ));
    }
}
