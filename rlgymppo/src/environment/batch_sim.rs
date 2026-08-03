use std::collections::VecDeque;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use burn::prelude::*;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::agent::model::Actic;
use crate::base::{Memory, TerminalState};
use crate::utils::shared_info::SharedInfoReport;
use crate::utils::{AvgTracker, Report};

const EPISODE_LENGTH_EMA_ALPHA: f64 = 0.1;
const MIN_TRAJECTORY_BASELINE_STEPS: usize = 32;

fn compute_trajectory_baseline_steps(
    episode_length_ema: Option<f64>,
    episode_length_std_ema: Option<f64>,
    fallback_steps: usize,
    max_episode_length: Option<usize>,
) -> usize {
    let Some(average) = episode_length_ema.filter(|average| average.is_finite() && *average > 0.0)
    else {
        return fallback_steps;
    };
    let standard_deviation = episode_length_std_ema
        .filter(|standard_deviation| standard_deviation.is_finite() && *standard_deviation >= 0.0)
        .unwrap_or(0.0);

    let baseline = (average - standard_deviation)
        .max(MIN_TRAJECTORY_BASELINE_STEPS as f64)
        .ceil() as usize;
    max_episode_length
        .filter(|&max_length| max_length > 0)
        .map_or(baseline, |max_length| baseline.min(max_length))
}

/// Per-player trajectory buffer that persists across collection calls
/// so incomplete episodes carry over to the next iteration.
#[derive(Default)]
struct PlayerTraj {
    /// Per-step observations stored row-major as `[step * state_width..]`.
    states: Vec<f32>,
    state_width: usize,
    /// Per-step observations from the old (teacher) obs builder, row-major
    /// as `[step * old_state_width..]`. Empty when no old obs is configured.
    old_states: Vec<f32>,
    old_state_width: usize,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<TerminalState>,
    /// Per-step action masks stored row-major.
    action_masks: Vec<bool>,
    action_mask_width: usize,
    /// Retained row capacity after a trajectory is cleared. This keeps a small
    /// reusable baseline without retaining an entire unusually long episode.
    baseline_steps: usize,
}

impl PlayerTraj {
    fn with_width_and_baseline(
        state_width: usize,
        old_state_width: usize,
        action_mask_width: usize,
        baseline_steps: usize,
    ) -> Self {
        Self {
            state_width,
            old_state_width,
            action_mask_width,
            baseline_steps,
            ..Self::default()
        }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    fn set_baseline_steps(&mut self, baseline_steps: usize) {
        self.baseline_steps = baseline_steps;
    }

    /// Move the accumulated samples out while keeping this slot ready for the
    /// next episode with the same row widths.
    fn take(&mut self) -> Self {
        let state_width = self.state_width;
        let old_state_width = self.old_state_width;
        let action_mask_width = self.action_mask_width;
        let baseline_steps = self.baseline_steps;
        mem::replace(
            self,
            Self::with_width_and_baseline(
                state_width,
                old_state_width,
                action_mask_width,
                baseline_steps,
            ),
        )
    }

    fn split_off(&mut self, at: usize) -> Self {
        Self {
            states: self.states.split_off(at * self.state_width),
            state_width: self.state_width,
            old_states: self.old_states.split_off(at * self.old_state_width),
            old_state_width: self.old_state_width,
            actions: self.actions.split_off(at),
            log_probs: self.log_probs.split_off(at),
            rewards: self.rewards.split_off(at),
            terminals: self.terminals.split_off(at),
            action_masks: self.action_masks.split_off(at * self.action_mask_width),
            action_mask_width: self.action_mask_width,
            baseline_steps: self.baseline_steps,
        }
    }

    fn state_at(&self, index: usize) -> Vec<f32> {
        let start = index * self.state_width;
        self.states[start..start + self.state_width].to_vec()
    }

    fn shrink_to_baseline(&mut self) {
        self.states
            .shrink_to(self.baseline_steps * self.state_width);
        self.old_states
            .shrink_to(self.baseline_steps * self.old_state_width);
        self.actions.shrink_to(self.baseline_steps);
        self.log_probs.shrink_to(self.baseline_steps);
        self.rewards.shrink_to(self.baseline_steps);
        self.terminals.shrink_to(self.baseline_steps);
        self.action_masks
            .shrink_to(self.baseline_steps * self.action_mask_width);
    }

    fn clear(&mut self) {
        self.states.clear();
        self.old_states.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.terminals.clear();
        self.action_masks.clear();
        self.shrink_to_baseline();
    }

    fn truncate(&mut self, len: usize) {
        self.states.truncate(len * self.state_width);
        self.old_states.truncate(len * self.old_state_width);
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

/// Claim an entire completed trajectory. The final trajectory is allowed to
/// consume the remaining budget rather than being cut at an arbitrary row.
fn claim_complete_steps(
    remaining_steps: &AtomicUsize,
    steps: usize,
    allow_final_overrun: bool,
) -> usize {
    if remaining_steps
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
            (remaining > 0).then_some(remaining.saturating_sub(steps))
        })
        .is_ok()
    {
        steps
    } else if allow_final_overrun {
        // A worker that has not completed any trajectory yet must still be
        // allowed to finish one after the shared budget reaches zero.
        steps
    } else {
        0
    }
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
        traj.old_states,
        traj.old_state_width,
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
    /// Pre-step observations from the old obs builders, parallel to `next_obs`.
    next_old_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    player_trajs: Vec<PlayerTraj>,
    overflow_trajs: VecDeque<(PlayerTraj, Option<Vec<f32>>)>,
    retain_overflow_episodes: bool,
    metrics: Report,
    device: B::Device,
    max_episode_length: Option<usize>,
    /// When enabled, only complete trajectories are returned. Incomplete
    /// buffers from the previous policy snapshot are discarded at collection
    /// start, so no learner update spans policy publications.
    complete_trajectories: bool,
    /// Retained per-player trajectory capacity, adjusted from the smoothed
    /// completed-trajectory length after each collection.
    trajectory_baseline_steps: usize,
    episode_length_ema: Option<f64>,
    /// EMA of the second moment, used with `episode_length_ema` to estimate σ.
    episode_length_second_moment_ema: Option<f64>,

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
    #[allow(clippy::too_many_arguments)]
    pub fn new<F, FO>(
        create_env_fn: F,
        make_old_obs: Option<FO>,
        thread_num: usize,
        num_games: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
        trajectory_capacity_hint: usize,
        max_episode_length: Option<usize>,
        retain_overflow_episodes: bool,
        complete_trajectories: bool,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
        FO: Fn() -> Box<dyn Obs<SI>>,
    {
        let mut games = Vec::with_capacity(num_games);
        let mut np = Vec::with_capacity(num_games);
        let mut player_offsets = Vec::with_capacity(num_games);
        let mut next_obs = Vec::with_capacity(num_games);
        let mut next_old_obs = Vec::with_capacity(num_games);
        let mut next_masks = Vec::with_capacity(num_games);
        let mut held_actions = Vec::with_capacity(num_games);
        let mut player_teams = Vec::new();

        let mut player_offset = 0;
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * (i + 1)));
            let old_obs = make_old_obs.as_ref().map(|make| make());
            let mut game = GameInstance::new(env, old_obs, reward_sampling.clone());
            let (obs, old_obs, masks) = game.reset();
            let n = game.num_players();
            next_obs.extend(obs);
            next_old_obs.extend(old_obs);
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
        let old_state_width = next_old_obs.first().map_or(0, Vec::len);
        let action_mask_width = next_masks.first().map_or(0, Vec::len);
        debug_assert!(state_width > 0);
        let baseline_steps = trajectory_capacity_hint.div_ceil(total_players.max(1));
        let player_trajs = (0..total_players)
            .map(|_| {
                PlayerTraj::with_width_and_baseline(
                    state_width,
                    old_state_width,
                    action_mask_width,
                    baseline_steps,
                )
            })
            .collect();

        Self {
            metrics: Report::default(),
            next_obs,
            next_old_obs,
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
            complete_trajectories,
            trajectory_baseline_steps: baseline_steps,
            episode_length_ema: None,
            episode_length_second_moment_ema: None,
        }
    }

    fn episode_length_std_ema(&self) -> Option<f64> {
        let (Some(average), Some(second_moment)) = (
            self.episode_length_ema,
            self.episode_length_second_moment_ema,
        ) else {
            return None;
        };

        Some((second_moment - average * average).max(0.0).sqrt())
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
        rollout_budget: usize,
        self_play: Option<(&Actic<B>, usize)>,
        overbatching: bool,
    ) -> (Memory, Report) {
        let (old_model, old_team) = self_play.unzip();

        let baseline_steps = compute_trajectory_baseline_steps(
            self.episode_length_ema,
            self.episode_length_std_ema(),
            self.trajectory_baseline_steps,
            self.max_episode_length,
        );
        self.trajectory_baseline_steps = baseline_steps;
        for trajectory in &mut self.player_trajs {
            trajectory.set_baseline_steps(baseline_steps);
        }
        for (trajectory, _) in &mut self.overflow_trajs {
            trajectory.set_baseline_steps(baseline_steps);
        }

        if self.complete_trajectories {
            // These rows were generated under the previous published model.
            // Keeping them would make the next update span policy snapshots.
            // Clearing also trims any episode-sized high-water allocations.
            for trajectory in &mut self.player_trajs {
                trajectory.clear();
            }
            self.overflow_trajs.clear();
        } else {
            // Preserve live carry-over rows, but do not retain capacity from a
            // much longer episode than the normal per-worker rollout share.
            for trajectory in &mut self.player_trajs {
                trajectory.shrink_to_baseline();
            }
        }

        // Build per-player tracking mask: `true` when player uses
        // the current policy.
        let player_is_tracked: Vec<bool> = if let Some(ot) = old_team {
            self.player_teams.iter().map(|&t| t != ot).collect()
        } else {
            vec![true; self.player_teams.len()]
        };

        let mut memory = Memory::with_capacity(memory_capacity_hint);
        let mut completed_for_update = false;

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
        let mut completed_episode_steps = 0_usize;
        let mut completed_episode_squared_steps = 0.0_f64;
        let mut completed_episode_count = 0_usize;

        while remaining_steps.load(Ordering::Relaxed) > 0
            || (self.complete_trajectories && !completed_for_update)
        {
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
            for (i, old_obs) in self.next_old_obs.iter().enumerate() {
                if player_is_tracked[i] {
                    self.player_trajs[i].old_states.extend_from_slice(old_obs);
                }
            }

            self.next_obs.clear();
            self.next_old_obs.clear();
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
                            completed_episode_steps += traj_len;
                            let traj_len_f64 = traj_len as f64;
                            completed_episode_squared_steps += traj_len_f64 * traj_len_f64;
                            completed_episode_count += 1;
                            let traj = self.player_trajs[ti].take();
                            if self.complete_trajectories {
                                let claimed = claim_complete_steps(
                                    remaining_steps,
                                    traj_len,
                                    !completed_for_update,
                                );
                                if claimed == traj_len {
                                    completed_for_update = true;
                                    push_claimed_trajectory(
                                        &mut memory,
                                        Some(ClaimedTrajectory {
                                            trajectory: traj,
                                            len: traj_len,
                                            next_state: trunc_next,
                                        }),
                                    );
                                }
                            } else if overbatching {
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
                        let (obs, old_obs, masks) = game.reset();
                        self.action_delay_primed[game_idx] = false;
                        self.next_obs.extend(obs);
                        self.next_old_obs.extend(old_obs);
                        self.next_masks.extend(masks);

                        // Refresh cached team info.
                        let teams = game.player_teams();
                        self.player_teams[player_start..(player_start + n)]
                            .copy_from_slice(&teams[..n]);
                    } else {
                        // Force truncation: game continues from current state
                        // (matches GGL behaviour).
                        self.next_obs.extend(result.obs);
                        self.next_old_obs.extend(result.old_obs);
                        self.next_masks.extend(result.action_masks);
                    }
                } else {
                    self.next_obs.extend(result.obs);
                    self.next_old_obs.extend(result.old_obs);
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

        if completed_episode_count > 0 {
            let collection_average =
                completed_episode_steps as f64 / completed_episode_count as f64;
            let collection_second_moment =
                completed_episode_squared_steps / completed_episode_count as f64;
            self.episode_length_ema = Some(match self.episode_length_ema {
                Some(previous) => {
                    previous * (1.0 - EPISODE_LENGTH_EMA_ALPHA)
                        + collection_average * EPISODE_LENGTH_EMA_ALPHA
                }
                None => collection_average,
            });
            self.episode_length_second_moment_ema =
                Some(match self.episode_length_second_moment_ema {
                    Some(previous) => {
                        previous * (1.0 - EPISODE_LENGTH_EMA_ALPHA)
                            + collection_second_moment * EPISODE_LENGTH_EMA_ALPHA
                    }
                    None => collection_second_moment,
                });
        }

        // Apply the newly observed line immediately. Live rows are retained;
        // only capacity above the new line is released.
        let baseline_steps = compute_trajectory_baseline_steps(
            self.episode_length_ema,
            self.episode_length_std_ema(),
            self.trajectory_baseline_steps,
            self.max_episode_length,
        );
        self.trajectory_baseline_steps = baseline_steps;
        for trajectory in &mut self.player_trajs {
            trajectory.set_baseline_steps(baseline_steps);
            trajectory.shrink_to_baseline();
        }
        for (trajectory, _) in &mut self.overflow_trajs {
            trajectory.set_baseline_steps(baseline_steps);
            trajectory.shrink_to_baseline();
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

        self.metrics["Collect/trajectory capacity baseline"] +=
            AvgTracker::from(self.trajectory_baseline_steps as f64);

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
            old_states: Vec::new(),
            old_state_width: 0,
            actions: (0..len).collect(),
            log_probs: vec![0.0; len],
            rewards: vec![1.0; len],
            terminals,
            action_masks: vec![true; len],
            action_mask_width: 1,
            baseline_steps: 0,
        }
    }

    #[test]
    fn regression_episode_baseline_uses_mean_minus_std_and_maximum() {
        assert_eq!(
            compute_trajectory_baseline_steps(None, None, 270, Some(1800)),
            270
        );
        assert_eq!(
            compute_trajectory_baseline_steps(Some(100.0), Some(20.0), 270, Some(1800)),
            80
        );
        assert_eq!(
            compute_trajectory_baseline_steps(Some(20.0), Some(30.0), 270, Some(1800)),
            32
        );
        assert_eq!(
            compute_trajectory_baseline_steps(Some(2_000.0), Some(0.0), 270, Some(1800)),
            1800
        );
    }

    #[test]
    fn regression_clear_trims_capacity_to_baseline() {
        let baseline = 2;
        let mut trajectory = PlayerTraj::with_width_and_baseline(2, 0, 3, baseline);
        for _ in 0..16 {
            trajectory.states.extend_from_slice(&[0.0, 0.0]);
            trajectory.actions.push(0);
            trajectory.log_probs.push(0.0);
            trajectory.rewards.push(1.0);
            trajectory.terminals.push(TerminalState::None);
            trajectory
                .action_masks
                .extend_from_slice(&[true, false, true]);
        }

        let grown_state_capacity = trajectory.states.capacity();
        let grown_mask_capacity = trajectory.action_masks.capacity();
        trajectory.clear();

        assert_eq!(trajectory.len(), 0);
        assert!(trajectory.states.capacity() >= baseline * 2);
        assert!(trajectory.states.capacity() < grown_state_capacity);
        assert!(trajectory.action_masks.capacity() >= baseline * 3);
        assert!(trajectory.action_masks.capacity() < grown_mask_capacity);
        assert_eq!(trajectory.actions.capacity(), baseline);
        assert_eq!(trajectory.log_probs.capacity(), baseline);
        assert_eq!(trajectory.rewards.capacity(), baseline);
        assert_eq!(trajectory.terminals.capacity(), baseline);
    }

    #[test]
    fn regression_episode_reuse_preserves_flat_row_widths() {
        let mut trajectory = PlayerTraj::with_width_and_baseline(2, 0, 1, 0);
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
    fn regression_complete_claim_never_splits_a_trajectory() {
        let remaining = AtomicUsize::new(3);
        assert_eq!(claim_complete_steps(&remaining, 5, true), 5);
        assert_eq!(remaining.load(Ordering::Acquire), 0);
        assert_eq!(claim_complete_steps(&remaining, 1, false), 0);
        assert_eq!(claim_complete_steps(&remaining, 1, true), 1);
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
