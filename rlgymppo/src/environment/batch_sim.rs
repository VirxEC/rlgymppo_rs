use std::collections::VecDeque;
use std::mem;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use burn::prelude::*;
use rayon::ThreadPool;
use rlgym::{
    Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
};
use rlgymppo_utils::shared_info::SharedInfoReport;
use rlgymppo_utils::{AvgTracker, Report};

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::agent::model::Actic;
use crate::base::{Memory, TerminalState};

const EPISODE_LENGTH_EMA_ALPHA: f64 = 0.1;
const MIN_TRAJECTORY_BASELINE_STEPS: usize = 32;

/// Report key for the per-pool inference wall time.
pub(crate) const COLLECT_INFERENCE_TIME_KEY: &str = "Collect/inference time";
/// Report key for the per-pool env-step wall time.
pub(crate) const COLLECT_ENV_STEP_TIME_KEY: &str = "Collect/env step time";

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

/// Per-player trajectory buffer. Incomplete episodes carry over to the
/// next collection.
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

/// A completed per-player trajectory, consumed in game order by the
/// claim pass.
struct FlushedTrajectory {
    trajectory: PlayerTraj,
    len: usize,
    next_state: Option<Vec<f32>>,
}

/// Per-game flush output. Terminal games produce flushes; non-terminal
/// games leave an empty outcome. It also accumulates episode-length
/// statistics.
#[derive(Default)]
struct GameOutcome {
    flushes: Vec<FlushedTrajectory>,
    episode_steps: usize,
    episode_squared_steps: f64,
    episode_count: usize,
}

struct TrajectoryPartition {
    claimed: Option<ClaimedTrajectory>,
    overflow: Option<OverflowTraj>,
}

/// Split a trajectory into the claimed prefix and the retained suffix.
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

/// Claim up to `steps` while the budget allows overbatching. The total
/// stays strictly below twice the budget.
fn claim_overbatch_steps(
    remaining_steps: &AtomicUsize,
    steps: usize,
    rollout_budget: usize,
) -> usize {
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

/// Claim up to `steps` without overbatching.
fn claim_available_steps(remaining_steps: &AtomicUsize, steps: usize) -> usize {
    remaining_steps
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
            Some(remaining.saturating_sub(steps))
        })
        .map(|remaining| remaining.min(steps))
        .unwrap_or(0)
}

/// Claim an entire completed trajectory. With `allow_final_overrun`,
/// one final overrun is allowed after the budget reaches zero.
fn claim_complete_steps(
    remaining_steps: &AtomicUsize,
    steps: usize,
    allow_final_overrun: bool,
) -> usize {
    let claimed = remaining_steps
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
            (remaining > 0).then_some(remaining.saturating_sub(steps))
        })
        .is_ok();
    if claimed || allow_final_overrun {
        steps
    } else {
        0
    }
}

/// Push a claimed trajectory into the memory.
fn push_claimed_trajectory(memory: &mut Memory, claimed: Option<ClaimedTrajectory>) {
    if let Some(claimed) = claimed {
        push_traj_prefix(memory, claimed.trajectory, claimed.len, claimed.next_state);
    }
}

/// Push a trajectory prefix into the memory. A cut boundary row gets a
/// truncated terminal state.
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

/// Check whether the next step brings any tracked player to the maximum
/// episode length.
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

/// The disjoint per-game state that one step job mutates.
struct StepJob<'a, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    game: &'a mut GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    trajs: &'a mut [PlayerTraj],
    next_obs: &'a mut [f32],
    next_old_obs: &'a mut [f32],
    next_masks: &'a mut [bool],
    held_actions: &'a mut Vec<usize>,
    action_delay_primed: &'a mut bool,
    player_teams: &'a mut [usize],
    outcome: &'a mut Option<GameOutcome>,
}

/// The read-only parameters that all step jobs share.
struct StepConfig<'a> {
    state_width: usize,
    old_state_width: usize,
    mask_width: usize,
    max_episode_length: Option<usize>,
    action_delay: u8,
    actions: &'a [usize],
    log_probs: &'a [f32],
    tracked: &'a [bool],
}

/// Advance one game through the remainder of the decision interval.
///
/// A force truncation applies when a tracked player reaches the maximum
/// episode length; the game then continues instead of resetting.
fn step_game<SS, OBS, ACT, REW, TERM, TRUNC, SI>(
    job: StepJob<'_, SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    config: &StepConfig<'_>,
) where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    let StepJob {
        game,
        trajs,
        next_obs,
        next_old_obs,
        next_masks,
        held_actions,
        action_delay_primed,
        player_teams,
        outcome,
    } = job;
    let n = config.actions.len();

    for (p, traj) in trajs.iter_mut().enumerate() {
        if config.tracked[p] {
            traj.states
                .extend_from_slice(&next_obs[p * config.state_width..(p + 1) * config.state_width]);
            traj.action_masks
                .extend_from_slice(&next_masks[p * config.mask_width..(p + 1) * config.mask_width]);
            traj.old_states.extend_from_slice(
                &next_old_obs[p * config.old_state_width..(p + 1) * config.old_state_width],
            );
        }
    }

    let result = if config.action_delay == 0 {
        game.step(config.actions)
    } else {
        game.finish_delayed_step(config.actions)
    };

    held_actions.clear();
    held_actions.extend_from_slice(config.actions);
    *action_delay_primed = true;

    let mut terminal_type = if result.truncated {
        TerminalState::Truncated
    } else if result.is_terminal {
        TerminalState::Normal
    } else {
        TerminalState::None
    };

    if terminal_type == TerminalState::None
        && reached_max_episode_length(trajs, config.tracked, 0, n, config.max_episode_length)
    {
        terminal_type = TerminalState::Truncated;
    }

    for (p, traj) in trajs.iter_mut().enumerate() {
        if config.tracked[p] {
            traj.rewards.push(result.rewards[p]);
            traj.actions.push(config.actions[p]);
            traj.log_probs.push(config.log_probs[p]);
            traj.terminals.push(terminal_type);
        }
    }

    let mut flush = GameOutcome::default();
    if terminal_type != TerminalState::None {
        for (p, traj) in trajs.iter_mut().enumerate() {
            if config.tracked[p] {
                let traj_len = traj.len();
                let trunc_next =
                    (terminal_type == TerminalState::Truncated).then(|| result.obs[p].clone());
                let traj_len_f64 = traj_len as f64;
                flush.episode_steps += traj_len;
                flush.episode_squared_steps += traj_len_f64 * traj_len_f64;
                flush.episode_count += 1;
                flush.flushes.push(FlushedTrajectory {
                    trajectory: traj.take(),
                    len: traj_len,
                    next_state: trunc_next,
                });
            } else {
                let _ = traj.take();
            }
        }
    }

    if result.is_terminal || result.truncated {
        let (obs, old_obs, masks) = game.reset();
        *action_delay_primed = false;
        for p in 0..n {
            next_obs[p * config.state_width..(p + 1) * config.state_width].copy_from_slice(&obs[p]);
            if !old_obs.is_empty() {
                next_old_obs[p * config.old_state_width..(p + 1) * config.old_state_width]
                    .copy_from_slice(&old_obs[p]);
            }
            next_masks[p * config.mask_width..(p + 1) * config.mask_width]
                .copy_from_slice(&masks[p]);
        }

        let teams = game.player_teams();
        player_teams.copy_from_slice(&teams[..n]);
    } else {
        for p in 0..n {
            next_obs[p * config.state_width..(p + 1) * config.state_width]
                .copy_from_slice(&result.obs[p]);
            if !result.old_obs.is_empty() {
                next_old_obs[p * config.old_state_width..(p + 1) * config.old_state_width]
                    .copy_from_slice(&result.old_obs[p]);
            }
            next_masks[p * config.mask_width..(p + 1) * config.mask_width]
                .copy_from_slice(&result.action_masks[p]);
        }
    }

    *outcome = Some(flush);
}

/// One collection batch: the game instances, the flat obs buffers, and
/// the per-player trajectory buffers.
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
    /// Per-game flush outputs for the current decision, consumed in game
    /// order. Reused across decisions.
    outcomes: Vec<Option<GameOutcome>>,
    /// Pre-step observations in global player order, stored flat.
    next_obs: Vec<f32>,
    /// Pre-step observations from the old obs builders, parallel to `next_obs`.
    next_old_obs: Vec<f32>,
    next_masks: Vec<bool>,
    state_width: usize,
    old_state_width: usize,
    mask_width: usize,
    total_players: usize,
    player_trajs: Vec<PlayerTraj>,
    overflow_trajs: VecDeque<(PlayerTraj, Option<Vec<f32>>)>,
    retain_overflow_episodes: bool,
    metrics: Report,
    device: B::Device,
    max_episode_length: Option<usize>,
    /// When enabled, only complete trajectories are returned. Incomplete
    /// buffers from the previous policy snapshot are discarded at
    /// collection start.
    complete_trajectories: bool,
    /// Retained per-player trajectory capacity, adjusted after each
    /// collection.
    trajectory_baseline_steps: usize,
    episode_length_ema: Option<f64>,
    /// EMA of the second moment, used with `episode_length_ema` to estimate σ.
    episode_length_second_moment_ema: Option<f64>,

    /// Per-player team index (0 = Blue, 1 = Orange).
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
    /// Build `num_games` seeded game instances. Game `i` gets the seed
    /// `thread_num * (i + 1)`.
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
        FO: Fn() -> Box<dyn Obs<SI> + Send>,
    {
        struct FreshObs {
            obs: FullObs,
            old_obs: FullObs,
            masks: Vec<Vec<bool>>,
        }

        let mut games = Vec::with_capacity(num_games);
        let mut np = Vec::with_capacity(num_games);
        let mut player_offsets = Vec::with_capacity(num_games);
        let mut held_actions = Vec::with_capacity(num_games);
        let mut player_teams = Vec::new();
        let mut fresh = Vec::with_capacity(num_games);

        let mut player_offset = 0;
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * (i + 1)));
            let old_obs = make_old_obs.as_ref().map(|make| make());
            let mut game = GameInstance::new(env, old_obs, reward_sampling.clone());
            let (obs, old_obs, masks) = game.reset();
            let n = game.num_players();
            fresh.push(FreshObs {
                obs,
                old_obs,
                masks,
            });
            np.push(n);
            player_offsets.push(player_offset);
            player_offset += n;
            held_actions.push(vec![0; n]);
            player_teams.extend(game.player_teams());
            games.push(game);
        }

        let total_players = player_offset;
        let state_width = fresh[0].obs.first().map_or(0, Vec::len);
        let old_state_width = fresh[0].old_obs.first().map_or(0, Vec::len);
        let mask_width = fresh[0].masks.first().map_or(0, Vec::len);
        debug_assert!(state_width > 0);

        let mut next_obs = vec![0.0; total_players * state_width];
        let mut next_old_obs = vec![0.0; total_players * old_state_width];
        let mut next_masks = vec![false; total_players * mask_width];
        for (game_idx, fresh) in fresh.into_iter().enumerate() {
            let start = player_offsets[game_idx];
            let n = np[game_idx];
            for p in 0..n {
                let row = start + p;
                next_obs[row * state_width..(row + 1) * state_width].copy_from_slice(&fresh.obs[p]);
                if !fresh.old_obs.is_empty() {
                    next_old_obs[row * old_state_width..(row + 1) * old_state_width]
                        .copy_from_slice(&fresh.old_obs[p]);
                }
                next_masks[row * mask_width..(row + 1) * mask_width]
                    .copy_from_slice(&fresh.masks[p]);
            }
        }

        let baseline_steps = trajectory_capacity_hint.div_ceil(total_players.max(1));
        let player_trajs = (0..total_players)
            .map(|_| {
                PlayerTraj::with_width_and_baseline(
                    state_width,
                    old_state_width,
                    mask_width,
                    baseline_steps,
                )
            })
            .collect();

        Self {
            metrics: Report::default(),
            games,
            np,
            player_offsets,
            held_actions,
            action_delay_primed: vec![false; num_games],
            outcomes: (0..num_games).map(|_| None).collect(),
            next_obs,
            next_old_obs,
            next_masks,
            state_width,
            old_state_width,
            mask_width,
            total_players,
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

    /// The smoothed standard deviation of the completed episode length.
    fn episode_length_std_ema(&self) -> Option<f64> {
        let (Some(average), Some(second_moment)) = (
            self.episode_length_ema,
            self.episode_length_second_moment_ema,
        ) else {
            return None;
        };

        Some((second_moment - average * average).max(0.0).sqrt())
    }

    /// Collect complete episodes until the shared budget is exhausted.
    ///
    /// Only current-policy player trajectories enter the returned memory.
    /// With `self_play`, the players on `old_team` use the old model for
    /// inference. With `pool`, per-game stepping runs on that rayon pool;
    /// with `None`, it runs serially (behavior-identical). Inference
    /// dispatches before the delayed action window, so CPU physics
    /// overlaps the GPU batch.
    #[allow(clippy::too_many_arguments)]
    pub fn run_with_budget(
        &mut self,
        model: &Actic<B>,
        remaining_steps: &AtomicUsize,
        memory_capacity_hint: usize,
        rollout_budget: usize,
        self_play: Option<(&Actic<B>, usize)>,
        overbatching: bool,
        pool: Option<&ThreadPool>,
    ) -> (Memory, Report)
    where
        SS: Send,
        SI: Send,
        OBS: Send,
        ACT: Send,
        REW: Send,
        TERM: Send,
        TRUNC: Send,
    {
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
            for trajectory in &mut self.player_trajs {
                trajectory.clear();
            }
            self.overflow_trajs.clear();
        } else {
            for trajectory in &mut self.player_trajs {
                trajectory.shrink_to_baseline();
            }
        }

        let player_is_tracked: Vec<bool> = if let Some(ot) = old_team {
            self.player_teams.iter().map(|&t| t != ot).collect()
        } else {
            vec![true; self.player_teams.len()]
        };

        let mut memory = Memory::with_capacity(memory_capacity_hint);
        let mut completed_for_update = false;

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
                    model.submit_react_indexed_flat(
                        &self.next_obs,
                        self.state_width,
                        &self.next_masks,
                        self.mask_width,
                        &self.self_play_current_indices,
                        &self.device,
                    )
                });
                let old_pending = (!self.self_play_old_indices.is_empty()).then(|| {
                    old_model.submit_react_indexed_flat(
                        &self.next_obs,
                        self.state_width,
                        &self.next_masks,
                        self.mask_width,
                        &self.self_play_old_indices,
                        &self.device,
                    )
                });

                if action_delay > 0 {
                    total_env_step_time += self.begin_delayed_phase(pool);
                }

                let (current_actions, current_log_probs) = current_pending
                    .map(|pending| pending.wait())
                    .unwrap_or_default();
                let (old_actions, _) = old_pending
                    .map(|pending| pending.wait())
                    .unwrap_or_default();

                let player_count = self.total_players;
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
                model.react_flat(
                    &self.next_obs,
                    self.total_players,
                    self.state_width,
                    &self.next_masks,
                    self.mask_width,
                    &self.device,
                )
            } else {
                let pending = model.submit_react_flat(
                    &self.next_obs,
                    self.total_players,
                    self.state_width,
                    &self.next_masks,
                    self.mask_width,
                    &self.device,
                );

                total_env_step_time += self.begin_delayed_phase(pool);

                pending.wait()
            };

            total_infer_time += infer_start.elapsed().as_secs_f64();

            let env_start = Instant::now();

            self.step_games_phase(pool, &actions, &log_probs, &player_is_tracked);

            let (episode_steps, episode_squared_steps, episode_count) = self.claim_game_outcomes(
                &mut memory,
                remaining_steps,
                rollout_budget,
                overbatching,
                &mut completed_for_update,
            );
            completed_episode_steps += episode_steps;
            completed_episode_squared_steps += episode_squared_steps;
            completed_episode_count += episode_count;

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
        report[COLLECT_INFERENCE_TIME_KEY] = total_infer_time.into();
        report[COLLECT_ENV_STEP_TIME_KEY] = total_env_step_time.into();

        (memory, report)
    }

    /// Advance every game through the delayed portion of the decision
    /// interval. With `pool`, one job per game; with `None`, a serial
    /// loop. Returns the wall time.
    fn begin_delayed_phase(&mut self, pool: Option<&ThreadPool>) -> f64
    where
        SS: Send,
        SI: Send,
        OBS: Send,
        ACT: Send,
        REW: Send,
        TERM: Send,
        TRUNC: Send,
    {
        let delay_start = Instant::now();
        if let Some(pool) = pool {
            let games = &mut self.games;
            let held_actions = &self.held_actions;
            let action_delay_primed = &self.action_delay_primed;
            pool.scope(|scope| {
                for (game_idx, game) in games.iter_mut().enumerate() {
                    let held = &held_actions[game_idx];
                    let primed = action_delay_primed[game_idx];
                    scope.spawn(move |_| {
                        if primed {
                            game.begin_delayed_step(held);
                        } else {
                            game.begin_neutral_delayed_step();
                        }
                    });
                }
            });
        } else {
            for (game_idx, game) in self.games.iter_mut().enumerate() {
                if self.action_delay_primed[game_idx] {
                    game.begin_delayed_step(&self.held_actions[game_idx]);
                } else {
                    game.begin_neutral_delayed_step();
                }
            }
        }
        delay_start.elapsed().as_secs_f64()
    }

    /// Advance every game through the remainder of the decision interval.
    /// With `pool`, one job per game; with `None`, the same work serially.
    /// Pool jobs never call `wait()` or touch the GPU.
    fn step_games_phase(
        &mut self,
        pool: Option<&ThreadPool>,
        actions: &[usize],
        log_probs: &[f32],
        player_is_tracked: &[bool],
    ) where
        SS: Send,
        SI: Send,
        OBS: Send,
        ACT: Send,
        REW: Send,
        TERM: Send,
        TRUNC: Send,
    {
        let action_delay = ACT::get_action_delay();
        self.outcomes.clear();
        self.outcomes.resize_with(self.games.len(), || None);

        if let Some(pool) = pool {
            let games = &mut self.games;
            let np = &self.np;
            let player_offsets = &self.player_offsets;
            let held_actions = &mut self.held_actions;
            let action_delay_primed = &mut self.action_delay_primed;
            let outcomes = &mut self.outcomes;
            let player_teams = &mut self.player_teams;
            let state_width = self.state_width;
            let old_state_width = self.old_state_width;
            let mask_width = self.mask_width;
            let max_episode_length = self.max_episode_length;
            let next_obs = &mut self.next_obs;
            let next_old_obs = &mut self.next_old_obs;
            let next_masks = &mut self.next_masks;
            let player_trajs = &mut self.player_trajs;

            pool.scope(|scope| {
                let mut obs = &mut next_obs[..];
                let mut old_obs = &mut next_old_obs[..];
                let mut masks = &mut next_masks[..];
                let mut trajs = &mut player_trajs[..];
                let mut teams = &mut player_teams[..];

                for (game_idx, (((game, held), primed), outcome)) in games
                    .iter_mut()
                    .zip(held_actions.iter_mut())
                    .zip(action_delay_primed.iter_mut())
                    .zip(outcomes.iter_mut())
                    .enumerate()
                {
                    let player_start = player_offsets[game_idx];
                    let n = np[game_idx];

                    let (game_obs, obs_rest) = obs.split_at_mut(n * state_width);
                    obs = obs_rest;
                    let (game_old_obs, old_obs_rest) = old_obs.split_at_mut(n * old_state_width);
                    old_obs = old_obs_rest;
                    let (game_masks, masks_rest) = masks.split_at_mut(n * mask_width);
                    masks = masks_rest;
                    let (game_trajs, trajs_rest) = trajs.split_at_mut(n);
                    trajs = trajs_rest;
                    let (game_teams, teams_rest) = teams.split_at_mut(n);
                    teams = teams_rest;

                    let config = StepConfig {
                        state_width,
                        old_state_width,
                        mask_width,
                        max_episode_length,
                        action_delay,
                        actions: &actions[player_start..player_start + n],
                        log_probs: &log_probs[player_start..player_start + n],
                        tracked: &player_is_tracked[player_start..player_start + n],
                    };

                    scope.spawn(move |_| {
                        step_game(
                            StepJob {
                                game,
                                trajs: game_trajs,
                                next_obs: game_obs,
                                next_old_obs: game_old_obs,
                                next_masks: game_masks,
                                held_actions: held,
                                action_delay_primed: primed,
                                player_teams: game_teams,
                                outcome,
                            },
                            &config,
                        );
                    });
                }
            });
        } else {
            for game_idx in 0..self.games.len() {
                let player_start = self.player_offsets[game_idx];
                let n = self.np[game_idx];
                let obs_start = player_start * self.state_width;
                let old_start = player_start * self.old_state_width;
                let mask_start = player_start * self.mask_width;
                step_game(
                    StepJob {
                        game: &mut self.games[game_idx],
                        trajs: &mut self.player_trajs[player_start..player_start + n],
                        next_obs: &mut self.next_obs[obs_start..obs_start + n * self.state_width],
                        next_old_obs: &mut self.next_old_obs
                            [old_start..old_start + n * self.old_state_width],
                        next_masks: &mut self.next_masks
                            [mask_start..mask_start + n * self.mask_width],
                        held_actions: &mut self.held_actions[game_idx],
                        action_delay_primed: &mut self.action_delay_primed[game_idx],
                        player_teams: &mut self.player_teams[player_start..player_start + n],
                        outcome: &mut self.outcomes[game_idx],
                    },
                    &StepConfig {
                        state_width: self.state_width,
                        old_state_width: self.old_state_width,
                        mask_width: self.mask_width,
                        max_episode_length: self.max_episode_length,
                        action_delay,
                        actions: &actions[player_start..player_start + n],
                        log_probs: &log_probs[player_start..player_start + n],
                        tracked: &player_is_tracked[player_start..player_start + n],
                    },
                );
            }
        }
    }

    /// Consume the flush outputs in game order and claim the budget for
    /// each flushed trajectory. It is the only place that mutates
    /// `memory`, the shared budget, and `overflow_trajs`.
    fn claim_game_outcomes(
        &mut self,
        memory: &mut Memory,
        remaining_steps: &AtomicUsize,
        rollout_budget: usize,
        overbatching: bool,
        completed_for_update: &mut bool,
    ) -> (usize, f64, usize) {
        let mut completed_episode_steps = 0_usize;
        let mut completed_episode_squared_steps = 0.0_f64;
        let mut completed_episode_count = 0_usize;

        for outcome in &mut self.outcomes {
            let Some(outcome) = outcome.take() else {
                continue;
            };
            for flush in outcome.flushes {
                let FlushedTrajectory {
                    trajectory,
                    len,
                    next_state,
                } = flush;
                if self.complete_trajectories {
                    let claimed =
                        claim_complete_steps(remaining_steps, len, !*completed_for_update);
                    if claimed == len {
                        *completed_for_update = true;
                        push_claimed_trajectory(
                            memory,
                            Some(ClaimedTrajectory {
                                trajectory,
                                len,
                                next_state,
                            }),
                        );
                    }
                } else if overbatching {
                    let claimed = claim_overbatch_steps(remaining_steps, len, rollout_budget);
                    let partition = partition_claimed_trajectory(
                        trajectory,
                        next_state,
                        claimed,
                        self.retain_overflow_episodes,
                    );
                    push_claimed_trajectory(memory, partition.claimed);
                    if let Some(overflow) = partition.overflow {
                        self.overflow_trajs.push_back(overflow);
                    }
                } else {
                    let claimed = claim_available_steps(remaining_steps, len);
                    let partition = partition_claimed_trajectory(
                        trajectory,
                        next_state,
                        claimed,
                        self.retain_overflow_episodes,
                    );
                    push_claimed_trajectory(memory, partition.claimed);
                    if let Some(overflow) = partition.overflow {
                        self.overflow_trajs.push_back(overflow);
                    }
                }
            }
            completed_episode_steps += outcome.episode_steps;
            completed_episode_squared_steps += outcome.episode_squared_steps;
            completed_episode_count += outcome.episode_count;
        }

        (
            completed_episode_steps,
            completed_episode_squared_steps,
            completed_episode_count,
        )
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

#[cfg(test)]
mod phase3_tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rlgym::GameState;
    use rlgym::rocketsim::{Arena, ArenaConfig, CarBodyConfig, GameMode, Team};
    use rlgymppo_utils::actions::DefaultAction;
    use rlgymppo_utils::obs::DefaultObs;
    use rlgymppo_utils::rewards::FaceBallReward;
    use rlgymppo_utils::shared_info::SharedInfoRng;
    use rlgymppo_utils::state_setters::RandomState;
    use rlgymppo_utils::terminal::{NoTouchCondition, OnGoalCondition};

    use super::*;
    use crate::environment::sim::StepResult;

    struct TestSharedInfo {
        rng: SmallRng,
        report: Report,
    }

    impl TestSharedInfo {
        fn seeded(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
                report: Report::default(),
            }
        }
    }

    impl SharedInfoProvider for TestSharedInfo {
        fn reset(&mut self, _initial_state: &GameState) {}
        fn update(&mut self, _game_state: &GameState) {}
    }

    impl SharedInfoRng for TestSharedInfo {
        type Rng = SmallRng;

        fn rng(&mut self) -> &mut Self::Rng {
            &mut self.rng
        }
    }

    impl SharedInfoReport for TestSharedInfo {
        fn report(&mut self) -> &mut Report {
            &mut self.report
        }
    }

    fn init_rocketsim() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let candidates = [
            cwd.join("collision_meshes"),
            cwd.join("../collision_meshes"),
            manifest.join("../collision_meshes"),
        ];
        let Some(root) = candidates
            .iter()
            .find(|candidate| candidate.join("soccar").is_dir())
        else {
            panic!(
                "could not locate RocketSim collision meshes; probed {candidates:?} (cwd: {})",
                cwd.display()
            );
        };
        crate::rocketsim::init(root, true).expect("failed to initialize RocketSim");
    }

    #[test]
    fn regression_phase3_budget_claim_order_exact_overbatch_complete() {
        let remaining = AtomicUsize::new(100);
        let claims: Vec<usize> = [60, 60, 30]
            .into_iter()
            .map(|len| claim_available_steps(&remaining, len))
            .collect();
        assert_eq!(claims, [60, 40, 0]);
        assert_eq!(claims.iter().sum::<usize>(), 100);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);

        for order in [[60, 60, 30], [60, 30, 60], [30, 60, 60]] {
            let remaining = AtomicUsize::new(100);
            let total: usize = order
                .into_iter()
                .map(|len| claim_available_steps(&remaining, len))
                .sum();
            assert_eq!(total, 100);
            assert_eq!(remaining.load(Ordering::Relaxed), 0);
        }

        let remaining = AtomicUsize::new(100);
        let claims: Vec<usize> = [60, 60, 30]
            .into_iter()
            .map(|len| claim_overbatch_steps(&remaining, len, 100))
            .collect();
        assert_eq!(claims, [60, 60, 0]);
        let total: usize = claims.iter().sum();
        assert_eq!(total, 120);
        assert!(total < 2 * 100);
        assert_eq!(remaining.load(Ordering::Relaxed), 0);

        let remaining = AtomicUsize::new(100);
        let mut completed_for_update = false;
        let mut claims = Vec::new();
        for len in [60, 60, 30] {
            let claimed = claim_complete_steps(&remaining, len, !completed_for_update);
            if claimed == len {
                completed_for_update = true;
            }
            claims.push(claimed);
        }
        assert_eq!(claims, [60, 60, 0]);
        assert_eq!(remaining.load(Ordering::Acquire), 0);

        let remaining = AtomicUsize::new(0);
        assert_eq!(claim_complete_steps(&remaining, 30, true), 30);
        assert_eq!(remaining.load(Ordering::Acquire), 0);
    }

    #[test]
    fn regression_phase3_panicking_job_propagates_from_scope() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let ok_job_ran = AtomicBool::new(false);
        let result = catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.spawn(|_| {
                    ok_job_ran.store(true, Ordering::Relaxed);
                });
                scope.spawn(|_| {
                    panic!("job panicked");
                });
            });
        }));

        assert!(ok_job_ran.load(Ordering::Relaxed));
        let payload = result.expect_err("the job panic must propagate out of pool.scope");
        let message = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("<non-string panic payload>");
        assert!(message.contains("job panicked"));
    }

    type T4StateSetter = RandomState<true, false, true>;
    type T4Action = DefaultAction<2, 8, 3>;
    type T4Env = Env<
        T4StateSetter,
        DefaultObs<1>,
        T4Action,
        FaceBallReward,
        OnGoalCondition,
        NoTouchCondition<24>,
        TestSharedInfo,
    >;

    fn make_env_t4(game_id: Option<usize>) -> T4Env {
        let game_id = game_id.unwrap_or(0);
        let mut config = ArenaConfig::new(GameMode::Soccar);
        config.rng_seed = Some(game_id as u64 * 7919 + 13);
        let mut arena = Arena::new_with_config(config);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
        Env::new(
            arena,
            RandomState::<true, false, true>,
            DefaultObs::<1>,
            T4Action::new(),
            FaceBallReward,
            OnGoalCondition,
            NoTouchCondition::<24>::default(),
            TestSharedInfo::seeded(game_id as u64 * 104729 + 17),
        )
    }

    #[test]
    fn regression_phase3_scoped_physics_matches_serial() {
        init_rocketsim();

        let make_game = |game_id: usize| {
            GameInstance::new(
                make_env_t4(Some(game_id)),
                None,
                RewardSamplingConfig::default(),
            )
        };

        let mut serial_games = [make_game(1), make_game(2)];
        let mut scoped_games = [make_game(1), make_game(2)];
        for game in serial_games.iter_mut().chain(scoped_games.iter_mut()) {
            game.reset();
        }

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .thread_name(|_| "parity".into())
            .build()
            .unwrap();

        let mut serial_held = [vec![0usize; 2], vec![0usize; 2]];
        let mut scoped_held = [vec![0usize; 2], vec![0usize; 2]];
        let mut serial_primed = [false, false];
        let mut scoped_primed = [false, false];
        let mut last_serial = Vec::new();

        for decision in 0..3 {
            let actions: Vec<usize> = (0..4).map(|i| decision * 4 + i).collect();

            for gi in 0..serial_games.len() {
                if serial_primed[gi] {
                    serial_games[gi].begin_delayed_step(&serial_held[gi]);
                } else {
                    serial_games[gi].begin_neutral_delayed_step();
                }
            }
            let serial_results: Vec<StepResult> = (0..serial_games.len())
                .map(|gi| {
                    let game_actions = &actions[gi * 2..gi * 2 + 2];
                    let result = serial_games[gi].finish_delayed_step(game_actions);
                    serial_held[gi].clear();
                    serial_held[gi].extend_from_slice(game_actions);
                    serial_primed[gi] = true;
                    result
                })
                .collect();

            let games = &mut scoped_games;
            let held = &scoped_held;
            let primed = &scoped_primed;
            pool.scope(|scope| {
                for (gi, game) in games.iter_mut().enumerate() {
                    let held = &held[gi];
                    let primed = primed[gi];
                    scope.spawn(move |_| {
                        if primed {
                            game.begin_delayed_step(held);
                        } else {
                            game.begin_neutral_delayed_step();
                        }
                    });
                }
            });

            let mut scoped_results: Vec<Option<StepResult>> = vec![None; scoped_games.len()];
            let games = &mut scoped_games;
            pool.scope(|scope| {
                for (gi, (game, slot)) in
                    games.iter_mut().zip(scoped_results.iter_mut()).enumerate()
                {
                    let game_actions = &actions[gi * 2..gi * 2 + 2];
                    scope.spawn(move |_| {
                        *slot = Some(game.finish_delayed_step(game_actions));
                    });
                }
            });

            for (gi, slot) in scoped_results.iter_mut().enumerate() {
                scoped_held[gi].clear();
                scoped_held[gi].extend_from_slice(&actions[gi * 2..gi * 2 + 2]);
                scoped_primed[gi] = true;
                let scoped = slot.take().expect("scoped stepping filled every slot");
                let serial = &serial_results[gi];
                assert_eq!(
                    serial.obs, scoped.obs,
                    "obs diverged at decision {decision}, game {gi}"
                );
                assert_eq!(serial.old_obs, scoped.old_obs);
                assert_eq!(serial.action_masks, scoped.action_masks);
                assert_eq!(serial.rewards, scoped.rewards);
                assert_eq!(serial.is_terminal, scoped.is_terminal);
                assert_eq!(serial.truncated, scoped.truncated);
            }
            last_serial = serial_results;
        }

        assert_ne!(last_serial[0].obs, last_serial[1].obs);
    }

    #[cfg(feature = "flex")]
    mod flex_gated {
        use std::iter::repeat_n;

        use burn::backend::Flex;

        use super::*;

        type StateSetter = RandomState<true, false, true>;
        type Action = DefaultAction<1, 8, 0>;
        type TruncateCond = NoTouchCondition<120>;
        type EnvType = Env<
            StateSetter,
            DefaultObs<1>,
            Action,
            FaceBallReward,
            OnGoalCondition,
            TruncateCond,
            TestSharedInfo,
        >;
        type BatchSimType = BatchSim<
            Flex,
            StateSetter,
            DefaultObs<1>,
            Action,
            FaceBallReward,
            OnGoalCondition,
            TruncateCond,
            TestSharedInfo,
        >;

        fn make_env(game_id: Option<usize>) -> EnvType {
            let game_id = game_id.unwrap_or(0);
            let mut config = ArenaConfig::new(GameMode::Soccar);
            config.rng_seed = Some(game_id as u64 * 7919 + 13);
            let mut arena = Arena::new_with_config(config);
            arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
            Env::new(
                arena,
                RandomState::<true, false, true>,
                DefaultObs::<1>,
                Action::new(),
                FaceBallReward,
                OnGoalCondition,
                TruncateCond::default(),
                TestSharedInfo::seeded(game_id as u64 * 104729 + 17),
            )
        }

        fn make_batch_sim() -> BatchSimType {
            BatchSimType::new(
                make_env,
                None::<fn() -> Box<dyn Obs<TestSharedInfo> + Send>>,
                1,
                3,
                Default::default(),
                RewardSamplingConfig::default(),
                100,
                None,
                true,
                false,
            )
        }

        fn push_marked_rows(traj: &mut PlayerTraj, count: usize, marker_base: usize) {
            for i in 0..count {
                let marker = marker_base + i;
                traj.states
                    .extend(repeat_n(marker as f32, traj.state_width));
                traj.actions.push(marker);
                traj.log_probs.push(0.5);
                traj.rewards.push(1.0);
                traj.terminals.push(TerminalState::None);
                traj.action_masks
                    .extend(repeat_n(true, traj.action_mask_width));
            }
        }

        fn push_terminal_row(traj: &mut PlayerTraj, action: usize, terminal: TerminalState) {
            traj.states
                .extend(repeat_n(action as f32, traj.state_width));
            traj.actions.push(action);
            traj.log_probs.push(0.5);
            traj.rewards.push(1.0);
            traj.terminals.push(terminal);
            traj.action_masks
                .extend(repeat_n(true, traj.action_mask_width));
        }

        fn flushed_episode(
            traj: &mut PlayerTraj,
            marker_base: usize,
            terminal: TerminalState,
        ) -> GameOutcome {
            push_marked_rows(traj, 59, marker_base);
            push_terminal_row(traj, marker_base + 59, terminal);
            GameOutcome {
                flushes: vec![FlushedTrajectory {
                    trajectory: traj.take(),
                    len: 60,
                    next_state: None,
                }],
                episode_steps: 60,
                episode_squared_steps: 3600.0,
                episode_count: 1,
            }
        }

        #[test]
        fn regression_phase3_step_phase_records_and_writes_flat() {
            init_rocketsim();

            let mut serial_sim = make_batch_sim();
            let mut scoped_sim = make_batch_sim();

            let pre_obs = serial_sim.next_obs.clone();
            let pre_masks = serial_sim.next_masks.clone();
            assert_eq!(serial_sim.next_obs.len(), 3 * 53);
            assert_eq!(serial_sim.next_masks.len(), 3 * 90);

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .thread_name(|_| "phase-a-parity".into())
                .build()
                .unwrap();

            let actions = [0usize, 1, 2];
            let log_probs = [0.5, 0.5, 0.5];
            let tracked = [true, true, true];

            serial_sim.step_games_phase(None, &actions, &log_probs, &tracked);
            scoped_sim.step_games_phase(Some(&pool), &actions, &log_probs, &tracked);

            assert_eq!(serial_sim.next_obs, scoped_sim.next_obs);
            assert_eq!(serial_sim.next_masks, scoped_sim.next_masks);
            for i in 0..3 {
                let serial = &serial_sim.player_trajs[i];
                let scoped = &scoped_sim.player_trajs[i];
                assert_eq!(serial.states, scoped.states);
                assert_eq!(serial.actions, scoped.actions);
                assert_eq!(serial.log_probs, scoped.log_probs);
                assert_eq!(serial.rewards, scoped.rewards);
                assert_eq!(serial.terminals, scoped.terminals);
            }

            for i in 0..3 {
                let traj = &serial_sim.player_trajs[i];
                assert_eq!(traj.states, &pre_obs[i * 53..(i + 1) * 53]);
                assert_eq!(traj.action_masks, &pre_masks[i * 90..(i + 1) * 90]);
                assert_eq!(traj.actions, [i]);
                assert_eq!(traj.log_probs, [0.5]);
                assert_eq!(traj.terminals, [TerminalState::None]);
                assert_eq!(serial_sim.held_actions[i], [i]);
                assert!(serial_sim.action_delay_primed[i]);
            }

            assert!(serial_sim.outcomes.iter().all(|outcome| {
                outcome
                    .as_ref()
                    .is_some_and(|outcome| outcome.flushes.is_empty())
            }));
            assert_ne!(&serial_sim.next_obs[0..53], &serial_sim.next_obs[53..106]);
            assert_ne!(
                &serial_sim.next_obs[53..106],
                &serial_sim.next_obs[106..159]
            );
        }

        #[test]
        fn regression_phase3_terminal_flush_claims_in_game_order() {
            init_rocketsim();
            let mut sim = make_batch_sim();

            sim.outcomes[0] = Some(flushed_episode(
                &mut sim.player_trajs[0],
                0,
                TerminalState::Normal,
            ));
            sim.outcomes[1] = Some(flushed_episode(
                &mut sim.player_trajs[1],
                100,
                TerminalState::Normal,
            ));
            sim.outcomes[2] = None;

            let mut memory = Memory::with_capacity(16);
            let remaining_steps = AtomicUsize::new(100);
            let mut completed_for_update = false;
            let (steps, squared_steps, count) = sim.claim_game_outcomes(
                &mut memory,
                &remaining_steps,
                100,
                false,
                &mut completed_for_update,
            );

            assert_eq!(memory.len(), 100);
            assert_eq!(
                &memory.actions()[..60],
                &(0..60).collect::<Vec<usize>>()[..]
            );
            assert_eq!(memory.actions()[59], 59);
            assert_eq!(memory.actions()[60], 100);
            assert_eq!(
                &memory.actions()[60..],
                &(100..140).collect::<Vec<usize>>()[..]
            );
            assert_eq!(
                &memory.states()[60 * 53..61 * 53],
                &[100.0; 53],
                "game 1's first claimed state row"
            );

            assert_eq!(memory.terminals()[59], TerminalState::Normal);
            assert_eq!(memory.terminals()[99], TerminalState::Truncated);
            assert_eq!(memory.trunc_next_states(), &[140.0; 53]);
            assert!(memory.validate().is_ok());

            assert_eq!(
                sim.overflow_trajs[0].0.actions,
                (140..160).collect::<Vec<usize>>()
            );

            assert_eq!(remaining_steps.load(Ordering::Relaxed), 0);
            assert_eq!((steps, count), (120, 2));
            assert_eq!(squared_steps, 7200.0);

            assert!(sim.outcomes.iter().all(Option::is_none));
        }
    }
}
