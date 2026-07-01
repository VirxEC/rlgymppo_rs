use std::time::Instant;

use burn::prelude::*;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::agent::model::Actic;
use crate::base::{Memory, TerminalState};
use crate::utils::Report;
use crate::utils::shared_info::SharedInfoReport;

/// Per-player trajectory buffer that persists across [`run`] calls
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
    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    player_trajs: Vec<PlayerTraj>,
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
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut games = Vec::with_capacity(num_games);
        let mut np = Vec::with_capacity(num_games);
        let mut next_obs = Vec::with_capacity(num_games);
        let mut next_masks = Vec::with_capacity(num_games);
        let mut player_teams = Vec::new();

        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * (i + 1)));
            let mut game = GameInstance::new(env, reward_sampling.clone());
            let (obs, masks) = game.reset();
            let n = game.num_players();
            next_obs.extend(obs);
            next_masks.extend(masks);
            np.push(n);
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
            player_trajs,
            device,
            player_teams,
            max_episode_length,
        }
    }

    /// Collect complete episodes until at least `min_steps` steps have been
    /// accumulated.
    ///
    /// When `self_play` is `Some((old_model, old_team))`, the players on
    /// `old_team` (0 = Blue, 1 = Orange) use `old_model` for inference
    /// while the rest use the current `model`.  Only trajectories from
    /// current-policy players are recorded in the returned [`Memory`].
    pub fn run(
        &mut self,
        model: &Actic<B>,
        min_steps: usize,
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

        let mut memory = Memory::with_capacity(min_steps * 2);
        let mut collected_steps: usize = 0;

        let mut total_infer_time = 0.0_f64;
        let mut total_env_step_time = 0.0_f64;

        while collected_steps < min_steps {
            let infer_start = Instant::now();

            let (actions, log_probs) = if let (Some(old_model), Some(_ot)) = (old_model, old_team) {
                // ── Self‑play: split observations by model group ──
                let (no, nm): (Vec<_>, Vec<_>) = self
                    .next_obs
                    .iter()
                    .zip(&self.next_masks)
                    .zip(&player_is_tracked)
                    .filter(|(_, tracked)| **tracked)
                    .map(|((o, m), _)| (o.clone(), m.clone()))
                    .unzip();
                let (oo, om): (Vec<_>, Vec<_>) = self
                    .next_obs
                    .iter()
                    .zip(&self.next_masks)
                    .zip(&player_is_tracked)
                    .filter(|(_, tracked)| !**tracked)
                    .map(|((o, m), _)| (o.clone(), m.clone()))
                    .unzip();

                // Infer actions for each group separately.
                let (mut na, mut nlp) = if no.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    model.react(&no, &nm, &self.device)
                };
                let (oa, _) = if oo.is_empty() {
                    (Vec::new(), Vec::new())
                } else {
                    old_model.react(&oo, &om, &self.device)
                };

                // Interleave back into the original player order.
                let pc = self.next_obs.len();
                let mut actions = vec![0usize; pc];
                let mut log_probs = vec![0.0f32; pc];
                let (mut ni, mut oi) = (0, 0);
                for p in 0..pc {
                    if player_is_tracked[p] {
                        actions[p] = na[ni];
                        log_probs[p] = nlp[ni];
                        ni += 1;
                    } else {
                        actions[p] = oa[oi];
                        oi += 1;
                    }
                }
                // Clean up moved-out vectors.
                na.clear();
                nlp.clear();

                (actions, log_probs)
            } else {
                // ── Normal inference ─────────────────────────────
                model.react(&self.next_obs, &self.next_masks, &self.device)
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

                let result = game.step(&actions[action_offset..action_offset + n]);

                let mut terminal_type = if result.truncated {
                    TerminalState::Truncated
                } else if result.is_terminal {
                    TerminalState::Normal
                } else {
                    TerminalState::None
                };

                let player_start: usize = self.np[..game_idx].iter().sum();

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
                            collected_steps += self.player_trajs[ti].states.len();
                            let trunc_next = (terminal_type == TerminalState::Truncated)
                                .then(|| result.obs[p].clone());
                            memory.push_player(
                                std::mem::take(&mut self.player_trajs[ti].states),
                                std::mem::take(&mut self.player_trajs[ti].actions),
                                std::mem::take(&mut self.player_trajs[ti].log_probs),
                                std::mem::take(&mut self.player_trajs[ti].rewards),
                                std::mem::take(&mut self.player_trajs[ti].terminals),
                                std::mem::take(&mut self.player_trajs[ti].action_masks),
                                trunc_next,
                            );
                        } else {
                            // Discard untracked player's buffers.
                            let _ = std::mem::take(&mut self.player_trajs[ti]);
                        }
                    }

                    if result.is_terminal || result.truncated {
                        // Env-level terminal — reset the game.
                        let (obs, masks) = game.reset();
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
