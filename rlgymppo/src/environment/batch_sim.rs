use std::time::Instant;

use burn::prelude::*;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::{
    agent::model::Actic,
    base::{Memory, TerminalState},
    utils::{Report, shared_info::SharedInfoReport},
};

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
    /// Per-game player counts (cached).
    np: Vec<usize>,
    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    /// Per-player trajectory buffers, indexed globally across all games.
    player_trajs: Vec<PlayerTraj>,
    metrics: Report,
    device: B::Device,
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
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut games = Vec::with_capacity(num_games);
        let mut np = Vec::with_capacity(num_games);

        let mut next_obs = Vec::with_capacity(num_games);
        let mut next_masks = Vec::with_capacity(num_games);
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * (i + 1)));
            let mut game = GameInstance::new(env, reward_sampling.clone());
            let (obs, masks) = game.reset();
            let n = game.num_players();
            next_obs.extend(obs);
            next_masks.extend(masks);
            np.push(n);
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
        }
    }

    /// Collect complete episodes until at least `min_steps` steps have been
    /// accumulated.  Incomplete episodes stay in `player_trajs` and carry over
    /// to the next call — no data is discarded.
    pub fn run(&mut self, model: &Actic<B>, min_steps: usize) -> (Memory, Report) {
        let mut memory = Memory::with_capacity(min_steps * 2);
        let mut collected_steps: usize = 0;

        let mut total_infer_time = 0.0_f64;
        let mut total_env_step_time = 0.0_f64;

        while collected_steps < min_steps {
            let infer_start = Instant::now();
            let (actions, log_probs) = model.react(&self.next_obs, &self.next_masks, &self.device);
            total_infer_time += infer_start.elapsed().as_secs_f64();

            let env_start = Instant::now();

            // Record pre‑step observations into per‑player trajectories.
            for (i, (obs, mask)) in self.next_obs.iter().zip(self.next_masks.iter()).enumerate() {
                self.player_trajs[i].states.push(obs.clone());
                self.player_trajs[i].action_masks.push(mask.clone());
            }

            self.next_obs.clear();
            self.next_masks.clear();

            // Step games in reverse order.
            let mut action_offset = actions.len();
            for (game_idx, game) in self.games.iter_mut().enumerate().rev() {
                let n = self.np[game_idx];
                action_offset -= n;

                let result = game.step(&actions[action_offset..action_offset + n]);

                let terminal_type = if result.truncated {
                    TerminalState::Truncated
                } else if result.is_terminal {
                    TerminalState::Normal
                } else {
                    TerminalState::None
                };

                let player_start: usize = self.np[..game_idx].iter().sum();
                for p in 0..n {
                    let ti = player_start + p;
                    self.player_trajs[ti].rewards.push(result.rewards[p]);
                    self.player_trajs[ti]
                        .actions
                        .push(actions[action_offset + p]);
                    self.player_trajs[ti]
                        .log_probs
                        .push(log_probs[action_offset + p]);
                    self.player_trajs[ti].terminals.push(terminal_type);
                }

                if result.is_terminal || result.truncated {
                    // Episode ended — flush this game's trajectories.
                    for p in 0..n {
                        let ti = player_start + p;
                        collected_steps += self.player_trajs[ti].states.len();

                        let trunc_next = result.truncated.then(|| result.obs[p].clone());

                        memory.push_player(
                            std::mem::take(&mut self.player_trajs[ti].states),
                            std::mem::take(&mut self.player_trajs[ti].actions),
                            std::mem::take(&mut self.player_trajs[ti].log_probs),
                            std::mem::take(&mut self.player_trajs[ti].rewards),
                            std::mem::take(&mut self.player_trajs[ti].terminals),
                            std::mem::take(&mut self.player_trajs[ti].action_masks),
                            trunc_next,
                        );
                    }

                    // Reset the game for the next episode.
                    let (obs, masks) = game.reset();
                    self.next_obs.extend(obs);
                    self.next_masks.extend(masks);
                } else {
                    self.next_obs.extend(result.obs);
                    self.next_masks.extend(result.action_masks);
                }
            }

            total_env_step_time += env_start.elapsed().as_secs_f64();

            self.next_obs.reverse();
            self.next_masks.reverse();
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
