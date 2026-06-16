use burn::prelude::*;
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim::ArenaState,
};

use super::sim::GameInstance;
use crate::{
    agent::model::Net,
    base::{Memory, TERMINAL_NONE, TERMINAL_NORMAL, TERMINAL_TRUNCATED},
    utils::Report,
};

pub struct BatchSim<B: Backend, C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &ArenaState),
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    games: Vec<GameInstance<C, SS, OBS, ACT, REW, TERM, TRUNC, SI>>,
    next_obs: Vec<Vec<f32>>,
    next_masks: Vec<Vec<bool>>,
    metrics: Report,
    device: B::Device,
}

impl<B, C, SS, OBS, ACT, REW, TERM, TRUNC, SI> BatchSim<B, C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend,
    C: Fn(&mut Report, &mut SI, &ArenaState) + Clone,
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new<F>(
        create_env_fn: F,
        step_callback: C,
        thread_num: usize,
        num_games: usize,
        device: B::Device,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut games = Vec::with_capacity(num_games);

        let mut next_obs = Vec::with_capacity(num_games);
        let mut next_masks = Vec::with_capacity(num_games);
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * i));
            let mut game = GameInstance::new(env, step_callback.clone());
            let (obs, masks) = game.reset();
            next_obs.extend(obs);
            next_masks.extend(masks);
            games.push(game);
        }

        Self {
            metrics: Report::default(),
            next_obs,
            next_masks,
            games,
            device,
        }
    }

    pub fn num_players(&self) -> usize {
        self.games.iter().map(|game| game.num_players()).sum()
    }

    pub fn run(&mut self, model: &Net<B>, num_steps: usize) -> (Memory, Report) {
        let mut memory = Memory::with_capacity(num_steps);

        while memory.len() < num_steps {
            let (actions, log_probs) = model.react(&self.next_obs, &self.next_masks, &self.device);

            let mut start_idx = self.next_obs.len();
            let masks = self.next_masks.drain(..).collect();
            memory.push_batch_part_1(self.next_obs.drain(..), masks, log_probs);

            for game in self.games.iter_mut().rev() {
                let end_idx = start_idx;
                start_idx -= game.num_players();
                let result = game.step(&actions[start_idx..end_idx]);

                // Encode the terminal type: NONE / NORMAL / TRUNCATED (like C++ TerminalType).
                let terminal_type = if result.truncated {
                    TERMINAL_TRUNCATED
                } else if result.is_terminal {
                    TERMINAL_NORMAL
                } else {
                    TERMINAL_NONE
                };

                // Save the next observation before reset so the critic can bootstrap
                // on truncated episodes (C++ GAE: `truncValPreds`).
                let trunc_obs = result.truncated.then(|| result.obs.clone());

                memory.push_batch_part_2(result.rewards, terminal_type);

                if result.is_terminal || result.truncated {
                    if let Some(ref obs) = trunc_obs {
                        memory.push_trunc_next_states(obs);
                    }
                    let (obs, masks) = game.reset();
                    self.next_obs.extend(obs);
                    self.next_masks.extend(masks);
                } else {
                    self.next_obs.extend(result.obs);
                    self.next_masks.extend(result.action_masks);
                }
            }

            memory.push_batch_part_3(actions);
            self.next_obs.reverse();
            self.next_masks.reverse();
        }

        (memory, self.get_metrics())
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
