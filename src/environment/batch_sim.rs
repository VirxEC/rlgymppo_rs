use super::sim::GameInstance;
use crate::{agent::model::Net, base::Memory, utils::Report};
use burn::prelude::*;
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};

pub struct BatchSim<B: Backend, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &GameStateA),
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    games: Vec<GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>>,
    next_obs: Vec<Vec<f32>>,
    metrics: Report,
    device: B::Device,
}

impl<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    BatchSim<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend,
    C: Fn(&mut Report, &mut SI, &GameStateA) + Clone,
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
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
        F: Fn(Option<usize>) -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        let mut games = Vec::with_capacity(num_games);

        let mut next_obs = Vec::with_capacity(num_games);
        for i in 0..num_games {
            let env = create_env_fn(Some(thread_num * i));
            let mut game = GameInstance::new(env, step_callback.clone());
            next_obs.extend(game.reset());
            games.push(game);
        }

        Self {
            metrics: Report::default(),
            next_obs,
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
            let (actions, log_probs) = model.react(&self.next_obs, &self.device);

            let mut start_idx = self.next_obs.len();
            memory.push_batch_part_1(self.next_obs.drain(..), log_probs);

            for game in self.games.iter_mut().rev() {
                let end_idx = start_idx;
                start_idx -= game.num_players();
                let result = game.step(&actions[start_idx..end_idx]);

                memory.push_batch_part_2(result.rewards, result.is_terminal, result.truncated);
                let obs = if result.is_terminal || result.truncated {
                    game.reset()
                } else {
                    result.obs
                };

                self.next_obs.extend(obs);
            }

            memory.push_batch_part_3(actions);
            self.next_obs.reverse();
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
