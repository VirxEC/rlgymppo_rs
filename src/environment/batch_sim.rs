use super::sim::GameInstance;
use crate::{agent::model::Net, base::Memory, utils::Report};
use burn::prelude::*;
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::mem;

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
    last_obs: Vec<Vec<f32>>,
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
    pub fn new<F>(create_env_fn: F, step_callback: C, num_games: usize, device: B::Device) -> Self
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        assert_ne!(num_games, 0, "num_games must be greater than 0");

        let mut games = Vec::with_capacity(num_games);

        let mut last_obs = Vec::with_capacity(num_games * 2);
        for _ in 0..num_games {
            let env = create_env_fn();
            let mut game = GameInstance::new(env, step_callback.clone());
            last_obs.extend(game.reset());
            games.push(game);
        }

        Self {
            next_obs: Vec::with_capacity(last_obs.len()),
            metrics: Report::default(),
            last_obs,
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
            let mut actions = model.react(&self.last_obs, &self.device);

            let mut start_idx = self.last_obs.len();
            for game in self.games.iter_mut().rev() {
                start_idx -= game.num_players();
                let game_actions = actions.split_off(start_idx);
                let result = game.step(&game_actions);

                memory.push_batch(
                    self.last_obs.split_off(start_idx),
                    game_actions,
                    result.rewards,
                    result.is_terminal,
                    result.truncated,
                );

                let obs = if result.is_terminal || result.truncated {
                    game.reset()
                } else {
                    result.obs
                };

                self.next_obs.extend(obs);
            }

            self.next_obs.reverse();
            mem::swap(&mut self.last_obs, &mut self.next_obs);
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
