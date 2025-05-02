use super::sim::GameInstance;
use crate::{agent::model::Actic, base::Memory, utils::Report};
use burn::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::mem;

#[derive(Clone, Copy)]
pub struct BatchSimConfig {
    pub num_games: usize,
    pub buffer_size: usize,
}

impl Default for BatchSimConfig {
    fn default() -> Self {
        Self {
            num_games: 5,
            buffer_size: 5_000,
        }
    }
}

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
    rng: SmallRng,
    config: BatchSimConfig,
    memory: Memory,
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
        config: BatchSimConfig,
        device: B::Device,
    ) -> Self
    where
        F: Fn() -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    {
        assert_ne!(config.num_games, 0, "num_games must be greater than 0");
        assert_ne!(config.buffer_size, 0, "buffer_size must be greater than 0");

        let mut games = Vec::with_capacity(config.num_games);

        let mut last_obs = Vec::with_capacity(config.num_games * 2);
        for _ in 0..config.num_games {
            let env = create_env_fn();
            let mut game = GameInstance::new(env, step_callback.clone());
            last_obs.extend(game.reset());
            games.push(game);
        }

        Self {
            memory: Memory::new(config.buffer_size),
            rng: SmallRng::from_os_rng(),
            next_obs: Vec::with_capacity(last_obs.len()),
            last_obs,
            games,
            config,
            device,
        }
    }

    pub fn run(&mut self, model: Actic<B>) -> &mut Memory {
        while self.memory.len() < self.config.buffer_size {
            let mut actions = model.react(&self.last_obs, &mut self.rng, &self.device);

            let mut start_idx = self.last_obs.len();
            for game in self.games.iter_mut().rev() {
                start_idx -= game.num_players();
                let game_actions = actions.split_off(start_idx);
                let result = game.step(&game_actions);

                self.memory.push_batch(
                    self.last_obs.split_off(start_idx),
                    &result.obs,
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

        &mut self.memory
    }

    pub fn get_metrics(&mut self) -> Report {
        let metrics = self.games[0].get_metrics();
        self.games
            .iter_mut()
            .skip(1)
            .fold(metrics, |mut acc, game| {
                acc += game.get_metrics();
                acc
            })
    }
}
