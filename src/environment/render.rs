use std::{
    sync::Arc,
    thread::sleep,
    time::{Duration, Instant},
};

use burn::prelude::*;
use parking_lot::{Condvar, Mutex};
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim::{ArenaState, consts},
};

use super::sim::GameInstance;
use crate::{agent::model::Net, utils::Report};

pub struct RendererControls<B: Backend> {
    pub model: Option<Net<B>>,
    pub deterministic: bool,
    pub render: bool,
    pub quit: bool,
}

impl<B: Backend> RendererControls<B> {
    pub fn new(render: bool) -> Self {
        Self {
            model: None,
            deterministic: false,
            quit: false,
            render,
        }
    }
}

pub struct Renderer<B: Backend, C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
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
    game: GameInstance<C, SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    last_obs: Vec<Vec<f32>>,
    controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
    device: B::Device,
}

impl<B, C, SS, OBS, ACT, REW, TERM, TRUNC, SI> Renderer<B, C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
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
    pub fn new(
        env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
        step_callback: C,
        controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
        device: B::Device,
    ) -> Self {
        let mut game = GameInstance::new(env, step_callback.clone());
        let (last_obs, _last_masks) = game.reset();

        Self {
            controller,
            last_obs,
            game,
            device,
        }
    }

    pub fn run(&mut self) {
        let (mut model, mut deterministic) = {
            let (controller, start_var) = &*self.controller;

            let mut guard = controller.lock();
            if guard.quit {
                return;
            }

            if guard.model.is_none() {
                start_var.wait(&mut guard);
            }

            while !guard.render {
                start_var.wait(&mut guard);
                if guard.quit {
                    return;
                }
            }

            (
                guard.model.take().unwrap().to_device(&self.device),
                guard.deterministic,
            )
        };

        self.game.set_rlviser_enabled(true);

        let mut last_controls_update_time = Instant::now();
        let controls_update_rate = Duration::from_secs(1);
        let tick_rate = Duration::from_secs_f32(ACT::get_tick_skip() as f32 * consts::TICK_TIME);
        let mut next_time = Instant::now();

        loop {
            // check for model updates every now and then
            let now = Instant::now();
            if now - last_controls_update_time >= controls_update_rate {
                let (controller, start_var) = &*self.controller;
                let mut guard = controller.lock();

                if guard.quit {
                    break;
                }

                while !guard.render {
                    start_var.wait(&mut guard);
                    if guard.quit {
                        return;
                    }
                    next_time = Instant::now();
                }

                if let Some(new_model) = guard.model.take() {
                    model = new_model.to_device(&self.device);
                }

                deterministic = guard.deterministic;
                last_controls_update_time = now;
            }

            let actions = if deterministic {
                model.react_deterministic(&self.last_obs, &[], &self.device)
            } else {
                model.react(&self.last_obs, &[], &self.device).0
            };

            // ensure real-time rendering
            // we get the action first to avoid stutters
            let wait_time = next_time - Instant::now();
            if !wait_time.is_zero() {
                sleep(wait_time);
            }
            next_time += tick_rate;

            let result = self.game.step(&actions);

            self.last_obs = if result.is_terminal || result.truncated {
                self.game.reset().0
            } else {
                result.obs
            };
        }
    }
}
