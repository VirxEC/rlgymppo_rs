use super::sim::GameInstance;
use crate::{agent::model::Actic, utils::Report};
use burn::prelude::*;
use rand::{SeedableRng, rngs::SmallRng};
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::{
    sync::{Arc, Condvar, Mutex},
    thread::sleep,
    time::{Duration, Instant},
};

pub struct RendererControls<B: Backend> {
    pub model: Option<Actic<B>>,
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

pub struct Renderer<B: Backend, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
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
    game: GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    last_obs: Vec<Vec<f32>>,
    rng: SmallRng,
    try_launch_exe: bool,
    controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
    device: B::Device,
}

impl<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    Renderer<B, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
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
    pub fn new(
        env: Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
        step_callback: C,
        try_launch_exe: bool,
        controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
    ) -> Self {
        let mut game = GameInstance::new(env, step_callback.clone());
        let last_obs = game.reset();

        Self {
            rng: SmallRng::from_os_rng(),
            try_launch_exe,
            controller,
            last_obs,
            game,
            device: B::Device::default(),
        }
    }

    pub fn start_rendering(&mut self) {
        self.game.open_rlviser(self.try_launch_exe);
        if !self.try_launch_exe {
            println!("Please open rlviser to view the simulation.");
        }
    }

    pub fn run(&mut self) {
        let (mut model, mut deterministic) = {
            let (controller, start_var) = &*self.controller;

            let mut guard = controller.lock().unwrap();
            if guard.quit {
                return;
            }

            if guard.model.is_none() {
                guard = start_var.wait(guard).unwrap();
            }

            while !guard.render {
                guard = start_var.wait(guard).unwrap();
                if guard.quit {
                    return;
                }
            }

            (guard.model.take().unwrap(), guard.deterministic)
        };

        let mut is_first = true;
        let mut last_controls_update_time = Instant::now();
        let controls_update_rate = Duration::from_secs(1);
        let mut tick_rate = Duration::from_secs_f32(ACT::get_tick_skip() as f32 / 120.);
        let mut next_time = Instant::now();

        loop {
            // ensure real-time rendering
            let now = Instant::now();
            let wait_time = next_time - now;
            if !wait_time.is_zero() {
                sleep(wait_time);
            }
            next_time += tick_rate;

            // enables in-renderer state setting
            self.game.handle_incoming_states(&mut tick_rate);

            // check for new controls every 15 iterations
            // ~1s, assuming tick skip is 8
            // it doesn't really matter for the render thread
            if now - last_controls_update_time >= controls_update_rate {
                let (controller, start_var) = &*self.controller;
                let mut guard = controller.lock().unwrap();

                if guard.quit {
                    break;
                }

                if !guard.render {
                    is_first = true;
                    self.game.close_rlviser();
                }

                while !guard.render {
                    guard = start_var.wait(guard).unwrap();
                    if guard.quit {
                        return;
                    }
                }

                if let Some(new_model) = guard.model.take() {
                    model = new_model;
                }

                deterministic = guard.deterministic;
                last_controls_update_time = now;
            }

            if self.game.is_paused() && !is_first {
                continue;
            }

            if is_first {
                self.start_rendering();
                is_first = false;
            }

            let actions = if deterministic {
                model.react_deterministic(&self.last_obs, &self.device)
            } else {
                model.react(&self.last_obs, &mut self.rng, &self.device)
            };
            let result = self.game.step(&actions);

            self.last_obs = if result.is_terminal || result.truncated {
                self.game.reset()
            } else {
                result.obs
            };
        }

        self.game.close_rlviser();
    }
}
