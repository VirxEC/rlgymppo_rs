use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use burn::prelude::*;
use parking_lot::{Condvar, Mutex};
use rlgym::rocketsim::consts;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};
use rlgymppo_utils::shared_info::SharedInfoReport;

use super::sim::{GameInstance, RewardSamplingConfig};
use crate::agent::model::Actic;

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

pub struct Renderer<B: Backend, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    game: GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    last_obs: Vec<Vec<f32>>,
    controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
    device: B::Device,
}

impl<B, SS, OBS, ACT, REW, TERM, TRUNC, SI> Renderer<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
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
    pub fn new(
        env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
        controller: Arc<(Mutex<RendererControls<B>>, Condvar)>,
        device: B::Device,
    ) -> Self {
        // The renderer doesn't track training metrics, so disable reward sampling.
        let mut game = GameInstance::new(
            env,
            None,
            RewardSamplingConfig {
                add_rewards_to_metrics: false,
                ..Default::default()
            },
        );
        let (last_obs, _old_obs, _last_masks) = game.reset();

        Self {
            controller,
            last_obs,
            game,
            device,
        }
    }

    pub fn run(&mut self) {
        let mut model = {
            let (controller, start_var) = &*self.controller;

            let mut guard = controller.lock();
            if guard.quit {
                return;
            }

            while guard.model.is_none() || !guard.render {
                start_var.wait(&mut guard);
                if guard.quit {
                    return;
                }
            }

            guard.model.take().unwrap().to_device(&self.device)
        };

        self.game.set_rlviser_enabled(true);

        let tick_rate = Duration::from_secs_f32(ACT::get_tick_skip() as f32 * consts::TICK_TIME);
        let mut next_time = Instant::now();

        loop {
            if let Err(error) = self.game.handle_rlviser_messages() {
                eprintln!("Error receiving messages from RLViser: {error}");
                break;
            }

            let speed = self.game.rlviser_speed();
            let paused = self.game.rlviser_paused();

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

            let deterministic = guard.deterministic;
            drop(guard);

            let actions = (!paused).then(|| {
                if deterministic {
                    model.react_deterministic(&self.last_obs, &[], &self.device)
                } else {
                    model.react(&self.last_obs, &[], &self.device).0
                }
            });

            // Keep the renderer in real time, adjusted by RLViser's speed.
            let interval = if paused {
                tick_rate
            } else {
                Duration::from_secs_f64(
                    tick_rate.as_secs_f64() / f64::from(speed.max(f32::EPSILON)),
                )
            };
            let now = Instant::now();
            if next_time > now {
                sleep(next_time - now);
            } else {
                // If inference or rendering takes too long, resync instead of
                // permanently trying to catch up.
                next_time = now;
            }
            next_time += interval;

            let Some(actions) = actions else {
                continue;
            };
            let result = self.game.step(&actions);

            self.last_obs = if result.is_terminal || result.truncated {
                self.game.reset().0
            } else {
                result.obs
            };
        }
    }
}
