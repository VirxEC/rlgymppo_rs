use crate::utils::{AvgTracker, Report};
use rlgym::{
    Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::{mem, time::Duration};

pub struct StepResult {
    pub obs: FullObs,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
}

pub struct GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &GameStateA),
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    env: Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>,
    step_callback: C,
    last_state: GameStateA,
    metrics: Report,
}

impl<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &GameStateA),
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(env: Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>, step_callback: C) -> Self {
        Self {
            env,
            step_callback,
            last_state: GameStateA::default(),
            metrics: Report::default(),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.env.is_paused()
    }

    pub fn open_rlviser(&mut self, try_launch_exe: bool) {
        self.env.enable_rendering(try_launch_exe);
    }

    pub fn close_rlviser(&mut self) {
        self.env.stop_rendering();
    }

    pub fn handle_incoming_states(&mut self, tick_rate: &mut Duration) {
        self.env.handle_incoming_states(tick_rate).unwrap();
    }

    pub fn num_players(&self) -> usize {
        self.last_state.cars.len()
    }

    pub fn reset(&mut self) -> FullObs {
        let (state, obs) = self.env.reset();
        self.last_state = state;

        obs
    }

    pub fn step(&mut self, actions: &[ACT::Input]) -> StepResult {
        let result = self.env.step(&self.last_state, actions);
        self.last_state = result.state;

        (self.step_callback)(
            &mut self.metrics,
            self.env.get_mut_shared_info(),
            &self.last_state,
        );

        let num_players = result.rewards.len();
        let total_rew = result.rewards.iter().sum::<f32>() as f64;

        self.metrics["Avg. step reward"] += AvgTracker::new(total_rew, num_players as u64).into();

        StepResult {
            obs: result.obs,
            rewards: result.rewards,
            is_terminal: result.is_terminal,
            truncated: result.truncated,
        }
    }

    pub fn get_metrics(&mut self) -> Report {
        mem::take(&mut self.metrics)
    }
}
