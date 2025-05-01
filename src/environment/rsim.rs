use rlgym::{
    Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, StepResult, Terminal,
    Truncate, rocketsim_rs::glam_ext::GameStateA,
};
use crate::utils::{AvgTracker, Report};

pub struct GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &SI, &GameStateA),
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
    metrics: Report,
}

impl<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
    GameInstance<C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &SI, &GameStateA),
    SS: StateSetter<SI>,
    SIP: SharedInfoProvider<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(env: Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>, step_callback: C) -> Self {
        Self { env, step_callback, metrics: Report::default() }
    }

    pub fn reset(&mut self) -> (GameStateA, FullObs) {
        self.env.reset()
    }

    pub fn step(&mut self, game_state: &GameStateA, actions: &[ACT::Input]) -> StepResult {
        let result = self.env.step(game_state, actions);
        (self.step_callback)(&mut self.metrics, self.env.shared_info(), game_state);

        let num_players = result.rewards.len();
        let total_rew = result.rewards.iter().sum::<f32>() as f64;

        self.metrics["Avg. step reward"] += AvgTracker::new(total_rew, num_players as u64).into();

        result
    }

    pub fn get_metrics(&self) -> &Report {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics.clear();
    }
}
