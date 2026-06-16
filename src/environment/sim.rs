use rlgym::{
    Action, Env, FullObs, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim::{ArenaState, GameMode},
};

use crate::utils::{AvgTracker, Report};

pub struct StepResult {
    pub obs: FullObs,
    pub action_masks: Vec<Vec<bool>>,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
}

pub struct GameInstance<C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &ArenaState),
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    step_callback: C,
    last_state: ArenaState,
    metrics: Report,
}

impl<C, SS, OBS, ACT, REW, TERM, TRUNC, SI> GameInstance<C, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    C: Fn(&mut Report, &mut SI, &ArenaState),
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>, step_callback: C) -> Self {
        Self {
            env,
            step_callback,
            last_state: ArenaState::new_empty(GameMode::Soccar),
            metrics: Report::default(),
        }
    }

    pub fn set_rlviser_enabled(&mut self, enabled: bool) {
        self.env.set_rlviser_enabled(enabled)
    }

    pub fn num_players(&self) -> usize {
        self.last_state.cars.len()
    }

    pub fn reset(&mut self) -> (FullObs, Vec<Vec<bool>>) {
        let (state, obs, masks) = self.env.reset();
        self.last_state = state;

        (obs, masks)
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

        self.metrics[".Avg. step reward"] += AvgTracker::new(total_rew, num_players as u64).into();

        StepResult {
            obs: result.obs,
            action_masks: result.action_masks,
            rewards: result.rewards,
            is_terminal: result.is_terminal,
            truncated: result.truncated,
        }
    }

    pub fn get_metrics(&self) -> &Report {
        &self.metrics
    }

    pub fn clear_metrics(&mut self) {
        self.metrics.clear();
    }
}
