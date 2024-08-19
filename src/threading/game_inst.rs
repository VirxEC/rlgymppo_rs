use crate::util::{avg_tracker::AvgTracker, report::Report};
use rlgym_rs::{Action, Env, FullObs, Obs, Reward, StateSetter, StepResult, Terminal, Truncate};
use std::rc::Rc;
use tch::no_grad_guard;

pub struct GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    cur_obs: Rc<FullObs>,
    total_steps: u64,
    cur_episode_reward: f32,
    avg_steps_reward: AvgTracker,
    avg_episode_reward: AvgTracker,
    metrics: Report,
}

impl<SS, OBS, ACT, REW, TERM, TRUNC, SI> GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = Vec<i32>>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>) -> Self {
        Self {
            env,
            cur_obs: Rc::default(),
            total_steps: 0,
            cur_episode_reward: 0.0,
            avg_steps_reward: AvgTracker::default(),
            avg_episode_reward: AvgTracker::default(),
            metrics: Report::default(),
        }
    }

    pub fn start(&mut self) {
        self.cur_obs = self.env.reset();
    }

    pub fn num_cars(&self) -> usize {
        self.env.num_cars()
    }

    pub fn step(&mut self, actions: ACT::Input) -> StepResult {
        let _no_grad = no_grad_guard();
        let result = self.env.step(actions);

        // Update avg rewards
        let num_players = result.rewards.len();
        let total_rew: f32 = result.rewards.iter().sum();

        self.avg_steps_reward += AvgTracker::new(total_rew, num_players as u64);
        self.cur_episode_reward += total_rew / num_players as f32;

        if result.is_terminal || result.truncated {
            self.cur_obs = self.env.reset();

            self.avg_episode_reward += self.cur_episode_reward;
            self.cur_episode_reward = 0.0;
        } else {
            self.cur_obs = result.obs.clone();
        }

        self.total_steps += 1;

        result
    }

    pub fn get_obs(&self) -> Rc<FullObs> {
        self.cur_obs.clone()
    }

    pub fn reset_metrics(&mut self) {
        self.avg_episode_reward.reset();
        self.avg_steps_reward.reset();
        self.metrics.clear();
    }
}
