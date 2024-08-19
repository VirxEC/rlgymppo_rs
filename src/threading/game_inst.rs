use crate::util::{avg_tracker::AvgTracker, report::Report};
use rlgym_rs::{Action, Env, FullObs, Obs, Reward, StateSetter, StepResult, Terminal, Truncate};
use std::{ops::AddAssign, rc::Rc};
use tch::no_grad_guard;

#[derive(Debug, Default)]
pub struct GameMetrics {
    pub avg_steps_reward: AvgTracker,
    pub avg_episode_reward: AvgTracker,
    pub report: Report,
}

impl GameMetrics {
    pub fn reset(&mut self) {
        self.avg_steps_reward.reset();
        self.avg_episode_reward.reset();
        self.report.clear();
    }
}

impl AddAssign<&GameMetrics> for GameMetrics {
    fn add_assign(&mut self, rhs: &GameMetrics) {
        self.avg_steps_reward += rhs.avg_steps_reward;
        self.avg_episode_reward += rhs.avg_episode_reward;
        self.report += &rhs.report;
    }
}

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
    last_obs: Option<Rc<FullObs>>,
    total_steps: u64,
    cur_episode_reward: f64,
    metrics: GameMetrics,
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
            last_obs: None,
            total_steps: 0,
            cur_episode_reward: 0.0,
            metrics: GameMetrics::default(),
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
        let total_rew: f64 = result.rewards.iter().sum::<f32>() as f64;

        self.metrics.avg_steps_reward += AvgTracker::new(total_rew, num_players as u64);
        self.cur_episode_reward += total_rew / num_players as f64;

        if result.is_terminal || result.truncated {
            self.last_obs = Some(self.cur_obs.clone());
            self.cur_obs = self.env.reset();

            println!("Episode reward: {}", self.cur_episode_reward);
            self.metrics.avg_episode_reward += self.cur_episode_reward;
            self.cur_episode_reward = 0.0;
        } else {
            self.last_obs = None;
            self.cur_obs = result.obs.clone();
        }

        self.total_steps += 1;

        result
    }

    pub fn get_next_obs(&self) -> Rc<FullObs> {
        match self.last_obs {
            Some(ref obs) => obs.clone(),
            None => self.cur_obs.clone(),
        }
    }

    pub fn get_obs(&self) -> Rc<FullObs> {
        self.cur_obs.clone()
    }

    pub fn get_metrics(&self) -> &GameMetrics {
        &self.metrics
    }

    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}
