use rlgym::{
    Action, Env, FullObs, GameState, Obs, Reward, SharedInfoProvider, StateSetter, Terminal,
    Truncate,
    rocketsim::{BallState, GameMode},
};

use crate::utils::{AvgTracker, Report, shared_info::SharedInfoReport};

pub struct StepResult {
    pub obs: FullObs,
    pub action_masks: Vec<Vec<bool>>,
    pub rewards: Vec<f32>,
    pub is_terminal: bool,
    pub truncated: bool,
}

/// Reward metrics sampling configuration.
#[derive(Clone, Debug)]
pub struct RewardSamplingConfig {
    /// Whether to include per-component reward metrics in the report.
    /// Reward components whose name starts with `"Reward/"` are tracked.
    pub add_rewards_to_metrics: bool,
    /// Sample reward metrics once every N steps (1 = every step).
    /// 0 means always sample (every step).
    pub reward_sample_interval: usize,
    /// Maximum number of random reward-component samples to include
    /// per metric report step (unused in the current Rust impl, kept
    /// for API compatibility with GigaLearn).
    #[allow(dead_code)]
    pub max_reward_samples: usize,
}

impl Default for RewardSamplingConfig {
    fn default() -> Self {
        Self {
            add_rewards_to_metrics: true,
            reward_sample_interval: 8,
            max_reward_samples: 50,
        }
    }
}

impl RewardSamplingConfig {
    /// Returns `true` when reward metrics should be recorded for this step.
    pub fn should_sample(&self, step_counter: usize) -> bool {
        if !self.add_rewards_to_metrics {
            return false;
        }
        self.reward_sample_interval == 0 || step_counter.is_multiple_of(self.reward_sample_interval)
    }
}

pub struct GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    last_state: GameState,
    metrics: Report,
    step_counter: usize,
    reward_sampling: RewardSamplingConfig,
}

impl<SS, OBS, ACT, REW, TERM, TRUNC, SI> GameInstance<SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport,
    OBS: Obs<SI>,
    ACT: Action<SI>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new(
        env: Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
        reward_sampling: RewardSamplingConfig,
    ) -> Self {
        Self {
            env,
            last_state: GameState {
                tick_count: 0,
                game_mode: GameMode::Soccar,
                ball: BallState::default(),
                cars: Vec::new(),
                boost_pads: Vec::new(),
                events: Vec::new(),
            },
            metrics: Report::default(),
            step_counter: 0,
            reward_sampling,
        }
    }

    pub fn set_rlviser_enabled(&mut self, enabled: bool) {
        self.env.set_rlviser_enabled(enabled)
    }

    pub fn num_players(&self) -> usize {
        self.last_state.cars.len()
    }

    /// Return the team index (0 = Blue, 1 = Orange) for each player
    /// in observation order (matches `cars` ordering).
    pub fn player_teams(&self) -> Vec<usize> {
        self.last_state
            .cars
            .iter()
            .map(|(info, _)| info.team as usize)
            .collect()
    }

    pub fn reset(&mut self) -> (FullObs, Vec<Vec<bool>>) {
        let (state, obs, masks) = self.env.reset();
        self.last_state = state;

        (obs, masks)
    }

    pub fn step(&mut self, actions: &[ACT::Input]) -> StepResult {
        let result = self.env.step(&self.last_state, actions);
        self.last_state = result.state;

        let shared_info = self.env.get_mut_shared_info();
        let report = shared_info.report();

        // Sample reward metrics: only merge reward entries on sampled steps.
        if self.reward_sampling.should_sample(self.step_counter) {
            self.metrics += &*report;
        } else {
            // Non-reward entries (e.g. player stats from SharedInfo) always flow
            // through, but "Reward/"-prefixed entries are dropped on non-sampled steps.
            report.remove_keys_with_prefix("Reward/");
            self.metrics += &*report;
        }
        report.clear();

        self.step_counter += 1;

        let num_players = result.rewards.len();
        let total_rew = result.rewards.iter().sum::<f32>() as f64;

        self.metrics["Collect/avg step reward"] += AvgTracker::new(total_rew, num_players as u64);

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
