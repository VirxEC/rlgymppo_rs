use rlgym::{GameState, Reward};

use crate::utils::AvgTracker;
use crate::utils::shared_info::SharedInfoReport;

#[macro_export]
macro_rules! combined_rewards {
    ($($name:expr, $reward:expr => $weight:expr;)+) => {
        $crate::utils::rewards::CombinedRewards::new(vec![
            $(($name, Box::new($reward), $weight),)+
        ])
    };
}

/// Combines multiple reward functions into a single reward function by summing their outputs, optionally with weights.
///
/// The `SharedInfoReport` is used to track the average contribution of each reward function for reporting purposes.
///
/// # Example
///
/// Simple default-constructible rewards use their path directly:
/// ```
/// let reward_fn = combined_rewards![
///     "reward1", MyUnitReward => 1.0;
///     "reward2", MyUnitReward2 => 0.5;
/// ];
/// ```
///
/// Parameterized or wrapped rewards use their constructor:
/// ```
/// let reward_fn = combined_rewards![
///     "reward1", ZeroSumReward::new(ChildReward, 0.3, 1.0) => 1.0;
///     "reward2", SomeReward::new(0.5) => 0.5;
/// ];
/// ```
pub struct CombinedRewards<SI: SharedInfoReport> {
    rewards: Vec<(&'static str, Box<dyn Reward<SI>>, f32)>,
}

impl<SI: SharedInfoReport> CombinedRewards<SI> {
    /// Prefer the [`combined_rewards!`] macro over calling this directly.
    pub fn new(rewards: Vec<(&'static str, Box<dyn Reward<SI>>, f32)>) -> Self {
        Self { rewards }
    }
}

impl<SI: SharedInfoReport> Reward<SI> for CombinedRewards<SI> {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<f32> {
        let mut rewards: Vec<f32> = vec![0.0; state.cars.len()];

        for (name, reward_fn, weight) in &mut self.rewards {
            let fn_rewards = reward_fn.get_rewards(state, shared_info);

            let mut reward_total = 0.0;
            for (total, extra) in rewards.iter_mut().zip(fn_rewards) {
                let adjusted = extra * *weight;
                *total += adjusted;
                reward_total += adjusted;
            }

            shared_info.report()[*name] +=
                AvgTracker::new((reward_total / *weight) as f64, rewards.len() as u64);
        }

        rewards
    }
}
