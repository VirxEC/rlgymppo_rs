use std::marker::PhantomData;

use rlgym::{GameState, Reward};

/// Wraps another reward function and applies a zero-sum team transform.
///
/// Per-player final reward:
/// `own * (1 - team_spirit) + avg_team * team_spirit - avg_opponent * opponent_scale`
///
/// - `team_spirit`: fraction of the team-average that is shared (0 = individual only, 1 = full team)
/// - `opponent_scale`: how much of the opponent's average to subtract (1 = full zero-sum)
///
/// When `team_spirit = 0` and `opponent_scale = 1`, the sum of all player rewards is zero
/// (the average opponent reward is subtracted from each player).
/// Based on GGL's `ZeroSumReward`.
pub struct ZeroSumReward<R: Reward<SI>, SI> {
    child: R,
    team_spirit: f32,
    opponent_scale: f32,
    _shared_info: PhantomData<SI>,
}

impl<R: Reward<SI>, SI> ZeroSumReward<R, SI> {
    pub fn new(reward: R, team_spirit: f32, opponent_scale: f32) -> Self {
        Self {
            child: reward,
            team_spirit,
            opponent_scale,
            _shared_info: PhantomData,
        }
    }
}

impl<R: Reward<SI> + Default, SI> Reward<SI> for ZeroSumReward<R, SI> {
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI) {
        self.child.reset(initial_state, shared_info);
    }

    fn get_rewards(&mut self, state: &GameState, shared_info: &mut SI) -> Vec<f32> {
        let mut rewards = self.child.get_rewards(state, shared_info);

        // Compute per-team averages
        let mut team_counts = [0usize; 2];
        let mut team_totals = [0.0f32; 2];

        for ((info, _), &reward) in state.cars.iter().zip(rewards.iter()) {
            let team_idx = info.team as usize;
            team_counts[team_idx] += 1;
            team_totals[team_idx] += reward;
        }

        let team_avgs = [
            team_totals[0] / team_counts[0].max(1) as f32,
            team_totals[1] / team_counts[1].max(1) as f32,
        ];

        // Apply transform
        for ((info, _), reward) in state.cars.iter().zip(rewards.iter_mut()) {
            let team_idx = info.team as usize;
            let opp_idx = 1 - team_idx;

            *reward = *reward * (1.0 - self.team_spirit) + team_avgs[team_idx] * self.team_spirit
                - team_avgs[opp_idx] * self.opponent_scale;
        }

        rewards
    }
}
