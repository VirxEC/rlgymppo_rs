use std::marker::PhantomData;

use rlgym::{GameState, Terminal};

use crate::utils::{AvgTracker, shared_info::SharedInfoReport};

/// A terminal condition that ends the episode when a goal is scored.
#[derive(Default)]
pub struct OnGoalCondition<SI: SharedInfoReport> {
    _shared_info: PhantomData<SI>,
}

impl<SI: SharedInfoReport> Terminal<SI> for OnGoalCondition<SI> {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn is_terminal(&mut self, state: &GameState, shared_info: &mut SI) -> bool {
        let scored = state.is_ball_scored();
        shared_info.report()["Game/Goal Speed"] += AvgTracker::from(state.ball.vel.length());

        scored
    }
}
