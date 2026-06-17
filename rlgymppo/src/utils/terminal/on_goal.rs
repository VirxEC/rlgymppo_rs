use rlgym::{GameState, Terminal};

/// A terminal condition that ends the episode when a goal is scored.
#[derive(Default)]
pub struct OnGoalCondition;

impl<SI> Terminal<SI> for OnGoalCondition {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn is_terminal(&mut self, state: &GameState, _shared_info: &mut SI) -> bool {
        state.is_ball_scored()
    }
}
