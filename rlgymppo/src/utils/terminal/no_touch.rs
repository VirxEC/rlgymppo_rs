use rlgym::rocketsim::ArenaEvent;
use rlgym::{GameState, Truncate};

/// Terminal condition that triggers when no car has touched the ball for a given duration.
#[derive(Default)]
pub struct NoTouchCondition<const MAX_TICKS: u64> {
    last_touch_tick: u64,
}

impl<const MAX_TICKS: u64, SI> Truncate<SI> for NoTouchCondition<MAX_TICKS> {
    fn reset(&mut self, initial_state: &GameState, _shared_info: &mut SI) {
        self.last_touch_tick = initial_state.tick_count;
    }

    fn should_truncate(&mut self, state: &GameState, _shared_info: &mut SI) -> bool {
        if state
            .events
            .iter()
            .any(|event| matches!(event, ArenaEvent::CarHitBall(_)))
        {
            self.last_touch_tick = state.tick_count;
        }

        let elapsed = state.tick_count - self.last_touch_tick;
        elapsed >= MAX_TICKS
    }
}
