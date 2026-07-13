use rlgymppo_utils::rlgym::GameState;
use rlgymppo_utils::rocketsim::ArenaState;

#[must_use]
pub fn to_rlgym_game_state(state: ArenaState) -> GameState {
    GameState {
        game_mode: state.game_mode(),
        tick_count: state.tick_count,
        ball: state.ball,
        cars: state.cars,
        boost_pads: state.boost_pads,
        events: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use rlgymppo_utils::rocketsim::{ArenaState, GameMode};

    use super::*;

    #[test]
    fn preserves_snapshot_fields() {
        let mut snapshot = ArenaState::new_empty(GameMode::Soccar);
        snapshot.tick_count = 123;

        let state = to_rlgym_game_state(snapshot);

        assert_eq!(state.game_mode, GameMode::Soccar);
        assert_eq!(state.tick_count, 123);
        assert!(state.cars.is_empty());
        assert!(state.events.is_empty());
    }
}
