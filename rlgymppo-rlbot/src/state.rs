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
    use rlgymppo_utils::rocketsim::{ArenaState, CarBodyConfig, CarInfo, CarState, GameMode, Team};

    use super::*;

    #[test]
    fn preserves_snapshot_fields() {
        let mut snapshot = ArenaState::new_empty(GameMode::Soccar);
        snapshot.tick_count = 123;
        let mut car = CarState::DEFAULT;
        car.phys.pos.x = 123.0;
        car.boost = 17.0;
        car.set_contact_flags(true, [true, false, false, false]);
        car.set_prev_controls(rlgymppo_utils::rocketsim::CarControls {
            jump: true,
            ..Default::default()
        });
        snapshot.cars.push((
            CarInfo {
                idx: 7,
                team: Team::Blue,
                config: CarBodyConfig::OCTANE,
            },
            car,
        ));

        let state = to_rlgym_game_state(snapshot);

        assert_eq!(state.game_mode, GameMode::Soccar);
        assert_eq!(state.tick_count, 123);
        assert_eq!(state.cars[0].0.idx, 7);
        assert_eq!(state.cars[0].1.phys.pos, car.phys.pos);
        assert_eq!(state.cars[0].1.boost, car.boost);
        assert_eq!(state.cars[0].1.is_on_ground(), car.is_on_ground());
        assert_eq!(
            state.cars[0].1.wheels_with_contact(),
            car.wheels_with_contact()
        );
        assert_eq!(state.cars[0].1.prev_controls(), car.prev_controls());
        assert!(state.events.is_empty());
    }
}
