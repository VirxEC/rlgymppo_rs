//! End-to-end checks that the Nexto policy actually plays Rocket League.
//!
//! The unit tests in `obs.rs` pin the observation layout channel by channel,
//! but nothing there notices when a channel is fed values on the wrong scale:
//! the policy still runs, it just plays badly. These tests drive a real
//! RocketSim arena with Nexto at the wheel and assert on the outcome, so a
//! regression in the observation contract shows up as a bot that stops
//! scoring.
//!
//! Everything here is deterministic: inference is argmax and the simulation is
//! seeded, so the assertions do not need slack for randomness.

use std::sync::Once;

use burn::backend::Flex;
use rlgym::GameState;
use rlgym::rocketsim::{
    Arena, BallState, BoostPadState, CarBodyConfig, CarState, GameMode, Mat3A, PhysState, Team,
    Vec3A, consts, init,
};
use rlgymppo_nexto::{BOOST, DEMO, IS_BOOST, NextoAction, NextoModel, NextoObsBuilder};

type Backend = Flex;

/// Load collision meshes from the workspace root
fn soccar_arena() -> Arena {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let meshes = concat!(env!("CARGO_MANIFEST_DIR"), "/../collision_meshes");
        init(meshes, true).expect("collision meshes are checked in at the workspace root");
    });
    Arena::new(GameMode::Soccar)
}

const TICK_SKIP: usize = 8;
const TICKS_PER_SECOND: f32 = 120.0;

/// Snapshot the arena the same way `rlgym::Env::get_game_state` does.
fn game_state(arena: &Arena) -> GameState {
    let cars = (0..arena.num_cars())
        .map(|i| {
            let (info, state) = arena.get_car_info_and_state(i);
            (*info, *state)
        })
        .collect();
    let boost_pads = (0..arena.num_boost_pads())
        .map(|i| (*arena.get_boost_pad_config(i), arena.get_boost_pad_state(i)))
        .collect();

    GameState {
        game_mode: arena.game_mode(),
        tick_count: arena.tick_count(),
        ball: *arena.get_ball_state(),
        cars,
        boost_pads,
        events: Vec::new(),
    }
}

/// Drive `nexto_cars` with the Nexto policy for at most `secs` of simulated
/// time, stopping early once `stop` accepts the state. Returns the final state.
fn drive(
    arena: &mut Arena,
    model: &NextoModel<Backend>,
    nexto_cars: &[usize],
    secs: f32,
    stop: impl Fn(&GameState) -> bool,
) -> GameState {
    let builder = NextoObsBuilder::default();
    let mut previous_actions = vec![NextoAction::ZERO; arena.num_cars()];
    let decisions = (secs * TICKS_PER_SECOND) as usize / TICK_SKIP;

    let mut state = game_state(arena);
    for _ in 0..decisions {
        if stop(&state) {
            return state;
        }

        let observations = builder.build(&state, &previous_actions);
        let selected: Vec<&_> = nexto_cars.iter().map(|&car| &observations[car]).collect();
        let actions = model.actions_from_observations(&selected);

        for (&car, action) in nexto_cars.iter().zip(actions) {
            let action = NextoAction::from_index(action);
            previous_actions[car] = action;
            arena.set_car_controls(state.cars[car].0.idx, action.to_car_controls());
        }

        for _ in 0..TICK_SKIP {
            arena.step_tick();
        }
        state = game_state(arena);
    }

    state
}

/// Blue scores in the orange net, at +y.
fn blue_scored(state: &GameState) -> bool {
    state.is_ball_scored() && state.ball.phys.pos.y > 0.0
}

fn place_car(arena: &mut Arena, car: usize, pos: Vec3A, yaw: f32, boost: f32) {
    arena.set_car_state(
        car,
        CarState {
            phys: PhysState {
                pos,
                rot_mat: Mat3A::from_rotation_z(yaw),
                vel: Vec3A::ZERO,
                ang_vel: Vec3A::ZERO,
            },
            boost,
            ..CarState::DEFAULT
        },
    );
}

fn place_ball(arena: &mut Arena, pos: Vec3A) {
    arena.set_ball_state(BallState {
        phys: PhysState {
            pos,
            rot_mat: Mat3A::IDENTITY,
            vel: Vec3A::ZERO,
            ang_vel: Vec3A::ZERO,
        },
        ..BallState::DEFAULT
    });
}

#[test]
fn nexto_scores_on_an_empty_net() {
    let mut arena = soccar_arena();
    let car = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    place_car(
        &mut arena,
        car,
        Vec3A::new(0.0, -2000.0, 17.0),
        std::f32::consts::FRAC_PI_2,
        consts::car::boost::MAX,
    );
    place_ball(&mut arena, Vec3A::new(0.0, -1000.0, 93.0));

    let model = NextoModel::<Backend>::default();
    let state = drive(&mut arena, &model, &[car], 10.0, blue_scored);

    assert!(
        blue_scored(&state),
        "Nexto did not score on an empty net in 10s; ball ended at {:?}",
        state.ball.phys.pos
    );
}

#[test]
fn nexto_beats_an_idle_opponent() {
    let mut arena = soccar_arena();
    let nexto = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let _idle = arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    arena.reset_to_random_kickoff(Some(6_741));

    let model = NextoModel::<Backend>::default();
    let state = drive(&mut arena, &model, &[nexto], 30.0, blue_scored);

    assert!(
        blue_scored(&state),
        "Nexto did not score against an idle opponent in 30s; ball ended at {:?}",
        state.ball.phys.pos
    );
}

#[test]
fn obs_channels_stay_in_the_trained_range() {
    // Nexto was trained on boost as a fraction and pad availability as a flag.
    // Play a while so pads get taken and boost gets spent, then check every
    // entity row against that contract.
    let mut arena = soccar_arena();
    let nexto = arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
    let _idle = arena.add_car(Team::Orange, CarBodyConfig::OCTANE);
    arena.reset_to_random_kickoff(Some(6_741));

    let model = NextoModel::<Backend>::default();
    drive(&mut arena, &model, &[nexto], 15.0, |_| false);

    // Put a couple of pads on cooldown so the availability channel is
    // exercised in both states rather than only the all-available case.
    arena.set_boost_pad_state(0, BoostPadState { cooldown: 10.0 });
    arena.set_boost_pad_state(1, BoostPadState { cooldown: 3.5 });

    let state = game_state(&arena);
    let observations = NextoObsBuilder::default().build(&state, &[NextoAction::ZERO; 2]);

    for observation in &observations {
        for (entity, row) in observation.kv.iter().enumerate() {
            assert!(
                row.iter().all(|value| value.is_finite()),
                "entity {entity} has a non-finite channel: {row:?}"
            );
            assert!(
                (0.0..=1.0).contains(&row[BOOST]),
                "entity {entity} has boost {} outside [0, 1]",
                row[BOOST]
            );
            if row[IS_BOOST] == 1.0 {
                assert!(
                    row[DEMO] == 0.0 || row[DEMO] == 1.0,
                    "boost pad {entity} has availability {}, expected a flag",
                    row[DEMO]
                );
            }
        }
    }

    // The scenario is only meaningful if the channels actually varied.
    assert!(
        observations[0].kv[..2].iter().any(|row| row[BOOST] < 1.0),
        "no boost was consumed, the range check proves nothing"
    );
    let pads = || observations[0].kv.iter().filter(|row| row[IS_BOOST] == 1.0);
    assert!(
        pads().any(|row| row[DEMO] == 0.0) && pads().any(|row| row[DEMO] == 1.0),
        "expected both available and unavailable pads in the observation"
    );
}
