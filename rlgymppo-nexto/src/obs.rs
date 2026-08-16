//! Nexto observation builder.
//!
//! This is a Rust port of the upstream `NextoObsBuilder`. It reads an
//! `rlgym::GameState` and builds
//! per-player observations in the q/kv/mask format used by the Nexto
//! transformer policy.
//!
//! Entity layout inside `kv`:
//! - player entities, one per slot, indices `0..player_slots`
//! - ball entity, index `player_slots`
//! - boost pad entities, the remaining indices
//!
//! Each row has 24 channels: 5 selectors, 5 vector fields of 3 floats, and
//! 4 scalars. The query adds the previous action: 32 floats in total.

use std::ops::Range;

use rlgym::GameState;
use rlgym::rocketsim::{BoostPadConfig, Team, Vec3A, consts};

pub use crate::action::NextoAction;

/// Length of a query vector: the self entity plus the previous action.
pub const Q_LEN: usize = 32;

/// Length of a key/value vector. All entities share this layout.
pub const KV_LEN: usize = 24;

// Entity selectors, indices 0..5.
pub const IS_SELF: usize = 0;
pub const IS_MATE: usize = 1;
pub const IS_OPP: usize = 2;
pub const IS_BALL: usize = 3;
pub const IS_BOOST: usize = 4;

// Vector fields, indices 5..20. Each field is 3 floats: x, y, z.
pub const POS: Range<usize> = 5..8;
pub const LIN_VEL: Range<usize> = 8..11;
pub const FW: Range<usize> = 11..14;
pub const UP: Range<usize> = 14..17;
pub const ANG_VEL: Range<usize> = 17..20;

// Scalar fields, indices 20..24.
pub const BOOST: usize = 20;
pub const DEMO: usize = 21;
pub const ON_GROUND: usize = 22;
pub const HAS_FLIP: usize = 23;

// Previous-action fields, indices 24..32. The caller fills these.
pub const ACTIONS: Range<usize> = 24..32;

/// The five per-entity 3-float vector fields, in layout order.
const VEC_FIELDS: [Range<usize>; 5] = [POS, LIN_VEL, FW, UP, ANG_VEL];

/// Boost amount of a small pad, as stored in the `BOOST` slot.
pub const SMALL_PAD_BOOST: f32 = 0.12;

/// Boost amount of a big pad, as stored in the `BOOST` slot.
pub const BIG_PAD_BOOST: f32 = 1.0;

/// Default per-channel normalization divisors.
///
/// Positions and velocities divide by the car max speed. Angular velocities
/// divide by the car max angular speed. All other channels divide by 1.
pub const DEFAULT_NORM: [f32; KV_LEN] = [
    1.0,
    1.0,
    1.0,
    1.0,
    1.0, // selectors
    consts::car::MAX_SPEED,
    consts::car::MAX_SPEED,
    consts::car::MAX_SPEED, // pos
    consts::car::MAX_SPEED,
    consts::car::MAX_SPEED,
    consts::car::MAX_SPEED, // lin vel
    1.0,
    1.0,
    1.0, // forward
    1.0,
    1.0,
    1.0, // up
    consts::car::MAX_ANG_SPEED,
    consts::car::MAX_ANG_SPEED,
    consts::car::MAX_ANG_SPEED, // ang vel
    1.0,
    1.0,
    1.0,
    1.0, // boost, demo, on ground, has flip
];

/// One player's Nexto observation.
#[derive(Clone, Debug)]
pub struct NextoObs {
    /// Query vector. The first 24 floats are the self entity. The last
    /// 8 floats are the previous action.
    pub q: [f32; Q_LEN],

    /// Key/value rows, one per entity.
    pub kv: Vec<[f32; KV_LEN]>,

    /// Per-entity additive attention bias. Empty player slots are marked with
    /// `1.0`, matching the upstream Nexto representation.
    pub mask: Vec<f32>,
}

/// Configuration for [`NextoObsBuilder`].
///
/// The default configuration matches the original Nexto observation builder.
#[derive(Clone, Debug)]
pub struct NextoObsConfig {
    /// Total player entity slots across both teams. `None` uses the number of
    /// cars in the state. The value must be at least the number of cars.
    pub player_slots: Option<usize>,

    /// Boost pads used for the boost entities. `None` uses the pads from the
    /// game state. The pads keep the order they appear in.
    pub boost_pads: Option<Vec<BoostPadConfig>>,

    /// Per-channel normalization divisors, applied to every kv row.
    /// Defaults to [`DEFAULT_NORM`].
    pub norm: [f32; KV_LEN],
}

impl Default for NextoObsConfig {
    fn default() -> Self {
        Self {
            player_slots: None,
            boost_pads: None,
            norm: DEFAULT_NORM,
        }
    }
}

/// Shared values computed once per `build` call.
struct ObsContext<'a> {
    state: &'a GameState,
    boosts: Vec<BoostPadConfig>,
    pads_from_state: bool,
    player_slots: usize,
    entity_count: usize,
    teams: Vec<f32>,
}

/// Builds Nexto observations from an `rlgym::GameState`.
///
/// The builder does not mutate between calls. `build` reads the state and
/// returns one observation per car.
pub struct NextoObsBuilder {
    config: NextoObsConfig,
}

impl NextoObsBuilder {
    /// Create a builder from a configuration.
    #[must_use]
    pub fn new(config: NextoObsConfig) -> Self {
        Self { config }
    }

    /// Build one observation per car in `state.cars` order.
    ///
    /// `previous_actions` must contain one action per car, in the same
    /// order as `state.cars`.
    pub fn build(&self, state: &GameState, previous_actions: &[NextoAction]) -> Vec<NextoObs> {
        let n_players = state.cars.len();
        assert_eq!(
            previous_actions.len(),
            n_players,
            "expected one previous action per car"
        );

        let ctx = self.context(state);

        (0..n_players)
            .map(|player_idx| self.build_one(&ctx, player_idx, previous_actions[player_idx]))
            .collect()
    }

    /// Build one observation for a single car.
    ///
    /// `player_idx` is the car's index in `state.cars`. The observation
    /// embeds `previous_action` in its query.
    pub fn build_for_player(
        &self,
        state: &GameState,
        player_idx: usize,
        previous_action: NextoAction,
    ) -> NextoObs {
        let ctx = self.context(state);
        assert!(
            player_idx < ctx.state.cars.len(),
            "player_idx ({player_idx}) must be less than the number of cars ({})",
            ctx.state.cars.len()
        );
        self.build_one(&ctx, player_idx, previous_action)
    }

    /// Compute the shared values used by every per-player observation.
    fn context<'a>(&self, state: &'a GameState) -> ObsContext<'a> {
        let n_players = state.cars.len();
        let pads_from_state = self.config.boost_pads.is_none();
        let boosts: Vec<BoostPadConfig> = if pads_from_state {
            state.boost_pads.iter().map(|(config, _)| *config).collect()
        } else {
            self.config.boost_pads.clone().expect("checked above")
        };
        let n_boosts = boosts.len();

        let player_slots = self.config.player_slots.unwrap_or(n_players);
        assert!(
            player_slots >= n_players,
            "player_slots ({player_slots}) must be at least the number of cars ({n_players})"
        );
        let entity_count = player_slots + 1 + n_boosts;

        let teams: Vec<f32> = state
            .cars
            .iter()
            .map(|(info, _)| f32::from(info.team == Team::Orange))
            .collect();

        ObsContext {
            state,
            boosts,
            pads_from_state,
            player_slots,
            entity_count,
            teams,
        }
    }

    fn build_one(
        &self,
        ctx: &ObsContext<'_>,
        player_idx: usize,
        previous_action: NextoAction,
    ) -> NextoObs {
        let n_players = ctx.state.cars.len();
        let ball_entity = ctx.player_slots;
        let invert = ctx.teams[player_idx] == 1.0;

        let mut kv = vec![[0.0; KV_LEN]; ctx.entity_count];

        // Ball entity.
        {
            let row = &mut kv[ball_entity];
            row[IS_BALL] = 1.0;
            set_vec3(row, POS, ctx.state.ball.phys.pos);
            set_vec3(row, LIN_VEL, ctx.state.ball.phys.vel);
            set_vec3(row, ANG_VEL, ctx.state.ball.phys.ang_vel);
        }

        // Boost pad entities.
        for (k, pad) in ctx.boosts.iter().enumerate() {
            let row = &mut kv[ball_entity + 1 + k];
            row[IS_BOOST] = 1.0;
            set_vec3(row, POS, pad.pos);
            row[BOOST] = if pad.is_big {
                BIG_PAD_BOOST
            } else {
                SMALL_PAD_BOOST
            };
            row[DEMO] = if ctx.pads_from_state {
                ctx.state.boost_pads[k].1.cooldown
            } else {
                0.0
            };
        }

        // Player entities.
        for (i, (_info, car)) in ctx.state.cars.iter().enumerate() {
            let row = &mut kv[i];
            row[IS_MATE] = 1.0 - ctx.teams[i];
            row[IS_OPP] = ctx.teams[i];
            set_vec3(row, POS, car.phys.pos);
            set_vec3(row, LIN_VEL, car.phys.vel);
            set_vec3(row, FW, car.phys.rot_mat.x_axis);
            set_vec3(row, UP, car.phys.rot_mat.z_axis);
            set_vec3(row, ANG_VEL, car.phys.ang_vel);
            row[BOOST] = car.boost;
            row[DEMO] = f32::from(car.is_demoed);
            row[ON_GROUND] = f32::from(car.is_on_ground);
            row[HAS_FLIP] = f32::from(car.has_flip_or_jump());
        }

        kv[player_idx][IS_SELF] = 1.0;

        // Orange players see the field mirrored: negate the x/y of every
        // vector field and swap the mate/opponent selectors.
        if invert {
            for row in &mut kv {
                row.swap(IS_MATE, IS_OPP);
                for range in VEC_FIELDS {
                    row[range.start] = -row[range.start];
                    row[range.start + 1] = -row[range.start + 1];
                }
            }
        }

        for row in &mut kv {
            for (value, &norm) in row.iter_mut().zip(self.config.norm.iter()) {
                *value /= norm;
            }
        }

        // The query starts as a copy of the self entity.
        let mut q = [0.0; Q_LEN];
        q[..KV_LEN].copy_from_slice(&kv[player_idx]);

        // Make everything relative to the self entity: translate positions
        // and rotate the x/y of every vector field so forward points to +y.
        convert_to_relative(&q, &mut kv);

        let mut mask = vec![0.0; ctx.entity_count];
        mask[n_players..ctx.player_slots].fill(1.0);

        q[ACTIONS].copy_from_slice(&previous_action.controls());

        NextoObs { q, kv, mask }
    }
}

impl Default for NextoObsBuilder {
    fn default() -> Self {
        Self::new(NextoObsConfig::default())
    }
}

/// Write a 3-float vector into the given channel range.
#[inline]
fn set_vec3(row: &mut [f32; KV_LEN], range: Range<usize>, value: Vec3A) {
    row[range.start] = value.x;
    row[range.start + 1] = value.y;
    row[range.start + 2] = value.z;
}

/// Make every kv row relative to the query.
///
/// First subtract the query position from every position. Then rotate the
/// x/y components of every vector field so the query forward points to +y.
fn convert_to_relative(q: &[f32; Q_LEN], kv: &mut [[f32; KV_LEN]]) {
    for row in kv.iter_mut() {
        for channel in POS {
            row[channel] -= q[channel];
        }
    }

    let forward_x = q[FW.start];
    let forward_y = q[FW.start + 1];
    let theta = forward_x.atan2(forward_y);
    let (sin_theta, cos_theta) = theta.sin_cos();

    for row in kv.iter_mut() {
        for range in VEC_FIELDS {
            let x = row[range.start];
            let y = row[range.start + 1];
            row[range.start] = cos_theta * x - sin_theta * y;
            row[range.start + 1] = sin_theta * x + cos_theta * y;
        }
    }
}

#[cfg(test)]
mod tests {
    use rlgym::GameState;
    use rlgym::rocketsim::{
        BallState, BoostPadConfig, BoostPadState, CarControls, CarInfo, CarState, GameMode, Mat3A,
        PhysState, Team, Vec3A, consts,
    };

    use super::*;

    const EPSILON: f32 = 1e-4;

    fn assert_approx(left: f32, right: f32) {
        assert!(
            (left - right).abs() <= EPSILON,
            "expected {left} to be within {EPSILON} of {right}"
        );
    }

    /// The standard 34 soccar pads, sorted like the RocketSim arena does.
    fn soccar_pads() -> Vec<BoostPadConfig> {
        let mut pads: Vec<BoostPadConfig> =
            consts::boost_pads::get_locations(GameMode::Soccar, false)
                .iter()
                .map(|pos| BoostPadConfig {
                    pos: *pos,
                    is_big: false,
                })
                .chain(
                    consts::boost_pads::get_locations(GameMode::Soccar, true)
                        .iter()
                        .map(|pos| BoostPadConfig {
                            pos: *pos,
                            is_big: true,
                        }),
                )
                .collect();
        pads.sort_by(|a, b| {
            a.pos
                .y
                .total_cmp(&b.pos.y)
                .then_with(|| a.pos.x.total_cmp(&b.pos.x))
        });
        pads
    }

    fn car(idx: usize, team: Team, pos: Vec3A, yaw: f32) -> (CarInfo, CarState) {
        let info = CarInfo {
            idx,
            team,
            ..CarInfo::default()
        };
        let state = CarState {
            phys: PhysState {
                pos,
                rot_mat: Mat3A::from_rotation_z(yaw),
                vel: Vec3A::ZERO,
                ang_vel: Vec3A::ZERO,
            },
            ..CarState::DEFAULT
        };
        (info, state)
    }

    fn state_with(cars: Vec<(CarInfo, CarState)>, ball_pos: Vec3A) -> GameState {
        let pads = soccar_pads();
        GameState {
            game_mode: GameMode::Soccar,
            tick_count: 0,
            ball: BallState {
                phys: PhysState {
                    pos: ball_pos,
                    rot_mat: Mat3A::IDENTITY,
                    vel: Vec3A::ZERO,
                    ang_vel: Vec3A::ZERO,
                },
                ..BallState::DEFAULT
            },
            boost_pads: pads
                .iter()
                .map(|config| (*config, BoostPadState::DEFAULT))
                .collect(),
            cars,
            events: Vec::new(),
        }
    }

    #[test]
    fn builds_one_obs_per_player_with_expected_shapes() {
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(1, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(2, Team::Orange, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(3, Team::Orange, Vec3A::new(0.0, 0.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(0.0, 0.0, 92.0));

        let builder = NextoObsBuilder::default();
        let obs = builder.build(&state, &[NextoAction::ZERO; 4]);

        // One observation per car, each with 4 players + 1 ball + 34 pads.
        assert_eq!(obs.len(), 4);
        for o in &obs {
            assert_eq!(o.q.len(), Q_LEN);
            assert_eq!(o.kv.len(), 4 + 1 + 34);
            assert_eq!(o.kv[0].len(), KV_LEN);
            assert_eq!(o.mask.len(), o.kv.len());
        }
    }

    #[test]
    fn player_slots_pad_and_mask_empty_slots() {
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(1, Team::Orange, Vec3A::new(0.0, 0.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(0.0, 0.0, 92.0));

        let config = NextoObsConfig {
            player_slots: Some(4),
            ..NextoObsConfig::default()
        };
        let builder = NextoObsBuilder::new(config);
        let obs = builder.build(&state, &[NextoAction::ZERO; 2]);

        // Entities: 4 player slots + 1 ball + 34 boost pads.
        assert_eq!(obs[0].kv.len(), 39);
        assert_eq!(obs[0].mask.len(), 39);

        // Only the empty player slots are masked.
        assert_eq!(&obs[0].mask[..2], &[0.0, 0.0]);
        assert_eq!(&obs[0].mask[2..4], &[1.0, 1.0]);
        // The ball and the boost pads are never masked.
        assert_eq!(&obs[0].mask[4..], &[0.0; 35]);
    }

    #[test]
    fn selectors_and_normalization() {
        // The car faces +y, so the relative transform is a pure translation.
        let cars = vec![car(
            0,
            Team::Blue,
            Vec3A::new(0.0, 0.0, 17.0),
            std::f32::consts::FRAC_PI_2,
        )];
        let state = state_with(cars, Vec3A::new(2300.0, 0.0, 92.0));
        let builder = NextoObsBuilder::default();
        let obs = builder.build(&state, &[NextoAction::ZERO; 1]);
        let o = &obs[0];

        // Query selectors: the self entity is the acting player. A blue
        // player is its own mate, so IS_MATE is set on the self row.
        assert_eq!(o.q[IS_SELF], 1.0);
        assert_eq!(o.q[IS_MATE], 1.0);
        assert_eq!(o.q[IS_OPP], 0.0);

        // The self entity is translated to the origin.
        assert_eq!(o.kv[0][POS.start], 0.0);
        assert_eq!(o.kv[0][POS.start + 1], 0.0);
        assert_eq!(o.kv[0][POS.start + 2], 0.0);

        // Ball entity at index 1: selectors and normalized position.
        let ball = &o.kv[1];
        assert_eq!(ball[IS_BALL], 1.0);
        assert_eq!(ball[IS_SELF], 0.0);
        assert_approx(ball[POS.start], 2300.0 / consts::car::MAX_SPEED);
        assert_approx(ball[POS.start + 1], 0.0);
        assert_approx(ball[POS.start + 2], 75.0 / consts::car::MAX_SPEED);

        // Boost entities start at index 2. The first pad is small, at
        // (0, -4240, 70). Position divides by the car max speed.
        let boost = &o.kv[2];
        assert_eq!(boost[IS_BOOST], 1.0);
        assert_eq!(boost[BOOST], SMALL_PAD_BOOST);
        assert_approx(boost[POS.start], 0.0);
        assert_approx(boost[POS.start + 1], -4240.0 / consts::car::MAX_SPEED);
        assert_approx(boost[POS.start + 2], 53.0 / consts::car::MAX_SPEED);
    }

    #[test]
    fn relative_transform_translates_and_rotates() {
        // The blue car faces +x. The ball and the orange car sit at known
        // offsets. The blue observation must show them in the blue frame.
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(50.0, 30.0, 17.0), 0.0),
            car(1, Team::Orange, Vec3A::new(-50.0, -30.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(150.0, 130.0, 92.0));
        let builder = NextoObsBuilder::default();
        let obs = builder.build(&state, &[NextoAction::ZERO; 2]);
        let blue = &obs[0];

        // Self entity: position zero, forward mapped to +y, up unchanged.
        assert_approx(blue.kv[0][POS.start], 0.0);
        assert_approx(blue.kv[0][POS.start + 1], 0.0);
        assert_approx(blue.kv[0][POS.start + 2], 0.0);
        assert_approx(blue.kv[0][FW.start], 0.0);
        assert_approx(blue.kv[0][FW.start + 1], 1.0);
        assert_approx(blue.kv[0][UP.start + 2], 1.0);

        // Ball offset (100, 100, 75) rotated +90 degrees about z.
        let ball = &blue.kv[2];
        assert_approx(ball[POS.start], -100.0 / consts::car::MAX_SPEED);
        assert_approx(ball[POS.start + 1], 100.0 / consts::car::MAX_SPEED);
        assert_approx(ball[POS.start + 2], 75.0 / consts::car::MAX_SPEED);

        // Orange offset (-100, -60, 0) rotated +90 degrees about z.
        let orange = &blue.kv[1];
        assert_approx(orange[POS.start], 60.0 / consts::car::MAX_SPEED);
        assert_approx(orange[POS.start + 1], -100.0 / consts::car::MAX_SPEED);
        assert_approx(orange[POS.start + 2], 0.0);
    }

    #[test]
    fn orange_observations_are_mirrored_and_team_flags_swapped() {
        // Both cars face +x. The ball is 100 units to the right of both.
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(1, Team::Orange, Vec3A::new(0.0, 0.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(100.0, 0.0, 92.0));
        let builder = NextoObsBuilder::default();
        let obs = builder.build(&state, &[NextoAction::ZERO; 2]);

        // Both players see the ball directly in front of them.
        let blue_ball = &obs[0].kv[2];
        let orange_ball = &obs[1].kv[2];
        assert_approx(blue_ball[POS.start], 0.0);
        assert_approx(blue_ball[POS.start + 1], 100.0 / consts::car::MAX_SPEED);
        assert_approx(orange_ball[POS.start], 0.0);
        assert_approx(orange_ball[POS.start + 1], 100.0 / consts::car::MAX_SPEED);

        // Blue view: entity 0 is a mate, entity 1 is an opponent.
        let blue = &obs[0];
        assert_eq!(blue.kv[0][IS_MATE], 1.0);
        assert_eq!(blue.kv[0][IS_OPP], 0.0);
        assert_eq!(blue.kv[1][IS_MATE], 0.0);
        assert_eq!(blue.kv[1][IS_OPP], 1.0);

        // Orange view: the flags are swapped.
        let orange = &obs[1];
        assert_eq!(orange.kv[0][IS_MATE], 0.0);
        assert_eq!(orange.kv[0][IS_OPP], 1.0);
        assert_eq!(orange.kv[1][IS_MATE], 1.0);
        assert_eq!(orange.kv[1][IS_OPP], 0.0);
        assert_eq!(orange.q[IS_SELF], 1.0);
    }

    #[test]
    fn previous_actions_are_written_into_the_query() {
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(1, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(0.0, 0.0, 92.0));
        let builder = NextoObsBuilder::default();

        let action = NextoAction::from_controls(&CarControls {
            throttle: 1.0,
            steer: -1.0,
            pitch: 0.5,
            yaw: -0.5,
            roll: 0.0,
            jump: true,
            boost: true,
            handbrake: false,
        });
        let obs = builder.build(&state, &[action, NextoAction::ZERO]);

        assert_eq!(obs[0].q[ACTIONS], action.controls());
        assert_eq!(obs[1].q[ACTIONS], NextoAction::ZERO.controls());
        // The selectors are untouched by the action.
        assert_eq!(obs[0].q[IS_SELF], 1.0);
    }

    #[test]
    fn build_for_player_sets_only_that_players_action() {
        let cars = vec![
            car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
            car(1, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0),
        ];
        let state = state_with(cars, Vec3A::new(0.0, 0.0, 92.0));
        let builder = NextoObsBuilder::default();

        let action = NextoAction::from_controls(&CarControls {
            steer: 1.0,
            handbrake: true,
            ..CarControls::default()
        });
        let obs = builder.build_for_player(&state, 0, action);
        assert_eq!(obs.q[ACTIONS], action.controls());
        assert_eq!(obs.q.len(), Q_LEN);
    }

    #[test]
    fn pad_cooldown_uses_state_seconds() {
        let cars = vec![car(0, Team::Blue, Vec3A::new(0.0, 0.0, 17.0), 0.0)];
        let mut state = state_with(cars, Vec3A::new(0.0, 0.0, 92.0));
        state.boost_pads[0].1 = BoostPadState { cooldown: 10.0 };

        let builder = NextoObsBuilder::default();
        let obs = builder.build(&state, &[NextoAction::ZERO; 1]);

        let boost = &obs[0].kv[2];
        assert_approx(boost[DEMO], 10.0);
    }
}
