use std::{iter::repeat_n, marker::PhantomData};

use rand::seq::SliceRandom;
use rlgym::{
    FullObs, GameState, Obs,
    rocketsim::{CarState, PhysState, Team, Vec3A, consts},
};

use crate::utils::shared_info::SharedInfoRng;

/// Default observation builder, ported from GigaLearn's `DefaultObs`/`DefaultObsPadded`.
///
/// Builds a fixed-size per-player observation vector consisting of:
/// - Ball state (pos, vel, ang_vel) – 9 floats
/// - Previous action – 8 floats
/// - Boost pad states – 34 floats
/// - Current car state – 19 floats
/// - Teammate car states – `(MAX_PLAYERS - 1) * 19` floats (padded with zeros, then shuffled)
/// - Opponent car states – `MAX_PLAYERS * 19` floats (padded with zeros, then shuffled)
///
/// Orange-team observations are mirrored (Y-axis flip) so all observations
/// use a consistent coordinate system (blue-team perspective).
///
/// Demoed cars have all observation fields zeroed except the last
/// (`demo_respawn_timer * DEMO_COEF`), matching the masking pattern from MyObs.
pub struct DefaultObs<const MAX_PLAYERS: usize, SI> {
    /// Component-wise position scaling factor (default: `[1/4096, 1/5120, 1/2044]`)
    pos_coef: Vec3A,
    /// Velocity scaling factor (default: [`Self::VEL_COEF`])
    vel_coef: f32,
    /// Angular velocity scaling factor (default: [`Self::ANG_VEL_COEF`])
    ang_vel_coef: f32,
    _phantom: PhantomData<SI>,
}

impl<const MAX_PLAYERS: usize, SI> DefaultObs<MAX_PLAYERS, SI> {
    pub const CAR_OBS: usize = 19;
    pub const BALL_OBS: usize = 9;
    pub const ACTION_OBS: usize = 8;
    pub const BOOST_PADS_OBS: usize = 34;

    pub const POS_COEF: Vec3A = Vec3A::new(1.0 / 4096.0, 1.0 / 5120.0, 1.0 / 2044.0);
    pub const VEL_COEF: f32 = 1.0 / consts::ball::MAX_SPEED;
    pub const ANG_VEL_COEF: f32 = 1.0 / consts::ball::MAX_ANG_SPEED;
    pub const BOOST_COEF: f32 = 1.0 / consts::car::boost::MAX;
    pub const DEMO_COEF: f32 = 1.0 / consts::car::spawn::RESPAWN_TIME;

    /// Create a new `DefaultObs` with default values matching GigaLearn's `DefaultObs`.
    pub fn new() -> Self {
        Self {
            pos_coef: Self::POS_COEF,
            vel_coef: Self::VEL_COEF,
            ang_vel_coef: Self::ANG_VEL_COEF,
            _phantom: PhantomData,
        }
    }

    /// Set the component-wise position scaling factor (default: `[1/4096, 1/5120, 1/2044]`).
    pub fn with_pos_coef(mut self, coef: Vec3A) -> Self {
        self.pos_coef = coef;
        self
    }

    /// Set the velocity scaling factor (default: [`Self::VEL_COEF`]).
    pub fn with_vel_coef(mut self, coef: f32) -> Self {
        self.vel_coef = coef;
        self
    }

    /// Set the angular velocity scaling factor (default: [`Self::ANG_VEL_COEF`]).
    pub fn with_ang_vel_coef(mut self, coef: f32) -> Self {
        self.ang_vel_coef = coef;
        self
    }

    /// Compute the full observation size in floats.
    fn obs_space(&self) -> usize {
        Self::BALL_OBS
            + Self::ACTION_OBS
            + Self::BOOST_PADS_OBS
            + Self::CAR_OBS
            + Self::CAR_OBS * (MAX_PLAYERS.saturating_sub(1))
            + Self::CAR_OBS * MAX_PLAYERS
    }

    /// Append a car's observation fields to `obs`.
    ///
    /// Observation layout per car (19 floats):
    ///   pos(3) + forward(3) + up(3) + vel(3) + ang_vel(3)
    ///   + boost(1) + isOnGround(1) + hasFlipOrJump(1) + demo_respawn_timer(1)
    ///
    /// When `car.is_demoed` is true, all fields are zeroed except the last
    /// (`demo_respawn_timer * DEMO_COEF`).
    #[inline]
    fn add_car_obs(obs: &mut Vec<f32>, phys: &PhysState, car: &CarState) {
        if car.is_demoed {
            obs.extend(repeat_n(0.0, Self::CAR_OBS - 1));
            obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
            return;
        }

        // Position (3)
        obs.extend(phys.pos.to_array());
        // Forward direction (rot_mat.x_axis) (3)
        obs.extend(phys.rot_mat.x_axis.to_array());
        // Up direction (rot_mat.z_axis) (3)
        obs.extend(phys.rot_mat.z_axis.to_array());
        // Velocity (3)
        obs.extend(phys.vel.to_array());
        // Angular velocity (3)
        obs.extend(phys.ang_vel.to_array());
        // Scalar state (5) – last field is the demo respawn timer
        obs.push(car.boost * Self::BOOST_COEF);
        obs.push(f32::from(car.is_on_ground));
        obs.push(f32::from(car.has_flip_or_jump()));
        obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
    }

    /// Build a scaled, optionally flipped, `PhysState` for a car.
    #[inline]
    fn build_car_phys(car: &CarState, invert: bool, coefs: &Self) -> PhysState {
        let phys = if invert { car.phys.flip_y() } else { car.phys };
        PhysState {
            pos: phys.pos * coefs.pos_coef,
            vel: phys.vel * coefs.vel_coef,
            ang_vel: phys.ang_vel * coefs.ang_vel_coef,
            ..phys
        }
    }
}

impl<const MAX_PLAYERS: usize, SI: SharedInfoRng> Obs<SI> for DefaultObs<MAX_PLAYERS, SI> {
    fn get_obs_space(&self, _shared_info: &SI) -> usize {
        self.obs_space()
    }

    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn build_obs(&mut self, state: &GameState, shared_info: &mut SI) -> FullObs {
        let num_cars = state.cars.len();
        let mut obs = Vec::with_capacity(num_cars);

        for (info, car_state) in &state.cars {
            let invert = info.team == Team::Orange;

            // ── Ball (flipped for orange team) ──────────────────────────
            let ball_phys = if invert {
                state.ball.phys.flip_y()
            } else {
                state.ball.phys
            };

            let mut obs_vec: Vec<f32> = Vec::with_capacity(self.obs_space());

            // Ball: pos(3) + vel(3) + ang_vel(3) = 9
            obs_vec.extend((ball_phys.pos * self.pos_coef).to_array());
            obs_vec.extend((ball_phys.vel * self.vel_coef).to_array());
            obs_vec.extend((ball_phys.ang_vel * self.ang_vel_coef).to_array());

            // ── Previous action (8 floats) ──────────────────────────────
            obs_vec.extend(car_state.prev_controls.to_floats());

            // ── Boost pads (34 floats) ──────────────────────────────────
            let mut pad_count = 0;
            for (_config, pstate) in &state.boost_pads {
                if pad_count >= Self::BOOST_PADS_OBS {
                    break;
                }

                obs_vec.push(f32::from(pstate.is_active()));
                pad_count += 1;
            }
            obs_vec.extend(repeat_n(0.0, Self::BOOST_PADS_OBS - pad_count));

            // ── Current car (19 floats) ─────────────────────────────────
            let self_phys = Self::build_car_phys(car_state, invert, self);
            Self::add_car_obs(&mut obs_vec, &self_phys, car_state);

            // ── Teammates ───────────────────────────────────────────────
            let mut teammates: Vec<Vec<f32>> = Vec::new();
            for (other_info, other_car) in &state.cars {
                if other_info.idx == info.idx {
                    continue;
                }
                if other_info.team == info.team {
                    let phys = Self::build_car_phys(other_car, invert, self);
                    let mut tm_obs = Vec::with_capacity(Self::CAR_OBS);
                    Self::add_car_obs(&mut tm_obs, &phys, other_car);
                    teammates.push(tm_obs);
                }
            }

            // Pad to (MAX_PLAYERS - 1) then shuffle to prevent slot bias
            while teammates.len() < MAX_PLAYERS.saturating_sub(1) {
                teammates.push(vec![0.0; Self::CAR_OBS]);
            }

            teammates.shuffle(shared_info.rng());
            for tm_obs in teammates {
                obs_vec.extend(tm_obs);
            }

            // ── Opponents ───────────────────────────────────────────────
            let mut opponents: Vec<Vec<f32>> = Vec::new();
            for (other_info, other_car) in &state.cars {
                if other_info.team != info.team {
                    let phys = Self::build_car_phys(other_car, invert, self);
                    let mut op_obs = Vec::with_capacity(Self::CAR_OBS);
                    Self::add_car_obs(&mut op_obs, &phys, other_car);
                    opponents.push(op_obs);
                }
            }

            // Pad to MAX_PLAYERS then shuffle
            while opponents.len() < MAX_PLAYERS {
                opponents.push(vec![0.0; Self::CAR_OBS]);
            }

            opponents.shuffle(shared_info.rng());
            for op_obs in opponents {
                obs_vec.extend(op_obs);
            }

            debug_assert_eq!(obs_vec.len(), self.obs_space());
            obs.push(obs_vec);
        }

        obs
    }
}

impl<const MAX_PLAYERS: usize, SI> Default for DefaultObs<MAX_PLAYERS, SI> {
    fn default() -> Self {
        Self::new()
    }
}
