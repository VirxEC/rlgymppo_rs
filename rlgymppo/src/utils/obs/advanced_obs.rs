use std::iter::repeat_n;

use rand::seq::SliceRandom;
use rlgym::{
    FullObs, GameState, Obs,
    rocketsim::{CarState, PhysState, Team, Vec3A, consts},
};

use crate::utils::shared_info::SharedInfoRng;

/// Advanced observation builder, ported from GigaLearn's `AdvancedObs`.
///
/// Extends the default observation with:
/// - Local-space angular velocity
/// - Local-space ball position and velocity (relative to each car)
/// - Boost pad blending (smooth timer-based values instead of binary)
/// - `hasJumped` flag for flip-reset detection
///
/// Observation layout per car (29 floats):
///   pos(3) + forward(3) + up(3) + vel(3) + ang_vel(3)
///   + local_ang_vel(3) + local_ball_pos(3) + local_ball_vel(3)
///   + boost(1) + isOnGround(1) + hasFlipOrJump(1) + demo_respawn_timer(1) + hasJumped(1)
///
/// Teammate and opponent lists are padded to `(MAX_PLAYERS - 1)` / `MAX_PLAYERS`
/// and then shuffled using `SI::rng()` to prevent slot-position bias.
///
/// Demoed cars have all observation fields zeroed except the last
/// (`demo_respawn_timer * DEMO_COEF`).
pub struct AdvancedObs<const MAX_PLAYERS: usize> {
    /// Position scaling factor (default: [`Self::POS_COEF`])
    pos_coef: Vec3A,
    /// Velocity scaling factor (default: [`Self::VEL_COEF`])
    vel_coef: f32,
    /// Angular velocity scaling factor (default: [`Self::ANG_VEL_COEF`])
    ang_vel_coef: f32,
}

impl<const MAX_PLAYERS: usize> AdvancedObs<MAX_PLAYERS> {
    pub const CAR_OBS: usize = 29;
    pub const BALL_OBS: usize = 9;
    pub const ACTION_OBS: usize = 8;
    pub const BOOST_PADS_OBS: usize = 34;

    pub const POS_COEF: Vec3A = Vec3A::new(1.0 / 4096.0, 1.0 / 5120.0, 1.0 / 2044.0);
    pub const VEL_COEF: f32 = 1.0 / consts::ball::MAX_SPEED;
    pub const ANG_VEL_COEF: f32 = 1.0 / consts::ball::MAX_ANG_SPEED;
    pub const BOOST_COEF: f32 = 1.0 / consts::car::boost::MAX;
    pub const DEMO_COEF: f32 = 1.0 / consts::car::spawn::RESPAWN_TIME;

    /// Create a new `AdvancedObs` with default values matching GigaLearn's `AdvancedObs`.
    pub fn new() -> Self {
        Self {
            pos_coef: Self::POS_COEF,
            vel_coef: Self::VEL_COEF,
            ang_vel_coef: Self::ANG_VEL_COEF,
        }
    }

    /// Set the position scaling factor (default: [`Self::POS_COEF`]).
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

    /// Append a car's AdvancedObs fields to `obs`.
    ///
    /// `ball_phys` is the (already-flipped) ball state used to compute local
    /// ball position and velocity relative to this car.
    ///
    /// When `car.is_demoed` is true, all fields are zeroed except the last
    /// (`demo_respawn_timer * DEMO_COEF`).
    #[inline]
    fn add_car_obs(
        obs: &mut Vec<f32>,
        phys: &PhysState,
        car: &CarState,
        ball_phys: &PhysState,
        coefs: &Self,
    ) {
        if car.is_demoed {
            obs.extend(repeat_n(0.0, Self::CAR_OBS - 1));
            obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
            return;
        }

        // ── World-space state (15 floats) ───────────────────────────────
        // Position (3)
        obs.extend((phys.pos * coefs.pos_coef).to_array());
        // Forward direction (rot_mat.x_axis) (3)
        obs.extend(phys.rot_mat.x_axis.to_array());
        // Up direction (rot_mat.z_axis) (3)
        obs.extend(phys.rot_mat.z_axis.to_array());
        // Velocity (3)
        obs.extend((phys.vel * coefs.vel_coef).to_array());
        // Angular velocity (3)
        obs.extend((phys.ang_vel * coefs.ang_vel_coef).to_array());

        // ── Local-space state (9 floats) ────────────────────────────────
        // Transform world-space vectors to the car's local frame.
        // rot_mat's transpose gives the inverse rotation (world → local).
        let local_ang_vel = phys.rot_mat.transpose() * phys.ang_vel;
        obs.extend((local_ang_vel * coefs.ang_vel_coef).to_array());

        let local_ball_pos = phys.rot_mat.transpose() * (ball_phys.pos - phys.pos);
        obs.extend((local_ball_pos * coefs.pos_coef).to_array());

        let local_ball_vel = phys.rot_mat.transpose() * (ball_phys.vel - phys.vel);
        obs.extend((local_ball_vel * coefs.vel_coef).to_array());

        // ── Scalar state (5 floats) ─────────────────────────────────────
        obs.push(car.boost * Self::BOOST_COEF);
        obs.push(f32::from(car.is_on_ground));
        obs.push(f32::from(car.has_flip_or_jump()));
        obs.push(f32::from(car.has_jumped));
        obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
    }
}

impl<const MAX_PLAYERS: usize, SI: SharedInfoRng> Obs<SI> for AdvancedObs<MAX_PLAYERS> {
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

            // ── Boost pads (34 floats, blending trick) ──────────────────
            // Active pads → 1.0; inactive pads → 1/(1 + cooldown),
            // smoothly approaching 1 as the pad becomes available.
            let mut pad_count = 0;
            for (_config, pstate) in &state.boost_pads {
                if pad_count >= Self::BOOST_PADS_OBS {
                    break;
                }

                let val = if pstate.is_active() {
                    1.0
                } else {
                    1.0 / (1.0 + pstate.cooldown)
                };

                obs_vec.push(val);
                pad_count += 1;
            }
            obs_vec.extend(repeat_n(0.0, Self::BOOST_PADS_OBS - pad_count));

            // ── Current car (29 floats) ─────────────────────────────────
            let self_phys = if invert {
                car_state.phys.flip_y()
            } else {
                car_state.phys
            };
            Self::add_car_obs(&mut obs_vec, &self_phys, car_state, &ball_phys, self);

            // ── Teammates ───────────────────────────────────────────────
            let mut teammates: Vec<Vec<f32>> = Vec::new();
            for (other_info, other_car) in &state.cars {
                if other_info.idx == info.idx {
                    continue;
                }

                if other_info.team == info.team {
                    let phys = if invert {
                        other_car.phys.flip_y()
                    } else {
                        other_car.phys
                    };

                    let mut tm_obs = Vec::with_capacity(Self::CAR_OBS);
                    Self::add_car_obs(&mut tm_obs, &phys, other_car, &ball_phys, self);
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
                    let phys = if invert {
                        other_car.phys.flip_y()
                    } else {
                        other_car.phys
                    };
                    let mut op_obs = Vec::with_capacity(Self::CAR_OBS);
                    Self::add_car_obs(&mut op_obs, &phys, other_car, &ball_phys, self);
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

impl<const MAX_PLAYERS: usize> Default for AdvancedObs<MAX_PLAYERS> {
    fn default() -> Self {
        Self::new()
    }
}
