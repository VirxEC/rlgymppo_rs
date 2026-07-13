use std::iter::repeat_n;

use rand::seq::SliceRandom;
use rlgym::rocketsim::{CarState, PhysState, Team, consts};
use rlgym::{FullObs, GameState, Obs};

use crate::shared_info::SharedInfoRng;

pub struct DefaultObs<const MAX_PLAYERS_PER_TEAM: usize>;

impl<const MAX_TEAM_PLAYERS: usize> DefaultObs<MAX_TEAM_PLAYERS> {
    const CAR_OBS: usize = 22;
    const BALL_OBS: usize = 9;

    const POS_COEF: f32 = 1.0 / 6000.0;
    const VEL_COEF: f32 = 1.0 / consts::ball::MAX_SPEED;
    const ANG_VEL_COEF: f32 = 1.0 / consts::ball::MAX_ANG_SPEED;
    const BOOST_COEF: f32 = 1.0 / consts::car::boost::MAX;
    const DEMO_COEF: f32 = 1.0 / consts::car::spawn::RESPAWN_TIME;

    /// Compute the full observation size in floats.
    const fn obs_space(&self) -> usize {
        Self::BALL_OBS + Self::CAR_OBS * MAX_TEAM_PLAYERS * 2
    }

    /// Build a car's observation fields as a `Vec<f32>`.
    ///
    /// Observation layout per car (22 floats):
    ///   pos(3) + forward(3) + up(3) + vel(3) + ang_vel(3)
    ///   + boost(1) + isOnGround(1) + hasFlipOrJump(1)
    ///   + isAutoFlipping(1) + airTimeSinceJump(1)
    ///   + handbrake(1) + demoRespawnTimer(1)
    ///
    /// When `car.is_demoed` is true, all fields are zeroed except the last
    /// (`demo_respawn_timer * DEMO_COEF`).
    #[inline]
    fn get_car_obs<const INVERT: bool>(car: &CarState) -> Vec<f32> {
        let mut obs = Vec::with_capacity(Self::CAR_OBS);

        if car.is_demoed {
            obs.extend(repeat_n(0.0, Self::CAR_OBS - 1));
            obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
        } else {
            let phys = if INVERT { car.phys.flip_y() } else { car.phys };

            // Position (3)
            obs.extend((phys.pos * Self::POS_COEF).to_array());
            // Forward direction (rot_mat.x_axis) (3)
            obs.extend(phys.rot_mat.x_axis.to_array());
            // Up direction (rot_mat.z_axis) (3)
            obs.extend(phys.rot_mat.z_axis.to_array());
            // Velocity (3)
            obs.extend((phys.vel * Self::VEL_COEF).to_array());
            // Angular velocity (3)
            obs.extend((phys.ang_vel * Self::ANG_VEL_COEF).to_array());
            // Scalar state (7) – last field is the demo respawn timer
            obs.push(car.boost * Self::BOOST_COEF);
            obs.push(f32::from(car.is_on_ground));
            obs.push(f32::from(car.has_flip_or_jump()));
            obs.push(f32::from(car.is_auto_flipping));
            obs.push(car.air_time_since_jump);
            obs.push(car.handbrake_val);
            obs.push(car.demo_respawn_timer * Self::DEMO_COEF);
        }

        assert_eq!(obs.len(), Self::CAR_OBS);
        obs
    }

    /// Build a ball's observation fields as a `Vec<f32>`.
    ///
    /// Observation layout: pos(3) + vel(3) + ang_vel(3) = 9 floats.
    fn get_ball_obs<const INVERT: bool>(phys: PhysState) -> Vec<f32> {
        let mut obs = Vec::with_capacity(Self::BALL_OBS);

        let phys = if INVERT { phys.flip_y() } else { phys };
        obs.extend((phys.pos * Self::POS_COEF).to_array());
        obs.extend((phys.vel * Self::VEL_COEF).to_array());
        obs.extend((phys.ang_vel * Self::ANG_VEL_COEF).to_array());

        assert_eq!(obs.len(), Self::BALL_OBS);
        obs
    }
}

impl<const MAX_TEAM_PLAYERS: usize, SI: SharedInfoRng> Obs<SI> for DefaultObs<MAX_TEAM_PLAYERS> {
    fn get_obs_space(&self, _shared_info: &SI) -> usize {
        self.obs_space()
    }

    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn build_obs(&mut self, state: &GameState, shared_info: &mut SI) -> FullObs {
        let num_cars = state.cars.len();
        let mut obs = Vec::with_capacity(num_cars);

        let car_obs: Vec<Vec<f32>> = state
            .cars
            .iter()
            .map(|(_, car)| Self::get_car_obs::<false>(car))
            .collect();
        let car_obs_inverted: Vec<Vec<f32>> = state
            .cars
            .iter()
            .map(|(_, car)| Self::get_car_obs::<true>(car))
            .collect();

        let ball_obs = Self::get_ball_obs::<false>(state.ball.phys);
        let ball_obs_inverted = Self::get_ball_obs::<true>(state.ball.phys);

        let car_pad_obs = vec![0.0; Self::CAR_OBS];
        let mut cars: Vec<Vec<f32>> = Vec::new();

        for (i, (info, _car_state)) in state.cars.iter().enumerate() {
            let invert = info.team == Team::Orange;

            // Ball
            let ball = if invert {
                &ball_obs_inverted
            } else {
                &ball_obs
            };

            let mut obs_vec: Vec<f32> = Vec::with_capacity(self.obs_space());

            obs_vec.extend(ball);
            assert_eq!(obs_vec.len(), Self::BALL_OBS);

            // Current car
            let car_obs = if invert { &car_obs_inverted } else { &car_obs };
            obs_vec.extend(&car_obs[i]);

            // Teammates
            for (j, (other_info, _)) in state.cars.iter().enumerate() {
                if other_info.idx == info.idx {
                    continue;
                }

                if other_info.team == info.team {
                    cars.push(car_obs[j].clone());
                }
            }

            while cars.len() < MAX_TEAM_PLAYERS - 1 {
                cars.push(car_pad_obs.clone());
            }

            cars.shuffle(shared_info.rng());
            for tm_obs in &cars {
                obs_vec.extend(tm_obs);
            }

            cars.clear();

            // Opponents
            for (j, (other_info, _)) in state.cars.iter().enumerate() {
                if other_info.team != info.team {
                    cars.push(car_obs[j].clone());
                }
            }

            while cars.len() < MAX_TEAM_PLAYERS {
                cars.push(car_pad_obs.clone());
            }

            cars.shuffle(shared_info.rng());
            for op_obs in &cars {
                obs_vec.extend(op_obs);
            }

            cars.clear();

            assert_eq!(obs_vec.len(), self.obs_space());
            obs.push(obs_vec);
        }

        obs
    }
}
