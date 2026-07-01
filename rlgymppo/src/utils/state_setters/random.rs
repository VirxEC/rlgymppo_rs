use std::f32::consts::{FRAC_PI_2, PI};

use rand::RngExt;
use rlgym::StateSetter;
use rlgym::rocketsim::{Arena, BallState, CarState, Mat3A, Vec3A};

use crate::utils::shared_info::SharedInfoRng;

/// Returns a uniformly distributed random unit vector (rejection-sampled on the unit cube).
fn rand_norm_vec(rng: &mut impl RngExt) -> Vec3A {
    loop {
        let v = Vec3A::new(
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
            rng.random_range(-1.0..1.0),
        );
        let sq_len = v.length_squared();
        if (1e-6..=1.0).contains(&sq_len) {
            return v / sq_len.sqrt();
        }
    }
}

/// Build a rotation matrix from (yaw, pitch, roll) applied in ZYX order.
fn euler_to_mat3(yaw: f32, pitch: f32, roll: f32) -> Mat3A {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();

    Mat3A::from_cols(
        // forward (x_axis)
        Vec3A::new(cy * cp, sy * cp, -sp),
        // right (y_axis)
        Vec3A::new(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr),
        // up (z_axis)
        Vec3A::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr),
    )
}

/// A state setter that randomizes the ball and car states.
///
/// `RAND_BALL_SPEED`: Whether to randomize the ball's velocity and angular velocity.
/// `RAND_CAR_SPEED`: Whether to randomize the cars' velocities and angular velocities.
/// `CARS_ON_GROUND`: Whether to force all cars to be on the ground (i.e. zero pitch and roll, z=17, zero vertical velocity and angular velocity).
#[derive(Default)]
pub struct RandomState<
    const RAND_BALL_SPEED: bool,
    const RAND_CAR_SPEED: bool,
    const CARS_ON_GROUND: bool,
>;

impl<
    const RAND_BALL_SPEED: bool,
    const RAND_CAR_SPEED: bool,
    const CARS_ON_GROUND: bool,
    SI: SharedInfoRng,
> StateSetter<SI> for RandomState<RAND_BALL_SPEED, RAND_CAR_SPEED, CARS_ON_GROUND>
{
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        // Reset boost pads and everything
        arena.reset_to_random_kickoff(None);

        const X_MAX: f32 = 3500.0;
        const Y_MAX: f32 = 4000.0;
        const Z_MAX: f32 = 1820.0;
        const CAR_Z_MIN: f32 = 150.0;
        const PITCH_MAX: f32 = FRAC_PI_2;
        const YAW_MAX: f32 = PI;
        const ROLL_MAX: f32 = PI;
        const ANGVEL_MAX: f32 = 5.5;
        const CAR_MAX_SPEED: f32 = 2300.0;

        let rng = shared_info.rng();
        let ball_radius = arena.get_config().mutators.ball_radius;

        // Randomize ball ------------------------------------------------------
        let mut ball_state = BallState::DEFAULT;
        ball_state.phys.pos = Vec3A::new(
            rng.random_range(-X_MAX..X_MAX),
            rng.random_range(-Y_MAX..Y_MAX),
            rng.random_range(ball_radius..Z_MAX),
        );
        if RAND_BALL_SPEED {
            ball_state.phys.vel = rand_norm_vec(rng) * rng.random_range(0.0..4000.0);
            ball_state.phys.ang_vel = Vec3A::new(
                rng.random_range(-4.0..4.0),
                rng.random_range(-4.0..4.0),
                rng.random_range(-4.0..4.0),
            );
        }
        arena.set_ball_state(ball_state);

        // Randomize cars ------------------------------------------------------
        for car_idx in 0..arena.num_cars() {
            let mut car_state = CarState::DEFAULT;
            car_state.phys.pos = Vec3A::new(
                rng.random_range(-X_MAX..X_MAX),
                rng.random_range(-Y_MAX..Y_MAX),
                rng.random_range(CAR_Z_MIN..Z_MAX),
            );

            if RAND_CAR_SPEED {
                car_state.phys.vel = rand_norm_vec(rng) * rng.random_range(0.0..CAR_MAX_SPEED);
                car_state.phys.ang_vel = rand_norm_vec(rng) * ANGVEL_MAX;
            }

            let yaw = rng.random_range(-YAW_MAX..YAW_MAX);
            let pitch = rng.random_range(-PITCH_MAX..PITCH_MAX);
            let roll = rng.random_range(-ROLL_MAX..ROLL_MAX);

            let on_ground = CARS_ON_GROUND || rng.random_bool(0.5);
            let (pitch, roll) = if on_ground { (0.0, 0.0) } else { (pitch, roll) };

            if on_ground {
                car_state.phys.pos.z = 17.0;
                car_state.phys.vel.z = 0.0;
                car_state.phys.ang_vel = Vec3A::ZERO;
            }

            car_state.phys.rot_mat = euler_to_mat3(yaw, pitch, roll);
            car_state.boost = rng.random_range(0.0..100.0);

            arena.set_car_state(car_idx, car_state);
        }
    }
}
