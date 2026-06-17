use rlgym::{
    GameState, Reward,
    rocketsim::{BallState, CarState},
};

/// Reward for how well the car is facing the ball.
///
/// Returns the dot product of the car's forward vector and the direction to the ball,
/// normalized so that facing directly towards the ball gives 1.0 and facing away gives -1.0.
#[derive(Default)]
pub struct FaceBallReward;

impl FaceBallReward {
    fn get_reward(car: &CarState, ball: &BallState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let car_to_ball = (ball.pos - car.pos).normalize();
        car.rot_mat.x_axis.dot(car_to_ball)
    }
}

impl<SI> Reward<SI> for FaceBallReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car, &state.ball))
            .collect()
    }
}

/// Reward for velocity towards the ball.
///
/// Returns the dot product of the car's velocity and the normalized direction to the ball,
/// scaled by the maximum car speed (2300) so the output is roughly in [-1, 1].
#[derive(Default)]
pub struct VelocityToBallReward;

impl VelocityToBallReward {
    fn get_reward(car: &CarState, ball: &BallState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let car_to_ball_norm = (ball.pos - car.pos).normalize_or_zero();
        let car_to_ball_dot = car.vel.dot(car_to_ball_norm);

        car_to_ball_dot / 2300.0
    }
}

impl<SI> Reward<SI> for VelocityToBallReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car, &state.ball))
            .collect()
    }
}
