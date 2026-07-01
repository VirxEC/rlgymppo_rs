use rlgym::rocketsim::{ArenaEvent, BallState, CarState, Team, Vec3A, consts};
use rlgym::{GameState, Reward};

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

        car_to_ball_dot / consts::car::MAX_SPEED
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

/// Reward for touching the ball.
///
/// Returns 1.0 when the car touches the ball (i.e. a `CarHitBall` event is fired for it)
/// and 0.0 otherwise.
#[derive(Default)]
pub struct BallTouchReward;

impl BallTouchReward {
    fn get_reward(car_idx: usize, car: &CarState, events: &[ArenaEvent]) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        let touched = events
            .iter()
            .any(|event| matches!(event, ArenaEvent::CarHitBall(hit) if hit.car_idx == car_idx));

        if touched { 1.0 } else { 0.0 }
    }
}

impl<SI> Reward<SI> for BallTouchReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| Self::get_reward(info.idx, car, &state.events))
            .collect()
    }
}

/// Reward for velocity in the direction the car is facing.
///
/// Returns the dot product of the car's velocity and its forward vector,
/// normalized by the maximum car speed (2300) so the output is roughly in [-1, 1].
#[derive(Default)]
pub struct VelocityReward;

impl VelocityReward {
    fn get_reward(car: &CarState) -> f32 {
        if car.is_demoed {
            return 0.0;
        }

        car.vel.dot(car.rot_mat.x_axis) / consts::car::MAX_SPEED
    }
}

impl<SI> Reward<SI> for VelocityReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car))
            .collect()
    }
}

/// Reward for being in the air.
///
/// Returns 1.0 when the car is not on the ground, 0.0 otherwise.
/// Based on GGL's `AirReward`.
#[derive(Default)]
pub struct AirReward;

impl AirReward {
    fn get_reward(car: &CarState) -> f32 {
        f32::from(!car.is_on_ground)
    }
}

impl<SI> Reward<SI> for AirReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car))
            .collect()
    }
}

/// Reward for having/conserving boost.
///
/// Returns `(boost / 100)^exponent` clamped to [0, 1], where the default exponent
/// 0.5 produces a diminishing-returns curve. Based on GGL's `SaveBoostReward`.
#[derive(Default)]
pub struct SaveBoostReward {
    pub exponent: f32,
}

impl SaveBoostReward {
    fn get_reward(&self, car: &CarState) -> f32 {
        (car.boost / 100.0).powf(self.exponent).clamp(0.0, 1.0)
    }
}

impl<SI> Reward<SI> for SaveBoostReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| self.get_reward(car))
            .collect()
    }
}

/// Reward for picking up boost.
///
/// Returns `sqrt(boost / 100)` when a `CarPickupBoost` event is fired for the car,
/// 0.0 otherwise. Based on GGL's `PickupBoostReward`.
#[derive(Default)]
pub struct PickupBoostReward;

impl PickupBoostReward {
    fn get_reward(car_idx: usize, car: &CarState, events: &[ArenaEvent]) -> f32 {
        let picked_up = events
            .iter()
            .any(|event| matches!(event, ArenaEvent::CarPickupBoost(p) if p.car_idx == car_idx));

        if picked_up {
            (car.boost / 100.0).sqrt()
        } else {
            0.0
        }
    }
}

impl<SI> Reward<SI> for PickupBoostReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| Self::get_reward(info.idx, car, &state.events))
            .collect()
    }
}

/// Reward for ball velocity towards the opponent's goal.
///
/// Returns the dot product of the ball's velocity (normalized by max ball speed)
/// with the direction from the ball to the opponent's goal face center.
/// Based on GGL's `VelocityBallToGoalReward`.
#[derive(Default)]
pub struct VelocityBallToGoalReward;

impl VelocityBallToGoalReward {
    fn get_reward(ball: &BallState, car_team: Team) -> f32 {
        let target_team = car_team.opposite();
        let goal_center = consts::goal::get_goal_face_center(target_team);
        let dir_to_goal = (goal_center - ball.pos).normalize();
        dir_to_goal.dot(ball.vel) / consts::ball::MAX_SPEED
    }
}

impl<SI> Reward<SI> for VelocityBallToGoalReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, _)| Self::get_reward(&state.ball, info.team))
            .collect()
    }
}

/// Reward for when a team scores a goal.
///
/// Returns 1.0 for cars on the scoring team and `concede_scale` for cars on
/// the conceding team on the tick a goal is first detected. Uses internal state
/// to detect the rising edge of `is_ball_scored()`. Based on GGL's `GoalReward`.
#[derive(Default)]
pub struct GoalReward {
    pub concede_scale: f32,
    prev_scored: bool,
}

impl GoalReward {
    pub fn new(concede_scale: f32) -> Self {
        Self {
            concede_scale,
            prev_scored: false,
        }
    }
}

impl<SI> Reward<SI> for GoalReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {
        self.prev_scored = false;
    }

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        let scored = state.is_ball_scored();
        let new_goal = scored && !self.prev_scored;
        self.prev_scored = scored;

        if !new_goal {
            return vec![0.0; state.cars.len()];
        }

        state
            .cars
            .iter()
            .map(|(info, _)| {
                let conceding_team = Team::from_team_y(state.ball.pos.y);
                if info.team != conceding_team {
                    1.0
                } else {
                    self.concede_scale
                }
            })
            .collect()
    }
}

/// Reward for bumping an opponent.
///
/// Returns 1.0 when the car bumps another car (non-demo `CarHitCar` event),
/// 0.0 otherwise. Based on GGL's `BumpReward`.
#[derive(Default)]
pub struct BumpReward;

impl BumpReward {
    fn get_reward(car_idx: usize, events: &[ArenaEvent]) -> f32 {
        let bumped = events.iter().any(|event| {
            matches!(
                event,
                ArenaEvent::CarHitCar(hit) if hit.bumper_car_idx == car_idx && !hit.is_demo
            )
        });
        f32::from(bumped)
    }
}

impl<SI> Reward<SI> for BumpReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, _)| Self::get_reward(info.idx, &state.events))
            .collect()
    }
}

/// Penalty for being bumped.
///
/// Returns -1.0 when the car is bumped by another car (non-demo `CarHitCar` event),
/// 0.0 otherwise. Based on GGL's `BumpedPenalty`.
#[derive(Default)]
pub struct BumpedPenalty;

impl BumpedPenalty {
    fn get_reward(car_idx: usize, events: &[ArenaEvent]) -> f32 {
        let bumped = events.iter().any(|event| {
            matches!(
                event,
                ArenaEvent::CarHitCar(hit) if hit.victim_car_idx == car_idx && !hit.is_demo
            )
        });
        -f32::from(bumped)
    }
}

impl<SI> Reward<SI> for BumpedPenalty {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, _)| Self::get_reward(info.idx, &state.events))
            .collect()
    }
}

/// Reward for demoing an opponent.
///
/// Returns 1.0 when the car demos another car (`CarHitCar` with `is_demo: true`),
/// 0.0 otherwise. Based on GGL's `DemoReward`.
#[derive(Default)]
pub struct DemoReward;

impl DemoReward {
    fn get_reward(car_idx: usize, events: &[ArenaEvent]) -> f32 {
        let demoed = events.iter().any(|event| {
            matches!(
                event,
                ArenaEvent::CarHitCar(hit) if hit.bumper_car_idx == car_idx && hit.is_demo
            )
        });
        f32::from(demoed)
    }
}

impl<SI> Reward<SI> for DemoReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, _)| Self::get_reward(info.idx, &state.events))
            .collect()
    }
}

/// Penalty for being demoed.
///
/// Returns -1.0 when the car is demoed (checked via `is_demoed` state field),
/// 0.0 otherwise. Based on GGL's `DemoedPenalty`.
#[derive(Default)]
pub struct DemoedPenalty;

impl DemoedPenalty {
    fn get_reward(car: &CarState) -> f32 {
        -f32::from(car.is_demoed)
    }
}

impl<SI> Reward<SI> for DemoedPenalty {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(_, car)| Self::get_reward(car))
            .collect()
    }
}

/// Reward for performing a wavedash.
///
/// Detects a wavedash by checking for the transition: car was flipping
/// and airborne in the previous tick, and is now on the ground.
/// Based on GGL's `WavedashReward`.
#[derive(Default)]
pub struct WavedashReward {
    prev_flipping: Vec<bool>,
}

impl WavedashReward {
    fn get_reward(&mut self, car_idx: usize, car: &CarState) -> f32 {
        let prev_flipping = self.prev_flipping.get(car_idx).copied().unwrap_or(false);

        // Ensure vec is large enough
        if self.prev_flipping.len() <= car_idx {
            self.prev_flipping.resize(car_idx + 1, false);
        }

        // Update for next tick
        self.prev_flipping[car_idx] = car.is_flipping;

        // Wavedash detected: car was flipping in the air, now on ground
        if car.is_on_ground && prev_flipping && !car.is_flipping {
            1.0
        } else {
            0.0
        }
    }
}

impl<SI> Reward<SI> for WavedashReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {
        self.prev_flipping.clear();
    }

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| self.get_reward(info.idx, car))
            .collect()
    }
}

/// Reward for accelerating the ball on touch.
///
/// Returns the increase in ball speed fraction (capped at `MAX_REWARDED_SPEED`)
/// when the car touches the ball, 0.0 otherwise.
/// Based on GGL's `TouchAccelReward`.
#[derive(Default)]
pub struct TouchAccelReward {
    pub max_rewarded_speed: f32,
    prev_ball_vel: Option<Vec3A>,
}

impl TouchAccelReward {
    pub fn new(max_rewarded_speed: f32) -> Self {
        Self {
            max_rewarded_speed,
            prev_ball_vel: None,
        }
    }

    fn get_reward(&mut self, car_idx: usize, car: &CarState, state: &GameState) -> f32 {
        if car.is_demoed {
            self.prev_ball_vel = Some(state.ball.vel);
            return 0.0;
        }

        let touched = state
            .events
            .iter()
            .any(|event| matches!(event, ArenaEvent::CarHitBall(hit) if hit.car_idx == car_idx));

        let reward = if touched {
            if let Some(prev_vel) = self.prev_ball_vel {
                let prev_speed = prev_vel.length();
                let cur_speed = state.ball.vel.length();

                if cur_speed > prev_speed {
                    let prev_frac = (prev_speed / self.max_rewarded_speed).min(1.0);
                    let cur_frac = (cur_speed / self.max_rewarded_speed).min(1.0);
                    cur_frac - prev_frac
                } else {
                    0.0
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.prev_ball_vel = Some(state.ball.vel);
        reward
    }
}

impl<SI> Reward<SI> for TouchAccelReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {
        self.prev_ball_vel = None;
    }

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| self.get_reward(info.idx, car, state))
            .collect()
    }
}

/// Reward for hard/strong touches.
///
/// Returns the hit force fraction (capped at `max_rewarded_speed`) when the car
/// touches the ball and the hit force exceeds `min_rewarded_speed`, 0.0 otherwise.
/// Based on GGL's `StrongTouchReward`.
#[derive(Default)]
pub struct StrongTouchReward {
    pub min_rewarded_speed: f32,
    pub max_rewarded_speed: f32,
    prev_ball_vel: Option<Vec3A>,
}

impl StrongTouchReward {
    pub fn new(min_rewarded_speed: f32, max_rewarded_speed: f32) -> Self {
        Self {
            min_rewarded_speed,
            max_rewarded_speed,
            prev_ball_vel: None,
        }
    }

    fn get_reward(&mut self, car_idx: usize, car: &CarState, state: &GameState) -> f32 {
        if car.is_demoed {
            self.prev_ball_vel = Some(state.ball.vel);
            return 0.0;
        }

        let touched = state
            .events
            .iter()
            .any(|event| matches!(event, ArenaEvent::CarHitBall(hit) if hit.car_idx == car_idx));

        let reward = if touched {
            if let Some(prev_vel) = self.prev_ball_vel {
                let hit_force = (state.ball.vel - prev_vel).length();

                if hit_force < self.min_rewarded_speed {
                    0.0
                } else {
                    (hit_force / self.max_rewarded_speed).min(1.0)
                }
            } else {
                0.0
            }
        } else {
            0.0
        };

        self.prev_ball_vel = Some(state.ball.vel);
        reward
    }
}

impl<SI> Reward<SI> for StrongTouchReward {
    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {
        self.prev_ball_vel = None;
    }

    fn get_rewards(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<f32> {
        state
            .cars
            .iter()
            .map(|(info, car)| self.get_reward(info.idx, car, state))
            .collect()
    }
}
