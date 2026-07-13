use rlbot_rocketsim::rlbot::flat::ControllerState;
use rlgymppo_utils::rocketsim::CarControls;

#[must_use]
pub fn to_rlbot_controls(controls: CarControls) -> ControllerState {
    ControllerState {
        throttle: controls.throttle,
        steer: controls.steer,
        pitch: controls.pitch,
        yaw: controls.yaw,
        roll: controls.roll,
        jump: controls.jump,
        boost: controls.boost,
        handbrake: controls.handbrake,
        use_item: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_all_shared_control_fields() {
        let controls = CarControls {
            throttle: 1.0,
            steer: -1.0,
            pitch: 0.5,
            yaw: -0.5,
            roll: 0.25,
            jump: true,
            boost: true,
            handbrake: true,
        };
        let expected_throttle = controls.throttle;
        let expected_steer = controls.steer;
        let expected_pitch = controls.pitch;
        let expected_yaw = controls.yaw;
        let expected_roll = controls.roll;
        let converted = to_rlbot_controls(controls);

        let throttle = converted.throttle;
        let steer = converted.steer;
        let pitch = converted.pitch;
        let yaw = converted.yaw;
        let roll = converted.roll;
        assert_eq!(throttle, expected_throttle);
        assert_eq!(steer, expected_steer);
        assert_eq!(pitch, expected_pitch);
        assert_eq!(yaw, expected_yaw);
        assert_eq!(roll, expected_roll);
        assert_eq!(converted.jump, controls.jump);
        assert_eq!(converted.boost, controls.boost);
        assert_eq!(converted.handbrake, controls.handbrake);
        assert!(!converted.use_item);
    }
}
