//! Nexto discrete action space.
//!
//! The policy outputs one logit per discrete action; the action with the
//! highest logit is played. Each action maps to 8 RocketSim controls:
//! `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`.
//!
//! The table mirrors the upstream Nexto action table.

use rlgym::rocketsim::CarControls;

/// Number of discrete actions in the Nexto action space.
pub const ACTION_COUNT: usize = 90;

/// Maps a discrete action index to 8 controls:
/// `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`.
const TABLE: [[f32; 8]; ACTION_COUNT] = [
    [-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [-1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, -1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, -1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, -1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, -1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, -1.0, 1.0, -1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
];

/// One discrete Nexto action: 8 RocketSim controls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NextoAction([f32; 8]);

impl NextoAction {
    /// The zero action: every control is off.
    pub const ZERO: Self = Self([0.0; 8]);

    /// The action at `index` in the fixed 90-action table.
    ///
    /// Panics if `index` is out of range.
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        assert!(index < ACTION_COUNT, "Nexto action index out of range");
        Self(TABLE[index])
    }

    /// Convert RocketSim car controls to a [`NextoAction`].
    ///
    /// Jump, boost, and handbrake become 1.0 or 0.0.
    #[must_use]
    pub fn from_controls(controls: &CarControls) -> Self {
        Self(previous_action(controls))
    }

    /// Return the 8 control values used in the Nexto observation.
    #[must_use]
    pub const fn controls(self) -> [f32; 8] {
        self.0
    }

    /// Convert back to RocketSim car controls.
    ///
    /// Jump, boost, and handbrake are `true` when their value is greater
    /// than 0. This is the inverse of [`NextoAction::from_controls`].
    #[must_use]
    pub fn to_car_controls(self) -> CarControls {
        CarControls {
            throttle: self.0[0],
            steer: self.0[1],
            pitch: self.0[2],
            yaw: self.0[3],
            roll: self.0[4],
            jump: self.0[5] > 0.0,
            boost: self.0[6] > 0.0,
            handbrake: self.0[7] > 0.0,
        }
    }
}

impl Default for NextoAction {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Convert RocketSim car controls to the 8 previous-action floats.
///
/// Jump, boost, and handbrake become 1.0 or 0.0.
#[must_use]
pub fn previous_action(controls: &CarControls) -> [f32; 8] {
    [
        controls.throttle,
        controls.steer,
        controls.pitch,
        controls.yaw,
        controls.roll,
        f32::from(controls.jump),
        f32::from(controls.boost),
        f32::from(controls.handbrake),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_90_actions_of_8_controls() {
        assert_eq!(TABLE.len(), ACTION_COUNT);
        for (index, action) in TABLE.iter().enumerate() {
            assert_eq!(action.len(), 8, "action {index}");
            for &value in action {
                assert!(
                    (-1.0..=1.0).contains(&value),
                    "action {index} has out-of-range control {value}"
                );
            }
        }
    }

    #[test]
    fn table_matches_nexto_python_generation() {
        // Ground actions: throttle x steer x boost x handbrake, skipping
        // boost without full throttle, with `throttle or boost` clamping.
        let mut expected: Vec<[f32; 8]> = Vec::new();
        for throttle in [-1.0, 0.0, 1.0] {
            for steer in [-1.0, 0.0, 1.0] {
                for boost in [0.0, 1.0] {
                    for handbrake in [0.0, 1.0] {
                        if boost == 1.0 && throttle != 1.0 {
                            continue;
                        }
                        expected.push([
                            if throttle != 0.0 { throttle } else { boost },
                            steer,
                            0.0,
                            steer,
                            0.0,
                            0.0,
                            boost,
                            handbrake,
                        ]);
                    }
                }
            }
        }
        // Aerial actions: pitch x yaw x roll x jump x boost, skipping
        // jump-with-yaw, duplicates of ground, with handbrake on air rolls.
        for pitch in [-1.0, 0.0, 1.0] {
            for yaw in [-1.0, 0.0, 1.0] {
                for roll in [-1.0, 0.0, 1.0] {
                    for jump in [0.0, 1.0] {
                        for boost in [0.0, 1.0] {
                            if jump == 1.0 && yaw != 0.0 {
                                continue;
                            }
                            if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                                continue;
                            }
                            let handbrake =
                                jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);
                            expected.push([
                                boost,
                                yaw,
                                pitch,
                                yaw,
                                roll,
                                jump,
                                boost,
                                f32::from(handbrake),
                            ]);
                        }
                    }
                }
            }
        }
        assert_eq!(expected.len(), ACTION_COUNT);
        assert_eq!(&expected[..], &TABLE[..]);
    }

    #[test]
    fn table_entries_round_trip_through_car_controls() {
        for (index, action) in TABLE.iter().enumerate() {
            let nexto = NextoAction::from_index(index);
            let controls = nexto.to_car_controls();
            let back = NextoAction::from_controls(&controls);
            // Jump/boost/handbrake are 0.0 or 1.0 in the table, so the
            // boolean threshold round-trips exactly.
            assert_eq!(back.controls(), *action, "action {index}");
        }
    }

    #[test]
    fn previous_action_and_controls_round_trip() {
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
        assert_eq!(
            previous_action(&controls),
            [1.0, -1.0, 0.5, -0.5, 0.25, 1.0, 1.0, 1.0]
        );

        let action = NextoAction::from_controls(&controls);
        assert_eq!(
            action.controls(),
            [1.0, -1.0, 0.5, -0.5, 0.25, 1.0, 1.0, 1.0]
        );

        let back = action.to_car_controls();
        // CarControls is a packed struct: copy fields before comparing.
        let (throttle, steer, jump, boost, handbrake) = (
            back.throttle,
            back.steer,
            back.jump,
            back.boost,
            back.handbrake,
        );
        assert_eq!(throttle, 1.0);
        assert_eq!(steer, -1.0);
        assert!(jump);
        assert!(boost);
        assert!(handbrake);

        let off = NextoAction::ZERO.to_car_controls();
        let (jump, boost, handbrake) = (off.jump, off.boost, off.handbrake);
        assert!(!jump && !boost && !handbrake);
    }

    #[test]
    fn zero_is_the_default() {
        assert_eq!(NextoAction::default(), NextoAction::ZERO);
        assert_eq!(NextoAction::ZERO.controls(), [0.0; 8]);
    }

    #[test]
    fn from_index_returns_the_table_row() {
        assert_eq!(NextoAction::from_index(0).controls(), TABLE[0]);
        assert_eq!(
            NextoAction::from_index(ACTION_COUNT - 1).controls(),
            TABLE[ACTION_COUNT - 1]
        );
    }

    #[test]
    #[should_panic]
    fn from_index_rejects_out_of_range() {
        let _ = NextoAction::from_index(ACTION_COUNT);
    }
}
