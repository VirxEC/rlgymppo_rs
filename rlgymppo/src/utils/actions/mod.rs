use arrayvec::ArrayVec;
use rlgym::{
    Action, GameState,
    rocketsim::{CarControls, CarState},
};

/// Generates a discrete action space table with 90 actions:
/// - 24 ground actions (throttle × steer × boost × handbrake, with filter)
/// - 66 aerial actions (pitch × yaw × roll × jump × boost, with filters)
///
/// Precomputes ground/air/jump/boost masks and provides `get_action_mask()`
/// to produce a per-car valid-action mask at inference time.
///
/// `MAX_NUM_AGENTS` is used to size the internal action buffer, which is
/// reused across ticks to avoid allocations. It should be set to the maximum
/// number of agents that will be controlled by this action parser.
pub struct DefaultAction<const MAX_NUM_AGENTS: usize, const TICK_SKIP: u8> {
    actions_table: Vec<CarControls>,
    ground_mask: Vec<bool>,
    air_mask: Vec<bool>,
    jump_mask: Vec<bool>,
    boost_mask: Vec<bool>,
    action_buffer: ArrayVec<(usize, CarControls), MAX_NUM_AGENTS>,
}

impl<const MAX_NUM_AGENTS: usize, const TICK_SKIP: u8> DefaultAction<MAX_NUM_AGENTS, TICK_SKIP> {
    pub fn new() -> Self {
        let mut actions_table = Vec::new();

        // ── Ground actions ──────────────────────────────────────────────
        for &throttle in &[-1.0, 0.0, 1.0] {
            for &steer in &[-1.0, 0.0, 1.0] {
                for &boost in &[false, true] {
                    for &handbrake in &[false, true] {
                        // Prevent useless throttle when boosting
                        if boost && throttle != 1.0 {
                            continue;
                        }

                        actions_table.push(CarControls {
                            throttle,
                            steer,
                            pitch: 0.0,
                            yaw: steer, // on ground, yaw follows steer
                            roll: 0.0,
                            jump: false,
                            boost,
                            handbrake,
                        });
                    }
                }
            }
        }

        let num_ground_actions = actions_table.len();

        // ── Aerial actions ──────────────────────────────────────────────
        for &pitch in &[-1.0, 0.0, 1.0] {
            for &yaw in &[-1.0, 0.0, 1.0] {
                for &roll in &[-1.0, 0.0, 1.0] {
                    for &jump in &[false, true] {
                        for &boost in &[false, true] {
                            // Only need roll for sideflip
                            if jump && yaw != 0.0 {
                                continue;
                            }

                            // Duplicate with ground
                            if pitch == 0.0 && roll == 0.0 && !jump {
                                continue;
                            }

                            // Enable handbrake for potential wavedashes
                            let handbrake = jump && (pitch != 0.0 || yaw != 0.0 || roll != 0.0);

                            actions_table.push(CarControls {
                                throttle: if boost { 1.0 } else { 0.0 },
                                steer: yaw,
                                pitch,
                                yaw,
                                roll,
                                jump,
                                boost,
                                handbrake,
                            });
                        }
                    }
                }
            }
        }

        // ── Precompute masks ────────────────────────────────────────────
        let num_actions = actions_table.len();
        let mut ground_mask = vec![false; num_actions];
        let mut air_mask = vec![false; num_actions];
        let mut jump_mask = vec![false; num_actions];
        let mut boost_mask = vec![false; num_actions];

        for (i, action) in actions_table.iter().enumerate() {
            if action.jump {
                jump_mask[i] = true;
            }

            if action.boost {
                boost_mask[i] = true;
            }

            if i < num_ground_actions {
                ground_mask[i] = true;
            }

            if i > num_ground_actions && !action.jump {
                air_mask[i] = true;
            }

            // Ground actions that are also valid in the air
            if i < num_ground_actions {
                let boost_f = if action.boost { 1.0 } else { 0.0 };
                if action.throttle == boost_f && (action.yaw != 0.0) == action.handbrake {
                    air_mask[i] = true;
                }
            }
        }

        Self {
            actions_table,
            ground_mask,
            air_mask,
            jump_mask,
            boost_mask,
            action_buffer: ArrayVec::new(),
        }
    }

    /// Returns a per-action validity mask for a given car state.
    ///
    /// Each element in the returned `Vec<bool>` corresponds to an action in
    /// the action table. `true` means the action is valid in the current
    /// state, `false` means it should be masked out.
    ///
    /// Logic:
    /// - Ground mask when `is_on_ground`, air mask otherwise.
    /// - Remove boost actions when `boost == 0`.
    /// - Enable jump actions when the car has a flip/jump available or
    ///   is turtled (on its back).
    pub fn get_action_mask(&self, car_state: &CarState) -> Vec<bool> {
        let num_actions = self.actions_table.len();
        let mut result = vec![false; num_actions];

        // Ground or air mask
        if car_state.is_on_ground {
            for (r, &m) in result.iter_mut().zip(self.ground_mask.iter()) {
                *r |= m;
            }
        } else {
            for (r, &m) in result.iter_mut().zip(self.air_mask.iter()) {
                *r |= m;
            }
        }

        // Remove boost actions when out of boost
        if car_state.boost == 0.0 {
            for (r, &m) in result.iter_mut().zip(self.boost_mask.iter()) {
                *r &= !m;
            }
        }

        // Enable jump actions when the car can still jump/flip or is turtled
        let is_turtled = car_state.world_contact_normal.is_some_and(|n| n.z > 0.9);
        if car_state.has_flip_or_jump() || is_turtled {
            for (r, &m) in result.iter_mut().zip(self.jump_mask.iter()) {
                *r |= m;
            }
        }

        result
    }
}

impl<const MAX_NUM_AGENTS: usize, const TICK_SKIP: u8> Default
    for DefaultAction<MAX_NUM_AGENTS, TICK_SKIP>
{
    fn default() -> Self {
        Self::new()
    }
}

impl<const MAX_NUM_AGENTS: usize, const TICK_SKIP: u8, SI> Action<SI>
    for DefaultAction<MAX_NUM_AGENTS, TICK_SKIP>
{
    type Input = usize;

    fn get_tick_skip() -> u8 {
        TICK_SKIP
    }

    fn get_action_space(&self, _shared_info: &SI) -> usize {
        self.actions_table.len()
    }

    fn reset(&mut self, _initial_state: &GameState, _shared_info: &mut SI) {}

    fn parse_actions<'a>(
        &'a mut self,
        actions: &[usize],
        state: &GameState,
        _shared_info: &'a mut SI,
    ) -> &'a [(usize, CarControls)] {
        self.action_buffer.clear();

        for ((info, _), &action_idx) in state.cars.iter().zip(actions.iter()) {
            self.action_buffer
                .push((info.idx, self.actions_table[action_idx]));
        }

        &self.action_buffer[..state.cars.len()]
    }

    fn get_action_masks(&mut self, state: &GameState, _shared_info: &mut SI) -> Vec<Vec<bool>> {
        state
            .cars
            .iter()
            .map(|(_, car)| self.get_action_mask(car))
            .collect()
    }
}
