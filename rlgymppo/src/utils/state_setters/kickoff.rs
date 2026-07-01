use rand::RngExt;
use rlgym::StateSetter;
use rlgym::rocketsim::Arena;

use crate::utils::shared_info::SharedInfoRng;

/// Sets the arena to a random kickoff state.
#[derive(Default)]
pub struct KickoffState;

impl<SI: SharedInfoRng> StateSetter<SI> for KickoffState {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        arena.reset_to_random_kickoff(Some(shared_info.rng().random()));
    }
}
