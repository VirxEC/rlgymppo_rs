use std::marker::PhantomData;

use rand::RngExt;
use rlgym::{GameState, Terminal, rocketsim::ArenaEvent};

use crate::utils::shared_info::SharedInfoRng;

/// Ends the episode after a random duration, but only if the ball has hit the floor at least once.
///
/// This is to avoid ending the episode while the ball is still in the air, which can lead to unrealistic training data.
///
/// The duration is randomly sampled from the range `[MIN_DURATION, MAX_DURATION]` at the start of each episode.
#[derive(Default)]
pub struct RandomGameEndedCondition<
    const MIN_DURATION: u64,
    const MAX_DURATION: u64,
    SI: SharedInfoRng,
> {
    start_tick: u64,
    episode_duration: u64,
    _shared_info: PhantomData<SI>,
}

impl<const MIN_DURATION: u64, const MAX_DURATION: u64, SI: SharedInfoRng> Terminal<SI>
    for RandomGameEndedCondition<MIN_DURATION, MAX_DURATION, SI>
{
    fn reset(&mut self, initial_state: &GameState, shared_info: &mut SI) {
        self.start_tick = initial_state.tick_count;
        self.episode_duration = shared_info.rng().random_range(MIN_DURATION..=MAX_DURATION);
    }

    fn is_terminal(&mut self, state: &GameState, _shared_info: &mut SI) -> bool {
        let elapsed = state.tick_count - self.start_tick;

        if elapsed < self.episode_duration {
            return false;
        }

        state.events.iter().any(|event| {
            if let ArenaEvent::BallHitWorld(info) = event {
                info.contact_point.z < 5.0
            } else {
                false
            }
        })
    }
}

/// Ends the episode after a fixed duration, but only if the ball has hit the floor at least once.
///
/// This is to avoid ending the episode while the ball is still in the air, which can lead to unrealistic training data.
#[derive(Default)]
pub struct GameEndedCondition<const DURATION: u64> {
    start_tick: u64,
}

impl<const DURATION: u64, SI> Terminal<SI> for GameEndedCondition<DURATION> {
    fn reset(&mut self, initial_state: &GameState, _shared_info: &mut SI) {
        self.start_tick = initial_state.tick_count;
    }

    fn is_terminal(&mut self, state: &GameState, _shared_info: &mut SI) -> bool {
        let elapsed = state.tick_count - self.start_tick;

        // reset after some minutes
        if elapsed < DURATION {
            return false;
        }

        // wait until the ball hits the floor
        state.events.iter().any(|event| {
            if let ArenaEvent::BallHitWorld(info) = event {
                info.contact_point.z < 5.0
            } else {
                false
            }
        })
    }
}
