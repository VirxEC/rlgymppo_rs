use rand::distr::{Distribution, weighted::WeightedIndex};
use rlgym::{StateSetter, rocketsim::Arena};

use crate::utils::shared_info::SharedInfoRng;

#[macro_export]
macro_rules! weighted_state {
    ($($state:ty, $weight:expr;)+) => {
        $crate::utils::state_setters::WeightedState::new(vec![
            $((Box::<$state>::default(), $weight),)+
        ])
    };
}

pub struct WeightedState<SI: SharedInfoRng> {
    state_setters: Vec<Box<dyn StateSetter<SI>>>,
    sampler: WeightedIndex<f32>,
}

impl<SI: SharedInfoRng> WeightedState<SI> {
    /// Prefer the [`weighted_state!`] macro over calling this directly.
    pub fn new(state_setters: Vec<(Box<dyn StateSetter<SI>>, f32)>) -> Self {
        let (state_setters, weights): (Vec<_>, Vec<_>) = state_setters.into_iter().unzip();
        let total_weight: f32 = weights.iter().sum();
        let weights: Vec<_> = weights.into_iter().map(|w| w / total_weight).collect();

        let sampler = WeightedIndex::new(weights).unwrap();

        Self {
            state_setters,
            sampler,
        }
    }
}

impl<SI: SharedInfoRng> StateSetter<SI> for WeightedState<SI> {
    fn apply(&mut self, arena: &mut Arena, shared_info: &mut SI) {
        self.state_setters[self.sampler.sample(shared_info.rng())].apply(arena, shared_info);
    }
}
