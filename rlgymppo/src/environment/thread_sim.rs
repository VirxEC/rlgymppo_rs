use std::marker::PhantomData;

use burn::prelude::Backend;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::batch_sim::{COLLECT_ENV_STEP_TIME_KEY, COLLECT_INFERENCE_TIME_KEY};
use super::pool_collector::PoolCollector;
use super::sim::RewardSamplingConfig;
use crate::agent::model::Actic;
use crate::base::Memory;
use rlgymppo_utils::Report;
use rlgymppo_utils::shared_info::SharedInfoReport;

/// Rollout supervisor for multiple independent collectors.
///
/// Each pool owns its games, its inference batch, and its rayon pool.
/// The results merge in pool order.
pub struct ThreadSim<B: Backend, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    collectors: Vec<PoolCollector<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>>,
    num_pools: usize,
    memory: Memory,
    metrics: Report,
    rollout_budget: usize,
    _marker: PhantomData<fn(SS, OBS, ACT, REW, TERM, TRUNC, SI)>,
}

impl<B, SS, OBS, ACT, REW, TERM, TRUNC, SI> ThreadSim<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend + Send + 'static,
    SS: StateSetter<SI> + Send,
    SI: SharedInfoProvider + SharedInfoReport + Send,
    OBS: Obs<SI> + Send,
    ACT: Action<SI, Input = usize> + Send,
    REW: Reward<SI> + Send,
    TERM: Terminal<SI> + Send,
    TRUNC: Truncate<SI> + Send,
{
    /// Build `num_pools` collectors. Pool `p` seeds its games with the
    /// multiplier `p + 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F, FO>(
        create_env_fn: F,
        make_old_obs: Option<FO>,
        rollout_budget: usize,
        num_threads_per_pool: usize,
        num_pools: usize,
        num_games_per_pool: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
        max_episode_length: Option<usize>,
        retain_overflow_episodes: bool,
        overbatching: bool,
        complete_trajectories: bool,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Clone + Send + 'static,
        FO: Fn() -> Box<dyn Obs<SI> + Send> + Clone + Send + 'static,
        B::Device: Send,
    {
        assert!(
            num_threads_per_pool > 0,
            "Each pool needs at least one thread."
        );
        assert!(num_pools > 0, "The trainer needs at least one pool.");
        assert!(num_games_per_pool > 0, "Each pool needs at least one game.");

        let collectors = (0..num_pools)
            .map(|pool_index| {
                PoolCollector::new(
                    create_env_fn.clone(),
                    make_old_obs.clone(),
                    pool_index,
                    num_games_per_pool,
                    num_threads_per_pool,
                    device.clone(),
                    reward_sampling.clone(),
                    max_episode_length,
                    retain_overflow_episodes,
                    overbatching,
                    complete_trajectories,
                )
            })
            .collect();

        Self {
            collectors,
            num_pools,
            memory: Memory::with_capacity(rollout_budget),
            metrics: Report::default(),
            rollout_budget,
            _marker: PhantomData,
        }
    }

    /// Publish the model (and optionally an old self-play model) and
    /// collect the resulting trajectories.
    pub fn run(
        &mut self,
        model: Actic<B>,
        self_play: Option<(Actic<B>, usize)>,
    ) -> (&Memory, Report) {
        self.run_internal(model, self_play, self.rollout_budget)
    }

    /// Like [`Self::run`], but with an explicit rollout budget. Used by
    /// transfer learning.
    pub fn run_with_budget(&mut self, model: Actic<B>, budget: usize) -> (&Memory, Report) {
        self.run_internal(model, None, budget)
    }

    /// Split the budget across the collectors, run them, and merge the
    /// results in pool order. With one collector, it runs inline. The
    /// `Collect/*` wall-clock keys are per-pool sums, so they are
    /// averaged for comparability.
    fn run_internal(
        &mut self,
        model: Actic<B>,
        self_play: Option<(Actic<B>, usize)>,
        budget: usize,
    ) -> (&Memory, Report) {
        self.memory.clear();
        self.metrics.clear();

        let shares = split_budget(budget, self.num_pools);
        let self_play = self_play.as_ref().map(|(model, team)| (model, *team));
        let model = &model;

        if self.num_pools == 1 {
            let (memory, metrics) = self.collectors[0].run(model, self_play, shares[0]);
            self.memory.merge(memory);
            self.metrics += metrics;
        } else {
            let results = std::thread::scope(|scope| {
                let handles: Vec<_> = self
                    .collectors
                    .iter_mut()
                    .zip(&shares)
                    .map(|(collector, &share)| {
                        scope.spawn(move || collector.run(model, self_play, share))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for (memory, metrics) in results {
                self.memory.merge(memory);
                self.metrics += metrics;
            }
        }

        let num_pools = self.num_pools as f64;
        *self.metrics[COLLECT_INFERENCE_TIME_KEY].as_float_mut() /= num_pools;
        *self.metrics[COLLECT_ENV_STEP_TIME_KEY].as_float_mut() /= num_pools;

        (&self.memory, self.metrics.clone())
    }
}

/// Split a rollout budget into one share per pool. The remainder goes to
/// the last pool.
fn split_budget(budget: usize, num_pools: usize) -> Vec<usize> {
    let mut shares = vec![budget / num_pools; num_pools];
    shares[num_pools - 1] += budget % num_pools;
    shares
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_budget_shares_sum_to_budget() {
        for (budget, num_pools) in [(100, 3), (1, 5), (0, 2)] {
            let shares = split_budget(budget, num_pools);
            assert_eq!(shares.len(), num_pools);
            assert_eq!(shares.iter().sum::<usize>(), budget);
        }
    }

    #[test]
    fn split_budget_remainder_lands_in_last_pool() {
        assert_eq!(split_budget(100, 3), vec![33usize, 33, 34]);
    }

    #[test]
    fn split_budget_below_pool_count_keeps_remainder_in_last_pool() {
        assert_eq!(split_budget(1, 5), vec![0usize, 0, 0, 0, 1]);
        assert_eq!(split_budget(2, 5), vec![0usize, 0, 0, 0, 2]);
    }

    #[test]
    fn split_budget_shares_are_base_or_base_plus_one() {
        let base = 100 / 3;
        assert!(
            split_budget(100, 3)
                .iter()
                .all(|&share| share == base || share == base + 1)
        );
    }
}
