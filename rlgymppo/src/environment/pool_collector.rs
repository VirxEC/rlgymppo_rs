use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use burn::prelude::Backend;
use rayon::ThreadPool;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};
use rlgymppo_utils::Report;
use rlgymppo_utils::shared_info::SharedInfoReport;

use super::batch_sim::BatchSim;
use super::sim::RewardSamplingConfig;
use crate::agent::model::Actic;
use crate::base::Memory;

/// One independent rollout collector: a [`BatchSim`], its rayon pool, and
/// its budget share. Pool jobs never touch the GPU or call `wait()`.
pub(crate) struct PoolCollector<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend,
    SS: StateSetter<SI>,
    SI: SharedInfoProvider,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    batch_sim: BatchSim<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>,
    pool: ThreadPool,
    remaining_steps: Arc<AtomicUsize>,
    overbatching: bool,
}

impl<B, SS, OBS, ACT, REW, TERM, TRUNC, SI> PoolCollector<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend + Send + 'static,
    B::Device: Send,
    SS: StateSetter<SI> + Send,
    SI: SharedInfoProvider + SharedInfoReport + Send,
    OBS: Obs<SI> + Send,
    ACT: Action<SI, Input = usize> + Send,
    REW: Reward<SI> + Send,
    TERM: Terminal<SI> + Send,
    TRUNC: Truncate<SI> + Send,
{
    /// Build one collector. The seed multiplier is `pool_index + 1`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new<F, FO>(
        create_env_fn: F,
        make_old_obs: Option<FO>,
        pool_index: usize,
        num_games_per_pool: usize,
        num_threads_per_pool: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
        max_episode_length: Option<usize>,
        retain_overflow_episodes: bool,
        overbatching: bool,
        complete_trajectories: bool,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI>,
        FO: Fn() -> Box<dyn Obs<SI> + Send>,
    {
        let batch_sim = BatchSim::new(
            create_env_fn,
            make_old_obs,
            pool_index + 1,
            num_games_per_pool,
            device,
            reward_sampling,
            num_games_per_pool * 4,
            max_episode_length,
            retain_overflow_episodes,
            complete_trajectories,
        );

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads_per_pool)
            .thread_name(move |thread_index| format!("collector-{pool_index}-{thread_index}"))
            .build()
            .expect("failed to build collector thread pool");

        Self {
            batch_sim,
            pool,
            remaining_steps: Arc::new(AtomicUsize::new(0)),
            overbatching,
        }
    }

    /// Collect `share` steps into a fresh memory. `share` is also the
    /// overbatch bound; it must be the pool's share, never the global
    /// budget.
    pub(crate) fn run(
        &mut self,
        model: &Actic<B>,
        self_play: Option<(&Actic<B>, usize)>,
        share: usize,
    ) -> (Memory, Report) {
        self.remaining_steps.store(share, Ordering::Release);
        self.batch_sim.run_with_budget(
            model,
            &self.remaining_steps,
            share,
            share,
            self_play,
            self.overbatching,
            Some(&self.pool),
        )
    }
}

#[cfg(all(test, feature = "flex"))]
mod flex_gated {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    use burn::backend::Flex;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rlgym::GameState;
    use rlgym::rocketsim::{Arena, ArenaConfig, CarBodyConfig, GameMode, Team};
    use rlgymppo_utils::actions::DefaultAction;
    use rlgymppo_utils::obs::DefaultObs;
    use rlgymppo_utils::rewards::FaceBallReward;
    use rlgymppo_utils::shared_info::SharedInfoRng;
    use rlgymppo_utils::state_setters::RandomState;
    use rlgymppo_utils::terminal::{NoTouchCondition, OnGoalCondition};

    use super::*;
    use crate::agent::model::Actic;
    use crate::environment::thread_sim::ThreadSim;

    type StateSetter = RandomState<true, false, true>;
    type Action = DefaultAction<1, 8, 0>;
    type TruncateCond = NoTouchCondition<120>;
    type EnvType = Env<
        StateSetter,
        DefaultObs<1>,
        Action,
        FaceBallReward,
        OnGoalCondition,
        TruncateCond,
        TestSharedInfo,
    >;
    type ThreadSimType = ThreadSim<
        Flex,
        StateSetter,
        DefaultObs<1>,
        Action,
        FaceBallReward,
        OnGoalCondition,
        TruncateCond,
        TestSharedInfo,
    >;

    struct TestSharedInfo {
        rng: SmallRng,
        report: Report,
    }

    impl TestSharedInfo {
        fn seeded(seed: u64) -> Self {
            Self {
                rng: SmallRng::seed_from_u64(seed),
                report: Report::default(),
            }
        }
    }

    impl SharedInfoProvider for TestSharedInfo {
        fn reset(&mut self, _initial_state: &GameState) {}
        fn update(&mut self, _game_state: &GameState) {}
    }

    impl SharedInfoRng for TestSharedInfo {
        type Rng = SmallRng;

        fn rng(&mut self) -> &mut Self::Rng {
            &mut self.rng
        }
    }

    impl SharedInfoReport for TestSharedInfo {
        fn report(&mut self) -> &mut Report {
            &mut self.report
        }
    }

    fn init_rocketsim() {
        static INIT: OnceLock<()> = OnceLock::new();
        INIT.get_or_init(|| {
            let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let candidates = [
                cwd.join("collision_meshes"),
                cwd.join("../collision_meshes"),
                manifest.join("../collision_meshes"),
            ];
            let Some(root) = candidates
                .iter()
                .find(|candidate| candidate.join("soccar").is_dir())
            else {
                panic!(
                    "could not locate RocketSim collision meshes; probed {candidates:?} (cwd: {})",
                    cwd.display()
                );
            };
            crate::rocketsim::init(root, true).expect("failed to initialize RocketSim");
        });
    }

    fn make_env(game_id: Option<usize>) -> EnvType {
        init_rocketsim();
        let game_id = game_id.unwrap_or(0);
        let mut config = ArenaConfig::new(GameMode::Soccar);
        config.rng_seed = Some(game_id as u64 * 7919 + 13);
        let mut arena = Arena::new_with_config(config);
        arena.add_car(Team::Blue, CarBodyConfig::OCTANE);
        Env::new(
            arena,
            RandomState::<true, false, true>,
            DefaultObs::<1>,
            Action::new(),
            FaceBallReward,
            OnGoalCondition,
            TruncateCond::default(),
            TestSharedInfo::seeded(game_id as u64 * 104729 + 17),
        )
    }

    fn make_model() -> Actic<Flex> {
        let probe = make_env(None);
        let obs_space = probe.get_obs_space();
        let action_space = probe.get_action_space();
        let device = Default::default();
        Actic::<Flex>::new(
            obs_space,
            action_space,
            vec![8],
            vec![8],
            &[],
            &device,
            None,
        )
    }

    fn make_thread_sim(num_pools: usize) -> ThreadSimType {
        ThreadSimType::new(
            make_env,
            None::<fn() -> Box<dyn Obs<TestSharedInfo> + Send>>,
            64,
            1,
            num_pools,
            2,
            Default::default(),
            RewardSamplingConfig::default(),
            None,
            true,
            false,
            false,
        )
    }

    #[test]
    fn multi_pool_exact_claim_returns_full_budget_twice() {
        let mut sim = make_thread_sim(2);

        let (memory, _) = sim.run_with_budget(make_model(), 64);
        assert_eq!(memory.len(), 64);

        let (memory, _) = sim.run_with_budget(make_model(), 64);
        assert_eq!(memory.len(), 64);
    }

    #[test]
    fn single_pool_exact_claim_matches_multi_pool_total() {
        let mut sim = make_thread_sim(1);

        let (memory, _) = sim.run_with_budget(make_model(), 64);
        assert_eq!(memory.len(), 64);
    }
}
