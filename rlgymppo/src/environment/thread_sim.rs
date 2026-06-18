use std::{
    marker::PhantomData,
    sync::{
        Arc, Barrier,
        mpsc::{Receiver, Sender, channel},
    },
    thread,
};

use burn::prelude::Backend;
use parking_lot::RwLock;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::{batch_sim::BatchSim, sim::RewardSamplingConfig};
use crate::{
    agent::model::Actic,
    base::Memory,
    utils::{Report, shared_info::SharedInfoReport},
};

pub struct DataResponse {
    pub memory: Memory,
    pub metrics: Report,
}

/// Shared model state passed to collector threads via the barrier.
/// The `model` field carries the current policy; `self_play`
/// optionally carries an old policy version and the team it
/// controls (0 = Blue, 1 = Orange).
struct ThreadControl<B: Backend> {
    model: RwLock<Option<Actic<B>>>,
    self_play: RwLock<Option<(Actic<B>, usize)>>,
    barrier: Barrier,
}

/// Multi‑threaded collector.  Each thread owns an independent pool of
/// `num_games_per_thread` games so completions stay frequent regardless of
/// the total thread count — scaling up just adds more parallel pools.
///
/// 1 thread  × 256 games  →  acts like GigaLearn
/// 2 threads × 256 games  →  2× CPU parallelism, same completion rate
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
    recv: Receiver<DataResponse>,
    control: Arc<ThreadControl<B>>,
    threads: Vec<thread::JoinHandle<()>>,
    metrics: Report,
    memory: Memory,
    _marker: PhantomData<fn(SS, OBS, ACT, REW, TERM, TRUNC, SI)>,
}

impl<B, SS, OBS, ACT, REW, TERM, TRUNC, SI> ThreadSim<B, SS, OBS, ACT, REW, TERM, TRUNC, SI>
where
    B: Backend + Send + 'static,
    SS: StateSetter<SI>,
    SI: SharedInfoProvider + SharedInfoReport,
    OBS: Obs<SI>,
    ACT: Action<SI, Input = usize>,
    REW: Reward<SI>,
    TERM: Terminal<SI>,
    TRUNC: Truncate<SI>,
{
    pub fn new<F>(
        create_env_fn: F,
        batch_size: usize,
        num_threads: usize,
        num_games_per_thread: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Clone + Send + 'static,
        B::Device: Send,
    {
        let (sender, recv) = channel();
        let control = Arc::new(ThreadControl {
            model: RwLock::new(None),
            self_play: RwLock::new(None),
            barrier: Barrier::new(num_threads + 1),
        });

        let steps_per_thread = batch_size.div_ceil(num_threads);
        let mut threads = Vec::with_capacity(num_threads);

        for t in 0..num_threads {
            let sender: Sender<DataResponse> = sender.clone();
            let create_env_fn = create_env_fn.clone();
            let device = device.clone();
            let control = control.clone();
            let reward_sampling = reward_sampling.clone();

            let thread = thread::spawn(move || {
                let mut batch_sim = BatchSim::new(
                    create_env_fn,
                    t + 1,
                    num_games_per_thread,
                    device,
                    reward_sampling,
                );

                loop {
                    control.barrier.wait(); // wait for model to be set
                    let model = {
                        let guard = control.model.read();
                        guard.clone()
                    };
                    let Some(model) = model else {
                        break; // None signals shutdown
                    };

                    let self_play = {
                        let guard = control.self_play.read();
                        guard.clone()
                    };

                    let (memory, metrics) = batch_sim.run(
                        &model,
                        steps_per_thread,
                        self_play.as_ref().map(|(m, t)| (m, *t)),
                    );

                    sender.send(DataResponse { memory, metrics }).unwrap();
                }
            });
            threads.push(thread);
        }

        Self {
            recv,
            control,
            threads,
            memory: Memory::with_capacity(batch_size * 2),
            metrics: Report::default(),
            _marker: PhantomData,
        }
    }

    /// Publish the model (and optionally an old self-play model), wake
    /// all collector threads, and collect the resulting trajectories.
    ///
    /// `self_play` optionally supplies an old policy version and which
    /// team (0 = Blue, 1 = Orange) should use it.  Only current-policy
    /// player trajectories are recorded in the returned memory.
    pub fn run(
        &mut self,
        model: Actic<B>,
        self_play: Option<(Actic<B>, usize)>,
    ) -> (&Memory, Report) {
        self.metrics.clear();
        self.memory.clear();

        // Publish the model and self-play assignment for all threads.
        *self.control.model.write() = Some(model);
        *self.control.self_play.write() = self_play;

        // Wake all collector threads.
        self.control.barrier.wait();

        // Collect results from every thread.
        for _ in 0..self.threads.len() {
            let response = self.recv.recv().unwrap();
            self.memory.merge(response.memory);
            self.metrics += response.metrics;
        }

        (&self.memory, self.metrics.clone())
    }

    pub fn join(self) {
        // Signal shutdown.
        *self.control.model.write() = None;
        self.control.barrier.wait();

        for thread in self.threads {
            thread.join().unwrap();
        }
    }
}
