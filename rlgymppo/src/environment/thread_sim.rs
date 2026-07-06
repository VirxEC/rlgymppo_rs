use std::marker::PhantomData;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Barrier};
use std::thread;

use burn::prelude::Backend;
use parking_lot::RwLock;
use rlgym::{Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate};

use super::batch_sim::BatchSim;
use super::sim::RewardSamplingConfig;
use crate::agent::model::Actic;
use crate::base::Memory;
use crate::utils::Report;
use crate::utils::shared_info::SharedInfoReport;

pub struct DataResponse {
    pub memory: Memory,
    pub metrics: Report,
}

#[derive(Clone)]
enum ThreadCommand<B: Backend> {
    Run {
        model: Arc<Actic<B>>,
        self_play: Option<(Arc<Actic<B>>, usize)>,
    },
    Shutdown,
}

/// Shared rollout state passed to collector threads via the barrier.
struct ThreadControl<B: Backend> {
    command: RwLock<ThreadCommand<B>>,
    remaining_steps: AtomicUsize,
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
    batch_size: usize,
    pending_responses: usize,
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
        max_episode_length: Option<usize>,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Clone + Send + 'static,
        B::Device: Send,
    {
        let (sender, recv) = channel();
        let control = Arc::new(ThreadControl {
            command: RwLock::new(ThreadCommand::Shutdown),
            remaining_steps: AtomicUsize::new(0),
            barrier: Barrier::new(num_threads + 1),
        });

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
                    max_episode_length,
                );

                loop {
                    control.barrier.wait();
                    let command = {
                        let guard = control.command.read();
                        guard.clone()
                    };

                    match command {
                        ThreadCommand::Run { model, self_play } => {
                            let (memory, metrics) = batch_sim.run_with_budget(
                                model.as_ref(),
                                &control.remaining_steps,
                                batch_size * 2,
                                self_play.as_ref().map(|(m, t)| (m.as_ref(), *t)),
                            );

                            sender.send(DataResponse { memory, metrics }).unwrap();
                        }
                        ThreadCommand::Shutdown => break,
                    }
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
            batch_size,
            pending_responses: 0,
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
        self.discard_pending_responses();
        self.metrics.clear();
        self.memory.clear();

        self.control
            .remaining_steps
            .store(self.batch_size, Ordering::Release);
        *self.control.command.write() = ThreadCommand::Run {
            model: Arc::new(model),
            self_play: self_play.map(|(model, team)| (Arc::new(model), team)),
        };
        self.control.barrier.wait();

        let mut received = 0;
        while received < self.threads.len() {
            let response = self.recv.recv().unwrap();
            received += 1;
            self.memory.merge(response.memory);
            self.metrics += response.metrics;

            if self.memory.len() >= self.batch_size {
                break;
            }
        }
        self.pending_responses = self.threads.len() - received;

        (&self.memory, self.metrics.clone())
    }

    fn discard_pending_responses(&mut self) {
        for _ in 0..self.pending_responses {
            let _ = self.recv.recv().unwrap();
        }
        self.pending_responses = 0;
    }

    pub fn join(mut self) {
        self.discard_pending_responses();

        // Signal shutdown.
        *self.control.command.write() = ThreadCommand::Shutdown;
        self.control.barrier.wait();

        for thread in self.threads {
            thread.join().unwrap();
        }
    }
}
