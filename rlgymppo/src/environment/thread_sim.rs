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
    control: Arc<(RwLock<Option<Actic<B>>>, Barrier)>,
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
        let control = Arc::new((RwLock::new(None), Barrier::new(num_threads + 1)));

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
                    control.1.wait(); // barrier: wait for model to be set
                    let model = {
                        let guard = control.0.read();
                        guard.clone()
                    };
                    let Some(model) = model else {
                        break; // None signals shutdown
                    };

                    let (memory, metrics) = batch_sim.run(&model, steps_per_thread);

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

    pub fn run(&mut self, model: Actic<B>) -> (&Memory, Report) {
        self.metrics.clear();
        self.memory.clear();

        // Publish the model for all threads.
        *self.control.0.write() = Some(model);

        // Wake all collector threads.
        self.control.1.wait();

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
        *self.control.0.write() = None;
        self.control.1.wait();

        for thread in self.threads {
            thread.join().unwrap();
        }
    }
}
