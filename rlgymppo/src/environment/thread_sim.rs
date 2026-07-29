use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;

use burn::prelude::Backend;
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

fn merge_worker_memory(
    target: &mut Memory,
    incoming: Memory,
    rollout_budget: usize,
    overbatching: bool,
) {
    if overbatching {
        target.merge(incoming);
        return;
    }

    let remaining = rollout_budget.saturating_sub(target.len());
    if incoming.len() <= remaining {
        target.merge(incoming);
    } else if remaining > 0 {
        target.merge_prefix(incoming, remaining);
    }
}

type WorkerResponse = Result<DataResponse, String>;

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| (*message).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic payload".to_owned())
}

fn collect_worker_responses(
    recv: &Receiver<WorkerResponse>,
    response_count: usize,
    target: &mut Memory,
    metrics: &mut Report,
    rollout_budget: usize,
    overbatching: bool,
) -> Result<(), String> {
    for _ in 0..response_count {
        let response = recv
            .recv()
            .map_err(|_| "collector worker disconnected without a response".to_owned())??;
        merge_worker_memory(target, response.memory, rollout_budget, overbatching);
        *metrics += response.metrics;
    }
    Ok(())
}

#[derive(Clone)]
enum ThreadCommand<B: Backend> {
    Run {
        model: Arc<Actic<B>>,
        self_play: Option<(Arc<Actic<B>>, usize)>,
    },
    Shutdown,
}

struct ThreadControl {
    remaining_steps: AtomicUsize,
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
    recv: Receiver<WorkerResponse>,
    command_senders: Vec<Sender<ThreadCommand<B>>>,
    control: Arc<ThreadControl>,
    threads: Vec<thread::JoinHandle<()>>,
    metrics: Report,
    memory: Memory,
    rollout_budget: usize,
    overbatching: bool,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new<F>(
        create_env_fn: F,
        rollout_budget: usize,
        num_threads: usize,
        num_games_per_thread: usize,
        device: B::Device,
        reward_sampling: RewardSamplingConfig,
        max_episode_length: Option<usize>,
        retain_overflow_episodes: bool,
        overbatching: bool,
        complete_trajectories: bool,
    ) -> Self
    where
        F: Fn(Option<usize>) -> Env<SS, OBS, ACT, REW, TERM, TRUNC, SI> + Clone + Send + 'static,
        B::Device: Send,
    {
        assert!(
            num_threads > 0,
            "Number of collector threads must be greater than zero"
        );
        assert!(
            num_games_per_thread > 0,
            "Number of games per collector thread must be greater than zero"
        );

        let (sender, recv) = channel();
        // These are initial capacities, not hard limits. Complete-trajectory
        // mode may grow for its bounded final-trajectory overrun.
        let coordinator_memory_capacity = rollout_budget;
        let worker_memory_capacity = rollout_budget.div_ceil(num_threads);

        let control = Arc::new(ThreadControl {
            remaining_steps: AtomicUsize::new(0),
        });

        let (startup_sender, startup_recv) = channel();
        let mut command_senders = Vec::with_capacity(num_threads);
        let mut threads = Vec::with_capacity(num_threads);

        for t in 0..num_threads {
            let sender: Sender<WorkerResponse> = sender.clone();
            let startup_sender = startup_sender.clone();
            let (command_sender, command_recv) = channel();
            command_senders.push(command_sender);
            let create_env_fn = create_env_fn.clone();
            let device = device.clone();
            let control = control.clone();
            let reward_sampling = reward_sampling.clone();

            let thread = thread::spawn(move || {
                let batch_sim = catch_unwind(AssertUnwindSafe(|| {
                    BatchSim::new(
                        create_env_fn,
                        t + 1,
                        num_games_per_thread,
                        device,
                        reward_sampling,
                        max_episode_length,
                        retain_overflow_episodes,
                        complete_trajectories,
                    )
                }));
                let mut batch_sim = match batch_sim {
                    Ok(batch_sim) => {
                        let _ = startup_sender.send(Ok(()));
                        batch_sim
                    }
                    Err(payload) => {
                        let _ = startup_sender.send(Err(panic_message(payload)));
                        return;
                    }
                };
                while let Ok(command) = command_recv.recv() {
                    match command {
                        ThreadCommand::Run { model, self_play } => {
                            let response = catch_unwind(AssertUnwindSafe(|| {
                                let (memory, metrics) = batch_sim.run_with_budget(
                                    model.as_ref(),
                                    &control.remaining_steps,
                                    worker_memory_capacity,
                                    rollout_budget,
                                    self_play.as_ref().map(|(m, t)| (m.as_ref(), *t)),
                                    overbatching,
                                );
                                DataResponse { memory, metrics }
                            }))
                            .map_err(panic_message);
                            let failed = response.is_err();
                            if sender.send(response).is_err() || failed {
                                break;
                            }
                        }
                        ThreadCommand::Shutdown => break,
                    }
                }
            });
            threads.push(thread);
        }
        drop(startup_sender);

        for _ in 0..num_threads {
            match startup_recv.recv() {
                Ok(Ok(())) => {}
                Ok(Err(message)) => {
                    drop(command_senders);
                    for thread in threads {
                        let _ = thread.join();
                    }
                    panic!("collector worker failed to initialize: {message}");
                }
                Err(_) => {
                    drop(command_senders);
                    for thread in threads {
                        let _ = thread.join();
                    }
                    panic!("collector worker disconnected during initialization");
                }
            }
        }

        Self {
            recv,
            command_senders,
            control,
            threads,
            memory: Memory::with_capacity(coordinator_memory_capacity),
            metrics: Report::default(),
            rollout_budget,
            overbatching,
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

        self.control
            .remaining_steps
            .store(self.rollout_budget, Ordering::Release);
        let command = ThreadCommand::Run {
            model: Arc::new(model),
            self_play: self_play.map(|(model, team)| (Arc::new(model), team)),
        };
        for sender in &self.command_senders {
            sender
                .send(command.clone())
                .expect("collector worker disconnected before rollout");
        }

        collect_worker_responses(
            &self.recv,
            self.threads.len(),
            &mut self.memory,
            &mut self.metrics,
            self.rollout_budget,
            self.overbatching,
        )
        .unwrap_or_else(|message| panic!("collector rollout failed: {message}"));

        (&self.memory, self.metrics.clone())
    }

    pub fn join(self) {
        for sender in self.command_senders {
            let _ = sender.send(ThreadCommand::Shutdown);
        }
        for thread in self.threads {
            thread.join().expect("collector worker panicked");
        }
    }
}

#[cfg(test)]
mod regression_tests {
    use super::*;
    use crate::base::TerminalState;

    fn memory(start: usize, count: usize) -> Memory {
        let mut memory = Memory::with_capacity(count);
        let mut terminals = vec![TerminalState::None; count];
        terminals[count - 1] = TerminalState::Normal;
        memory.push_player(
            (start..start + count).map(|i| i as f32).collect::<Vec<_>>(),
            1,
            (start..start + count).collect(),
            vec![0.0; count],
            vec![1.0; count],
            terminals,
            vec![true; count],
            1,
            None,
        );
        memory
    }

    #[test]
    fn regression_overbatching_receives_and_merges_every_worker_response() {
        let (sender, receiver) = channel();
        for (start, count) in [(0, 60), (60, 60), (120, 30)] {
            sender
                .send(Ok(DataResponse {
                    memory: memory(start, count),
                    metrics: Report::default(),
                }))
                .unwrap();
        }

        let mut target = Memory::with_capacity(100);
        let mut metrics = Report::default();
        collect_worker_responses(&receiver, 3, &mut target, &mut metrics, 100, true).unwrap();

        assert_eq!(target.len(), 150);
        assert_eq!(target.actions().first(), Some(&0));
        assert_eq!(target.actions().last(), Some(&149));
        assert!(matches!(
            receiver.try_recv(),
            Err(std::sync::mpsc::TryRecvError::Empty)
        ));
    }

    #[test]
    fn regression_worker_failure_returns_without_waiting_for_other_responses() {
        let (sender, receiver) = channel();
        sender
            .send(Err("simulated worker panic".to_owned()))
            .unwrap();
        sender
            .send(Ok(DataResponse {
                memory: memory(0, 10),
                metrics: Report::default(),
            }))
            .unwrap();

        let mut target = Memory::with_capacity(10);
        let mut metrics = Report::default();
        let error = collect_worker_responses(&receiver, 2, &mut target, &mut metrics, 10, false)
            .unwrap_err();

        assert_eq!(error, "simulated worker panic");
        assert!(target.is_empty());
    }

    #[test]
    fn regression_exact_batching_discards_worker_overflow() {
        let mut target = Memory::with_capacity(100);
        merge_worker_memory(&mut target, memory(0, 60), 100, false);
        merge_worker_memory(&mut target, memory(60, 60), 100, false);
        merge_worker_memory(&mut target, memory(120, 30), 100, false);

        assert_eq!(target.len(), 100);
        assert_eq!(target.actions().first(), Some(&0));
        assert_eq!(target.actions().last(), Some(&99));
    }

    #[test]
    fn regression_complete_trajectories_respect_exact_batching() {
        let mut target = Memory::with_capacity(100);
        merge_worker_memory(&mut target, memory(0, 60), 100, false);
        merge_worker_memory(&mut target, memory(60, 60), 100, false);

        assert_eq!(target.len(), 100);
        assert_eq!(target.actions().last(), Some(&99));
    }
}
