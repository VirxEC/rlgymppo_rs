use super::batch_sim::BatchSim;
use crate::{agent::model::Net, base::Memory, utils::Report};
use burn::prelude::Backend;
use parking_lot::RwLock;
use rlgym::{
    Action, Env, Obs, Reward, SharedInfoProvider, StateSetter, Terminal, Truncate,
    rocketsim_rs::glam_ext::GameStateA,
};
use std::{
    sync::{
        Arc, Barrier,
        mpsc::{Receiver, channel},
    },
    thread,
};

#[derive(Default)]
pub struct DataRequest<B: Backend> {
    pub model: Option<Net<B>>,
    pub total_num_players: usize,
}

pub struct DataResponse {
    pub memory: Memory,
    pub metrics: Report,
}

pub struct ThreadSim<B: Backend> {
    recv: Receiver<DataResponse>,
    thread_controls: Arc<(RwLock<DataRequest<B>>, Barrier)>,
    threads: Vec<thread::JoinHandle<()>>,
    metrics: Report,
    memory: Memory,
}

impl<B: Backend> ThreadSim<B> {
    pub fn new<F, C, SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>(
        create_env_fn: F,
        step_callback: C,
        batch_size: usize,
        exp_buffer_size: usize,
        num_threads: usize,
        num_games_per_thread: usize,
        device: B::Device,
    ) -> Self
    where
        B: Backend,
        F: Fn(Option<usize>) -> Env<SS, SIP, OBS, ACT, REW, TERM, TRUNC, SI>
            + Clone
            + Send
            + 'static,
        C: Fn(&mut Report, &mut SI, &GameStateA) + Clone + Send + 'static,
        SS: StateSetter<SI>,
        SIP: SharedInfoProvider<SI>,
        OBS: Obs<SI>,
        ACT: Action<SI, Input = usize>,
        REW: Reward<SI>,
        TERM: Terminal<SI>,
        TRUNC: Truncate<SI>,
    {
        let (pc_sender, pc_recv) = channel();
        let (sender, recv) = channel();

        let mut threads = Vec::with_capacity(num_threads);
        let data_request = DataRequest::default();
        let thread_controls = Arc::new((RwLock::new(data_request), Barrier::new(num_threads + 1)));

        for i in 0..num_threads {
            let pc_sender = pc_sender.clone();
            let sender = sender.clone();
            let create_env_fn = create_env_fn.clone();
            let step_callback = step_callback.clone();
            let device = device.clone();
            let thread_controls = thread_controls.clone();

            let thread = thread::spawn(move || {
                let mut batch_sim = BatchSim::new(
                    create_env_fn,
                    step_callback,
                    i,
                    num_games_per_thread,
                    device,
                );

                pc_sender.send(batch_sim.num_players()).unwrap();

                loop {
                    thread_controls.1.wait();
                    let request = thread_controls.0.read();
                    let Some(model) = request.model.clone() else {
                        break;
                    };

                    let steps_per_player = batch_size.div_ceil(request.total_num_players);
                    let (memory, metrics) =
                        batch_sim.run(&model, steps_per_player * batch_sim.num_players());

                    sender.send(DataResponse { memory, metrics }).unwrap();
                }
            });
            threads.push(thread);
        }

        let mut total_num_players = 0;
        for _ in 0..num_threads {
            total_num_players += pc_recv.recv().unwrap();
        }

        thread_controls.0.write().total_num_players = total_num_players;

        Self {
            recv,
            threads,
            thread_controls,
            memory: Memory::with_capacity(exp_buffer_size),
            metrics: Report::default(),
        }
    }

    pub fn run(&mut self, model: Net<B>) -> (&Memory, Report) {
        self.metrics.clear();

        {
            let mut thread_controls = self.thread_controls.0.write();
            thread_controls.model = Some(model);
        }

        // Notify all threads to start processing
        self.thread_controls.1.wait();

        for _ in 0..self.threads.len() {
            let response = self.recv.recv().unwrap();

            self.memory.merge(response.memory);
            self.metrics += response.metrics;
        }

        (&self.memory, self.metrics.clone())
    }

    pub fn join(self) {
        {
            let mut thread_controls = self.thread_controls.0.write();
            thread_controls.model = None;
        }

        self.thread_controls.1.wait();

        for thread in self.threads {
            thread.join().unwrap();
        }
    }
}
