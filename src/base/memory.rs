use burn::prelude::*;
use itertools::izip;
use ringbuffer::{AllocRingBuffer, RingBuffer};

pub type MemoryIndices = Vec<usize>;

pub fn get_batch_1d<T: Copy>(data: &AllocRingBuffer<T>, indices: &MemoryIndices) -> Vec<T> {
    indices
        .iter()
        .filter_map(|i| data.get(*i))
        .copied()
        .collect::<Vec<_>>()
}

pub fn get_states_batch<B: Backend>(
    data: &AllocRingBuffer<Vec<f32>>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [indices.len(), data.get(0).unwrap().len()];
    let mut states: Vec<f32> = Vec::with_capacity(shape[0] * shape[1]);
    for i in indices {
        states.extend(data.get(*i).unwrap());
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub fn get_action_batch<B: Backend>(
    data: &AllocRingBuffer<usize>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let shape = [indices.len(), 1];
    let mut states: Vec<u32> = Vec::with_capacity(shape[0] * shape[1]);
    for i in indices {
        states.push(*data.get(*i).unwrap() as u32);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub struct Memory {
    state: AllocRingBuffer<Vec<f32>>,
    next_state: AllocRingBuffer<Vec<f32>>,
    action: AllocRingBuffer<usize>,
    reward: AllocRingBuffer<f32>,
    done: AllocRingBuffer<bool>,
    truncated: AllocRingBuffer<bool>,
}

impl Memory {
    pub fn new(capacity: usize) -> Self {
        Self {
            state: AllocRingBuffer::new(capacity),
            next_state: AllocRingBuffer::new(capacity),
            action: AllocRingBuffer::new(capacity),
            reward: AllocRingBuffer::new(capacity),
            done: AllocRingBuffer::new(capacity),
            truncated: AllocRingBuffer::new(capacity),
        }
    }

    pub fn push(
        &mut self,
        state: Vec<f32>,
        next_state: Vec<f32>,
        action: usize,
        reward: f32,
        done: bool,
        truncated: bool,
    ) {
        #[cfg(debug_assertions)]
        {
            // ensure no NaN values
            assert!(!state.iter().any(|&x| x.is_nan()));
            assert!(!next_state.iter().any(|&x| x.is_nan()));
            assert!(!reward.is_nan());
        }

        self.state.push(state);
        self.next_state.push(next_state);
        self.action.push(action);
        self.reward.push(reward);
        self.done.push(done);
        self.truncated.push(truncated);
    }

    pub fn push_batch(
        &mut self,
        state: Vec<Vec<f32>>,
        next_state: &[Vec<f32>],
        action: Vec<usize>,
        reward: Vec<f32>,
        done: bool,
        truncated: bool,
    ) {
        #[cfg(debug_assertions)]
        {
            let n = state.len();
            assert_eq!(n, next_state.len());
            assert_eq!(n, action.len());
            assert_eq!(n, reward.len());
        }

        for (i, (state, action, reward)) in izip!(state, action, reward).enumerate() {
            self.push(
                state,
                next_state[i].clone(),
                action,
                reward,
                done,
                truncated,
            );
        }
    }

    pub fn states(&self) -> &AllocRingBuffer<Vec<f32>> {
        &self.state
    }

    pub fn next_states(&self) -> &AllocRingBuffer<Vec<f32>> {
        &self.next_state
    }

    pub fn actions(&self) -> &AllocRingBuffer<usize> {
        &self.action
    }

    pub fn rewards(&self) -> &AllocRingBuffer<f32> {
        &self.reward
    }

    pub fn dones(&self) -> &AllocRingBuffer<bool> {
        &self.done
    }

    pub fn truncateds(&self) -> &AllocRingBuffer<bool> {
        &self.truncated
    }

    pub fn len(&self) -> usize {
        self.state.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.is_empty()
    }

    pub fn clear(&mut self) {
        self.state.clear();
        self.next_state.clear();
        self.action.clear();
        self.reward.clear();
        self.done.clear();
        self.truncated.clear();
    }
}
