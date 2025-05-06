use burn::prelude::*;
use ringbuffer::{AllocRingBuffer, RingBuffer};
use std::iter;

pub fn get_batch_1d<T: Copy>(data: &AllocRingBuffer<T>, indices: &[usize]) -> Vec<T> {
    indices.iter().map(|i| data[*i]).collect::<Vec<_>>()
}

pub fn get_states_batch<B: Backend>(
    data: &AllocRingBuffer<Vec<f32>>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [indices.len(), data[0].len()];
    let mut states: Vec<f32> = Vec::with_capacity(shape[0] * shape[1]);
    for i in indices {
        states.extend(&data[*i]);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub fn get_log_probs_batch<B: Backend>(
    data: &AllocRingBuffer<f32>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut states: Vec<f32> = Vec::with_capacity(indices.len());
    for i in indices {
        states.push(data[*i]);
    }

    Tensor::from_data(TensorData::new(states, [indices.len(), 1]), device)
}

pub fn get_action_batch<B: Backend>(
    data: &AllocRingBuffer<usize>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let shape = [indices.len(), 1];
    let mut states: Vec<u32> = Vec::with_capacity(shape[0] * shape[1]);
    for i in indices {
        states.push(data[*i] as u32);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

#[derive(Clone)]
pub struct Memory {
    states: AllocRingBuffer<Vec<f32>>,
    actions: AllocRingBuffer<usize>,
    log_probs: AllocRingBuffer<f32>,
    rewards: AllocRingBuffer<f32>,
    dones: AllocRingBuffer<bool>,
    truncateds: AllocRingBuffer<bool>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Memory {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: AllocRingBuffer::new(capacity),
            actions: AllocRingBuffer::new(capacity),
            log_probs: AllocRingBuffer::new(capacity),
            rewards: AllocRingBuffer::new(capacity),
            dones: AllocRingBuffer::new(capacity),
            truncateds: AllocRingBuffer::new(capacity),
        }
    }

    pub fn push_batch_part_1(&mut self, states: std::vec::Drain<Vec<f32>>, log_probs: Vec<f32>) {
        debug_assert_eq!(states.len(), log_probs.len());

        self.states.extend(states);
        self.log_probs.extend(log_probs);

        debug_assert_eq!(self.states.len(), self.log_probs.len());
    }

    pub fn push_batch_part_2(&mut self, rewards: Vec<f32>, done: bool, truncated: bool) {
        let n = rewards.len();
        debug_assert_eq!(n, rewards.len());

        self.rewards.extend(rewards);
        self.dones.extend(iter::repeat_n(done, n));
        self.truncateds.extend(iter::repeat_n(truncated, n));
    }

    pub fn push_batch_part_3(&mut self, actions: Vec<usize>) {
        self.actions.extend(actions);
        debug_assert_eq!(self.states.len(), self.actions.len());
        debug_assert_eq!(self.states.len(), self.rewards.len());
    }

    pub fn merge(&mut self, other: Memory) {
        self.states.extend(other.states);
        self.actions.extend(other.actions);
        self.log_probs.extend(other.log_probs);
        self.rewards.extend(other.rewards);
        self.dones.extend(other.dones);
        self.truncateds.extend(other.truncateds);
    }

    pub fn states(&self) -> &AllocRingBuffer<Vec<f32>> {
        &self.states
    }

    pub fn actions(&self) -> &AllocRingBuffer<usize> {
        &self.actions
    }

    pub fn log_probs(&self) -> &AllocRingBuffer<f32> {
        &self.log_probs
    }

    pub fn rewards(&self) -> &AllocRingBuffer<f32> {
        &self.rewards
    }

    pub fn dones(&self) -> &AllocRingBuffer<bool> {
        &self.dones
    }

    pub fn truncateds(&self) -> &AllocRingBuffer<bool> {
        &self.truncateds
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.actions.clear();
        self.log_probs.clear();
        self.rewards.clear();
        self.dones.clear();
        self.truncateds.clear();
    }
}
