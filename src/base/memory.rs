use std::iter;

use burn::prelude::*;
use ringbuffer::{AllocRingBuffer, RingBuffer};

/// Terminal-state encoding.
pub type TerminalState = u8;
pub const TERMINAL_NONE: TerminalState = 0;
pub const TERMINAL_NORMAL: TerminalState = 1;
pub const TERMINAL_TRUNCATED: TerminalState = 2;

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

pub fn get_generic_batch<B: Backend>(
    data: &[f32],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut states: Vec<f32> = Vec::with_capacity(indices.len());
    for i in indices {
        states.push(data[*i]);
    }

    Tensor::from_data(TensorData::new(states, [indices.len(), 1]), device)
}

/// Flatten per-player action masks into a [N, n_actions] f32 tensor (1.0 = valid, 0.0 = invalid).
pub fn get_action_masks_batch<B: Backend>(
    data: &AllocRingBuffer<Vec<bool>>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [indices.len(), data[0].len()];
    let mut masks: Vec<f32> = Vec::with_capacity(shape[0] * shape[1]);
    for i in indices {
        for &v in &data[*i] {
            masks.push(if v { 1.0 } else { 0.0 });
        }
    }
    Tensor::from_data(TensorData::new(masks, shape), device)
}

#[derive(Clone)]
pub struct Memory {
    states: AllocRingBuffer<Vec<f32>>,
    actions: AllocRingBuffer<usize>,
    log_probs: AllocRingBuffer<f32>,
    rewards: AllocRingBuffer<f32>,
    /// Unified terminal encoding per step: TERMINAL_NONE / NORMAL / TRUNCATED.
    terminals: AllocRingBuffer<TerminalState>,
    /// Observations immediately after a truncated step, used for critic bootstrapping.
    /// Stored in the same order as TERMINAL_TRUNCATED entries appear in `terminals`.
    trunc_next_states: AllocRingBuffer<Vec<f32>>,
    /// Action-validity mask per player-step, stored in the same order as `states`.
    action_masks: AllocRingBuffer<Vec<bool>>,
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
            terminals: AllocRingBuffer::new(capacity),
            trunc_next_states: AllocRingBuffer::new(capacity),
            action_masks: AllocRingBuffer::new(capacity),
        }
    }

    pub fn push_batch_part_1(
        &mut self,
        states: std::vec::Drain<Vec<f32>>,
        masks: Vec<Vec<bool>>,
        log_probs: Vec<f32>,
    ) {
        debug_assert_eq!(states.len(), log_probs.len());
        debug_assert_eq!(states.len(), masks.len());

        self.states.extend(states);
        self.action_masks.extend(masks);
        self.log_probs.extend(log_probs);

        debug_assert_eq!(self.states.len(), self.log_probs.len());
        debug_assert_eq!(self.states.len(), self.action_masks.len());
    }

    pub fn push_batch_part_2(&mut self, rewards: Vec<f32>, terminal: TerminalState) {
        let n = rewards.len();
        debug_assert_eq!(n, rewards.len());

        self.rewards.extend(rewards);
        self.terminals.extend(iter::repeat_n(terminal, n));
    }

    /// Store the next-state observations for a truncated step, one per player.
    pub fn push_trunc_next_states(&mut self, states: &[Vec<f32>]) {
        self.trunc_next_states.extend(states.iter().cloned());
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
        self.terminals.extend(other.terminals);
        self.trunc_next_states.extend(other.trunc_next_states);
        self.action_masks.extend(other.action_masks);
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

    pub fn terminals(&self) -> &AllocRingBuffer<TerminalState> {
        &self.terminals
    }

    /// Mutate the terminal type at a specific index (used for batch-boundary truncation).
    pub fn set_terminal_at(&mut self, idx: usize, terminal: TerminalState) {
        self.terminals[idx] = terminal;
    }

    pub fn trunc_next_states(&self) -> &AllocRingBuffer<Vec<f32>> {
        &self.trunc_next_states
    }

    pub fn action_masks(&self) -> &AllocRingBuffer<Vec<bool>> {
        &self.action_masks
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
        self.terminals.clear();
        self.trunc_next_states.clear();
        self.action_masks.clear();
    }
}
