use burn::prelude::*;
use ringbuffer::{AllocRingBuffer, RingBuffer};

/// Terminal-state encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminalState {
    #[default]
    None,
    Normal,
    Truncated,
}

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
    for &i in indices {
        states.extend(&data[i]);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub fn get_states_batch_range<B: Backend>(
    data: &AllocRingBuffer<Vec<f32>>,
    start: usize,
    end: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let width = data[0].len();
    let mut states: Vec<f32> = Vec::with_capacity((end - start) * width);
    for i in start..end {
        states.extend(&data[i]);
    }
    Tensor::from_data(TensorData::new(states, [end - start, width]), device)
}

pub fn get_log_probs_batch<B: Backend>(
    data: &AllocRingBuffer<f32>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut states: Vec<f32> = Vec::with_capacity(indices.len());
    for &i in indices {
        states.push(data[i]);
    }

    Tensor::from_data(TensorData::new(states, [indices.len(), 1]), device)
}

pub fn get_action_batch<B: Backend>(
    data: &AllocRingBuffer<usize>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let shape = [indices.len(), 1];
    let mut states: Vec<u32> = Vec::with_capacity(shape[0]);
    for &i in indices {
        states.push(data[i] as u32);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub fn get_generic_batch<B: Backend>(
    data: &[f32],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut states: Vec<f32> = Vec::with_capacity(indices.len());
    for &i in indices {
        states.push(data[i]);
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
    for &i in indices {
        for &v in &data[i] {
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

    /// Push a complete per-player trajectory.
    /// All vectors must have the same length.
    /// `trunc_next_state` is `Some` only when the last terminal is `TERMINAL_TRUNCATED`.
    #[allow(clippy::too_many_arguments)]
    pub fn push_player(
        &mut self,
        states: Vec<Vec<f32>>,
        actions: Vec<usize>,
        log_probs: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<TerminalState>,
        action_masks: Vec<Vec<bool>>,
        trunc_next_state: Option<Vec<f32>>,
    ) {
        let n = states.len();
        debug_assert_eq!(n, actions.len());
        debug_assert_eq!(n, log_probs.len());
        debug_assert_eq!(n, rewards.len());
        debug_assert_eq!(n, terminals.len());
        debug_assert_eq!(n, action_masks.len());

        self.states.extend(states);
        self.actions.extend(actions);
        self.log_probs.extend(log_probs);
        self.rewards.extend(rewards);
        self.terminals.extend(terminals);
        self.action_masks.extend(action_masks);
        if let Some(ns) = trunc_next_state {
            self.trunc_next_states.enqueue(ns);
        }
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
