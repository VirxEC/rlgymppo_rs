use burn::prelude::*;

/// Terminal-state encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerminalState {
    #[default]
    None,
    Normal,
    Truncated,
}

pub fn get_batch_1d<T: Copy>(data: &[T], indices: &[usize]) -> Vec<T> {
    indices.iter().map(|i| data[*i]).collect::<Vec<_>>()
}

pub fn get_states_batch<B: Backend>(
    data: &[Vec<f32>],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [indices.len(), data[0].len()];
    let mut states = vec![0.0; shape[0] * shape[1]];
    for (row, &i) in states.chunks_exact_mut(shape[1]).zip(indices) {
        row.copy_from_slice(&data[i]);
    }

    Tensor::from_data(TensorData::new(states, shape), device)
}

pub fn get_states_batch_range<B: Backend>(
    data: &[Vec<f32>],
    start: usize,
    end: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let width = data[0].len();
    let mut states = vec![0.0; (end - start) * width];
    for (row, i) in states.chunks_exact_mut(width).zip(start..end) {
        row.copy_from_slice(&data[i]);
    }
    Tensor::from_data(TensorData::new(states, [end - start, width]), device)
}

pub fn get_log_probs_batch<B: Backend>(
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

pub fn get_action_batch<B: Backend>(
    data: &[usize],
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
    data: &[Vec<bool>],
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let shape = [indices.len(), data[0].len()];
    let mut masks = vec![0.0; shape[0] * shape[1]];
    for (row, &i) in masks.chunks_exact_mut(shape[1]).zip(indices) {
        for (mask, &valid) in row.iter_mut().zip(&data[i]) {
            *mask = valid as u8 as f32;
        }
    }

    Tensor::from_data(TensorData::new(masks, shape), device)
}

#[derive(Clone)]
pub struct Memory {
    states: Vec<Vec<f32>>,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    /// Unified terminal encoding per step: TERMINAL_NONE / NORMAL / TRUNCATED.
    terminals: Vec<TerminalState>,
    /// Observations immediately after a truncated step, used for critic bootstrapping.
    /// Stored in the same order as TERMINAL_TRUNCATED entries appear in `terminals`.
    trunc_next_states: Vec<Vec<f32>>,
    /// Action-validity mask per player-step, stored in the same order as `states`.
    action_masks: Vec<Vec<bool>>,
}

impl Default for Memory {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Memory {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::with_capacity(capacity),
            actions: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminals: Vec::with_capacity(capacity),
            // Truncations are sparse relative to rollout steps, so reserving one
            // slot per step wastes substantial CPU memory.
            trunc_next_states: Vec::new(),
            action_masks: Vec::with_capacity(capacity),
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
            self.trunc_next_states.push(ns);
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

    /// Move at most `max_steps` samples from `other` into this memory.
    ///
    /// Truncation next states are stored independently, in terminal order, so
    /// only the entries associated with truncated steps in the moved prefix are
    /// retained. The unclaimed suffix is dropped with `other`.
    pub fn merge_prefix(&mut self, other: Memory, max_steps: usize) {
        let steps = max_steps.min(other.len());
        if steps == 0 {
            return;
        }

        let truncations = other
            .terminals
            .iter()
            .take(steps)
            .filter(|&&terminal| terminal == TerminalState::Truncated)
            .count();
        let Memory {
            states,
            actions,
            log_probs,
            rewards,
            terminals,
            trunc_next_states,
            action_masks,
        } = other;

        self.states.extend(states.into_iter().take(steps));
        self.actions.extend(actions.into_iter().take(steps));
        self.log_probs.extend(log_probs.into_iter().take(steps));
        self.rewards.extend(rewards.into_iter().take(steps));
        self.terminals.extend(terminals.into_iter().take(steps));
        self.action_masks
            .extend(action_masks.into_iter().take(steps));
        self.trunc_next_states
            .extend(trunc_next_states.into_iter().take(truncations));
    }

    pub fn states(&self) -> &[Vec<f32>] {
        &self.states
    }

    pub fn actions(&self) -> &[usize] {
        &self.actions
    }

    pub fn log_probs(&self) -> &[f32] {
        &self.log_probs
    }

    pub fn rewards(&self) -> &[f32] {
        &self.rewards
    }

    pub fn terminals(&self) -> &[TerminalState] {
        &self.terminals
    }

    pub fn trunc_next_states(&self) -> &[Vec<f32>] {
        &self.trunc_next_states
    }

    pub fn action_masks(&self) -> &[Vec<bool>] {
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

#[cfg(test)]
mod regression_tests {
    use super::*;

    fn push_steps(memory: &mut Memory, start: usize, count: usize, terminal: TerminalState) {
        let mut terminals = vec![TerminalState::None; count];
        if let Some(last) = terminals.last_mut() {
            *last = terminal;
        }
        memory.push_player(
            (start..start + count).map(|i| vec![i as f32]).collect(),
            (start..start + count).collect(),
            vec![0.0; count],
            vec![1.0; count],
            terminals,
            vec![vec![true]; count],
            (terminal == TerminalState::Truncated).then(|| vec![(start + count) as f32]),
        );
    }

    #[test]
    fn regression_capacity_hint_does_not_drop_rollout_samples() {
        let mut memory = Memory::with_capacity(2);
        push_steps(&mut memory, 0, 5, TerminalState::None);

        assert_eq!(memory.len(), 5);
        assert_eq!(memory.actions(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn regression_prefix_merge_keeps_matching_truncation_states() {
        let mut source = Memory::with_capacity(1);
        push_steps(&mut source, 0, 1, TerminalState::Truncated);
        push_steps(&mut source, 1, 1, TerminalState::Truncated);
        push_steps(&mut source, 2, 2, TerminalState::Truncated);

        let mut destination = Memory::with_capacity(3);
        destination.merge_prefix(source, 3);

        assert_eq!(destination.actions(), &[0, 1, 2]);
        assert_eq!(
            destination.terminals(),
            &[
                TerminalState::Truncated,
                TerminalState::Truncated,
                TerminalState::None,
            ]
        );
        assert_eq!(destination.trunc_next_states(), &[vec![1.0], vec![2.0]]);
    }

    #[test]
    fn regression_truncation_storage_is_not_preallocated_per_step() {
        let memory = Memory::with_capacity(10_000);

        assert_eq!(memory.trunc_next_states.capacity(), 0);
        assert!(memory.states.capacity() >= 10_000);
    }
}
