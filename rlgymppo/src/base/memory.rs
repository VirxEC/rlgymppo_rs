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
    data: &[f32],
    width: usize,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut states = Vec::with_capacity(indices.len() * width);
    for &index in indices {
        let start = index * width;
        states.extend_from_slice(&data[start..start + width]);
    }

    Tensor::from_data(TensorData::new(states, [indices.len(), width]), device)
}

pub fn get_states_batch_range<B: Backend>(
    data: &[f32],
    width: usize,
    start: usize,
    end: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let start_offset = start * width;
    let end_offset = end * width;
    Tensor::from_data(
        TensorData::new(
            data[start_offset..end_offset].to_vec(),
            [end - start, width],
        ),
        device,
    )
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
    data: &[bool],
    width: usize,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut masks = Vec::with_capacity(indices.len() * width);
    for &index in indices {
        let start = index * width;
        masks.extend(
            data[start..start + width]
                .iter()
                .map(|&valid| valid as u8 as f32),
        );
    }

    Tensor::from_data(TensorData::new(masks, [indices.len(), width]), device)
}

#[derive(Clone)]
pub struct Memory {
    /// Per-step observations stored row-major as `[step * state_width..]`.
    states: Vec<f32>,
    state_width: usize,
    actions: Vec<usize>,
    log_probs: Vec<f32>,
    rewards: Vec<f32>,
    /// Unified terminal encoding per step: TERMINAL_NONE / NORMAL / TRUNCATED.
    terminals: Vec<TerminalState>,
    /// Observations immediately after truncated steps, stored row-major.
    /// Entries are ordered the same way as TERMINAL_TRUNCATED entries appear in
    /// `terminals`.
    trunc_next_states: Vec<f32>,
    /// Action-validity masks stored row-major as `[step * action_mask_width..]`.
    action_masks: Vec<bool>,
    action_mask_width: usize,
    /// Per-step observations from the old (teacher) obs builder, stored
    /// row-major as `[step * old_state_width..]`. Empty when no old obs
    /// builder is configured (same-obs transfer learning).
    old_states: Vec<f32>,
    old_state_width: usize,
    /// Baseline capacity in rollout rows. Flat buffers convert this to scalar
    /// capacity using their stable row width.
    baseline_steps: usize,
}

impl Default for Memory {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl Memory {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            states: Vec::new(),
            state_width: 0,
            actions: Vec::with_capacity(capacity),
            log_probs: Vec::with_capacity(capacity),
            rewards: Vec::with_capacity(capacity),
            terminals: Vec::with_capacity(capacity),
            // Truncations are sparse relative to rollout steps, so reserving one
            // slot per step wastes substantial CPU memory.
            trunc_next_states: Vec::new(),
            action_masks: Vec::new(),
            action_mask_width: 0,
            old_states: Vec::new(),
            old_state_width: 0,
            baseline_steps: capacity,
        }
    }

    fn set_widths(&mut self, state_width: usize, action_mask_width: usize, old_state_width: usize) {
        debug_assert!(state_width > 0);
        if self.state_width == 0 {
            self.state_width = state_width;
            self.action_mask_width = action_mask_width;
            self.old_state_width = old_state_width;
            self.states.reserve(self.baseline_steps * state_width);
            self.action_masks
                .reserve(self.baseline_steps * action_mask_width);
            self.old_states
                .reserve(self.baseline_steps * old_state_width);
        } else {
            debug_assert_eq!(self.state_width, state_width);
            debug_assert_eq!(self.action_mask_width, action_mask_width);
            debug_assert_eq!(self.old_state_width, old_state_width);
        }
    }

    /// Push a complete per-player trajectory.
    /// All vectors must have the same length.
    /// `trunc_next_state` is `Some` only when the last terminal is `Truncated`.
    /// `old_states` holds the per-step observations produced by the teacher's
    /// (old) obs builder, row-major with width `old_state_width`; pass empty
    /// buffers when no old obs builder is configured.
    #[allow(clippy::too_many_arguments)]
    pub fn push_player(
        &mut self,
        states: Vec<f32>,
        state_width: usize,
        actions: Vec<usize>,
        log_probs: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<TerminalState>,
        action_masks: Vec<bool>,
        action_mask_width: usize,
        old_states: Vec<f32>,
        old_state_width: usize,
        trunc_next_state: Option<Vec<f32>>,
    ) {
        let n = actions.len();
        debug_assert_eq!(states.len(), n * state_width);
        debug_assert_eq!(n, log_probs.len());
        debug_assert_eq!(n, rewards.len());
        debug_assert_eq!(n, terminals.len());
        debug_assert_eq!(action_masks.len(), n * action_mask_width);
        debug_assert_eq!(old_states.len(), n * old_state_width);
        if n == 0 {
            return;
        }

        self.set_widths(state_width, action_mask_width, old_state_width);
        if let Some(ref ns) = trunc_next_state {
            debug_assert_eq!(ns.len(), state_width);
        }

        self.states.extend(states);
        self.actions.extend(actions);
        self.log_probs.extend(log_probs);
        self.rewards.extend(rewards);
        self.terminals.extend(terminals);
        self.action_masks.extend(action_masks);
        self.old_states.extend(old_states);
        if let Some(ns) = trunc_next_state {
            self.trunc_next_states.extend(ns);
        }
    }

    pub fn merge(&mut self, other: Memory) {
        let Memory {
            states,
            state_width,
            actions,
            log_probs,
            rewards,
            terminals,
            trunc_next_states,
            action_masks,
            action_mask_width,
            old_states,
            old_state_width,
            ..
        } = other;

        if !actions.is_empty() {
            self.set_widths(state_width, action_mask_width, old_state_width);
        }
        self.states.extend(states);
        self.actions.extend(actions);
        self.log_probs.extend(log_probs);
        self.rewards.extend(rewards);
        self.terminals.extend(terminals);
        self.trunc_next_states.extend(trunc_next_states);
        self.action_masks.extend(action_masks);
        self.old_states.extend(old_states);
    }

    /// Move at most `max_steps` samples from `other` into this memory.
    ///
    /// Truncation next states are stored independently, in terminal order, so
    /// only the entries associated with truncated steps in the moved prefix are
    /// retained. If the prefix cuts a non-terminal trajectory, the retained
    /// final row is repaired with the incoming observation as its bootstrap
    /// state. The unclaimed suffix is dropped with `other`.
    pub fn merge_prefix(&mut self, other: Memory, max_steps: usize) {
        let steps = max_steps.min(other.len());
        if steps == 0 {
            return;
        }

        let Memory {
            states,
            state_width,
            actions,
            log_probs,
            rewards,
            terminals,
            trunc_next_states,
            action_masks,
            action_mask_width,
            old_states,
            old_state_width,
            ..
        } = other;
        let mut terminals = terminals;
        let truncations = terminals
            .iter()
            .take(steps)
            .filter(|&&terminal| terminal == TerminalState::Truncated)
            .count();
        let cut_boundary = steps < actions.len() && terminals[steps - 1] == TerminalState::None;
        if cut_boundary {
            // The first discarded row is still available in `states`, so use
            // it as the critic bootstrap for the repaired boundary.
            terminals[steps - 1] = TerminalState::Truncated;
        }

        self.set_widths(state_width, action_mask_width, old_state_width);
        self.states
            .extend(states.iter().copied().take(steps * state_width));
        self.actions.extend(actions.into_iter().take(steps));
        self.log_probs.extend(log_probs.into_iter().take(steps));
        self.rewards.extend(rewards.into_iter().take(steps));
        self.terminals.extend(terminals.into_iter().take(steps));
        self.action_masks
            .extend(action_masks.into_iter().take(steps * action_mask_width));
        self.old_states
            .extend(old_states.into_iter().take(steps * old_state_width));
        self.trunc_next_states.extend(
            trunc_next_states
                .into_iter()
                .take(truncations * state_width),
        );
        if cut_boundary {
            let next_start = steps * state_width;
            self.trunc_next_states
                .extend_from_slice(&states[next_start..next_start + state_width]);
        }
    }

    /// Validate the row and boundary contract expected by the learner.
    pub fn validate(&self) -> Result<(), String> {
        let rows = self.len();
        if self.states.len() != rows.saturating_mul(self.state_width) {
            return Err(format!(
                "states has {} scalars for {rows} rows of width {}",
                self.states.len(),
                self.state_width
            ));
        }
        if self.log_probs.len() != rows || self.rewards.len() != rows {
            return Err("log_probs and rewards must be row-aligned with actions".into());
        }
        if self.terminals.len() != rows {
            return Err("terminals must be row-aligned with actions".into());
        }
        if self.action_masks.len() != rows.saturating_mul(self.action_mask_width) {
            return Err(format!(
                "action_masks has {} values for {rows} rows of width {}",
                self.action_masks.len(),
                self.action_mask_width
            ));
        }
        if self.old_states.len() != rows.saturating_mul(self.old_state_width) {
            return Err(format!(
                "old_states has {} values for {rows} rows of width {}",
                self.old_states.len(),
                self.old_state_width
            ));
        }
        let truncations = self
            .terminals
            .iter()
            .filter(|&&terminal| terminal == TerminalState::Truncated)
            .count();
        if self.trunc_next_states.len() != truncations.saturating_mul(self.state_width) {
            return Err(format!(
                "trunc_next_states has {} scalars for {truncations} truncation rows of width {}",
                self.trunc_next_states.len(),
                self.state_width
            ));
        }
        if rows > 0 && self.terminals[rows - 1] == TerminalState::None {
            return Err("the final learner row must have an explicit terminal boundary".into());
        }
        Ok(())
    }

    pub fn states(&self) -> &[f32] {
        &self.states
    }

    pub fn state_width(&self) -> usize {
        self.state_width
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

    pub fn trunc_next_states(&self) -> &[f32] {
        &self.trunc_next_states
    }

    pub fn truncation_len(&self) -> usize {
        self.trunc_next_states
            .len()
            .checked_div(self.state_width)
            .unwrap_or(0)
    }

    pub fn action_masks(&self) -> &[bool] {
        &self.action_masks
    }

    pub fn action_mask_width(&self) -> usize {
        self.action_mask_width
    }

    /// Per-step observations from the old (teacher) obs builder, row-major.
    /// Empty when no old obs builder is configured.
    pub fn old_states(&self) -> &[f32] {
        &self.old_states
    }

    pub fn old_state_width(&self) -> usize {
        self.old_state_width
    }

    pub fn len(&self) -> usize {
        self.actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    pub fn clear(&mut self) {
        // Keep a small baseline for the next collection, but discard any
        // high-water growth from an unusually large rollout or episode.
        self.states.clear();
        self.states
            .shrink_to(self.baseline_steps * self.state_width);
        self.actions.clear();
        self.actions.shrink_to(self.baseline_steps);
        self.log_probs.clear();
        self.log_probs.shrink_to(self.baseline_steps);
        self.rewards.clear();
        self.rewards.shrink_to(self.baseline_steps);
        self.terminals.clear();
        self.terminals.shrink_to(self.baseline_steps);
        self.trunc_next_states.clear();
        self.trunc_next_states.shrink_to(0);
        self.action_masks.clear();
        self.action_masks
            .shrink_to(self.baseline_steps * self.action_mask_width);
        self.old_states.clear();
        self.old_states
            .shrink_to(self.baseline_steps * self.old_state_width);
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
            (start..start + count).map(|i| i as f32).collect::<Vec<_>>(),
            1,
            (start..start + count).collect(),
            vec![0.0; count],
            vec![1.0; count],
            terminals,
            vec![true; count],
            1,
            Vec::new(),
            0,
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
                TerminalState::Truncated,
            ]
        );
        assert_eq!(destination.trunc_next_states(), &[1.0, 2.0, 3.0]);
        assert!(destination.validate().is_ok());
    }

    #[test]
    fn regression_prefix_merge_preserves_an_existing_terminal_boundary() {
        let mut source = Memory::with_capacity(3);
        push_steps(&mut source, 0, 2, TerminalState::Normal);
        push_steps(&mut source, 2, 2, TerminalState::Normal);

        let mut destination = Memory::with_capacity(1);
        destination.merge_prefix(source, 3);

        assert_eq!(
            destination.terminals(),
            &[
                TerminalState::None,
                TerminalState::Normal,
                TerminalState::Truncated
            ]
        );
        assert_eq!(destination.trunc_next_states(), &[3.0]);
        assert!(destination.validate().is_ok());
    }

    #[test]
    fn regression_flat_storage_preserves_observation_rows() {
        let mut memory = Memory::with_capacity(2);
        memory.push_player(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            vec![0, 1],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![TerminalState::None, TerminalState::Normal],
            vec![true, false, false, true],
            2,
            Vec::new(),
            0,
            None,
        );

        assert_eq!(memory.states(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(memory.state_width(), 2);
        assert!(memory.states.capacity() >= 2 * memory.state_width());
        assert_eq!(memory.action_masks(), &[true, false, false, true]);
        assert_eq!(memory.action_mask_width(), 2);
        assert!(memory.action_masks.capacity() >= 2 * memory.action_mask_width());
    }

    #[test]
    fn regression_clear_shrinks_growth_to_baseline() {
        let baseline = 8;
        let mut memory = Memory::with_capacity(baseline);
        push_steps(&mut memory, 0, 10_000, TerminalState::None);
        let grown_state_capacity = memory.states.capacity();
        let grown_mask_capacity = memory.action_masks.capacity();

        memory.clear();

        assert!(memory.states.capacity() >= baseline);
        assert!(memory.states.capacity() < grown_state_capacity);
        assert!(memory.action_masks.capacity() >= baseline);
        assert!(memory.action_masks.capacity() < grown_mask_capacity);
        assert_eq!(memory.state_width(), 1);
        assert_eq!(memory.action_mask_width(), 1);
        assert_eq!(memory.actions.capacity(), baseline);
    }
}
