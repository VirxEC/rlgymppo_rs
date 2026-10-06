use burn::prelude::*;
use burn::tensor::{DType, f16};
use rayon::prelude::*;

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

/// Convert one state row range from `f32` to `f16`. Runs on all CPU cores.
/// Values round to nearest; inputs are normalized observations that fit
/// easily (magnitude far below 65504, precision around 3 decimal digits).
fn convert_states_to_f16(data: &[f32]) -> Vec<f16> {
    data.par_iter().map(|&x| f16::from_f32(x)).collect()
}

/// Upload state rows `[start, end)` as device `f16`: half the bytes cross
/// the bus and half the VRAM stays resident.
///
/// The returned tensor holds half-precision storage despite its `Tensor<B,
/// 2>` type. Slice it with `narrow` or `select` first, then upcast the
/// slice with `.cast(DType::F32)` before running the model. Never feed it
/// to the model directly: mixed-precision matmuls fail on dtype mismatch.
/// See [`convert_states_to_f16`] for the precision bounds.
pub fn get_states_batch_range<B: Backend>(
    data: &[f32],
    width: usize,
    start: usize,
    end: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let rows = end - start;
    let half = convert_states_to_f16(&data[start * width..end * width]);
    Tensor::from_data(TensorData::new(half, [rows, width]), (device, DType::F16))
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
    /// One byte per entry (1 = valid, 0 = invalid): same RAM as `bool`,
    /// but staging converts to `f32` with a single vectorized pass.
    action_masks: Vec<u8>,
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

/// Destination for claimed trajectory rows.
///
/// `Memory` appends (legacy path); [`MemoryShard`] writes positionally
/// into an exclusive row range of a shared memory (exact-budget path).
/// Both funnel through [`Memory::push_player`]'s argument list so the
/// claim code is identical for either destination.
#[allow(clippy::too_many_arguments)]
pub trait ClaimTarget {
    fn push_player(
        &mut self,
        states: Vec<f32>,
        state_width: usize,
        actions: Vec<usize>,
        log_probs: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<TerminalState>,
        action_masks: Vec<u8>,
        action_mask_width: usize,
        old_states: Vec<f32>,
        old_state_width: usize,
        trunc_next_state: Option<Vec<f32>>,
    );
}

#[allow(clippy::too_many_arguments)]
impl ClaimTarget for Memory {
    fn push_player(
        &mut self,
        states: Vec<f32>,
        state_width: usize,
        actions: Vec<usize>,
        log_probs: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<TerminalState>,
        action_masks: Vec<u8>,
        action_mask_width: usize,
        old_states: Vec<f32>,
        old_state_width: usize,
        trunc_next_state: Option<Vec<f32>>,
    ) {
        Memory::push_player(
            self,
            states,
            state_width,
            actions,
            log_probs,
            rewards,
            terminals,
            action_masks,
            action_mask_width,
            old_states,
            old_state_width,
            trunc_next_state,
        );
    }
}

/// Exclusive row range of a shared [`Memory`] that one pool fills directly.
/// Per-row buffers are disjoint slices (no synchronization needed); the
/// sparse truncation tail is kept aside and concatenated in pool order at
/// the join, exactly like [`Memory::merge`] does today.
#[allow(clippy::too_many_arguments)]
pub struct MemoryShard<'a> {
    states: &'a mut [f32],
    actions: &'a mut [usize],
    log_probs: &'a mut [f32],
    rewards: &'a mut [f32],
    terminals: &'a mut [TerminalState],
    action_masks: &'a mut [u8],
    old_states: &'a mut [f32],
    trunc_next_states: Vec<f32>,
    state_width: usize,
    action_mask_width: usize,
    old_state_width: usize,
    rows_written: usize,
    capacity_rows: usize,
}

impl<'a> MemoryShard<'a> {
    /// Rows written so far. The join asserts this equals the share.
    pub fn rows_written(&self) -> usize {
        self.rows_written
    }

    /// Take the shard's truncation tail for pool-order concatenation.
    pub fn take_trunc_next_states(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.trunc_next_states)
    }
}

#[allow(clippy::too_many_arguments)]
impl ClaimTarget for MemoryShard<'_> {
    fn push_player(
        &mut self,
        states: Vec<f32>,
        state_width: usize,
        actions: Vec<usize>,
        log_probs: Vec<f32>,
        rewards: Vec<f32>,
        terminals: Vec<TerminalState>,
        action_masks: Vec<u8>,
        action_mask_width: usize,
        old_states: Vec<f32>,
        old_state_width: usize,
        trunc_next_state: Option<Vec<f32>>,
    ) {
        let n = actions.len();
        debug_assert_eq!(states.len(), n * state_width);
        debug_assert_eq!(state_width, self.state_width);
        debug_assert_eq!(action_mask_width, self.action_mask_width);
        debug_assert_eq!(old_state_width, self.old_state_width);
        debug_assert!(self.rows_written + n <= self.capacity_rows);
        let start = self.rows_written;
        let end = start + n;
        self.states[start * state_width..end * state_width].copy_from_slice(&states);
        self.actions[start..end].copy_from_slice(&actions);
        self.log_probs[start..end].copy_from_slice(&log_probs);
        self.rewards[start..end].copy_from_slice(&rewards);
        self.terminals[start..end].copy_from_slice(&terminals);
        self.action_masks[start * action_mask_width..end * action_mask_width]
            .copy_from_slice(&action_masks);
        self.old_states[start * old_state_width..end * old_state_width]
            .copy_from_slice(&old_states);
        if let Some(ns) = trunc_next_state {
            debug_assert_eq!(ns.len(), state_width);
            self.trunc_next_states.extend(ns);
        }
        self.rows_written = end;
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
        action_masks: Vec<u8>,
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
        // No `..` catch-all: listing every field forces a compile error when
        // a field is added, so merges can never silently drop a buffer again.
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
            baseline_steps: _,
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
            baseline_steps: _,
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
        if self.log_probs.len() != rows {
            return Err(format!(
                "log_probs has {} values for {rows} rows",
                self.log_probs.len()
            ));
        }
        if self.rewards.len() != rows {
            return Err(format!(
                "rewards has {} values for {rows} rows",
                self.rewards.len()
            ));
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

    pub fn action_masks(&self) -> &[u8] {
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

    /// Append one pool's truncation tail, preserving pool order at the join.
    pub fn append_trunc_next_states(&mut self, states: Vec<f32>) {
        self.trunc_next_states.extend(states);
    }

    /// Reset the sparse truncation tail, retaining capacity. The per-row
    /// buffers are left at full length: the exact-budget path overwrites
    /// every row in place (asserted full at the join).
    pub fn clear_trunc_next_states(&mut self) {
        self.trunc_next_states.clear();
    }

    /// Split exclusive row ranges for `shares` (prefix sums, pool order).
    /// The memory must already hold full length at the sum of `shares`
    /// with widths set. The sparse truncation tail is not split: each
    /// shard keeps its own and the join concatenates them in pool order.
    pub fn shard_mut(&mut self, shares: &[usize]) -> Vec<MemoryShard<'_>> {
        let state_width = self.state_width;
        let mask_width = self.action_mask_width;
        let old_width = self.old_state_width;
        let mut states = self.states.as_mut_slice();
        let mut actions = self.actions.as_mut_slice();
        let mut log_probs = self.log_probs.as_mut_slice();
        let mut rewards = self.rewards.as_mut_slice();
        let mut terminals = self.terminals.as_mut_slice();
        let mut masks = self.action_masks.as_mut_slice();
        let mut old = self.old_states.as_mut_slice();
        let mut shards = Vec::with_capacity(shares.len());
        for &share in shares {
            let (s_head, s_tail) = states.split_at_mut(share * state_width);
            states = s_tail;
            let (a_head, a_tail) = actions.split_at_mut(share);
            actions = a_tail;
            let (l_head, l_tail) = log_probs.split_at_mut(share);
            log_probs = l_tail;
            let (r_head, r_tail) = rewards.split_at_mut(share);
            rewards = r_tail;
            let (t_head, t_tail) = terminals.split_at_mut(share);
            terminals = t_tail;
            let (m_head, m_tail) = masks.split_at_mut(share * mask_width);
            masks = m_tail;
            let (o_head, o_tail) = old.split_at_mut(share * old_width);
            old = o_tail;
            shards.push(MemoryShard {
                states: s_head,
                actions: a_head,
                log_probs: l_head,
                rewards: r_head,
                terminals: t_head,
                action_masks: m_head,
                old_states: o_head,
                trunc_next_states: Vec::new(),
                state_width,
                action_mask_width: mask_width,
                old_state_width: old_width,
                rows_written: 0,
                capacity_rows: share,
            });
        }
        shards
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
            vec![1u8; count],
            1,
            Vec::new(),
            0,
            (terminal == TerminalState::Truncated).then(|| vec![(start + count) as f32]),
        );
    }

    #[test]
    fn f16_conversion_stays_within_half_ulp_bounds() {
        // Normalized-obs magnitudes: exact for small integers, tiny relative
        // error elsewhere. Guards against a wrong conversion routine.
        let values = [
            0.0, 1.0, -1.0, 0.5, 0.12, 100.0, -5120.0, 2044.0, 0.001, 123.456,
        ];
        let half = convert_states_to_f16(&values);
        assert_eq!(half.len(), values.len());
        for (&original, &converted) in values.iter().zip(half.iter()) {
            let back = converted.to_f32();
            if original == 0.0 {
                assert_eq!(back, 0.0);
            } else {
                let rel_err = ((back - original) / original).abs();
                assert!(rel_err < 0.001, "{original} -> {back}");
            }
        }
    }

    #[cfg(all(test, feature = "flex"))]
    mod flex_gated {
        use burn::backend::Flex;
        use burn::tensor::FloatDType;

        use super::*;

        #[test]
        fn f16_upload_narrow_cast_roundtrip() {
            // Exercises the learner's exact plumbing: `f16` device storage,
            // `narrow` a slice, upcast, read back. Values must match within
            // half precision.
            let device = Default::default();
            let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
            let states = get_states_batch_range::<Flex>(&data, 8, 0, 4, &device);
            let back: Vec<f32> = states
                .narrow(0, 1, 2)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .unwrap();
            assert_eq!(back.len(), 16);
            for (actual, expected) in back.iter().zip(data[8..24].iter()) {
                assert!((actual - expected).abs() < 0.01, "{actual} vs {expected}");
            }
        }

        #[test]
        fn u8_mask_upload_narrow_cast_is_exact() {
            // Masks upload as `u8` once and slices upcast on the device.
            // Values are exactly 0/1, so the readback must match bit-exactly.
            // Mirrors the transfer-learning resident mask plumbing.
            let device = Default::default();
            let data: Vec<u8> = vec![1, 0, 1, 1, 0, 0, 1, 0];
            let masks = Tensor::<Flex, 2, Int>::from_data(TensorData::new(data, [2, 4]), &device)
                .narrow(0, 0, 2)
                .cast(FloatDType::F32);
            let back: Vec<f32> = masks.into_data().to_vec::<f32>().unwrap();
            assert_eq!(back, vec![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
        }
    }

    #[test]
    fn shard_mut_writes_exclusive_row_ranges_in_order() {
        // Two pools claim into one pre-sized memory: rows must land in
        // positional order with no gaps, and truncation tails must stay
        // separable for pool-order concatenation at the join.
        // Steady state: full-length buffers, widths set, no clear.
        let mut memory = Memory::with_capacity(4);
        memory.push_player(
            vec![0.0; 4],
            1,
            vec![0; 4],
            vec![0.0; 4],
            vec![0.0; 4],
            vec![TerminalState::None; 4],
            vec![1u8; 4],
            1,
            Vec::new(),
            0,
            None,
        );
        assert_eq!(memory.len(), 4);

        let mut shards = memory.shard_mut(&[1, 3]);
        shards[1].push_player(
            vec![10.0, 20.0, 30.0],
            1,
            vec![1, 2, 3],
            vec![0.0; 3],
            vec![0.0; 3],
            vec![TerminalState::None; 3],
            vec![0u8; 3],
            1,
            Vec::new(),
            0,
            Some(vec![99.0]),
        );
        shards[0].push_player(
            vec![7.0],
            1,
            vec![0],
            vec![0.0],
            vec![0.0],
            vec![TerminalState::Normal],
            vec![1u8],
            1,
            Vec::new(),
            0,
            None,
        );
        assert_eq!(shards[0].rows_written(), 1);
        assert_eq!(shards[1].rows_written(), 3);
        let tail = shards[1].take_trunc_next_states();
        drop(shards);

        assert_eq!(memory.states(), &[7.0, 10.0, 20.0, 30.0]);
        assert_eq!(memory.actions(), &[0, 1, 2, 3]);
        assert_eq!(
            memory.terminals(),
            &[
                TerminalState::Normal,
                TerminalState::None,
                TerminalState::None,
                TerminalState::None
            ]
        );
        assert_eq!(tail, vec![99.0]);
        memory.append_trunc_next_states(tail);
        assert_eq!(memory.trunc_next_states(), &[99.0]);
    }

    #[test]
    fn regression_capacity_hint_does_not_drop_rollout_samples() {
        let mut memory = Memory::with_capacity(2);
        push_steps(&mut memory, 0, 5, TerminalState::None);

        assert_eq!(memory.len(), 5);
        assert_eq!(memory.actions(), &[0, 1, 2, 3, 4]);
    }

    #[test]
    fn regression_merge_preserves_every_buffer() {
        // Guards the multi-pool path: `ThreadSim` merges one memory per pool,
        // and a dropped buffer surfaces only at learner validation time.
        // Distinct values per row catch silent drops.
        let mut first = Memory::with_capacity(2);
        first.push_player(
            vec![1.0, 2.0],
            1,
            vec![0, 1],
            vec![0.1, 0.2],
            vec![1.0, 2.0],
            vec![TerminalState::None, TerminalState::Normal],
            vec![1u8, 1u8],
            1,
            Vec::new(),
            0,
            None,
        );
        let mut second = Memory::with_capacity(2);
        second.push_player(
            vec![3.0],
            1,
            vec![2],
            vec![0.3],
            vec![3.0],
            vec![TerminalState::Normal],
            vec![1u8],
            1,
            Vec::new(),
            0,
            None,
        );

        first.merge(second);

        assert!(first.validate().is_ok());
        assert_eq!(first.len(), 3);
        assert_eq!(first.states(), &[1.0, 2.0, 3.0]);
        assert_eq!(first.actions(), &[0, 1, 2]);
        assert_eq!(first.log_probs(), &[0.1, 0.2, 0.3]);
        assert_eq!(first.rewards(), &[1.0, 2.0, 3.0]);
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
            vec![1u8, 0, 0, 1],
            2,
            Vec::new(),
            0,
            None,
        );

        assert_eq!(memory.states(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(memory.state_width(), 2);
        assert!(memory.states.capacity() >= 2 * memory.state_width());
        assert_eq!(memory.action_masks(), &[1u8, 0, 0, 1]);
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
