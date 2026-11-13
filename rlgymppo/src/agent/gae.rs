use burn::tensor::{Int, Tensor};

use super::config::GaeEstimator;
use crate::base::TerminalState;

/// CPU reference implementation for GAE; kept for the parity tests that
/// validate the device scan against it.
#[cfg(test)]
pub(crate) struct GAEOutput {
    pub(crate) returns: Vec<f32>,
    pub(crate) target_vals: Vec<f32>,
    pub(crate) advantages: Vec<f32>,
    /// Per-row TD residual (`pred_ret - value`), kept for coherence-based
    /// extensions such as adaptive clipping. Raw (pre-standardization).
    pub(crate) deltas: Vec<f32>,
}

/// Calculate returns and GAE advantages from row-aligned rollout data.
///
/// Truncation bootstrap predictions are supplied in forward terminal order.
/// The reverse scan consumes them from the end so each truncated row receives
/// the prediction for its own boundary observation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_gae(
    values: Vec<f32>,
    rewards: Vec<f32>,
    terminals: &[TerminalState],
    trunc_val_preds: &[f32],
    gamma: f32,
    lambda: f32,
    return_std: f32,
    clip_range: f32,
    estimator: GaeEstimator,
) -> GAEOutput {
    let n_returns = values.len();
    assert_eq!(
        rewards.len(),
        n_returns,
        "GAE values and rewards must have the same length"
    );
    assert_eq!(
        terminals.len(),
        n_returns,
        "GAE values and terminals must have the same length"
    );
    let truncation_count = terminals
        .iter()
        .filter(|&&terminal| terminal == TerminalState::Truncated)
        .count();
    assert_eq!(
        trunc_val_preds.len(),
        truncation_count,
        "GAE requires exactly one truncation bootstrap prediction per truncated row"
    );
    if estimator == GaeEstimator::TerminationTime {
        assert!(
            lambda > 0.0 && lambda < 1.0,
            "Termination-time GAE requires lambda in the open interval (0, 1)"
        );
    }

    let mut returns = vec![0.0; n_returns];
    let mut target_vals = vec![0.0; n_returns];
    let mut advantages = vec![0.0; n_returns];
    let mut deltas = vec![0.0; n_returns];

    let mut return_scale = 1.0 / return_std;
    if !return_scale.is_finite() || return_scale == 0.0 {
        return_scale = 1.0;
    }

    let mut last_return = 0.0;
    let mut running_advantage = 0.0;
    let mut trunc_cursor = trunc_val_preds.len();
    let mut remaining = 0_usize;

    for i in (0..n_returns).rev() {
        let term = terminals[i];
        let is_done = term == TerminalState::Normal;
        let is_trunc = term == TerminalState::Truncated;

        // A terminal row starts a fresh finite suffix. Non-terminal rows extend
        // the suffix while scanning toward its beginning.
        if term != TerminalState::None {
            remaining = 1;
        } else {
            remaining += 1;
        }
        assert!(remaining > 0, "GAE suffix length must be positive");

        let not_done = f32::from(!is_done);
        let not_trunc = f32::from(!is_trunc);

        let mut norm_reward = rewards[i] * return_scale;
        if clip_range > 0.0 {
            norm_reward = norm_reward.clamp(-clip_range, clip_range);
        }

        let next_val = if is_trunc {
            trunc_cursor -= 1;
            trunc_val_preds[trunc_cursor]
        } else if !is_done && i + 1 < n_returns {
            values[i + 1]
        } else {
            0.0
        };

        let pred_ret = norm_reward + gamma * next_val * not_done;
        let delta = pred_ret - values[i];
        deltas[i] = delta;

        last_return = rewards[i] + last_return * gamma * not_done * not_trunc;
        returns[i] = last_return;

        let recursive_factor = match estimator {
            GaeEstimator::Truncated => 1.0,
            GaeEstimator::TerminationTime => finite_time_factor(lambda, remaining),
        };
        running_advantage =
            delta + gamma * lambda * not_done * not_trunc * recursive_factor * running_advantage;
        advantages[i] = running_advantage;
        target_vals[i] = values[i] + running_advantage;
    }

    debug_assert_eq!(trunc_cursor, 0);
    GAEOutput {
        returns,
        target_vals,
        advantages,
        deltas,
    }
}

/// Chunk width for the two-level reversed device scan. The transient scan
/// workspace is `n * GAE_CHUNK` f32 values (the per-chunk pairwise weight
/// matrix) plus an `M x M` level-two matrix with `M = ceil(n / GAE_CHUNK)` —
/// about 50 MiB for a 100k-row rollout, independent of anything larger.
const GAE_CHUNK: usize = 128;
/// Floor for scan carry coefficients that are logically zero (trajectory
/// boundaries). The scan runs in the log domain, where a zero factor must
/// contribute an enormous negative exponent — erasing every term that
/// crosses the boundary — rather than a NaN. `ln(1e-15)` keeps a single
/// boundary's leakage at ~1e-15 relative (far below the parity tolerance)
/// while two boundaries underflow to exactly zero.
const GAE_COEF_FLOOR: f32 = 1e-15;
/// Additive exponent fill for masked-out pairwise weights. Valid-region
/// exponents are always <= 0; masked-region exponents are bounded by
/// `(#rows in span) * |ln(GAE_COEF_FLOOR)|` — about 4.4e3 per 128-row chunk
/// and 3.5e6 across 782 chunks — so -1e7 keeps every masked weight
/// underflowing to exactly zero without ever overflowing the valid region.
const GAE_MASK_FILL: f32 = -1.0e7;

/// Device-tensor GAE results. All tensors are `[n, 1]` on the caller's
/// device; the two scalar reductions stay on the device so they can be read
/// back in the learner's end-of-iteration transaction.
pub(crate) struct DeviceGaeOutput {
    pub(crate) returns: Tensor<2>,
    pub(crate) target_vals: Tensor<2>,
    pub(crate) advantages: Tensor<2>,
    pub(crate) deltas: Tensor<2>,
    /// `sum |rewards * return_scale|` (before clipping).
    pub(crate) reward_abs_sum: Tensor<1>,
    /// `sum |clipped reward|` (after clipping).
    pub(crate) clipped_abs_sum: Tensor<1>,
}

/// Per-row scan side-data computed on the CPU from `terminals` in one
/// reverse pass (mirroring `get_gae`'s bookkeeping exactly), returned
/// row-major as an `[n, 4]` grid ready for a single upload:
///
/// - col 0: `is_trunc` (0/1)
/// - col 1: `not_done` (0/1)
/// - col 2: advantage carry `gamma*lambda*not_done*not_trunc*f(remaining)`
/// - col 3: return carry `gamma*not_done*not_trunc`
///
/// Zero carries are floored at `GAE_COEF_FLOOR` (see that constant).
pub(crate) fn gae_cpu_flags(
    terminals: &[TerminalState],
    gamma: f32,
    lambda: f32,
    estimator: GaeEstimator,
) -> Vec<f32> {
    if estimator == GaeEstimator::TerminationTime {
        assert!(
            lambda > 0.0 && lambda < 1.0,
            "Termination-time GAE requires lambda in the open interval (0, 1)"
        );
    }
    let n = terminals.len();
    let mut flags = vec![0.0; n * 4];
    let mut remaining = 0_usize;
    for i in (0..n).rev() {
        let term = terminals[i];
        let is_done = term == TerminalState::Normal;
        let is_trunc = term == TerminalState::Truncated;
        if term != TerminalState::None {
            remaining = 1;
        } else {
            remaining += 1;
        }
        assert!(remaining > 0, "GAE suffix length must be positive");
        let not_done = f32::from(!is_done);
        let not_trunc = f32::from(!is_trunc);
        let recursive_factor = match estimator {
            GaeEstimator::Truncated => 1.0,
            GaeEstimator::TerminationTime => finite_time_factor(lambda, remaining),
        };
        let o = i * 4;
        flags[o] = f32::from(is_trunc);
        flags[o + 1] = not_done;
        flags[o + 2] =
            (gamma * lambda * not_done * not_trunc * recursive_factor).max(GAE_COEF_FLOOR);
        flags[o + 3] = (gamma * not_done * not_trunc).max(GAE_COEF_FLOOR);
    }
    flags
}

/// Calculate returns and GAE advantages on the device with tensor ops.
///
/// Semantics match `get_gae` exactly (see the parity tests in this
/// module); differences are only float-level: the reversed scan runs as a
/// two-level log-domain chunked scan over fixed-size chunks, so results are
/// accurate to a few ulps of accumulated exp/log rounding rather than
/// bit-identical. Coefficients arrive precomputed via [`gae_cpu_flags`];
/// values, rewards, truncation bootstraps and all outputs stay on the
/// device, so the whole pass costs no PCIe crossing.
///
/// `values` and `rewards` are `[n, 1]` f32 on `device`; `flags` is the
/// `[n, 4]` tensor described in [`gae_cpu_flags`]; `trunc_val_preds` is
/// `[m, 1]` in forward terminal order (`m` may be 0).
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_gae_device(
    values: Tensor<2>,
    rewards: Tensor<2>,
    flags: Tensor<2>,
    trunc_val_preds: Tensor<2>,
    gamma: f32,
    return_std: f32,
    clip_range: f32,
) -> DeviceGaeOutput {
    let device = values.device();
    let n = values.shape().dims::<2>()[0];
    assert_eq!(
        rewards.shape().dims::<2>()[0],
        n,
        "GAE values and rewards must have the same length"
    );
    assert_eq!(
        flags.shape().dims::<2>()[0],
        n,
        "GAE flags and values must have the same length"
    );
    let truncation_count = trunc_val_preds.shape().dims::<2>()[0];
    // The caller guarantees `truncation_count` equals the number of
    // Truncated rows in `flags` (it builds both from the same terminal
    // slice); the parity tests enforce that contract against `get_gae`.

    if n == 0 {
        let empty = Tensor::<2>::zeros([0, 1], &device);
        return DeviceGaeOutput {
            returns: empty.clone(),
            target_vals: empty.clone(),
            advantages: empty.clone(),
            deltas: empty,
            reward_abs_sum: Tensor::zeros([1], &device),
            clipped_abs_sum: Tensor::zeros([1], &device),
        };
    }

    let return_scale = {
        let scale = 1.0 / return_std;
        if !scale.is_finite() || scale == 0.0 {
            1.0
        } else {
            scale
        }
    };

    let is_trunc_f = flags.clone().narrow(1, 0, 1);
    let not_done_f = flags.clone().narrow(1, 1, 1);
    let coef_adv = flags.clone().narrow(1, 2, 1);
    let coef_ret = flags.narrow(1, 3, 1);
    let is_trunc = is_trunc_f.clone().greater_scalar(0.5);
    let is_done = not_done_f.clone().lower_scalar(0.5);

    // Scaled and clipped rewards, plus the clip-portion metric reductions.
    let norm_reward = rewards.clone().mul_scalar(return_scale);
    let reward_abs_sum = norm_reward.clone().abs().sum();
    let clipped_reward = if clip_range > 0.0 {
        norm_reward.clone().clamp(-clip_range, clip_range)
    } else {
        norm_reward.clone()
    };
    let clipped_abs_sum = clipped_reward.clone().abs().sum();

    // next_val: values shifted by one row (zero past the rollout end),
    // overridden to 0 on done rows and to the row's own truncation
    // bootstrap on truncated rows. The bootstrap index for each row is the
    // number of truncated rows seen so far (inclusive) minus one, floored
    // at 0 so non-truncated rows gather a harmless in-range value.
    let shifted_next = if n > 1 {
        Tensor::cat(
            vec![
                values.clone().narrow(0, 1, n - 1),
                Tensor::<2>::zeros([1, 1], &device),
            ],
            0,
        )
    } else {
        Tensor::<2>::zeros([n, 1], &device)
    };
    let next_val = if truncation_count > 0 {
        let bootstrap_index = is_trunc_f
            .clone()
            .cumsum(0)
            .add_scalar(-1.0)
            .clamp_min(0.0)
            .int();
        shifted_next
            .mask_fill(is_done.clone(), 0.0)
            .mask_where(is_trunc, trunc_val_preds.gather(0, bootstrap_index))
    } else {
        shifted_next.mask_fill(is_done.clone(), 0.0)
    };

    let deltas = clipped_reward + (next_val * not_done_f).mul_scalar(gamma) - values.clone();

    let advantages = reverse_weighted_scan(deltas.clone(), coef_adv);
    let returns = reverse_weighted_scan(rewards, coef_ret);
    let target_vals = values + advantages.clone();

    DeviceGaeOutput {
        returns,
        target_vals,
        advantages,
        deltas,
        reward_abs_sum,
        clipped_abs_sum,
    }
}

/// Reversed weighted scan over `[n, 1]` rows:
/// `out[i] = input[i] + coef[i] * out[i + 1]`, with `out[n] = 0`.
///
/// Two-level chunked scan: rows are padded into `M` chunks of
/// `GAE_CHUNK`; within a chunk every row's suffix scan is evaluated with
/// one pairwise weight matrix (`exp` of log-coefficient suffix
/// differences); chunk-level carries are then scanned with the same trick
/// over `M` chunk aggregates. Kernel count is constant in `n`.
fn reverse_weighted_scan(input: Tensor<2>, coef: Tensor<2>) -> Tensor<2> {
    let device = input.device();
    let n = input.shape().dims::<2>()[0];
    let chunk = GAE_CHUNK.min(n.max(1));
    let m = n.div_ceil(chunk);
    let padded = m * chunk;

    // Pad with zero inputs and unit coefficients: unit carries pass scan
    // state through unchanged and zero inputs add nothing.
    let input_m = pad_rows_to(input, m, chunk, 0.0).reshape([m, chunk]);
    let coef_m = pad_rows_to(coef, m, chunk, 1.0).reshape([m, chunk]);
    let log_coef = coef_m.log();

    // Suffix-inclusive sums of log coefficients within each chunk.
    let suffix_log = log_coef.clone().flip([1]).cumsum(1).flip([1]);

    // Pairwise weights: weight of input[k] in row i's local scan is
    // `exp(suffix_log[i] - suffix_log[k])` for `k >= i`, zero otherwise.
    // Valid exponents are <= 0 (coefficients are <= 1); the -1000 additive
    // fill drives masked entries to exactly zero without overflowing.
    let row_idx = Tensor::<1, Int>::arange(0..chunk as i64, &device).reshape([1, chunk, 1]);
    let col_idx = Tensor::<1, Int>::arange(0..chunk as i64, &device).reshape([1, 1, chunk]);
    let masked = row_idx.greater(col_idx).float().mul_scalar(GAE_MASK_FILL);
    let pair_weights = (suffix_log
        .clone()
        .unsqueeze_dim::<3>(2)
        .sub(suffix_log.clone().unsqueeze_dim::<3>(1))
        + masked)
        .exp();

    // Local scan value per row: weighted sum of inputs to its right. The
    // input broadcasts along the column (k) axis.
    let local = pair_weights
        .mul(input_m.unsqueeze_dim::<3>(1))
        .sum_dim(2)
        .reshape([m, chunk]);

    // Chunk aggregates for the level-two scan: the scan value at each
    // chunk's first row and the log product of the whole chunk.
    let local_start = local.clone().narrow(1, 0, 1).reshape([m]);
    let log_chunk_product = suffix_log.clone().narrow(1, 0, 1).reshape([m]);

    // Level two: identical construction over the M chunk aggregates.
    let suffix_chunk = log_chunk_product.clone().flip([0]).cumsum(0).flip([0]);
    let row_idx2 = Tensor::<1, Int>::arange(0..m as i64, &device).reshape([m, 1]);
    let col_idx2 = Tensor::<1, Int>::arange(0..m as i64, &device).reshape([1, m]);
    let masked2 = row_idx2.greater(col_idx2).float().mul_scalar(GAE_MASK_FILL);
    let chunk_weights = (suffix_chunk
        .clone()
        .unsqueeze_dim::<2>(1)
        .sub(suffix_chunk.unsqueeze_dim::<2>(0))
        + masked2)
        .exp();
    let chunk_scan = chunk_weights
        .mul(local_start.unsqueeze_dim::<2>(0))
        .sum_dim(1)
        .reshape([m]);

    // Row-level assembly: out[i] = local[i] + exp(suffix_log[i]) * V[c+1],
    // where V[c+1] is the scan value at the next chunk's first row (0 past
    // the last chunk). The product over the full chunk includes any
    // boundary's floored coefficient, so carries cannot cross a boundary.
    let chunk_scan_next = if m > 1 {
        Tensor::cat(
            vec![
                chunk_scan.narrow(0, 1, m - 1),
                Tensor::<1>::zeros([1], &device),
            ],
            0,
        )
    } else {
        Tensor::<1>::zeros([1], &device)
    };
    let per_row = local + suffix_log.exp().mul(chunk_scan_next.reshape([m, 1]));
    per_row.reshape([1, padded]).narrow(1, 0, n).reshape([n, 1])
}

/// Pad an `[n, 1]` tensor with `value` rows up to `m * chunk` rows.
fn pad_rows_to(tensor: Tensor<2>, m: usize, chunk: usize, value: f32) -> Tensor<2> {
    let n = tensor.shape().dims::<2>()[0];
    if n < m * chunk {
        let pad = Tensor::<2>::full([m * chunk - n, 1], value, &tensor.device());
        Tensor::cat(vec![tensor, pad], 0)
    } else {
        tensor
    }
}

fn finite_time_factor(lambda: f32, remaining: usize) -> f32 {
    assert!(
        remaining > 0,
        "finite-time GAE requires a positive suffix length"
    );
    if remaining == 1 {
        return 0.0;
    }
    if lambda == 0.0 {
        return 1.0;
    }
    if lambda == 1.0 {
        return (remaining - 1) as f32 / remaining as f32;
    }

    // expm1 avoids losing precision when lambda is close to one.
    let log_lambda = lambda.ln();
    (log_lambda * (remaining - 1) as f32).exp_m1() / (log_lambda * remaining as f32).exp_m1()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(actual: f32, expected: f32) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "actual {actual} != expected {expected}"
        );
    }

    fn calculate(
        values: &[f32],
        rewards: &[f32],
        terminals: &[TerminalState],
        trunc_values: &[f32],
        gamma: f32,
        lambda: f32,
        estimator: GaeEstimator,
    ) -> Vec<f32> {
        get_gae(
            values.to_vec(),
            rewards.to_vec(),
            terminals,
            trunc_values,
            gamma,
            lambda,
            1.0,
            0.0,
            estimator,
        )
        .advantages
    }

    #[test]
    fn truncated_mode_matches_ordinary_recursion_and_reverse_bootstrap_order() {
        let advantages = calculate(
            &[0.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0, 0.0],
            &[
                TerminalState::Truncated,
                TerminalState::None,
                TerminalState::Truncated,
                TerminalState::Normal,
            ],
            &[10.0, 20.0],
            0.9,
            0.5,
            GaeEstimator::Truncated,
        );

        // Reverse traversal must assign the gamma-discounted bootstraps:
        // delta = r + gamma * next_val - value, so row 2 gets 0.9 * 20 and
        // row 0 gets 0.9 * 10 (matches GGL's GAE: predReturn = curReward +
        // gamma * nextValPred * (1 - done)).
        approx_eq(advantages[0], 0.9 * 10.0);
        approx_eq(advantages[1], 0.9 * 0.5 * 0.9 * 20.0);
        approx_eq(advantages[2], 0.9 * 20.0);
        approx_eq(advantages[3], 0.0);
    }

    #[test]
    fn finite_time_one_step_suffix_is_delta() {
        let advantages = calculate(
            &[0.0],
            &[2.0],
            &[TerminalState::Normal],
            &[],
            0.9,
            0.95,
            GaeEstimator::TerminationTime,
        );
        approx_eq(advantages[0], 2.0);
    }

    #[test]
    fn finite_time_two_step_suffix_uses_normalized_recursive_mass() {
        let gamma = 0.9;
        let lambda = 0.5;
        let advantages = calculate(
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[TerminalState::None, TerminalState::Normal],
            &[],
            gamma,
            lambda,
            GaeEstimator::TerminationTime,
        );
        approx_eq(advantages[1], 1.0);
        approx_eq(advantages[0], 1.0 + gamma * lambda / (1.0 + lambda));
    }

    #[test]
    fn finite_time_recursion_matches_the_normalized_direct_mixture() {
        let gamma = 0.8;
        let lambda = 0.6;
        let deltas = [1.0, 2.0, 3.0];
        let advantages = calculate(
            &[0.0, 0.0, 0.0],
            &deltas,
            &[
                TerminalState::None,
                TerminalState::None,
                TerminalState::Normal,
            ],
            &[],
            gamma,
            lambda,
            GaeEstimator::TerminationTime,
        );

        let mut mixture = 0.0;
        for k in 0..deltas.len() {
            let k_step = (0..=k)
                .map(|j| gamma.powi(j as i32) * deltas[j])
                .sum::<f32>();
            mixture += lambda.powi(k as i32) * k_step;
        }
        mixture *= (1.0 - lambda) / (1.0 - lambda.powi(deltas.len() as i32));
        approx_eq(advantages[0], mixture);
    }

    #[test]
    fn finite_time_factor_handles_endpoints_and_near_one() {
        approx_eq(finite_time_factor(0.0, 1), 0.0);
        approx_eq(finite_time_factor(0.0, 4), 1.0);
        approx_eq(finite_time_factor(1.0, 1), 0.0);
        approx_eq(finite_time_factor(1.0, 4), 0.75);
        let near_one = finite_time_factor(1.0 - 1e-7, 1000);
        approx_eq(near_one, 999.0 / 1000.0);
    }

    #[test]
    #[should_panic(expected = "exactly one truncation bootstrap")]
    fn sparse_bootstrap_count_must_match_rows() {
        let _ = calculate(
            &[0.0],
            &[1.0],
            &[TerminalState::Truncated],
            &[],
            0.99,
            0.95,
            GaeEstimator::Truncated,
        );
    }

    #[test]
    fn truncated_bootstrap_is_included_in_final_delta_but_does_not_continue() {
        let advantages = calculate(
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[TerminalState::Truncated, TerminalState::Normal],
            &[3.0],
            0.9,
            0.95,
            GaeEstimator::TerminationTime,
        );
        approx_eq(advantages[0], 0.9 * 3.0 - 1.0);
        approx_eq(advantages[1], 0.0);
    }
}

/// Parity tests for the device scan against the CPU reference scan. Both
/// paths must agree on seeded random rollouts, including edge cases:
/// truncated first row, terminal boundaries mid-trajectory, the
/// TerminationTime estimator, reward clipping, and degenerate
/// standardization scales.
#[cfg(all(test, any(feature = "torch", feature = "flex")))]
mod device_parity_tests {
    use burn::tensor::{Device, Tensor, TensorData};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::{
        DeviceGaeOutput, GaeEstimator, TerminalState, gae_cpu_flags, get_gae, get_gae_device,
    };

    /// Relative tolerance: the log-domain scan differs from the sequential
    /// reference by accumulated exp/log rounding (a few ulps per chained
    /// factor); the sequential reference itself accumulates ~n·eps error on
    /// long chains, so the tolerance is dominated by the reference, not the
    /// scan. Boundary leakage through the coefficient floor is 1e-15.
    fn assert_close(what: &str, index: usize, actual: f32, expected: f32) {
        let tolerance = 2e-3 * expected.abs().max(1.0) + 1e-4;
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what}[{index}]: device {actual} != reference {expected} (tolerance {tolerance})"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn compare(
        device: Device,
        seed: u64,
        n: usize,
        gamma: f32,
        lambda: f32,
        estimator: GaeEstimator,
        return_std: f32,
        clip_range: f32,
    ) {
        let mut rng = SmallRng::seed_from_u64(seed);

        // Build a rollout of random-length trajectories; every trajectory
        // ends on a boundary row, some of which are truncations with
        // bootstrap predictions in forward terminal order.
        let mut terminals = vec![TerminalState::None; n];
        let mut trunc_preds = Vec::new();
        let mut row = 0;
        while row < n {
            let len = rng.random_range(1..=40).min(n - row);
            let truncated = rng.random_bool(0.25);
            let end = row + len;
            terminals[end - 1] = if truncated {
                TerminalState::Truncated
            } else {
                TerminalState::Normal
            };
            if truncated {
                trunc_preds.push(rng.random_range(-5.0..5.0));
            }
            row = end;
        }

        let values = (0..n)
            .map(|_| rng.random_range(-2.0..2.0))
            .collect::<Vec<_>>();
        let rewards = (0..n)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect::<Vec<_>>();
        let rewards_ref = rewards.clone();

        let reference = get_gae(
            values.clone(),
            rewards.clone(),
            &terminals,
            &trunc_preds,
            gamma,
            lambda,
            return_std,
            clip_range,
            estimator,
        );

        let values_t = Tensor::<2>::from_data(TensorData::new(values, [n, 1]), &device);
        let rewards_t = Tensor::<2>::from_data(TensorData::new(rewards, [n, 1]), &device);
        let flags = Tensor::<2>::from_data(
            TensorData::new(gae_cpu_flags(&terminals, gamma, lambda, estimator), [n, 4]),
            &device,
        );
        let trunc_count = trunc_preds.len();
        let preds_t =
            Tensor::<2>::from_data(TensorData::new(trunc_preds, [trunc_count, 1]), &device);
        let DeviceGaeOutput {
            returns,
            target_vals,
            advantages,
            deltas,
            reward_abs_sum,
            clipped_abs_sum,
        } = get_gae_device(
            values_t, rewards_t, flags, preds_t, gamma, return_std, clip_range,
        );

        let device_advantages = advantages.into_data().try_into_vec::<f32>().unwrap();
        let device_returns = returns.into_data().try_into_vec::<f32>().unwrap();
        let device_targets = target_vals.into_data().try_into_vec::<f32>().unwrap();
        let device_deltas = deltas.into_data().try_into_vec::<f32>().unwrap();

        for (index, (&actual, &expected)) in device_deltas.iter().zip(&reference.deltas).enumerate()
        {
            // Deltas use plain elementwise ops: much tighter tolerance.
            let tolerance = 1e-4 * expected.abs().max(1.0) + 1e-5;
            assert!(
                (actual - expected).abs() <= tolerance,
                "deltas[{index}]: device {actual} != reference {expected}"
            );
        }
        for (name, device_out, reference_out) in [
            ("advantages", &device_advantages, &reference.advantages),
            ("returns", &device_returns, &reference.returns),
            ("target_vals", &device_targets, &reference.target_vals),
        ] {
            for (index, (&actual, &expected)) in device_out.iter().zip(reference_out).enumerate() {
                assert_close(name, index, actual, expected);
            }
        }

        // Clip-portion metric: recompute the reference reduction sums.
        let scale = if !return_std.is_finite() || return_std == 0.0 {
            1.0
        } else {
            1.0 / return_std
        };
        let mut total_raw = 0.0_f32;
        let mut total_clipped = 0.0_f32;
        for &reward in &rewards_ref {
            let scaled = reward * scale;
            total_raw += scaled.abs();
            let clipped = if clip_range > 0.0 {
                scaled.clamp(-clip_range, clip_range)
            } else {
                scaled
            };
            total_clipped += clipped.abs();
        }
        let reference_portion = (total_raw - total_clipped) / total_raw.max(f32::EPSILON);
        let raw = reward_abs_sum.into_data().try_into_vec::<f32>().unwrap()[0];
        let clipped = clipped_abs_sum.into_data().try_into_vec::<f32>().unwrap()[0];
        let device_portion = (raw - clipped) / raw.max(f32::EPSILON);
        assert!(
            (device_portion - reference_portion).abs() <= 1e-5,
            "clip portion: device {device_portion} != reference {reference_portion}"
        );
    }

    #[cfg(feature = "torch")]
    #[test]
    #[allow(deprecated)]
    fn device_scan_matches_reference_on_torch_cpu() {
        let device = Device::libtorch();
        // Seed matrix over estimator, clipping, and standardization scales.
        for (seed, (gamma, lambda, estimator, return_std, clip_range, n)) in [
            (
                1u64,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 1000),
            ),
            (
                2,
                (0.9, 0.5, GaeEstimator::TerminationTime, 1.0, 10.0, 1000),
            ),
            (3, (0.99, 0.95, GaeEstimator::Truncated, 1.0, 10.0, 1000)),
            (
                4,
                (0.99, 0.95, GaeEstimator::TerminationTime, 2.5, 10.0, 1000),
            ),
            (
                5,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 0.0, 1000),
            ),
            (
                6,
                (0.99, 0.95, GaeEstimator::TerminationTime, 0.0, 10.0, 1000),
            ),
            (
                7,
                (
                    0.99,
                    0.95,
                    GaeEstimator::TerminationTime,
                    f32::NAN,
                    10.0,
                    1000,
                ),
            ),
            // Chunk-boundary sizes: exact chunk, chunk-1, chunk+1.
            (
                8,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 127),
            ),
            (
                9,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 128),
            ),
            (
                10,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 129),
            ),
            (
                11,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 256),
            ),
            (
                12,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 257),
            ),
            // Multi-chunk with boundaries crossing chunk edges.
            (
                13,
                (0.95, 0.9, GaeEstimator::TerminationTime, 1.0, 5.0, 2000),
            ),
            (14, (0.95, 0.9, GaeEstimator::Truncated, 1.0, 5.0, 2000)),
            // Tiny rollouts.
            (
                15,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 1),
            ),
            (
                16,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 2),
            ),
        ] {
            compare(
                device.clone(),
                seed,
                n,
                gamma,
                lambda,
                estimator,
                return_std,
                clip_range,
            );
        }
    }

    #[cfg(feature = "flex")]
    #[test]
    fn device_scan_matches_reference_on_flex() {
        let device = Device::flex();
        for (seed, (gamma, lambda, estimator, return_std, clip_range, n)) in [
            (
                1u64,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 1000),
            ),
            (14, (0.95, 0.9, GaeEstimator::Truncated, 1.0, 5.0, 2000)),
            (
                13,
                (0.95, 0.9, GaeEstimator::TerminationTime, 1.0, 5.0, 2000),
            ),
            (
                12,
                (0.99, 0.95, GaeEstimator::TerminationTime, 1.0, 10.0, 257),
            ),
        ] {
            compare(
                device.clone(),
                seed,
                n,
                gamma,
                lambda,
                estimator,
                return_std,
                clip_range,
            );
        }
    }
}
