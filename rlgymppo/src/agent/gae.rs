use super::config::GaeEstimator;
use crate::base::TerminalState;

pub(crate) struct GAEOutput {
    pub(crate) returns: Vec<f32>,
    pub(crate) target_vals: Vec<f32>,
    pub(crate) advantages: Vec<f32>,
    pub(crate) rew_clip_portion: f32,
}

/// Calculate returns and GAE advantages from row-aligned rollout data.
///
/// Truncation bootstrap predictions are supplied in forward terminal order.
/// The reverse scan consumes them from the end so each truncated row receives
/// the prediction for its own boundary observation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_gae(
    values: Vec<f32>,
    rewards: Vec<f32>,
    terminals: Vec<TerminalState>,
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

    let mut return_scale = 1.0 / return_std;
    if return_scale.is_nan() || return_scale == 0.0 {
        return_scale = 1.0;
    }

    let mut last_return = 0.0;
    let mut running_advantage = 0.0;
    let mut trunc_cursor = trunc_val_preds.len();
    let mut remaining = 0_usize;
    let mut total_reward = 0.0_f32;
    let mut total_clipped_reward = 0.0_f32;

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
        total_reward += norm_reward.abs();
        if clip_range > 0.0 {
            norm_reward = norm_reward.clamp(-clip_range, clip_range);
        }
        total_clipped_reward += norm_reward.abs();

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
    let rew_clip_portion = (total_reward - total_clipped_reward) / total_reward.max(f32::EPSILON);
    GAEOutput {
        returns,
        target_vals,
        advantages,
        rew_clip_portion,
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
            terminals.to_vec(),
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

        // Reverse traversal must assign 20 to row 2 and 10 to row 0.
        approx_eq(advantages[0], 10.0);
        approx_eq(advantages[1], 0.9 * 0.5 * 20.0);
        approx_eq(advantages[2], 20.0);
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
