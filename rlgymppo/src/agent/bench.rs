//! Consumption-path benchmark for [`Ppo::learn`].
//!
//! Not a correctness test: gated behind the `RLBENCH` env var so `cargo test`
//! skips it. Run with:
//!
//! ```sh
//! RLBENCH=1 RLBENCH_DEVICE=cuda cargo test -p rlgymppo --no-default-features \
//!   --features torch --lib bench_consumption -- --nocapture
//! ```
//!
//! Mirrors `default_config` in `rlgymppo-trainer` (TerminationTime GAE,
//! 100k-step rollout, one epoch, 20k mini-batches, 256-wide shared head +
//! 256x3 actor/critic) over a synthetic rollout with realistic episode
//! structure. Prints the per-phase timing metrics and consumption steps/s.

use std::collections::HashMap;
use std::time::Instant;

use burn::optim::AdamWConfig;
use rlgymppo_utils::Report;

use super::Ppo;
use super::model::Actic;
use crate::base::{Memory, TerminalState};
use crate::{PpoLearnerConfig, Stats};

const STATE_WIDTH: usize = 141; // DefaultObs<3>
const ACTION_WIDTH: usize = 90; // DefaultAction
const HIDDEN: usize = 256;

/// Build a synthetic rollout with realistic trajectory/terminal structure.
fn synthetic_memory(n: usize) -> Memory {
    use rand::RngExt;
    let mut rng = rand::rng();
    let mut memory = Memory::with_capacity(n);
    let mut total = 0;
    while total < n {
        let mut len = rng.random_range(80..=400);
        if total + len > n {
            len = n - total;
        }
        let truncated = rng.random_bool(0.15);
        let mut states = vec![0.0f32; len * STATE_WIDTH];
        let mut actions = Vec::with_capacity(len);
        let mut log_probs = Vec::with_capacity(len);
        let mut rewards = vec![0.0f32; len];
        let mut terminals = vec![TerminalState::None; len];
        let masks = vec![1u8; len * ACTION_WIDTH];

        for row in states.chunks_mut(STATE_WIDTH).by_ref() {
            for value in row {
                *value = rng.random_range(-1.5..1.5);
            }
        }
        for _ in 0..len {
            actions.push(rng.random_range(0..ACTION_WIDTH));
            log_probs.push(-4.5 + rng.random_range(-2.0..2.0));
        }
        for value in &mut rewards {
            *value = rng.random_range(-1.0..1.0);
        }
        *terminals.last_mut().unwrap() = if truncated {
            TerminalState::Truncated
        } else {
            TerminalState::Normal
        };

        let trunc_next_state = truncated.then(|| {
            (0..STATE_WIDTH)
                .map(|_| rng.random_range(-1.5..1.5))
                .collect::<Vec<_>>()
        });

        memory.push_player(
            states,
            STATE_WIDTH,
            actions,
            log_probs,
            rewards,
            terminals,
            masks,
            ACTION_WIDTH,
            Vec::new(),
            0,
            trunc_next_state,
        );
        total += len;
    }
    memory
}

#[test]
#[allow(deprecated)]
fn bench_consumption() {
    if std::env::var("RLBENCH").is_err() {
        return;
    }
    let device = match std::env::var("RLBENCH_DEVICE").as_deref() {
        Ok("cpu") => burn::tensor::Device::libtorch().autodiff(),
        _ => burn::tensor::Device::libtorch_cuda(0).autodiff(),
    };
    let n: usize = std::env::var("RLBENCH_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100_000);
    let warmup: usize = std::env::var("RLBENCH_WARMUP")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(3);
    let iters: usize = std::env::var("RLBENCH_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(10);

    let config = PpoLearnerConfig {
        timesteps_per_iteration: n,
        batch_size: n,
        mini_batch_size: 20_000,
        gpu_timestep_buffer_size: n,
        truncation_value_batch_size: n,
        epochs: 1,
        learning_rate: 1e-3,
        entropy_scale: 0.024,
        gae_estimator: crate::GaeEstimator::TerminationTime,
        max_episode_length: None,
        ..Default::default()
    };

    let net = Actic::new(
        STATE_WIDTH,
        ACTION_WIDTH,
        vec![HIDDEN; 3],
        vec![HIDDEN; 3],
        &[HIDDEN; 2],
        &device,
        None,
    );
    let mut net = net;
    let mut ppo = Ppo::new(config, device, || AdamWConfig::new().init());
    let memory = synthetic_memory(n);
    memory.validate().unwrap();

    let mut rng = rand::rng();
    let mut stats = Stats::default();
    let mut sums: HashMap<String, f64> = HashMap::new();
    let mut total_secs = 0.0;

    for iteration in 0..(warmup + iters) {
        let mut metrics = Report::default();
        let start = Instant::now();
        let (next_net, _rows) = ppo.learn(
            net,
            &memory,
            &mut rng,
            &mut metrics,
            &mut stats,
            iteration == 0,
        );
        net = next_net;
        let secs = start.elapsed().as_secs_f64();
        if iteration >= warmup {
            total_secs += secs;
            for (key, value) in metrics.to_flat_map() {
                *sums.entry(key).or_default() += value;
            }
        }
    }

    println!(
        "\n=== consumption bench: device={:?} n={n} iters={iters} ===",
        std::env::var("RLBENCH_DEVICE").as_deref().unwrap_or("cuda")
    );
    println!(
        "wall per iteration: {:.1} ms -> {:.0} player steps/s",
        total_secs / iters as f64 * 1e3,
        n as f64 / (total_secs / iters as f64)
    );
    let mut keys: Vec<_> = sums.keys().cloned().collect();
    keys.sort();
    for key in keys {
        println!("  {key}: {:.4} ms", sums[&key] / iters as f64 * 1e3);
    }
}
