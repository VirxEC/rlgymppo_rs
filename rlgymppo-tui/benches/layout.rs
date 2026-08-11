//! Criterion benchmarks for the TUI layout planner (`LayoutPlanCache`).
//!
//! Run with:
//! ```sh
//! cargo bench -p rlgymppo-tui --features bench
//! ```
//!
//! The `bench` feature exposes the owned-data shim in `src/render.rs`; without
//! it the planner internals are private and cannot be driven from here.
//!
//! All cases measure the *cold* planning path (fresh cache per call), which is
//! the pathological case: every frame after metric values change the group
//! dimensions shift, the cache key changes, and the full search re-runs.

use std::collections::{HashMap, VecDeque};
use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use rlgymppo_tui::bench;

/// (group prefix, entry names) — mirrors a real training dashboard and must
/// stay in sync with `GROUP_DEFS` in the `layout_bench` test module of
/// `src/render.rs` so both harnesses exercise the same shapes.
const GROUP_DEFS: &[(&str, &[&str])] = &[
    (
        "Collect",
        &["avg step reward", "episode length", "timesteps"],
    ),
    ("GAE", &["time"]),
    ("Loss", &["policy", "value", "entropy"]),
    ("Update", &["gradient norm", "weight norm"]),
    ("Timing", &["total", "collect", "gae", "update"]),
    ("Throughput", &["steps/s"]),
    ("Cumulative", &["steps", "updates"]),
    ("Rating", &["overall"]),
    ("Rewards", &["Total", "Positive", "Negative"]),
    ("Rewards/NormFactor", &["Vel2Ball", "Vel2BallVel"]),
    ("Env", &["boost avg", "car height", "action rate"]),
    ("Skill", &["overall", "kickoff"]),
    ("Curriculum", &["level", "progress"]),
    ("Match", &["win rate", "goal diff", "avg goals"]),
    ("Rewards/Positive", &["BallTouch", "Goal"]),
    ("Rewards/Negative", &["Concede", "OwnGoal"]),
    ("Demo", &["demos", "demoed"]),
    ("Save", &["shots", "saves"]),
];

/// Terminal sizes: small, typical, large, and "way larger than needed".
const SIZES: &[(u16, u16)] = &[(80, 24), (120, 30), (200, 60), (300, 100), (500, 200)];

/// Group counts: the 8 built-in groups up to a heavy 18-group stress case.
const GROUP_COUNTS: &[usize] = &[8, 10, 12, 14, 16, 18];

/// Groups + sparkline history for one benchmark case.
type Inputs = (
    Vec<(String, Vec<(String, f64)>)>,
    HashMap<String, VecDeque<f64>>,
);

/// Build `(groups, history)` inputs for `group_count` groups. Values and
/// history mirror steady-state training data (sparklines enabled, 12 samples).
fn build_inputs(group_count: usize) -> Inputs {
    let groups: Vec<(String, Vec<(String, f64)>)> = GROUP_DEFS
        .iter()
        .take(group_count)
        .enumerate()
        .map(|(i, (prefix, entries))| {
            (
                (*prefix).to_string(),
                entries
                    .iter()
                    .enumerate()
                    .map(|(j, name)| {
                        (
                            (*name).to_string(),
                            12_345.678 * (i as f64 + 1.0) + j as f64 * 0.001,
                        )
                    })
                    .collect(),
            )
        })
        .collect();

    let mut history = HashMap::new();
    for (i, (prefix, entries)) in GROUP_DEFS.iter().enumerate().take(group_count) {
        for (j, name) in entries.iter().enumerate() {
            history.insert(
                format!("{prefix}/{name}"),
                (0..12)
                    .map(|t| t as f64 * 0.1 + i as f64 + j as f64)
                    .collect(),
            );
        }
    }

    (groups, history)
}

fn bench_layout_planning(c: &mut Criterion) {
    let mut group = c.benchmark_group("layout_cold");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    // Criterion sizes the samples to fill this window *per benchmark*, so keep
    // it modest: fast cases finish in a few seconds, and the pathological
    // large-terminal cases (which can take seconds per iteration) still get a
    // full 10-sample set because `sample_size` has a floor of 10.
    group.measurement_time(Duration::from_secs(5));

    for &group_count in GROUP_COUNTS {
        let (groups, history) = build_inputs(group_count);
        for &(width, height) in SIZES {
            group.bench_with_input(
                BenchmarkId::new("cold", format!("{group_count}g_{width}x{height}")),
                &(width, height),
                |b, &(width, height)| {
                    b.iter(|| {
                        let columns = bench::plan_metric_columns(
                            black_box(width),
                            black_box(height),
                            black_box(&groups),
                            black_box(&history),
                        );
                        black_box(columns)
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_layout_planning);
criterion_main!(benches);
