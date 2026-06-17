## rlgymppo-rs

A Rust implementation of Proximal Policy Optimization (PPO) for Rocket League
training, built on [RocketSim v3](https://github.com/ZealanL/RocketSim/tree/v3-rust) +
[RLViser](https://github.com/VirxEC/rlviser) via
[RLGym-rs](https://github.com/VirxEC/rlgym_rs).

### Project structure

The workspace is split into three crates:

| Crate | Purpose |
|---|---|
| `rlgymppo` | Core PPO learner, multi-threaded environment runner, and training loop. |
| `rlgymppo-tui` | Terminal-based dashboard that renders live training metrics (ratatui). |
| `rlgymppo-wandb` | Weights & Biases integration via an embedded Python interpreter (pyo3). |

### Quick start

See [`rlgymppo/examples/generic.rs`](rlgymppo/examples/generic.rs) for a
complete training example. It includes:

- A custom `SharedInfo` that tracks player metrics (distance to ball, speed,
  boost, air time, demo status)
- A `MyObs` builder with a zero-padded multi-agent observation space
- 1v1, 2v2, and 3v3 random game selection
- Several state setters (kickoff, random positions) weighted by probability
- Combined rewards (face ball + velocity to ball)
- Terminal conditions (goal scored, game time elapsed, no-touch timeout)

At a high level, training looks like this:

```rust
let config = LearnerConfig::<Autodiff<LibTorch>> {
    num_threads: 4,
    num_games_per_thread: 64,
    ppo: PpoLearnerConfig {
        batch_size: 50_000,
        mini_batch_size: 50_000,
        learning_rate: 2e-4,
        entropy_scale: 0.035,
        ..Default::default()
    },
    device: LibTorchDevice::Cuda(0),
    ..Default::default()
};

let mut learner = config.init(create_env);
learner.load();    // resume from checkpoint if one exists
learner.learn();   // train forever (or until num_additional_iterations)
```

### Training controls

While training, you can press:

| Key | Action |
|---|---|
| `Q` | Quit |
| `S` | Quick-save a checkpoint |
| `R` | Toggle the RocketSim visualizer on/off |
| `D` | Toggle deterministic mode for the renderer |

If the `tui` feature is enabled, these are shown in the status bar of the
terminal dashboard. Without `tui`, you type the letter and press enter.

### Logging & metrics

**Weights & Biases** — Enable the `wandb` feature and set a project name in
`LearnerConfig`. The embedded Python interpreter calls `wandb.init()` and
`wandb.log()` directly. You'll also need the `_WANDB_CORE_PATH` environment
variable — see [wandb integration](#wandb-integration) below.

**Terminal dashboard** — Enable the `tui` feature for a live-updating ratatui
dashboard that organizes metrics into groups (Collect, GAE, Loss, Update,
Timing, Throughput, Cumulative).

**Reward metrics** — Reward components with names starting with `Reward/` are
automatically tracked. Configure sampling with `reward_sample_interval` and
`add_rewards_to_metrics` in `PpoLearnerConfig`.

### Checkpoints

Models, optimizer states, and training stats are saved to the folder specified
by `checkpoints_folder` (defaults to `./checkpoints`). On restart, `learner.load()`
resumes from the latest checkpoint — safe to call unconditionally.

### wandb integration

The environment variable `_WANDB_CORE_PATH` must be set. The easiest way to do
this is `pip install wandb`, then find where wandb was installed and set
`_WANDB_CORE_PATH` to the directory containing the `wandb-core` binary
(e.g. `/path/to/venv/lib/python3.12/site-packages/wandb/bin`).

### torch

Using the `torch` backend is recommended — it's the most mature, fastest backend
and supports a wide range of devices.

To point Rust to your LibTorch installation, you may need to set environment
variables like `LIBTORCH`, `LIBTORCH_INCLUDE`, and `LIBTORCH_LIB`. See
[tch-rs docs](https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started)
for help getting started.
