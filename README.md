## rlgymppo-rs

A Rust implementation of Proximal Policy Optimization (PPO) for Rocket League
training, built on [RocketSim v3](https://github.com/ZealanL/RocketSim/tree/v3-rust) +
[RLViser](https://github.com/VirxEC/rlviser) via
[RLGym-rs](https://github.com/VirxEC/rlgym_rs).

### Project structure

The workspace is split into seven crates:

| Crate | Purpose |
|---|---|
| `rlgymppo` | Core PPO learner, multi-threaded environment runner, and training loop. |
| `rlgymppo-model` | Backend-generic policy/model definitions and checkpoint-compatible inference loading. |
| `rlgymppo-rlbot` | RLBot v5 agent that runs trained policies with Burn Flex and RocketSim state enrichment. |
| `rlgymppo-utils` | Reusable RLGym observation builders, action parsers, and shared-info traits, without depending on the PPO learner. |
| `rlgymppo-tui` | Terminal-based dashboard that renders live training metrics (ratatui). |
| `rlgymppo-wandb` | Weights & Biases integration via an embedded Python interpreter (pyo3). |
| `rlgymppo-trainer` | Bundled training example with shared logic, self-play, and skill tracking. |

`rlgymppo-utils` contains the shared `DefaultObs`, `AdvancedObs`, and
`DefaultAction` implementations used by the trainer. It re-exports `rlgym` and
RocketSim types so inference applications can reuse the same observation and
action logic without depending on the PPO training stack.

### Quick start

See [`rlgymppo-trainer/examples/run.rs`](rlgymppo-trainer/examples/run.rs) for a
complete training example. The core logic lives in
[`rlgymppo-trainer/src/lib.rs`](rlgymppo-trainer/src/lib.rs). It includes:

- A custom `SharedInfo` that tracks player metrics (distance to ball, speed,
  boost, air time, demo status, touch height)
- 1v1, 2v2, and 3v3 random game selection
- Several state setters (kickoff, random positions) weighted by probability
- Combined rewards (air time, face ball, velocity to ball)
- Terminal conditions (goal scored, random game end, no-touch timeout)
- `SelfPlayConfig` & `SkillTrackerConfig` for policy versioning and Elo rating

Run with your chosen backend (replace `torch` with `cuda`, `wgpu`, `metal`, etc.):

```sh
cargo run -p rlgymppo-trainer --example run --features torch
```

At a high level, training looks like this:

```rust
let config = LearnerConfig {
    num_threads: 4,
    num_games_per_thread: 64,
    ppo: PpoLearnerConfig {
        timesteps_per_iteration: 80_000,
        batch_size: 80_000,
        mini_batch_size: 40_000,
        epochs: 1,
        learning_rate: 1.5e-4,
        entropy_scale: 0.036,
        ..Default::default()
    },
    shared_head_layer_sizes: vec![256; 2],
    policy_layer_sizes: vec![256; 3],
    critic_layer_sizes: vec![256; 3],
    timesteps_per_save: 10_000_000,
    checkpoints_limit: Some(10),
    self_play: SelfPlayConfig {
        save_policy_versions: true,
        ts_per_version: 100_000_000,
        ..Default::default()
    },
    skill_tracker: SkillTrackerConfig {
        enabled: true,
        ..Default::default()
    },
    device: LibTorchDevice::Cuda(0),
    render_device: LibTorchDevice::Cpu,
    wandb_project_name: Some("rlgym-ppo".into()),
    wandb_run_name: Some("ppo-bot-v1".into()),
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

### Backends

The project uses [Burn](https://burn.dev) and supports all its backends. Enable
exactly one via a feature flag:

| Feature | Backend | Device types |
|---|---|---|
| `torch` | LibTorch (libtorch C++) | `Cuda(N)`, `Cpu`, `Mps`, `Vulkan` |
| `cuda` | Pure Rust CUDA | `CudaDevice` |
| `metal` | Apple Metal | `WgpuDevice` |
| `rocm` | AMD ROCm | `RocmDevice` |
| `wgpu` | Cross-platform GPU | `WgpuDevice` |
| `flex` | CPU fallback | `Default` |
| `candle` | Candle ML framework | `CandleDevice` |

**torch** is the most mature and fastest backend, supporting CUDA, CPU, MPS,
and Vulkan devices.

To point Rust to your LibTorch installation, you may need to set environment
variables like `LIBTORCH`, `LIBTORCH_INCLUDE`, and `LIBTORCH_LIB`. See
[tch-rs docs](https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started)
for help getting started.
