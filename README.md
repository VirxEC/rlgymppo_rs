## rlgymppo-rs

A Rust implementation of Proximal Policy Optimization (PPO) for Rocket League
training, built on [RocketSim v3](https://github.com/ZealanL/RocketSim/tree/v3-rust) +
[RLViser](https://github.com/VirxEC/rlviser) via
[RLGym-rs](https://github.com/VirxEC/rlgym_rs).

### Project structure

The workspace is split into nine crates:

| Crate | Purpose |
|---|---|
| `rlgymppo` | Core PPO learner, multi-threaded environment runner, and training loop. |
| `rlgymppo-model` | Backend-generic policy/model definitions and checkpoint-compatible inference loading. |
| `rlgymppo-nexto` | Nexto model inference, observation builder, action table, and pre-generated model weights. |
| `rlgymppo-rlbot` | RLBot v5 agent that runs trained policies with Burn Flex and RocketSim state enrichment. |
| `rlgymppo-utils` | Reusable RLGym observation builders, action parsers, and shared-info traits, without depending on the PPO learner. |
| `rlgymppo-tui` | Terminal-based dashboard that renders live training metrics (ratatui). |
| `rlgymppo-wandb` | Weights & Biases integration via an embedded Python interpreter (pyo3). |
| `rlgymppo-trainer` | Bundled training example with shared logic, self-play, and skill tracking. |
| `rlgymppo-transfer` | Transfer-learning example that pretends `rlgymppo-trainer` is the parent model. |

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
- `SelfPlayConfig` for policy versioning
- `SkillTrackerConfig` for optional Elo ratings against saved previous policy versions and the fixed Nexto baseline

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
        batch_size: 40_000,
        mini_batch_size: 10_000,
        gpu_timestep_buffer_size: 40_000,
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
        // Some(mmr) enables Nexto evals. None disables only Nexto.
        nexto_mmr: Some(1500.0),
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

### Skill tracking

`SkillTrackerConfig::default()` is disabled and sets `nexto_mmr` to
`Some(1500.0)`. Set `enabled` to `true` to run evaluations. When enabled,
periodic evaluations run the current policy against a randomly selected saved
previous policy version and against Nexto when `nexto_mmr` is `Some`.

Old-version matches use two-sided Elo. Both sides update their ratings. Each
saved policy version keeps its own rating. Nexto is the only fixed bot; its
MMR stays at `nexto_mmr`. Set `skill_tracker.nexto_mmr` to `Some(value)` to
change it.

Set `nexto_mmr` to `None` to disable Nexto. The Nexto model is not loaded.
Old-version comparisons still run when `enabled` is `true`. They require policy
version saving (`SelfPlayConfig.save_policy_versions`) and can continue when it
is enabled. Set `enabled` to `false` to disable all evaluations.

The `rlgymppo-nexto` crate contains the Rust observation and action functions
and a pre-generated Burn model. The generated model source and BurnPack
weights are committed in `rlgymppo-nexto/nexto/`. Normal builds do not run
model conversion or code generation and do not need the original model files.

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
variable — see [wandb integration](#wandb-integration) below. Skill ratings use
`Rating/{mode}` for the current policy. All metrics use `Cumulative/steps` as
the chart axis.

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

### Transfer learning

If you have an already-trained model that is too large, you can distill it
into a smaller policy with `learner.transfer_learn(...)`. The student acts in
the environment while the frozen teacher (described by a `TeacherConfig`)
scores the same states; the student's actor and shared head are trained to
match the teacher's action distribution (mean-absolute-difference or KL-div
loss, scaled by `loss_scale`). The critic is not trained.

The teacher and student must share the same action space, but the observation
space may differ: pass an old (teacher) obs builder factory to
`init_with_old_obs` and it runs alongside the student's obs builder in the
collector, scoring the same game states with a different layout. Everything
else (actions, rewards, terminals, shared info) must be identical.

```rust
// same obs space: nothing extra needed at init
let mut learner = config.init(create_env, default_adamw_optimizer::<B>());

// different obs space: the teacher's obs builder, run in lockstep
let mut learner = config.init_with_old_obs(
    create_env,
    default_adamw_optimizer::<B>(),
    || Box::new(OldObs) as Box<dyn Obs<SharedInfo>>,
);

learner.load(); // resume an earlier distillation run if one exists
learner.transfer_learn(
    TeacherConfig {
        models_path: PathBuf::from("checkpoints"), // the big model
        policy_layer_sizes: vec![256; 3],
        shared_head_layer_sizes: vec![256; 2],
        norm: NormSelection::RmsNorm,
    },
    TransferLearnConfig::default(), // distillation hyperparameters
);
```

`TeacherConfig` describes the teacher (its `policy_layer_sizes`,
`shared_head_layer_sizes`, and `norm`); its observation size is probed
automatically from the old obs builder. `models_path` may point directly at a
checkpoint folder (containing `actor.mpk.gz`) or at a directory of timestamped
checkpoints (the latest is used). Keep it distinct from the folder this run
saves its own checkpoints to. The distillation hyperparameters (learning rate,
batch sizes, epochs, loss) live in `TransferLearnConfig`.

Run it with a small `config` (smaller `policy_layer_sizes` etc.) and watch
`Transfer/loss` and `Transfer/accuracy` (argmax agreement with the teacher).
Once distillation has converged, stop with `Q`, then continue with normal PPO
training (`learner.load()` + `learner.learn()`) to fine-tune.

The `rlgymppo-transfer` crate bundles a ready-to-run transfer-learning
example that pretends `rlgymppo-trainer` is the parent model: it loads the
model trained by the trainer's `run` example (architecture, checkpoint folder,
and obs builder are copied from the trainer) and distills it into a smaller
student that observes with `DefaultObs<1>` (53 floats) instead of the parent's
`DefaultObs<3>` (141 floats):

```sh
cargo run -p rlgymppo-transfer --example transfer_learn --features torch
```

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
