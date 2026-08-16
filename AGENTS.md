# AGENTS.md

Guidance for AI agents working in this repository.

## Project overview

`rlgymppo-rs` is a Rust implementation of [GigaLearn](https://github.com/VirxEC/GigaLearn),
a PPO trainer for Rocket League. It is built on RocketSim v3 + RLViser via
RLGym-rs, with neural networks and training written in [Burn](https://burn.dev) 0.21
instead of PyTorch.

## Reference implementation

The original C++ project lives in `GigaLearnCPP/` (a git submodule-style checkout
of GigaLearn + RLGymCPP). It is **gitignored**, so search tools (`grep`, `find_path`)
will not see it — use shell commands (`ls`, `cat`, `grep -r`, `find`) to consult it.
The Rust code is a port; when behavior is ambiguous, check the C++ headers first
(e.g. `GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h`).

## Workspace layout

Nine crates, all Rust edition 2024:

| Crate | Purpose |
|---|---|
| `rlgymppo` | Core PPO learner, multi-threaded environment runner, training loop. |
| `rlgymppo-model` | Backend-generic policy/model definitions and checkpoint-compatible inference loading. |
| `rlgymppo-nexto` | Nexto model inference, observation builder, action table, and pre-generated model weights. |
| `rlgymppo-rlbot` | RLBot v5 agent that runs trained policies (Burn Flex + RocketSim state enrichment). |
| `rlgymppo-utils` | Reusable RLGym observation builders, action parsers, shared-info traits — no dependency on the learner. |
| `rlgymppo-tui` | Terminal dashboard for live training metrics (ratatui). |
| `rlgymppo-wandb` | Weights & Biases integration via embedded Python (pyo3). |
| `rlgymppo-trainer` | Bundled training example (`examples/run.rs`) with self-play and skill tracking. |
| `rlgymppo-transfer` | Transfer-learning example (`examples/transfer_learn.rs`). |

Key modules inside `rlgymppo`:

- `src/agent/` — PPO (`config.rs`, `gae.rs`, `model.rs`), self-play (`self_play.rs`), skill tracking (`skill_tracker.rs`), transfer learning (`transfer_learn.rs`)
- `src/base/` — rollout memory (`memory.rs`)
- `src/environment/` — game simulation: `sim.rs` (single game), `thread_sim.rs` (worker threads), `batch_sim.rs` (batch stepping), `render.rs` (RLViser renderer)
- `src/utils/` — observations, rewards, terminal conditions, state setters, `running_stat.rs` (running stats), `avg_tracker.rs`, `report.rs` (metric reporting)

- Entry points: `LearnerConfig` + `Learner` in `rlgymppo/src/lib.rs`.
`Learner::load()` is safe to call unconditionally — it resumes from the latest
checkpoint if one exists. `Learner::learn()` trains forever unless
`num_additional_iterations` is set. `Learner::transfer_learn()` distills a frozen
teacher into a smaller student.

## Writing custom obs, actions, state setters, rewards & terminals

Everything is generic over the `SharedInfo` type, which is the single hook for
sharing data between components (RNG, per-tick metrics, episode state).

**Traits** — all extension traits come from `rlgym` (re-exported as
`rlgymppo::rlgym`, defined in `rlgym/src/lib.rs` of the git checkout):

- `Obs<SI>` — `get_obs_space()`, `reset()`, `build_obs() -> FullObs`
- `Action<SI>` — `get_tick_skip()`, `get_action_delay()`, `get_action_space()`,
  `parse_actions() -> &[(usize, CarControls)]`, `get_action_masks()`
- `StateSetter<SI>` — `apply(&mut self, arena: &mut Arena, shared_info: &mut SI)`
- `Reward<SI>` — `get_rewards() -> Vec<f32>` (one float per car)
- `Terminal<SI>` / `Truncate<SI>` — `is_terminal()` / `should_truncate()`
- `SharedInfoProvider` — `reset()`, `update()` (called each tick; the natural
  place to track player metrics)

**Where things live**

- Reusable, learner-independent builders go in `rlgymppo-utils`:
  `obs/` (`DefaultObs<N>`, `AdvancedObs<N>`), `actions/`
  (`DefaultAction<MAX_PLAYERS, TICK_SKIP, ACTION_DELAY>` — 90-entry discrete
  action table with validity masks), `shared_info.rs` (`SharedInfoRng`). The
  core crate re-exports these as `rlgymppo::utils::{obs, actions}`.
- Nexto inference and its Rust observation/action functions live in
  `rlgymppo-nexto`. The crate embeds pre-generated Burn model code and BurnPack
  weights from `rlgymppo-nexto/nexto/`. Cargo does not run model conversion or
  model code generation.
- Trainer-side pieces go in `rlgymppo/src/utils/`: `rewards/`
  (`FaceBallReward`, `VelocityToBallReward`, `GoalReward`, `DemoReward`, …),
  `terminal/` (`OnGoalCondition`, `NoTouchCondition<MAX_TICKS>`,
  `RandomGameEndedCondition<MIN, MAX>`, …), `state_setters/` (`KickoffState`,
  `RandomState<CARS_ON_GROUND, BALL_ON_GROUND, RANDOM_STRENGTH>`,
  `WeightedState`), `shared_info/mod.rs` (`SharedInfoReport`), `report.rs`
  (`Report`, `AvgTracker`).
- A custom `SharedInfo` implements `SharedInfoProvider` + `SharedInfoRng` (state
  setters need the RNG) + `SharedInfoReport` (reward averaging; metric names
  starting with `Reward/` are auto-tracked by the learner).

**Wiring** — `create_env` in `rlgymppo-trainer/src/lib.rs` is the canonical
example: `Env::new(arena, state_setter, obs, action, reward, terminal, truncate,
shared_info)` with concrete types flowing into `LearnerConfig::init(create_env,
optimizer)`. The `Learner` is generic over `SS, OBS, ACT, REW, TERM, TRUNC, SI`.

**Composition macros** (exported at `rlgymppo`'s root, each with doctests that
must stay in sync): `weighted_state![Type, weight; ...]`,
`combined_rewards!["name", RewardType => weight; ...]`, `any_terminal![Type, ...]`.

**Watch out**

- Obs space and action space are derived from the builders (no config knobs),
  but checkpoints are architecture-specific: changing obs size or action count
  makes old checkpoints unloadable. Keep them stable mid-run.
- Transfer learning requires identical action spaces; the obs space may differ
  (pass the teacher's obs builder via `init_with_old_obs`).
- `rlgymppo-rlbot` reuses the `rlgymppo-utils` obs/action builders for
  inference, so custom obs/actions used in training must also be runnable there
  (and the agent's `PolicyConfig` sizes must match the trained model).
- `DefaultObs<3>` is 141 floats, `DefaultObs<1>` is 53 — the const generic is
  players per team.

## Build & run

Backends are mutually exclusive — enable **exactly one** feature: `torch`,
`cuda`, `metal`, `rocm`, `wgpu`, `flex`, `candle`. `torch` is the most mature.

```sh
# train
cargo run -p rlgymppo-trainer --example run --features torch

# transfer-learn (distill rlgymppo-trainer's model into a smaller student)
cargo run -p rlgymppo-transfer --example transfer_learn --features torch
```

`rlgymppo` defaults to `tui` + `wandb`; the trainer crate defaults to
`wandb` + `tui` + `torch`.

### Environment prerequisites

- **wandb**: the `_WANDB_CORE_PATH` env var must point at the directory
  containing the `wandb-core` binary (e.g. a venv's `wandb/bin`, or the
  global Python install's site-packages `wandb/bin` — a plain `pip install
  wandb` into the global interpreter is fine).
- **torch backend**: set `LIBTORCH`, `LIBTORCH_INCLUDE`, `LIBTORCH_LIB` to a
  LibTorch installation if needed. The devcontainer (`.devcontainer/Dockerfile`)
  instead uses the system PyTorch via `LIBTORCH_USE_PYTORCH=1` and
  `LIBTORCH_BYPASS_VERSION_CHECK=1` (ROCm container).
- **RLBot**: `rlgymppo-rlbot/examples/rlbot/main.rs` embeds collision meshes from
  `collision_meshes/` via `include_bytes!` — those files must exist to build it.

### Gotchas

- `Cargo.lock` is **gitignored** (not tracked) — do not try to commit it.
- `rlgym` comes from a git branch (`native-rust`), and RocketSim is patched via
  `[patch."https://github.com/ZealanL/RocketSim.git"]` to the
  `fix-jump` branch — keep the patch table intact.
- `SkillTrackerConfig::default()` has `enabled: false` and
  `nexto_mmr: Some(1500.0)`. Set `enabled` to `true` to run evaluations.
  Enabled evaluations use randomly selected saved previous policy versions and
  Nexto when configured. Old-version matches use two-sided Elo with saved
  version ratings. Nexto is the only fixed bot; its MMR is configurable. Set
  `nexto_mmr` to `None` to disable Nexto without loading its model. Set
  `enabled` to `false` to disable all evaluations.
- `rand` is 0.10 — note the newer API (e.g. `rand::seq::SliceRandom`,
  `Rng::random_range`).
- Checkpoints are `NamedMpkGzFileRecorder` files (`actor.mpk.gz`, etc.) in
  `checkpoints_folder` (default `./checkpoints`, gitignored).
- `checkpoints`, `wandb/`, `target/`, `collision_meshes/`, `rlviser*`,
  `GigaLearnCPP/`, `Cargo.lock` are all gitignored.
- Keep `default = ["tui", "wandb"]` on `rlgymppo` unless a consumer explicitly
  opts out with `default-features = false` (as `rlgymppo-trainer` does).

## Development workflow

- **Format**: always `cargo +nightly fmt` (the `rustfmt.toml` uses unstable
  options: `imports_granularity = "Module"`, `group_imports = "StdExternalCrate"`,
  `use_field_init_shorthand = true`). If nightly Rust isn't installed, remove
  `imports_granularity` and `group_imports` from `rustfmt.toml` and use plain
  `cargo fmt` instead.
- **Lint**: `cargo clippy` must be clean.
- **Check/tests**: `cargo check`, `cargo test` (includes doctests — reward and
  terminal macros have doctests that must stay in sync).
- Build only what you touched when iterating:
  `cargo check -p rlgymppo`, `cargo test -p rlgymppo-trainer --example run`.

## Conventions

- Match the existing structure: one module per concern inside `rlgymppo/src/`,
  generic over `B: AutodiffBackend` for backend-agnostic code.
- Obs/action/shared-info traits used by the trainer live in `rlgymppo-utils`
  (`DefaultObs`, `AdvancedObs`, `DefaultAction`); the trainer re-exports them.
- When adding reward/terminal logic to the trainer, keep the macro-based
  implementations and their doctests in sync.
- Training-control keys (handled in `stdin_reader`/`handle_input_char` in
  `rlgymppo/src/lib.rs`): `Q` quit, `S` quick-save, `R` renderer toggle,
  `D` deterministic renderer.
