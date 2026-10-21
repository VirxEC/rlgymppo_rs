# Migrate a downstream bot to RocketSim 0.3 and Burn 0.22

This guide is for an agent that maintains a bot built with `rlgymppo-rs`,
RLGym-rs, and RLViser. The bot builds or runs in Docker. Update simulation,
visualization, and neural-network code together without changing the trained
policy's observation space or action space.

## 1. Read the upstream guides and inspect the project

Read these guides before editing:

- [RocketSim 0.3 migration](https://github.com/ZealanL/RocketSim/blob/v3-rust/MIGRATING.md)
- [Burn 0.22 migration](https://burn.dev/books/burn/migrating-to-0.22.html)

Use the updated `rlgymppo-rs` source as the integration reference:

- [Trainer configuration and initialization](rlgymppo-trainer/src/lib.rs)
- [Runtime device selection](rlgymppo-trainer/examples/run.rs)
- [Viewer connection and render loop](rlgymppo/src/environment/render.rs)
- [Viewer controls and action requests](rlgymppo/src/environment/sim.rs)
- [Inference policy loading](rlgymppo-model/src/policy.rs)
- [RLBot configuration and inference](rlgymppo-rlbot/src/agent.rs)

Before editing the downstream project:

1. Read its agent instructions and inspect its uncommitted changes.
2. Find every direct dependency on Burn, RocketSim, RLGym, RLBot, and RLViser.
3. Find custom observation builders, action parsers, rewards, and state setters.
4. Find copied renderer code and custom checkpoint loaders.
5. Identify the Docker container, workspace mount, and networking mode.
6. Record the checkpoint format, model architecture, and observation/action sizes.
7. Keep an unchanged copy of the existing checkpoints.

Run builds and tests in the project's container. Do not substitute a host build.
`hardcore_lovelace` is the container used to validate `rlgymppo-rs`; do not assume
that a downstream project uses that name or the same mount paths.

## 2. Update dependencies as one compatible set

Update the downstream `rlgymppo-rs` dependency to a revision that contains both
migrations. Preserve the project's dependency source unless a source change is
required. Do not leave a Git revision pinned to the old API.

| Dependency | Target |
|---|---|
| `burn` and direct Burn subcrates | `0.22` |
| `rocketsim`, when used directly | `0.3` |
| RLGym-rs | Updated revision with the `vis` feature and RocketSim 0.3 |
| `rlbot-rocketsim`, when used | `0.2` |
| `rlbot`, when used directly with `rlbot-rocketsim` | `0.7` |
| `glam`, when exchanged with these simulation/bot types | Compatible version; this integration uses `0.34` |
| RLViser dependency | Remove after replacing its call sites |

The current integration resolves Burn 0.22.1 and RocketSim 0.3.1. It uses
`rocketsim_vis` 0.3.1 through RLGym. Install a compatible viewer separately.
Do not add a direct `rocketsim_vis` dependency unless downstream code imports it.

Use Rust 1.95 or later for Burn 0.22. Check the container's toolchain.

**Keep `burn-tch` for this migration.** Keep the project's `torch` feature mapped
to `burn/tch`. Burn deprecates LibTorch in 0.22, but it remains supported here.
Do not replace it with a CubeCL backend as part of this task. Remove Candle
features and code because Burn 0.22 removed Candle.

Preserve the existing LibTorch/PyTorch configuration in Docker. Check version
compatibility with the resolved `tch` dependency. Do not treat a version-check
bypass as proof of runtime compatibility.

Inspect `cargo tree -d` and the reverse dependency trees for `burn`, `rocketsim`,
and `glam`. Remove duplicate versions that make shared types incompatible.
Prefer the `rlgymppo` or utility crate re-exports for shared simulation types.
Do not add an old-to-new state adapter to keep RocketSim 0.2 in the graph.

Follow the downstream project's lockfile policy. This repository ignores
`Cargo.lock`; the downstream project may track it.

### RLGym visualization feature

RLGym now has a default-enabled `vis` feature. It enables the optional
`rocketsim_vis` dependency and the viewer methods on `Env`.

If a direct RLGym dependency disables default features, enable `vis` explicitly
when the code uses viewer methods. A build without `vis` must gate those call
sites. Cargo features are additive: another dependency can still enable `vis`.
Disabling defaults on one dependency declaration does not prove the resolved
build has no viewer support.

## 3. Migrate RocketSim and custom RLGym components

Follow the RocketSim migration guide for the full API changes. In downstream
bot code, check these integration points first:

- Replace `state.is_on_ground` with `state.is_on_ground()`.
- Replace `state.world_contact_normal` with `state.world_contact_normal()`.
- Replace external `CarState` struct literals that access private fields.
  Construct a default state and set the supported fields or use the new API.
- Use `Arena::set_car_state` with `CarCore` for ordinary state placement.
  Existing mutable `CarState` values can supply `state.core()`.
- Use `set_car_state_full` when restoring a complete snapshot with history.
  Do not use the core setter when that history must survive restoration.
- Do not replace measured contacts with fabricated flags to make code compile.
- Use `rlbot-rocketsim` 0.2 directly for state enrichment. Do not retain a
  RocketSim 0.2-to-0.3 conversion layer.

Keep custom components generic over their RLGym `SharedInfo` type where required.
The Burn migration does not remove RLGym component generics.

Keep observation ordering, normalization, masks, and action-table ordering stable.
A matching tensor length alone does not establish policy compatibility.

## 4. Replace RLViser with the separate viewer

### Run the viewer on the host

The usual Docker setup runs the bot in the container and the viewer on the host.
The user starts the viewer manually. The container connects to it through TCP.
Do not install or launch a GUI inside Docker to solve a connection failure.

For the current 0.3.1 integration, run on the host:

```sh
cargo install --locked rocketsim_viewer --version 0.3.1
rocketsim-viewer --bind 127.0.0.1 39201
```

With Linux host networking, configure the learner:

```rust
render: true,
render_viewer_addr: Some("127.0.0.1:39201".into()),
```

With bridge networking, container loopback does not reach the host. Use a host
IP reachable from the container. Bind the viewer to that interface and allow
TCP port `39201` through the firewall. A hostname alias alone is not sufficient
for `render_viewer_addr`, which expects a numeric `IP:port`.

The viewer is the TCP listener. Publishing a port on the bot container does not
make a host listener reachable. If the viewer runs in another container, expose
or publish the viewer's listening port as required by that network setup.
Do not expose this control connection to untrusted networks.

With `render_viewer_addr: None`, the learner uses local viewer startup/reuse.
Set an explicit address for a manually started host viewer.

### Replace custom viewer calls

If the downstream project uses the learner's renderer unchanged, configure
`render_viewer_addr` and remove obsolete RLViser setup. The learner already
applies the new viewer controls and requests.

If it has a custom renderer, migrate the renderer's loop as well:

| Purpose | New `Env` API |
|---|---|
| Enable local viewer startup/reuse | `set_vis_enabled(true)` |
| Connect to a manually started viewer | `set_net_vis(Some(addr))` |
| Check connection state | `vis_connected()` |
| Read recent car controls | `latest_vis_controls()` |
| Consume pending action requests | `take_vis_actions()` |
| Access debug drawing | `debug_renderer()` |

`set_vis_enabled` and `set_net_vis` return `io::Result<()>`. Propagate the error
or report it with the address and operation. Do not discard it.

The new API does not provide RLViser's pause, speed, or message-polling methods.
Remove those calls. If the bot needs application-level pause or speed control,
implement it in the application. The learner's renderer uses real-time pacing.

For a custom loop, apply viewer input in this order:

1. Consume pending action requests before policy inference.
2. Validate the selected car index against the current arena.
3. Apply reset, dribble, or launch requests. Reset takes priority.
4. After a reset, reset the environment components through the normal reset path.
5. After a ball edit, refresh state, shared info, observations, and action masks.
   Do not calculate another reward or terminal result without a physics step.
6. Apply policy controls, then override the selected car with valid viewer controls.
7. Step physics, then perform the normal post-step processing.

The current request bits are reset at bit 0, dribble at bit 1, and launch at bit 2.
Check these against the resolved viewer API before copying constants.
`latest_vis_controls()` can return `None` when controls expire. Leave policy
controls active in that case. Do not retain stale manual controls indefinitely.

Keep manual controls and viewer state edits out of training collection arenas.
They belong in the rendered arena unless the application explicitly requires
interactive data collection.

## 5. Migrate Burn callers to runtime devices

Use the Burn migration guide for tensor, module, optimizer, and numeric API
changes. The downstream `rlgymppo-rs` integration needs these public API changes:

- Remove backend parameters from `LearnerConfig`, `Ppo`, `Actic`, `Net`, `Policy`,
  `PPOOutput`, `NextoModel`, and `NextoOutput`.
- Remove the Burn backend parameter from `Learner`. Keep its RLGym component
  parameters.
- Replace `B::Device` with `burn::tensor::Device` in application code.
- Replace `Tensor<B, D>` with `Tensor<D>`. Keep rank and tensor-kind parameters.
- Remove backend type arguments from optimizer factories.
- Custom optimizer factories now return `ModuleOptimizer`.
- RLBot custom `PpoBotConfig::load_policy` takes `&Device` and returns `Policy`.
  The old `rlgymppo_rlbot::Backend` alias is removed.

For a LibTorch learner, use explicit runtime devices:

```rust
use rlgymppo::burn::tensor::Device;
use rlgymppo::{LearnerConfig, default_adamw_optimizer};

#[allow(deprecated)] // Retain burn-tch during this migration.
let config = LearnerConfig {
    device: Device::libtorch_cuda(0).autodiff(),
    render_device: Device::libtorch(),
    skill_tracker_device: Some(Device::libtorch()),
    render: true,
    render_viewer_addr: Some("127.0.0.1:39201".into()),
    ..Default::default()
};

let mut learner = config.init(create_env, default_adamw_optimizer());
```

Use `Device::libtorch().autodiff()` for a CPU training test. Use `Device::flex()`
for Flex inference when that feature is enabled. Do not rely on
`Device::default()` to select the same backend after Cargo feature changes.

Learner initialization enables autodiff on its training device. Custom training
loops must enable it before model and input initialization. Use plain devices
and `model.valid()` snapshots for inference.

Read the guide's module movement rules before changing `to_device` calls.
Moving a tensor does not change its autodiff context. Moving initialized model
parameters can produce graph intermediates rather than independent trainable
leaves. Use the documented `train().fork(&device)` path when independent training
parameters are required. Preserve parameter IDs where optimizer state needs them.

Runtime devices can use different integer dtypes. Do not replace a generic
integer read with a hard-coded `i32` read. Use `try_into_vec_as::<i64>()` when
conversion is intended, then validate conversion to action indices. Exact reads
such as `try_into_vec::<i64>()` require the stored dtype to match.

## 6. Convert checkpoints before loading them

**Burn 0.22 cannot directly load `.mpk` or `.mpk.gz` checkpoints.** Renaming files
to `.bpk` does not convert them. Do not catch a load failure and continue with
random weights.

Follow [Burn's checkpoint migration procedure](https://burn.dev/books/burn/migrating-to-0.22.html#migrating-checkpoints).
Use a separate conversion project with the old Burn version and original model
definition. Do not put incompatible Burn versions into the bot's normal runtime.

Convert each required component with the same architecture and parameter paths:

| Component | New filename |
|---|---|
| Actor | `actor.bpk` |
| Critic, required for training resume | `critic.bpk` |
| Shared head, when configured | `shared_head.bpk` |

Export the old weights through `burn-store`. Import them into the corresponding
0.22 module and save its burnpack file. Compare outputs on fixed inputs before
using the converted policy. For bot inference, load the actor and configured
shared head. Training also needs the critic.

Update custom loaders to the new fallible `try_load_file` or store APIs.
Update deployment scripts, Docker copies, checkpoint discovery, and RLBot
configuration to use the new filenames.

Legacy optimizer state has no automatic migration. Start with fresh optimizer
state or implement a separate conversion. Do not describe converted model weights
as an exact training resume. Saved self-play policies, transfer teachers, and
regen anchors also need compatible weights if the application uses them.
Keep those assets separate from unchanged legacy checkpoints.

## 7. Validate inside Docker

Run commands through the discovered container and workspace mount. Check the
project's selected backend features before choosing test arguments. Do not use
`--all-features` when the examples require one backend.

Validate in this order:

1. Check the dependency graph and compile the bot and training entry points.
2. Run formatting and Clippy with the project's toolchain.
3. Run unit tests and doctests. Record unrelated failures separately.
4. Load the actual converted checkpoint. Compare fixed-input logits and actions.
5. Run two PPO updates if the project trains. Check finite losses and changed
   parameters. Exercise masks, truncation bootstrap, and mini-batch accumulation.
6. Save and reload model and optimizer state from a new checkpoint. Compare the
   next update against uninterrupted training. Include regen when configured.
7. Run a short simulation with the bot's real observation and action builders.
8. Ask the user to start the host viewer. Test TCP connection, debug drawing,
   manual controls, reset, dribble, launch, and disconnect behavior.
9. Test GPU execution separately from CPU tests when GPU hardware is available.

A successful compile does not prove checkpoint compatibility or TCP connectivity.
A CPU LibTorch test does not prove that the container can execute GPU kernels.

## 8. Report the migration and test limits

In the completion report, state:

- The dependency versions and RLGym/learner revisions used.
- The source files and Docker configuration changed.
- Whether checkpoints were converted and fixed-input outputs compared.
- Whether optimizer state started fresh or resumed from a new-format checkpoint.
- The validation commands and their results.
- Whether a real viewer connection and GPU execution were tested.
- Any application controls removed because the new API has no replacement.

Do not claim that the downstream bot is validated from this repository's tests
alone. This repository's migration passed Flex and LibTorch CPU training tests.
The downstream bot's custom components, checkpoints, host viewer connection, and
GPU setup still require their own tests.
