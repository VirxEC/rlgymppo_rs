# rlgymppo-rlbot

Runs an `rlgymppo-trainer` checkpoint as an RLBot v5 agent using Burn Flex.

RLBot packets are enriched through `rlbot-rocketsim`, then adapted to an RLGym
`GameState` inside this crate. The example selects its shared info, observation
builder, and action parser directly in `examples/rlbot/main.rs`:

```rust
type Bot = PpoBot<DefaultObs<3>, DefaultAction<6, 8>, SharedInfo>;
```

Shared info must implement `From<u64>` for per-agent seeding. Custom actions
must implement the RLGym `Action` trait, `Default`, and
`rlgymppo_rlbot::RlbotAction`. Observation and action-space sizes are read from
the configured implementations when the policy is loaded.

## Checkpoint

Set `RLGYMPPO_CHECKPOINT` to a checkpoint directory containing:

- `actor.mpk.gz`
- `shared_head.mpk.gz`

If unset, the bot selects the newest numeric directory under `./checkpoints`.
The bundled policy architecture matches `rlgymppo-trainer::default_config`:

- 141 observation values
- shared head `[256, 256, 256]`
- actor hidden layers `[256, 256]`
- RMSNorm
- 90 discrete actions
- 8-frame action repeat

## Build

```sh
cargo build --release -p rlgymppo-rlbot --example rlbot
```

Point RLBot at `rlgymppo-rlbot/examples/rlbot/bot.toml`. The example directory
also contains `loadout.toml` and `logo.svg`. The local run command invokes Cargo
with the workspace manifest. Soccar collision meshes are embedded into the
binary; the bot only searches parent directories for `checkpoints/`. Set an
absolute `RLGYMPPO_CHECKPOINT` path if the checkpoint lives elsewhere.

## Botpack

`examples/rlbot/` contains `bob.toml` and `bob.Dockerfile`. Bob cross-compiles
the `rlbot` Cargo example for Windows GNU and Linux musl, packages the RLBot
configuration/loadout/logo and `checkpoints/`. Collision meshes remain embedded.
A trained checkpoint must be present under `checkpoints/` before producing a
Botpack submission.

```sh
bob build rlgymppo-rlbot/examples/rlbot/bob.toml
```
