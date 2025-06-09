## rlgymppo-rs

Custom implementation of PPO using [RocketSim](https://github.com/VirxEC/rocketsim-rs) + [RLViser](https://github.com/VirxEC/rlviser) via [RLGym-rs](https://github.com/VirxEC/rlgym_rs).

### wandb integration

The environment variable `_WANDB_CORE_PATH` must be set.
The easiest way to do this is `pip install wandb`,
then find where wandb was installed and set it to something like `/path/to/venv/lib/python3.12/site-packages/wandb/bin` - there should be a binary called `wandb-core` in it.

### torch

Using the `torch` backend is preferable as it's the most mature, fastest backend that also supports a wide range of devices.

To properly point to torch, some environment variables might have to be set. See <https://github.com/LaurentMazare/tch-rs?tab=readme-ov-file#getting-started> for help getting started.
