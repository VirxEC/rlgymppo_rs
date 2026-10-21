// Transfer learning: distill `rlgymppo-trainer`'s parent model (the big
// policy trained by the `rlgymppo-trainer` `run` example) into a smaller
// student policy. PPO fine-tuning requires a separate run.
//
// Before running, train the parent model with the `rlgymppo-trainer` `run`
// example (its checkpoints land in `./checkpoints`). This example then:
//   - loads the parent as the frozen teacher,
//   - collects experience with a smaller student that observes with
//     `DefaultObs<1>` (53 floats) instead of the parent's `DefaultObs<3>`
//     (141 floats) — everything else is identical,
//   - trains the student's actor and shared head to match the teacher's
//     action distribution. The student critic stays untrained. Rewards are
//     metrics only. This example does not run PPO updates.
//
// ```sh
// cargo run -p rlgymppo-transfer --example transfer_learn --features torch
// ```
//
// Student checkpoints are saved to `./checkpoints_transfer`. Once distillation
// has converged (watch `Transfer/loss` and `Transfer/accuracy`), stop with `Q`
// and run the `rlgymppo-trainer` `run` example with `checkpoints_folder`
// pointed at `checkpoints_transfer` to continue with normal PPO training.

#[cfg(not(any(
    feature = "torch",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "wgpu",
    feature = "flex"
)))]
compile_error!(
    "enable exactly one backend feature to run this example, e.g. `cargo run -p rlgymppo-transfer --example transfer_learn --features torch`"
);

#[cfg(any(
    all(
        feature = "torch",
        any(
            feature = "cuda",
            feature = "metal",
            feature = "rocm",
            feature = "wgpu",
            feature = "flex"
        )
    ),
    all(
        feature = "cuda",
        any(
            feature = "metal",
            feature = "rocm",
            feature = "wgpu",
            feature = "flex"
        )
    ),
    all(
        feature = "metal",
        any(feature = "rocm", feature = "wgpu", feature = "flex")
    ),
    all(feature = "rocm", any(feature = "wgpu", feature = "flex")),
    all(feature = "wgpu", feature = "flex"),
))]
compile_error!(
    "enable only one backend feature to run this example; backend features are mutually exclusive"
);

fn main() {
    use burn::tensor::Device;

    #[cfg(feature = "torch")]
    #[allow(
        deprecated,
        reason = "Keep burn-tch support until a later backend migration."
    )]
    {
        rlgymppo_transfer::transfer_learn(
            Device::libtorch_cuda(0).autodiff(),
            Device::libtorch(),
            Some(Device::libtorch()),
            true,
        );
    }

    #[cfg(feature = "cuda")]
    rlgymppo_transfer::transfer_learn(Device::cuda(0).autodiff(), Device::cuda(0), None, false);

    #[cfg(feature = "metal")]
    rlgymppo_transfer::transfer_learn(
        Device::metal(Default::default()).autodiff(),
        Device::metal(Default::default()),
        None,
        false,
    );

    #[cfg(feature = "rocm")]
    rlgymppo_transfer::transfer_learn(Device::rocm(0).autodiff(), Device::rocm(0), None, false);

    #[cfg(feature = "wgpu")]
    rlgymppo_transfer::transfer_learn(
        Device::wgpu(Default::default()).autodiff(),
        Device::wgpu(Default::default()),
        Some(Device::wgpu(burn::tensor::DeviceKind::Cpu)),
        true,
    );

    #[cfg(feature = "flex")]
    rlgymppo_transfer::transfer_learn(Device::flex().autodiff(), Device::flex(), None, true);
}
