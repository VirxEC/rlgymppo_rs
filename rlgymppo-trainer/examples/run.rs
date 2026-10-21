#[cfg(not(any(
    feature = "torch",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "wgpu",
    feature = "flex"
)))]
compile_error!(
    "enable exactly one backend feature to run this example, e.g. `cargo run -p rlgymppo-trainer --example run --features torch`"
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
        rlgymppo_trainer::run(
            Device::libtorch_cuda(0).autodiff(),
            Device::libtorch(),
            Some(Device::libtorch()),
            true,
        );
    }

    #[cfg(feature = "cuda")]
    rlgymppo_trainer::run(Device::cuda(0).autodiff(), Device::cuda(0), None, false);

    #[cfg(feature = "metal")]
    rlgymppo_trainer::run(
        Device::metal(Default::default()).autodiff(),
        Device::metal(Default::default()),
        None,
        false,
    );

    #[cfg(feature = "rocm")]
    rlgymppo_trainer::run(Device::rocm(0).autodiff(), Device::rocm(0), None, false);

    #[cfg(feature = "wgpu")]
    rlgymppo_trainer::run(
        Device::wgpu(Default::default()).autodiff(),
        Device::wgpu(Default::default()),
        Some(Device::wgpu(burn::tensor::DeviceKind::Cpu)),
        true,
    );

    #[cfg(feature = "flex")]
    rlgymppo_trainer::run(Device::flex().autodiff(), Device::flex(), None, true);
}
