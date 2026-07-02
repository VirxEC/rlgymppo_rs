#[cfg(not(any(
    feature = "torch",
    feature = "cuda",
    feature = "metal",
    feature = "rocm",
    feature = "wgpu",
    feature = "flex",
    feature = "candle"
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
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "cuda",
        any(
            feature = "metal",
            feature = "rocm",
            feature = "wgpu",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "metal",
        any(
            feature = "rocm",
            feature = "wgpu",
            feature = "flex",
            feature = "candle"
        )
    ),
    all(
        feature = "rocm",
        any(feature = "wgpu", feature = "flex", feature = "candle")
    ),
    all(feature = "wgpu", any(feature = "flex", feature = "candle")),
    all(feature = "flex", feature = "candle"),
))]
compile_error!(
    "enable only one backend feature to run this example; backend features are mutually exclusive"
);

fn main() {
    #[cfg(feature = "torch")]
    {
        use burn::backend::LibTorch;
        use burn::backend::libtorch::LibTorchDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<LibTorch>>(LibTorchDevice::Cuda(0), LibTorchDevice::Cpu);
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::Cuda;
        use burn::backend::cuda::CudaDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Cuda>>(CudaDevice::new(0), CudaDevice::default());
    }

    #[cfg(feature = "metal")]
    {
        use burn::backend::Metal;
        use burn::backend::wgpu::WgpuDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Metal>>(WgpuDevice::default(), WgpuDevice::default());
    }

    #[cfg(feature = "rocm")]
    {
        use burn::backend::Rocm;
        use burn::backend::rocm::RocmDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Rocm>>(RocmDevice::new(0), RocmDevice::default());
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::Wgpu;
        use burn::backend::wgpu::WgpuDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Wgpu>>(WgpuDevice::default(), WgpuDevice::default());
    }

    #[cfg(feature = "flex")]
    {
        use burn::backend::Flex;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Flex>>(Default::default(), Default::default());
    }

    #[cfg(feature = "candle")]
    {
        use burn::backend::Candle;
        use burn::backend::candle::CandleDevice;
        use rlgymppo::backend::Autodiff;

        rlgymppo_trainer::run::<Autodiff<Candle>>(CandleDevice::default(), CandleDevice::default());
    }
}
