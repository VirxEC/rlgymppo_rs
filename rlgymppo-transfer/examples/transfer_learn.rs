// Transfer learning: distill `rlgymppo-trainer`'s parent model (the big
// policy trained by the `rlgymppo-trainer` `run` example) into a smaller
// student policy, then continue with normal PPO training afterwards.
//
// Before running, train the parent model with the `rlgymppo-trainer` `run`
// example (its checkpoints land in `./checkpoints`). This example then:
//   - loads the parent as the frozen teacher,
//   - collects experience with a smaller student that observes with
//     `DefaultObs<1>` (53 floats) instead of the parent's `DefaultObs<3>`
//     (141 floats) — everything else is identical,
//   - trains the student's actor and shared head to match the teacher's
//     action distribution.
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
    feature = "flex",
    feature = "candle"
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
        use burn::backend::libtorch::LibTorchDevice;
        use burn::backend::{Autodiff, LibTorch};

        rlgymppo_transfer::transfer_learn::<Autodiff<LibTorch>>(
            LibTorchDevice::Cuda(0),
            LibTorchDevice::Cpu,
            Some(LibTorchDevice::Cpu),
            true,
        );
    }

    #[cfg(feature = "cuda")]
    {
        use burn::backend::cuda::CudaDevice;
        use burn::backend::{Autodiff, Cuda};

        rlgymppo_transfer::transfer_learn::<Autodiff<Cuda>>(
            CudaDevice::new(0),
            CudaDevice::default(),
            None,
            false,
        );
    }

    #[cfg(feature = "metal")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::{Autodiff, Metal};

        rlgymppo_transfer::transfer_learn::<Autodiff<Metal>>(
            WgpuDevice::default(),
            WgpuDevice::default(),
            None,
            false,
        );
    }

    #[cfg(feature = "rocm")]
    {
        use burn::backend::rocm::RocmDevice;
        use burn::backend::{Autodiff, Rocm};

        rlgymppo_transfer::transfer_learn::<Autodiff<Rocm>>(
            RocmDevice::new(0),
            RocmDevice::default(),
            None,
            false,
        );
    }

    #[cfg(feature = "wgpu")]
    {
        use burn::backend::wgpu::WgpuDevice;
        use burn::backend::{Autodiff, Wgpu};

        rlgymppo_transfer::transfer_learn::<Autodiff<Wgpu>>(
            WgpuDevice::default(),
            WgpuDevice::default(),
            Some(WgpuDevice::Cpu),
            true,
        );
    }

    #[cfg(feature = "flex")]
    {
        use burn::backend::{Autodiff, Flex};

        rlgymppo_transfer::transfer_learn::<Autodiff<Flex>>(
            Default::default(),
            Default::default(),
            None,
            true,
        );
    }

    #[cfg(feature = "candle")]
    {
        use burn::backend::candle::CandleDevice;
        use burn::backend::{Autodiff, Candle};

        rlgymppo_transfer::transfer_learn::<Autodiff<Candle>>(
            CandleDevice::default(),
            CandleDevice::default(),
            Some(CandleDevice::Cpu),
            true,
        );
    }
}
