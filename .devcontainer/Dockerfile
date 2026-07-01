FROM rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.9.1

ENV DEBIAN_FRONTEND=noninteractive

# 6700xt fix
# ENV HSA_OVERRIDE_GFX_VERSION=10.3.0
# ENV PYTORCH_ROCM_ARCH=gfx1030

# 7700S fix
ENV HSA_OVERRIDE_GFX_VERSION=11.0.0

ENV HIP_VISIBLE_DEVICES=0

# BLAS acceleration
ENV TORCH_BLAS_PREFER_HIPBLASLT=1

# Reduce kernel launch overhead
ENV HIP_FORCE_DEV_KERNARG=1

# Reduce implicit sync stalls
ENV AMD_SERIALIZE_COPY=0
ENV AMD_SERIALIZE_KERNEL=0

# tch-rs uses system PyTorch backend
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

# Library path fixes (ROCm + PyTorch runtime)
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    htop \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup toolchain install beta && \
    rustup toolchain install nightly && \
    rustup default beta

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

CMD ["/bin/bash"]
