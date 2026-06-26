FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        curl \
        git \
        ninja-build \
        python3.11 \
        python3.11-dev \
        python3-pip \
        ripgrep \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 100 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 100

# PyTorch build dependencies
RUN python -m pip install --no-cache-dir \
        astunparse \
        cmake \
        cffi \
        expecttest \
        future \
        hypothesis \
        ninja \
        numpy \
        packaging \
        psutil \
        pytest \
        pytest-xdist \
        pyyaml \
        requests \
        setuptools \
        six \
        types-dataclasses \
        typing-extensions

# Install a pre-built PyTorch matching the development branch (fallback runtime)
# The agent will rebuild from source via inplace_build; this provides a baseline
# install so imports work before the first build.
RUN python -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/cusparselt/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_cupti/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_nvrtc/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cufft/lib:/usr/local/lib/python3.11/dist-packages/nvidia/curand/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusolver/lib:/usr/local/lib/python3.11/dist-packages/nvidia/cusparse/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nccl/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nvjitlink/lib:/usr/local/lib/python3.11/dist-packages/nvidia/nvtx/lib:${LD_LIBRARY_PATH}

# Use ccache via the standard symlink dir (/usr/lib/ccache/gcc -> ccache).
# Do NOT set CC="ccache gcc" — CMake may misparse the compound value and pass
# the source file as the compiler binary, causing "Permission denied" errors.
ENV PATH=/usr/lib/ccache:${PATH}
ENV CCACHE_DIR=/workspace/.ccache
ENV CCACHE_MAXSIZE=10G

# PyTorch build env
ENV USE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
ENV MAX_JOBS=8
ENV PYTORCH_BUILD_VERSION=2.6.0
ENV PYTORCH_BUILD_NUMBER=0

WORKDIR /workspace
