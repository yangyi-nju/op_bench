FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# System build dependencies
# Note: cmake is installed via pip below (pinned to 3.x) so we skip apt's cmake
# to avoid PATH conflicts between /usr/bin/cmake and /usr/local/bin/cmake.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        curl \
        git \
        libomp-dev \
        libopenblas-dev \
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
# Pin cmake to 3.31.x (last CMake 3 series). PyTorch 2.6 third_party (protobuf)
# uses cmake_minimum_required < 3.5, which CMake 4.0 removed. Pinning to 3.x
# plus CMAKE_POLICY_VERSION_MINIMUM=3.5 below covers both new and old projects.
RUN python -m pip install --no-cache-dir \
        astunparse \
        "cmake<4" \
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
        "setuptools>=70,<81" \
        six \
        types-dataclasses \
        typing-extensions \
        "wheel>=0.42"

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

# Allow building third-party CMake projects (e.g. protobuf) that declare
# cmake_minimum_required < 3.5. CMake 4.0+ removed that compatibility band.
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# PyTorch build env
ENV USE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"
# MAX_JOBS controls both setup.py's parallel compile and cmake ninja concurrency.
# Default to the container's visible CPU count so we saturate whatever host the
# operator has provisioned (V100 machine: 20 cores). Override at build/run time
# via `--build-arg MAX_JOBS=N` or `docker run -e MAX_JOBS=N`.
ARG MAX_JOBS
ENV MAX_JOBS=${MAX_JOBS:-20}
# Disable optional components that require network access during CMake FetchContent
# (post-2.6 nightlies pull NCCL / gloo / tensorpipe sources on the fly). The
# benchmark tasks are single-GPU CUDA kernel bugs — none need distributed backends.
ENV USE_NCCL=0
ENV USE_DISTRIBUTED=0
ENV USE_TENSORPIPE=0
ENV USE_GLOO=0
ENV USE_MPI=0
ENV USE_KINETO=0
ENV PYTORCH_BUILD_VERSION=2.6.0
ENV PYTORCH_BUILD_NUMBER=0

WORKDIR /workspace
