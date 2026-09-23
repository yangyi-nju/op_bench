# Dependency image recipe, not a built or admitted environment claim.
# Complete candidate source is supplied separately at the declared revision.
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ninja-build libopenblas-dev libomp-dev \
    python3 python3-dev python3-pip python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/opbench-logsoftmax/requirements.txt
ARG PIP_NO_INDEX=0
RUN --mount=type=bind,source=wheels,target=/wheels,readonly \
    PIP_NO_INDEX=${PIP_NO_INDEX} python3 -m pip install --no-cache-dir --timeout 120 \
    --index-url https://pypi.org/simple --find-links=/wheels \
    -r /opt/opbench-logsoftmax/requirements.txt \
    && python3 -m pip check \
    && python3 -m pip freeze --all > /opt/opbench-logsoftmax/python-packages.txt \
    && dpkg-query -W > /opt/opbench-logsoftmax/system-packages.txt

# Softmax uses ATen's CUDA implementation. The disabled optional backends are
# outside this candidate's eager CPU/CUDA numerical scope. This still builds the
# complete configured framework; no kernel extraction or installed-wheel overlay.
ENV USE_CUDA=1 TORCH_CUDA_ARCH_LIST=7.0 USE_CUDNN=0 USE_DISTRIBUTED=0 \
    USE_NCCL=0 USE_MAGMA=0 USE_KINETO=0 USE_MKLDNN=0 USE_FBGEMM=0 \
    USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_CUSPARSELT=0 USE_CUDSS=0 \
    USE_NNPACK=0 USE_XNNPACK=0 BUILD_TEST=0 MAX_JOBS=12 \
    BLAS=OpenBLAS PYTORCH_BUILD_VERSION=2.6.0a0+opbench PYTORCH_BUILD_NUMBER=1
WORKDIR /workspace
