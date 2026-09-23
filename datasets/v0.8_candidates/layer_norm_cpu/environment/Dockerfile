# Candidate CPU build environment. The task/source is provided separately;
# no installed torch wheel, original repository history or private tests.
FROM python:3.12.7-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libopenblas-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/opbench-layernorm/requirements.txt
ARG PIP_NO_INDEX=0
RUN --mount=type=bind,source=wheels,target=/wheels,readonly \
    PIP_NO_INDEX=${PIP_NO_INDEX} python -m pip install --no-cache-dir --timeout 120 \
    --index-url https://pypi.org/simple --find-links=/wheels \
    -r /opt/opbench-layernorm/requirements.txt \
    && python -m pip check \
    && python -m pip freeze --all > /opt/opbench-layernorm/python-packages.txt \
    && dpkg-query -W > /opt/opbench-layernorm/system-packages.txt

COPY build_source.py source_torch.py reproduce.py /opt/opbench-layernorm/
ENV USE_CUDA=0 USE_ROCM=0 USE_XPU=0 USE_DISTRIBUTED=0 BUILD_TEST=0 \
    MAX_JOBS=16 BLAS=OpenBLAS CMAKE_BUILD_TYPE=Release \
    PYTORCH_BUILD_VERSION=2.6.0a0+opbench.240aa77 PYTORCH_BUILD_NUMBER=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
