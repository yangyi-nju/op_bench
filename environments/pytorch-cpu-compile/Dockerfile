FROM op-bench/pytorch-cpu:torch2.6.0-py311

ENV DEBIAN_FRONTEND=noninteractive

# torch.compile/Inductor emits and compiles C++ kernels even for Python-only
# source overlays. Keep this toolchain in a dedicated image so historical CPU
# tasks retain their admitted runtime unchanged.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
