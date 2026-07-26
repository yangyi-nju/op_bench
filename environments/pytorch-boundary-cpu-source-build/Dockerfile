FROM op-bench/pytorch-cpu-compile:torch2.6.0-py311

ENV DEBIAN_FRONTEND=noninteractive
ENV CCACHE_DIR=/workspace/.ccache
ENV CCACHE_BASEDIR=/workspace
ENV CCACHE_NOHASHDIR=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends ccache cmake ninja-build \
 && rm -rf /var/lib/apt/lists/* \
 && ccache --version

RUN python -m pip install --no-cache-dir \
        astunparse \
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

WORKDIR /workspace
