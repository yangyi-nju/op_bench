FROM op-bench/pytorch-cuda:torch2.6.0-cu124-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp311-cp311-linux_x86_64.whl
ARG TORCH_WHEEL_SHA256=81397ff1c84a3f2c666d2627144ecac268665325726267e092a80113385ad3e8
ARG TORCH_WHEEL_PATH=/tmp/torch-2.4.0+cu124-cp311-cp311-linux_x86_64.whl

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --output "${TORCH_WHEEL_PATH}" "${TORCH_WHEEL_URL}" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import torch; assert torch.__version__ == '2.4.0+cu124'; print(torch.__version__)"

WORKDIR /workspace
