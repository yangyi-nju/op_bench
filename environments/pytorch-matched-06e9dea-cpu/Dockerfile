FROM op-bench/pytorch-cpu-compile:torch2.6.0-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cpu/torch-2.7.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=6b7edcbf8bb0b9ac2e6c001434797c5ec3f25394f91eb0ed7aeeeeed9ad4500f
ARG TORCH_WHEEL_PATH=/tmp/torch-2.7.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN wget --quiet --output-document="${TORCH_WHEEL_PATH}" "${TORCH_WHEEL_URL}" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -m pip install --no-cache-dir \
      expecttest \
      hypothesis \
      ninja \
      numpy \
      pytest \
      pytest-xdist \
 && python -c "import torch; assert torch.__version__ == '2.7.0+cpu'; print(torch.__version__)"

WORKDIR /workspace
