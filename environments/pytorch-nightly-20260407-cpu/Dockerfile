FROM op-bench/pytorch-boundary-cpu-source-build:py311

ARG TORCH_WHEEL_URL=https://download.pytorch.org/whl/nightly/cpu/torch-2.12.0.dev20260407%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=b20f1e71842d9f1a7c21a0525058d1d2f56efe8e6c32b4978113bb606b4cf385
ARG TORCH_WHEEL_PATH=/tmp/torch-2.12.0.dev20260407+cpu-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN wget --quiet --output-document="${TORCH_WHEEL_PATH}" "${TORCH_WHEEL_URL}" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import torch; assert torch.__version__ == '2.12.0.dev20260407+cpu'; assert torch.version.git_version == '6a13e444ee88996ff01cd2bab41d7f2857291646'; print(torch.__version__, torch.version.git_version)"

WORKDIR /workspace
