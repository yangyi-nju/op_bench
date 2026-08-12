FROM op-bench/pytorch-cuda:torch2.6.0-cu124-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cu126/torch-2.12.0.dev20260407%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=676cf93c822752750a4dd78bbaa5b1da96ea4334e60007704f327e7b5c0156b0
ARG TORCH_WHEEL_PATH=/tmp/torch-2.12.0.dev20260407+cu126-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -m pip download --no-cache-dir --no-deps \
      --dest /tmp \
      "${TORCH_WHEEL_URL}" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --force-reinstall \
      --index-url https://download.pytorch.org/whl/nightly/cu126 \
      "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import torch; assert torch.__version__ == '2.12.0.dev20260407+cu126'; assert torch.version.git_version == '6a13e444ee88996ff01cd2bab41d7f2857291646'; assert torch.version.cuda == '12.6'; print(torch.__version__, torch.version.git_version, torch.version.cuda)"

WORKDIR /workspace
