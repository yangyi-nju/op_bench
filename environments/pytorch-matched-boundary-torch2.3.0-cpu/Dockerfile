FROM python:3.11-slim AS wheel

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cpu/torch-2.3.0%2Bcpu-cp311-cp311-linux_x86_64.whl
ARG TORCH_WHEEL_SHA256=97a38b25ee0e3d020691e7846efbca62a3d8a57645c027dcb5ba0adfec36fe55

RUN python -m pip download --no-deps "$TORCH_WHEEL_URL" -d /tmp/wheels \
 && echo "$TORCH_WHEEL_SHA256  /tmp/wheels/$(basename "$TORCH_WHEEL_URL" | sed 's/%2B/+/')" | sha256sum -c -

FROM op-bench/pytorch-cpu-compile:torch2.6.0-py311

COPY --from=wheel /tmp/wheels /tmp/wheels

RUN python -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/wheels/*.whl \
 && python -m pip install --no-cache-dir --force-reinstall "triton==2.3.0" \
 && rm -rf /tmp/wheels \
 && python -c "import torch; assert torch.__version__ == '2.3.0+cpu'; print(torch.__version__)"

WORKDIR /workspace
