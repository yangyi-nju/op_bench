FROM python:3.11-slim AS wheel

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp311-cp311-linux_x86_64.whl
ARG TORCH_WHEEL_SHA256=6b2966ede9affe2fd69e0765691ca723ec870e0c34c7761f4d5b8e318383fdaf

RUN python -m pip download --no-deps "$TORCH_WHEEL_URL" -d /tmp/wheels \
 && echo "$TORCH_WHEEL_SHA256  /tmp/wheels/$(basename "$TORCH_WHEEL_URL" | sed 's/%2B/+/')" | sha256sum -c -

FROM op-bench/pytorch-cuda:torch2.6.0-cu124-py311

COPY --from=wheel /tmp/wheels /tmp/wheels

RUN python -m pip install --no-cache-dir --no-deps --force-reinstall /tmp/wheels/*.whl \
 && python -m pip install --no-cache-dir --force-reinstall "triton==3.1.0" \
 && rm -rf /tmp/wheels \
 && python -c "import torch; assert torch.__version__ == '2.5.1+cu124'; assert torch.version.cuda == '12.4'; print(torch.__version__)"

WORKDIR /workspace
