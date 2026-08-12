FROM op-bench/pytorch-boundary-cpu-source-build:py311 AS legacy_dependencies

FROM python:3.11-slim AS runtime_dependencies

COPY --from=legacy_dependencies /usr/local/lib/python3.11/dist-packages/ /opt/runtime-dependencies/

RUN rm -rf \
      /opt/runtime-dependencies/functorch \
      /opt/runtime-dependencies/torch \
      /opt/runtime-dependencies/torch-*.dist-info \
      /opt/runtime-dependencies/torchgen

FROM python:3.11-slim

COPY --from=runtime_dependencies /opt/runtime-dependencies/ /usr/local/lib/python3.11/site-packages/

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260707%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=2404762b4bd7f4da79f336aa8a55b1be5467a360a06a2b84d7f8d85ed65de1e2
ARG TORCH_WHEEL_PATH=/tmp/torch-2.14.0.dev20260707+cpu-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -c "import urllib.request; request=urllib.request.Request('${TORCH_WHEEL_URL}', headers={'User-Agent': 'op-bench-wheel-build/1'}); open('${TORCH_WHEEL_PATH}', 'wb').write(urllib.request.urlopen(request).read())" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import sys, torch; assert hasattr(sys, 'get_int_max_str_digits'); assert torch.__version__ == '2.14.0.dev20260707+cpu'; assert torch.version.git_version == '91ee0e8a8ae2ab8df31662af070af6ae572ce0df'; print(sys.version); print(torch.__version__, torch.version.git_version)"

WORKDIR /workspace
