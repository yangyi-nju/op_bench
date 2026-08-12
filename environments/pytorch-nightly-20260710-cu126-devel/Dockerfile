FROM op-bench/pytorch-nightly-cu126:2.12.0.dev20260407-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cu126/torch-2.14.0.dev20260710%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=12ff1b41862cd2573864e1e15d720697fee49726c13a5f1e21c218baf6849fb6
ARG TORCH_WHEEL_PATH=/tmp/torch-2.14.0.dev20260710+cu126-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -c "import urllib.request; request=urllib.request.Request('${TORCH_WHEEL_URL}', headers={'User-Agent': 'op-bench-wheel-build/1'}); open('${TORCH_WHEEL_PATH}', 'wb').write(urllib.request.urlopen(request).read())" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import shutil, sys, torch, triton; assert hasattr(sys, 'get_int_max_str_digits'); assert torch.__version__ == '2.14.0.dev20260710+cu126'; assert torch.version.git_version == '0ce05af6274d975672dd30447bf00010317288ef'; assert torch.version.cuda == '12.6'; assert shutil.which('nvcc'); assert shutil.which('g++'); print(sys.version); print(torch.__version__, torch.version.git_version, torch.version.cuda, triton.__version__)"

WORKDIR /workspace
