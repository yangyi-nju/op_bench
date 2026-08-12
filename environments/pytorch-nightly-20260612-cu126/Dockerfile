FROM op-bench/pytorch-nightly-cu126:2.14.0.dev20260710-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cu126/torch-2.14.0.dev20260612%2Bcu126-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=24378ffefa5746d2edb98d39d815190e68afa4d2fe086fb873da5e43acc37192
ARG TORCH_WHEEL_PATH=/tmp/torch-2.14.0.dev20260612+cu126-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -c "import urllib.request; request=urllib.request.Request('${TORCH_WHEEL_URL}', headers={'User-Agent': 'op-bench-wheel-build/1'}); open('${TORCH_WHEEL_PATH}', 'wb').write(urllib.request.urlopen(request).read())" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import sys, torch, triton; assert hasattr(sys, 'get_int_max_str_digits'); assert torch.__version__ == '2.14.0.dev20260612+cu126'; assert torch.version.git_version == 'b858202e04fee5be87860df073caef017babd896'; assert torch.version.cuda == '12.6'; print(sys.version); print(torch.__version__, torch.version.git_version, torch.version.cuda, triton.__version__)"

WORKDIR /workspace
