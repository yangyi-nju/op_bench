FROM op-bench/pytorch-nightly-cpu:2.14.0.dev20260710-py311

ARG TORCH_WHEEL_URL=https://download-r2.pytorch.org/whl/nightly/cpu/torch-2.14.0.dev20260612%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl
ARG TORCH_WHEEL_SHA256=fcb88e9690c95f7a813afa2d47359c3f805938b88781de8d84d1b5c5cc3514e9
ARG TORCH_WHEEL_PATH=/tmp/torch-2.14.0.dev20260612+cpu-cp311-cp311-manylinux_2_28_x86_64.whl

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN python -c "import urllib.request; request=urllib.request.Request('${TORCH_WHEEL_URL}', headers={'User-Agent': 'op-bench-wheel-build/1'}); open('${TORCH_WHEEL_PATH}', 'wb').write(urllib.request.urlopen(request).read())" \
 && echo "${TORCH_WHEEL_SHA256}  ${TORCH_WHEEL_PATH}" | sha256sum --check --strict \
 && python -m pip install --no-cache-dir --no-deps --force-reinstall "${TORCH_WHEEL_PATH}" \
 && rm -f "${TORCH_WHEEL_PATH}" \
 && python -c "import sys, torch; assert hasattr(sys, 'get_int_max_str_digits'); assert torch.__version__ == '2.14.0.dev20260612+cpu'; assert torch.version.git_version == 'b858202e04fee5be87860df073caef017babd896'; print(sys.version); print(torch.__version__, torch.version.git_version)"

WORKDIR /workspace
