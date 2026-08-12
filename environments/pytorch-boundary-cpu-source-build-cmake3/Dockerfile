FROM op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311 AS cmake_tools

FROM op-bench/pytorch-boundary-cpu-source-build:py311

COPY --from=cmake_tools /usr/local/lib/python3.11/dist-packages/cmake/ /usr/local/lib/python3.11/dist-packages/cmake/

RUN ln -sf /usr/local/lib/python3.11/dist-packages/cmake/data/bin/cmake /usr/local/bin/cmake \
 && ln -sf /usr/local/lib/python3.11/dist-packages/cmake/data/bin/cpack /usr/local/bin/cpack \
 && ln -sf /usr/local/lib/python3.11/dist-packages/cmake/data/bin/ctest /usr/local/bin/ctest \
 && python -c "import re, subprocess; output=subprocess.check_output(['cmake', '--version'], text=True); version=tuple(map(int, re.search(r'cmake version ([0-9]+)\\.([0-9]+)\\.([0-9]+)', output).groups())); assert version >= (3, 27), version; print(output.splitlines()[0])"

WORKDIR /workspace
