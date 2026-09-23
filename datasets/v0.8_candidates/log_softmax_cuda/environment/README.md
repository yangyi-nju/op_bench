# CUDA dependency image and optional offline wheels

This image contains public build dependencies for the complete framework.
`requirements.txt` preserves the original recipe's 17 pinned direct dependencies.
Task source, private graders and candidate/reference patches are supplied separately.

The declared Ubuntu 22.04 image uses **CPython 3.10 on Linux x86_64**. From the
OpBench repository root, prefetch compatible binary wheels from official PyPI:

```sh
python3 -m pip --isolated download --index-url https://pypi.org/simple \
  --timeout 120 --only-binary=:all: --platform manylinux2014_x86_64 \
  --python-version 310 --implementation cp --abi cp310 \
  -r datasets/v0.8_candidates/log_softmax_cuda/environment/requirements.txt \
  -d .op_bench_cache/v08-wheelhouse/cuda-py310
cp .op_bench_cache/v08-wheelhouse/cuda-py310/*.whl \
  datasets/v0.8_candidates/log_softmax_cuda/environment/wheels/
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg PIP_NO_INDEX=1 \
  -t opbench/pytorch-cuda-source:cu124-sm70-r1 \
  datasets/v0.8_candidates/log_softmax_cuda/environment
```

The download command can run under the controller's Python 3.12: explicit target
flags select Python 3.10 Linux wheels rather than controller-platform wheels.
Keep this wheelhouse separate from the CPU candidate's Python 3.12 dependencies.

`wheels/` is a read-only BuildKit bind mount during installation; the wheel
archives are not copied into an image layer. Binary wheels are Git-ignored and
`.gitkeep` keeps the empty default build context valid. With the default
`PIP_NO_INDEX=0`, pip may fetch missing dependencies from PyPI. With `=1`, pip
uses only the supplied wheels and fails if any dependency is missing. Apt and
base-image retrieval still need their own network access or cache.

Prefetch includes transitive dependencies. Their originally unpinned versions
are selected at download time; retain the selected wheelhouse for repeat builds.
The image runs `pip check` and records `python-packages.txt` and
`system-packages.txt` under `/opt/opbench-logsoftmax`. Dependency installation
does not establish successful source compilation or task admission.
