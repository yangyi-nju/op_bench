# CPU dependency image and optional offline wheels

`requirements.txt` pins the candidate's direct and transitive Python
dependencies. Environment r2 changes only the Ninja wheel from 1.11.1.1 to
1.11.1.3: the earlier wheel has malformed platform-tag metadata and failed
`pip check` in the r1 image. Both package versions wrap Ninja 1.11.1.
This public dependency image also carries the full-source build
helper and public reproducer. Task source, private graders and patches are
supplied separately.

The declared image uses **CPython 3.12 on Linux x86_64** for the selected build
host. From the OpBench repository root:

```sh
python3 -m pip --isolated download --index-url https://pypi.org/simple \
  --timeout 120 --only-binary=:all: --platform manylinux2014_x86_64 \
  --python-version 312 --implementation cp --abi cp312 \
  -r datasets/v0.8_candidates/layer_norm_cpu/environment/requirements.txt \
  -d .op_bench_cache/v08-wheelhouse/cpu-py312-r2
cp .op_bench_cache/v08-wheelhouse/cpu-py312-r2/*.whl \
  datasets/v0.8_candidates/layer_norm_cpu/environment/wheels/
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 --build-arg PIP_NO_INDEX=1 \
  -t opbench/pytorch-layernorm-source:240aa77-cpu-r2 \
  datasets/v0.8_candidates/layer_norm_cpu/environment
```

The target flags select Linux x86_64 wheels even when downloading on macOS.
Keep this wheelhouse separate from CUDA's Python 3.10 dependencies. A different
CPU architecture needs a separately selected and validated wheelhouse.

BuildKit mounts `wheels/` read-only for the pip step; wheel archives are not copied
into image layers. They are Git-ignored; `.gitkeep` preserves an empty default
directory. The default `PIP_NO_INDEX=0` allows PyPI fallback. Pass `=1` to require
all Python packages to come from the supplied wheels. Apt and base-image
retrieval still need their own network access or cache.

The image checks installed dependencies and records Python/system package
versions under `/opt/opbench-layernorm`. Keep the wheelhouse with the resolved
image identity for repeat builds. See the [candidate README](../README.md) for
the complete source-build contract and the recorded runtime controls.

Task environment r3 continues to use this r2 dependency image and wheelhouse.
Its declared environment adds `USE_NNPACK=0` and `USE_FBGEMM=0` for the complete
CPU source build after observed optional-backend failures. When building the
framework manually, pass those same environment variables; rebuilding this
dependency image alone does not select the task's r3 framework configuration.

The [r1 Ninja installation failure](../validation/dependency_environment_r1_failure.json)
and [r2 complete-build failures](../validation/r2_full_build_failure.json) remain
as historical evidence. The [r3 full build](../validation/r3_full_build.json)
completed successfully using this r2 dependency image and the declared r3 flags;
the record includes the resolved image identity, stage times, stopped execution
and captured workspace. It supplied no candidate patch, private grader or model
channel and used no network or GPU.

The [four CPU controls](../validation/runtime_controls_r3.json) subsequently
matched expectations, and the public baseline reproducer observed the declared
dtype mismatch. Baseline and mutation had 13 and 10 subtest failures respectively
inside one F2P unittest; reference/alternative passed, and all four controls
passed three P2P unittests. All saved evaluations passed offline evidence
verification without cleanup errors. These shared-host correctness records do
not establish performance, model capability or formal task admission.
