# CPU LayerNorm candidate: PyTorch #140557

**Candidate.2, not formally admitted.** The complete configured CPU framework
and all four controls completed under environment r3 on 2026-09-09, with the
expected behavioral outcomes. This `operator_integration` task concerns CPU
metadata and regressions. [Runtime evidence](validation/runtime_controls_r3.json)
records the controls and public baseline reproduction; no GPU, model or
performance evaluation was performed in these controls.

The exact base is `240aa77ad01c4f0cd9b2417748272f2f617c112f`, the parent of the
landed upstream fix. The original issue concerns CPU bfloat16
`aten.native_layer_norm.default` metadata under FakeTensorMode with the Python
dispatcher: the decomposition's device-dependent behavior was lost in the meta
registration path. [Upstream PR](https://github.com/pytorch/pytorch/pull/140557),
[landed change and parent](https://github.com/pytorch/pytorch/commit/9ae19ff).

## Contract and oracle

The public task covers nonempty, concrete CPU shapes; homogeneous bfloat16
input/optional affine tensors; one or several normalized dimensions; all four
weight/bias-presence combinations; and noncontiguous input. Eager and fake
output, mean and reciprocal standard deviation must agree on shape and dtype;
all three are bfloat16. Statistic shapes retain leading input dimensions and
replace normalized dimensions with singletons. The public reproducer contains
the original full-affine case; the task statement explicitly requires the other
cases as well.

Regression checks preserve float32 fake metadata, eager numerical results in
bfloat16/float32/float64, and first-order float64 input/weight/bias gradients.
An explicit population-variance formula supplies the numerical oracle. The
statement publishes the bounded input domain and dtype-dependent tolerances.
Tests run one F2P unittest and three P2P unittests with behavioral subcases.
These execution counts describe the declared suite, not hard-coded success
output. No source text, helper name or reference-patch file is an acceptance rule.

Mixed bfloat16-input/float32-affine operation is deliberately outside this narrow
contract: eager CPU mixed-type statistics can be float32. CUDA, empty/symbolic
shapes, the `out` overload, higher-order gradients and performance are also out
of scope. This candidate must not be advertised as covering all LayerNorm
semantics. [Base CPU implementation](https://github.com/pytorch/pytorch/blob/240aa77ad01c4f0cd9b2417748272f2f617c112f/aten/src/ATen/native/layer_norm.cpp).

## Full source and environment

The source recipe needs the exact base tree plus **all recursively pinned
submodules**. Evaluation must export those contents without upstream Git
history. The baseline has 37 direct gitlinks; upstream build preparation also
checks dependencies that may not be exercised by this CPU operator. Do not
substitute a sparse checkout, a wheel, or an extracted Python function.

The dependency-only image is `opbench/pytorch-layernorm-source:240aa77-cpu-r2`;
environment identity is `pytorch-layernorm-cpu-py312` /
`240aa77-cpu-gcc12-openblas-r3`. It uses Python 3.12.7 on Debian 12, GCC/G++ 12,
OpenBLAS, CMake 3.30.5, Ninja 1.11.1.3, and the explicit Python versions in
`environment/requirements.txt`. Setuptools 72.1.0 and SymPy 1.13.1 satisfy the
base's stated constraints. The image records installed Python and system
packages in `/opt/opbench-layernorm`; record its resolved image identity with
runtime evidence. Apt repositories and image tags alone are not an immutable
dependency lock. [Base requirements](https://github.com/pytorch/pytorch/blob/240aa77ad01c4f0cd9b2417748272f2f617c112f/requirements.txt),
[base build instructions](https://github.com/pytorch/pytorch/blob/240aa77ad01c4f0cd9b2417748272f2f617c112f/README.md#from-source).

Environment r1 failed during dependency installation: the Ninja 1.11.1.1 wheel
placed a blank line before its platform `Tag` headers, and `pip check` rejected
the installed package as an unsupported platform. Environment r2 selects Ninja
1.11.1.3, a packaging revision of the same Ninja 1.11.1 release. Task/scoring
revision `candidate.2` and the other dependency versions are unchanged. The
[r1 failure record](validation/dependency_environment_r1_failure.json) is retained;
an r2 wheel check alone does not imply a successful image or complete source build.

The deployment controller subsequently built the r2 dependency image, but the
complete CPU framework build failed in two optional backends: NNPACK attempted
an online `six-1.11.0.tar.gz` download, and FBGEMM triggered GCC 12 AVX512
`-Werror=maybe-uninitialized` errors. The [r2 full-build failure record](validation/r2_full_build_failure.json)
is retained. Environment r3 sets `USE_NNPACK=0` and `USE_FBGEMM=0` in the task's
explicit build environment. These backends are not used by the declared CPU
`native_layer_norm` and FakeTensor contract. The r2 dependency image, all other
environment settings, complete source-build command and grader remain the same.
The build stays offline and still compiles the complete configured framework;
it does not suppress compiler errors, extract an operator or overlay a wheel.
The [r3 full-build record](validation/r3_full_build.json) now records a ready
baseline: preparation and the complete configured build exited successfully,
the network remained disabled, execution stopped and the workspace was captured.
No candidate patch, private grader or model channel was supplied to that build.

The chosen execution envelope is **16 CPUs, 64 GiB RAM, 2,048 PIDs, MAX_JOBS=16,
and a four-hour build timeout**. No GPU, CUDA toolkit, NCCL or Triton is needed.
This envelope supported the recorded build; it is not a measured minimum.
The build record separates command duration from total preparation/capture time;
peak memory and disk requirements still need measurement.
Allow substantial local disk space for the complete recursive source, native
object files and independent control workspaces. Distributed/CUDA/ROCm/XPU,
NNPACK/FBGEMM and upstream test-target builds are disabled; the CPU operator
implementation under test remains enabled.

After each candidate patch, `build_source.py` invokes the real upstream
`setup.py build --build-lib /workspace/build/opbench-package`. It recreates
packaged Python output so stale copies cannot hide edits, while preserving
ordinary CMake/Ninja dependency rebuilding. It does not overlay selected files.
`source_torch.py` then verifies the loaded Torch Python modules, `_C` extension
and mapped `libtorch_cpu.so`, `libtorch_python.so`, and `libc10.so` all come from
that rebuilt package. The same public loader is used by the reproducer and
private grader. Any normal source path can be edited.

These loading checks detect accidental stale/wheel artifacts; they are not a
security proof against hostile code in the same interpreter as a Python grader.
Control execution and the benchmark's information/isolation policy remain
separate requirements.

## Execution

From the OpBench repository root, prepare the image and complete pinned source:

```sh
docker build -t opbench/pytorch-layernorm-source:240aa77-cpu-r2 \
  datasets/v0.8_candidates/layer_norm_cpu/environment
opbench prepare --fetch \
  --task datasets/v0.8_candidates/layer_norm_cpu/task.json \
  --output /tmp/opbench-layernorm-prepared
opbench check-task \
  --task datasets/v0.8_candidates/layer_norm_cpu/task.json \
  --reference datasets/v0.8_candidates/layer_norm_cpu/controls/reference.patch \
  --alternative datasets/v0.8_candidates/layer_norm_cpu/controls/alternative.patch \
  --mutation datasets/v0.8_candidates/layer_norm_cpu/controls/mutation.patch \
  --output /tmp/opbench-layernorm-controls
```

Inside a source workspace using this image and the task's declared environment
(including r3's two optional-backend flags), the public commands are:

```sh
python3 -I /opt/opbench-layernorm/build_source.py /workspace
python3 -I /opt/opbench-layernorm/reproduce.py /workspace
```

The reference is the upstream five-line fake implementation registration. The
alternative places the registration in `torch/_subclasses/fake_impls.py` and
normalizes schema arguments before invoking the existing mathematical
decomposition. It demonstrates a repair outside the old Gold file restriction;
it shares the decomposition and is not a second independent numerical oracle.
The mutation has a plausible incomplete fix: full-affine metadata is correct,
but missing weight or bias incorrectly promotes statistics to float32. Actual
execution reached those behavioral assertions; the failures were not syntax,
import or build errors.

The [r3 control matrix](validation/runtime_controls_r3.json) is:

| Control | F2P: one unittest with subtests | P2P: three unittests |
| --- | --- | --- |
| baseline | Failed, 13 subtest failures | All passed |
| reference | Passed | All passed |
| alternative | Passed | All passed |
| mutation | Failed, 10 subtest failures | All passed |

Each control has one F2P protocol case and one P2P protocol case; the latter
runs three unittest methods. Thirteen or ten failed subtests remain failures
inside one unittest, not independent tasks or extra scoring opportunities.
All four saved evaluations pass offline evidence verification, with no recorded
errors, skips or cleanup errors. Offline verification checks saved evidence;
it does not rerun the framework assertions.

The public baseline reproducer also ran inside the declared image and observed
the bfloat16 eager versus float32 fake statistic mismatch. The runtime record
links that reproduction and the original assertion/loading logs. These are
correctness controls on a shared host, without GPU or model calls, and provide
no performance result.

`validation/patch_applicability_candidate2.json` records only successful patch
application checks and parsing of the entire modified modules. Older validation
files describe the previous draft and unprepared environment; they do not
validate this revision. See [migration review](migration_review.md) for the
evidence boundary. The completed controls do not replace independent contract
and release/provenance review; the task remains development-exposed and
unadmitted. Upstream source and patches retain their original attribution
and licensing; this candidate does not decide OpBench's own code license.
