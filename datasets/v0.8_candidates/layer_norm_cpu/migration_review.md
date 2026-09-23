# Review of CPU LayerNorm candidate.2

Status: **candidate; full-source CPU controls match expectations; not admitted**.
The current contract, environment recipe and run commands are in [README](README.md).

## What changed after the first migration draft

The task is now schema 2 with explicit task, scoring, defect-group and environment
identities. Scope is `operator_integration`, tied to the narrow upstream CPU
bfloat16 FakeTensor metadata problem. The public statement includes optional
affine arguments, normalized dimensions/layout, numerical domains/tolerances and
eager regression requirements. It explicitly excludes mixed input/affine dtypes:
CPU mixed-type statistics need not have the input dtype. The parent of the
[landed upstream fix](https://github.com/pytorch/pytorch/commit/9ae19ff) is
`240aa77ad01c4f0cd9b2417748272f2f617c112f`.

The old issue's fix instructions and `torch/_refs/__init__.py` restriction are
absent. A complete source build is required after candidate application. A
public image helper recreates the packaged Python tree through upstream
`setup.py build`, and checks the actual loaded native extension and CPU shared
libraries. The grader uses that rebuilt package; it does not extract functions
or install a one-file overlay on a prebuilt Torch wheel.

The prior grader checked only dtype/shape plus float64 eager output and input
gradients. That could miss damaged bfloat16 eager values or affine gradients.
The revised independent formula checks bfloat16/float32/float64 values and
float64 input/weight/bias gradients. The retained alternative uses another
production source path; it shares the upstream decomposition, so its structural
independence is limited and is not claimed as an independent oracle. The revised
mutation correctly handles the original full-affine example but incorrectly
promotes statistics when an affine argument is absent. Its expected failure is
a semantic assertion, not an invalid patch or build failure.

## Evidence actually available

`validation/patch_applicability_candidate2.json` records `git apply --check`
against complete production files from the exact base and parsing of each
entire modified module. This is static evidence only. The task schema and helper
syntax are checked separately; neither proves an executable operator contract.
The first dependency image attempts failed before any complete CPU source build:
`pip check` rejected Ninja 1.11.1.1 as an unsupported platform. Local inspection
of the downloaded wheel confirms a blank line before its `Tag` headers; a wheel
metadata parser consequently sees no platform tags. The observation is retained
in `validation/dependency_environment_r1_failure.json`. The earlier successful
offline dependency resolution did not inspect installed wheel metadata and must
not be mistaken for successful image validation.

Environment r2 replaces only that Ninja packaging version with 1.11.1.3. The
task/scoring revision remains `candidate.2`; the old r1 review record is preserved
as historical evidence. R2 offline dependency and wheel metadata checks are
preparation evidence. The deployment controller later built the r2 dependency
image successfully; its complete CPU source build then failed in optional
NNPACK and FBGEMM backends. NNPACK's CMake preparation attempted to download
`six-1.11.0.tar.gz` in the offline build session. FBGEMM compilation hit GCC 12
AVX512 `maybe-uninitialized` warnings promoted to errors. These are environment
preparation failures, not behavioral failures of the LayerNorm task. They are
preserved in `validation/r2_full_build_failure.json`; no operator controls are
claimed from that incomplete build.

Environment r3 explicitly sets `USE_NNPACK=0` and `USE_FBGEMM=0`. The declared
CPU native LayerNorm/FakeTensor API does not depend on either optional backend.
All other environment settings, the r2 dependency image, full-source build
recipe and private grader stay unchanged. No compiler errors are hidden and
build networking remains disabled. The full configured framework must still
build from all selected source files and pinned submodules; only these two
optional backends are outside the configuration. The [r3 baseline](validation/r3_full_build.json)
completed the full configured build, stopped execution and captured a ready
workspace without network, candidate patch, private grader or model channel.
The task/scoring revision remains `candidate.2`. This bundle has no authoring
generator to synchronize.

The subsequent [four-control run](validation/runtime_controls_r3.json) confirms
the intended behavior. Baseline produced 13 subtest failures in its one F2P
unittest; reference and alternative passed all tests; mutation produced 10
subtest failures in that same F2P unittest. All controls passed the three P2P
unittests. These counts do not create 13 or 10 separate tasks: the task remains
one logical problem, with two protocol cases running one and three unittest
methods respectively. Errors and skips were zero. All four saved results passed
offline evidence verification and recorded no cleanup errors; this verification
does not execute the framework tests again.

The public baseline reproducer ran in the declared image and observed the eager
bfloat16 versus fake float32 statistic mismatch. Runtime records retain the
assertion/loading evidence and the resolved image identity. This control phase
used CPU execution on a shared host, with no GPU or model calls and no
performance measurement.

The older `validation/inspection.json`, `patch_applicability.json`,
`existing_cache_preparation_check.json` and `unprepared_baseline/` are historical
observations about the previous draft. In particular, the old cache directory
was named after the base but had a different actual HEAD, sparse checkout, and
37 uninitialized direct submodules. The old runtime image was an arm64 Torch
wheel without the native build toolchain. Its recorded environment error is
not a baseline bug reproduction, and those old patch checks do not validate
candidate.2 controls.

Read-only comparison with the available CUDA candidate's `2409b49` Git cache
found 31 matching direct gitlinks with usable commit objects; the six different
CPU requirements were not present there. New source preparation can reuse the
31 matching objects, then obtain the six exact commits and recursively verify
their submodules. This does not justify reusing the CUDA revision's source tree
as the CPU baseline. The original cache was not reset or modified by this review.

## Remaining release gates

The complete matrix and public baseline reproduction are now recorded. Review
their coverage and alternatives independently; successful execution is not
formal admission or proof of sufficient coverage outside the declared domain.
Measure peak build resource requirements: the recorded 16 CPU / 64 GiB envelope
worked for this build, but is not a measured minimum or a performance benchmark.

Then obtain an independent prompt/oracle review and record upstream provenance,
release exposure and rights. The Python grader executes candidate code in its
process: import-location checks catch accidental artifacts, not all malicious
same-process interference. That residual trust property belongs in the runtime
admission assessment. No claim of general LayerNorm coverage, agent performance
or formal task admission follows from these controls.
