# Changelog

This file records user-visible OpBench version milestones. Detailed design,
implementation, and experiment evidence remain in the versioned documents
under `docs/`.

## v0.5 Precision Phase - Completed

Precision-phase development completed on 2026-07-11. The v0.5 release remains
open for additional problem dimensions; the cumulative manifest is intentionally
marked `draft` until those dimensions are admitted and the final full run is done.

Implemented:

- Added the precision taxonomy (`problem_dimension=precision`, subclasses P1-P5) and ghstack-aware PyTorch candidate screening workflow.
- Added extended experiment reporting: resolved rate, patch conciseness, pass-to-pass kept rate, strict resolved rate, regression rate, tier-weighted score, per-problem breakdown, and median runtime.
- Added persistent remote ccache reuse, replay-spec evidence hashes, per-source-load build environments, and one source load per evaluation phase.
- Added the `pytorch-cpu-compile` environment for CPU Inductor/`torch.compile` tasks.
- Corrected the basic CPU image ID declaration and re-admitted affected precision tasks.
- Optimized CUDA kernel builds with `BUILD_TEST=0` and `TORCH_CUDA_ARCH_LIST=7.0`; the warm incremental compile dropped from tens of minutes to roughly 3 minutes.

v0.5 cumulative dataset (`datasets/pytorch_v0.5/dataset.json`): **17 verified tasks** in a draft manifest (13 inherited from v0.4 plus #140557, #139999, #129138, and #139372). Deprecated #129154 and #144073 are excluded.

Precision-phase experiment result (Codex CLI 0.144.0-alpha.4, 6 tasks x 3 repeats):

- **13/18 = 72.2% resolved**; tier-weighted score 78.8%.
- Patch conciseness 1.000; pass-to-pass kept rate 83.3%; strict regression rate 0%.
- P1 0/3, P2 3/3, P3 4/6, P4 N/A, P5 6/6.
- CPU overlay 6/9, CUDA overlay 1/3, CUDA kernel build 6/6.

Deferred before final v0.5 completion:

- Add boundary and compatibility dimensions, then run the frozen cumulative dataset end to end.
- Admit a real P4 numerical-instability task; P4 remains N/A rather than being filled with a non-matching task.
- #129154 and #144073 require a matched wheel or source-build environment and remain candidates for a later dataset revision.
- Align the evaluator's broad `pass_to_pass_regressed` status label with the strict regression metric and add bounded retry for transient remote rsync failures.

Documents:

- `docs/v0.5/design.md`
- `docs/v0.5/candidate_search.md`
- `docs/v0.5/setup_remote_agent.md`
- `docs/v0.5/experiment_report.md`

## v0.4 - Completed

Development started on 2026-06-21. Experiment completed on 2026-07-08.

Implemented:

- Remote GPU Docker executor via SSH (`src/op_bench/remote.py`) with rsync workspace sync (excludes `.ccache/`, `build/`, `torch.egg-info/`, `__pycache__/`), `--gpus all` flag injection, `ServerAliveInterval=30` keep-alive, and `_kill_remote_container_processes` fallback on subprocess timeout.
- Two new runtime tiers: `cuda_python_overlay`, `cuda_kernel_build`.
- `inplace_build` source loading mode for full PyTorch source rebuilds (cuda_kernel_build tier); default build command streams `setup.py develop` output to `.op_bench_build.log` and preserves progress on timeout.
- `kernel_full` snapshot mode (`scripts/setup_sources.py`) with recursive submodule init from `.gitmodules`.
- CUDA Docker images: `pytorch-cuda` (overlay, torch 2.6.0 wheel) and `pytorch-cuda-devel` (with nvcc + ccache via `/usr/lib/ccache` symlinks + `cmake<4` + `CMAKE_POLICY_VERSION_MINIMUM=3.5` + `setuptools>=70,<81` + `USE_NCCL=0 USE_DISTRIBUTED=0 USE_TENSORPIPE=0 USE_GLOO=0 USE_MPI=0 USE_KINETO=0`).
- Codex CLI rate-limit auto-retry (`_run_codex`): detects 429 / "rate limit" / "quota exceeded" signatures, sleeps `OP_BENCH_CODEX_RATE_LIMIT_WAIT_SEC` (default 5h5min), retries up to `OP_BENCH_CODEX_RATE_LIMIT_MAX_RETRIES` (default 3).
- `TaskManifest.build_timeout_sec` property (defaults to 6h for `cuda_kernel_build` / `inplace_build`).
- Empty `hidden_test.patch` support (skip apply when PR fixes a pre-existing test).
- Patch apply fuzz fallback (`patch -F 3`) for minor base-commit drift.
- `--no-public-tests` ablation flag (mechanism present, not exercised in v0.4).
- Preflight script (`scripts/preflight_task.py`) to verify task admission readiness offline: snapshot exists, patches apply, test names resolve.

v0.4 dataset (`datasets/pytorch_v0.4/dataset.json`): **13 tasks verified** (10 from v0.3 + 2 cuda_python_overlay: 132616, 132835 + 1 cuda_kernel_build: 144009).

v0.4 experiment result (Codex CLI, 3-repeat):

- **33/39 = 84.6% resolved** (v0.3 was 76.7%).
- Batch A (CPU, 10 tasks × 3): 24/30, median 45.9s. 2 stable failures (162340, 163961) carried over from v0.3.
- Batch B (GPU, 3 tasks × 3, remote_docker on 4× V100): 9/9, median 82.2s.
- `cuda_kernel_build` (144009): 3/3, median ~91min per attempt (build-heavy).

Deferred to v0.5:

- Multi-agent comparison with Claude Code (blocked on external conditions; agent adapter design retained).
- Public test ablation (no task ships `public_test.patch` yet; mechanism kept, content deferred).

Documents:

- `docs/v0.4/design.md`
- `docs/v0.4/experiment_report.md`
- `docs/v0.4/candidate_tasks_cuda.md`
- `docs/v0.4/public_test_ablation.md`

## v0.3 - Completed

Development started on 2026-06-05.

Implemented milestones:

- Expanded PyTorch verified dataset from 3 to 10 tasks across 5 subsystems.
- Added patch scope validation with `enforced` mode (`src/op_bench/patch_scope.py`).
- Added public/hidden test separation (`hidden_test_patch` + `public_test_patch`).
- Added multi-file Python overlay support (verified with conv.py + utils.py task).
- Added `--filter-tasks` for incremental experiment runs on task subsets.
- Added batch admission runner (`scripts/run_admission_batch.py`).
- Upgraded agent prompt to communicate patch scope and public test visibility.
- Upgraded evaluator to check patch scope before scoring.
- Ran 3-repeat Codex CLI evaluation on all 10 tasks: 76.7% resolved (23/30).

Documents:

- `docs/v0.3/design.md`
- `docs/v0.3/experiment_report.md`
- `docs/v0.3/candidate_tasks.md`

## v0.2 - Completed

Development started on 2026-06-04.

Planned scope:

- Expand the verified PyTorch operator dataset to 3-5 tasks.
- Add reusable environment and source snapshot registries.
- Add a formal task admission pipeline with stable replay evidence.
- Add evidence-aware dataset validation and environment lifecycle management.
- Continue using the v0.1 Codex action-bridge path for isolated agent scoring.

Implemented milestones:

- Added committed environment and source snapshot registries with typed loaders.
- Added backward-compatible task references and v0.2 runtime/admission fields.
- Added `run_admission.py` and stable task-local admission evidence.
- Added evidence-aware dataset validation with registry reference checks.
- Added registry-driven task resolution for dataset, admission, replay, source, and environment CLIs.
- Added asset cache inspection and OpBench-managed Docker container lifecycle utilities.
- Added dataset curation utilities for verified-only slices and summaries.
- Migrated `pytorch__149693__lazylinear_init` and re-verified baseline/gold replay.
- Promoted `pytorch__160952__bilinear_lazy_check` to verified after fixing its hidden test replay.
- Added and verified `pytorch__147599__lazylinear_state_forward` from PyTorch PR #147599.
- Updated `datasets/pytorch_mini` to a 3-task verified PyTorch CPU operator slice.
- Ran the `gold` agent loop on the 3-task slice; all three tasks resolved.
- Ran a real Codex CLI `codex_action_bridge` experiment on the 3-task verified slice; all three tasks resolved with isolated final scoring.

Documents:

- `docs/v0.2/design.md`
- `docs/v0.2/developer_guide.md`
- `docs/v0.2/experiment_report.md`
- `docs/v0.2/implementation_plan.md`

## v0.1 - Completed

OpBench v0.1 established the minimum isolated benchmark loop:

- Built task bundles from real PyTorch issues and PRs.
- Replayed fail-to-pass and pass-to-pass tests in task-specific Docker environments.
- Managed local full-repository source snapshots for reproducible workspaces.
- Ran a real Codex CLI agent through the OpBench action interface.
- Scored agent patches in fresh isolated workspaces and recorded experiment evidence.

Documents:

- `docs/v0.1/product_requirements.md`
- `docs/v0.1/experiment_report.md`
- `docs/v0.1/developer_guide.md`
- `docs/v0.1/manual_validation.md`
