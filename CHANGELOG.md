# Changelog

This file records user-visible OpBench version milestones. Detailed design,
implementation, and experiment evidence remain in the versioned documents
under `docs/`.

## Unreleased

Planned direction as of 2026-07-17:

- v0.6 upgrades the existing real-Codex benchmark demo into a standardized Agent evaluation platform with versioned contracts, canonical CLI/MCP actions, explicit AttemptSession lifecycle and budgets, trajectory artifacts, patch freeze, fresh evaluation, failure attribution, replay, and rebuildable summaries.
- v0.6 M1 is complete: strict versioned runtime contracts, canonical JSON/SHA-256, deterministic RunManifest/Cohort/Attempt identities, a frozen expected matrix, v0.5 compatibility projection, independent JSON Schema validation, and offline manifest CLIs are implemented. This is platform evidence, not a new benchmark score.
- v0.6 M2 is complete: AgentTaskView is an explicit scanned public projection and a Manifest/Attempt identity axis; Authoritative Workspace applies regular-file/path/scope/mode/size/symlink/binary policy; Freeze converges concurrent mutations into one canonical add/modify/delete/empty patch, verifies strict clean-base application, and binds one patch hash across Session, artifact metadata, and EvaluationSpec. Existing v0.5 Action Bridge patch export remains compatible.
- v0.6 is one platform release with M1–M7 internal milestones. It does not expand the dataset or run the formal multi-Agent study.
- v0.7 builds the reusable Dataset Factory, adds the Boundary Slice, and revisits the two matched-runtime Precision candidates.
- v0.8 adds Device/API Compatibility and freezes Evaluation/Scoring Specification RC; v0.9 runs the formal multi-Agent and feedback-ablation cohorts.

Planning documents:

- `docs/project_plan.md`
- `docs/project_state.md`
- `docs/v0.6/design.md`
- `docs/v0.6/implementation_plan.md`
- `docs/v0.6/acceptance_matrix.md`
- `docs/v0.7/design.md`

## v0.5 - Completed

Development and the full experiment completed on 2026-07-11. v0.5 establishes
the first problem-dimension slice (numerical precision) and freezes a verified
17-task cumulative dataset. Boundary and compatibility are separate future
versions, not blockers for this release.

Implemented:

- Added the precision taxonomy (`problem_dimension=precision`, subclasses P1-P5) and ghstack-aware PyTorch candidate screening workflow.
- Added extended experiment reporting: resolved rate, patch conciseness, pass-to-pass kept rate, strict resolved rate, regression rate, tier-weighted score, per-problem breakdown, and median runtime.
- Added `datasets/pytorch_v0.5_precision/dataset.json` as a reusable 6-task precision slice; P4 remains an explicit N/A coverage gap.
- Added persistent remote ccache reuse, replay-spec evidence hashes, per-source-load build environments, and one source load per evaluation phase.
- Added the `pytorch-cpu-compile` environment for CPU Inductor/`torch.compile` tasks.
- Unified all official task replay on Linux `remote_docker`; re-admitted the 10 inherited CPU tasks under that policy.
- Optimized CUDA kernel builds with `BUILD_TEST=0` and `TORCH_CUDA_ARCH_LIST=7.0`; the warm incremental compile dropped from tens of minutes to roughly 3 minutes.
- Made resume task-content-aware, excluded explicit environment failures from completed attempt keys, retained append-only audit rows, and deduplicated retries for scoring.
- Added bounded rsync retry (including mutable Git pack exit 23), incremental summaries, strict F2P/P2P status classification, and aggregate completeness enforcement.

v0.5 cumulative dataset (`datasets/pytorch_v0.5/dataset.json`): **17 verified tasks** (13 inherited from v0.4 plus #140557, #139999, #129138, and #139372). Deprecated #129154 and #144073 are excluded.

Full experiment result (Codex CLI 0.144.0-alpha.4, 17 tasks x 3 repeats):

- **37/51 = 72.5% resolved**; tier-weighted score 76.8%.
- Patch conciseness 1.000; pass-to-pass kept rate 94.1%; strict regression rate 0%.
- CPU overlay 27/39, CUDA overlay 4/6, CUDA kernel build 6/6.
- Precision slice: **13/18 = 72.2%**; P1 3/3, P2 0/3, P3 4/6, P4 N/A, P5 6/6.
- Aggregate integrity: 17/17 baselines and 51/51 logical attempts, with zero logical environment transient.

Deferred to later versions:

- Admit a real P4 numerical-instability task; P4 remains N/A rather than being filled with a non-matching task.
- v0.6 standardizes the Agent evaluation platform; v0.7 adds boundary tasks; v0.8 adds device/API compatibility tasks.
- #129154 and #144073 require a matched wheel or source-build environment and remain v0.7 backlog candidates.

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
