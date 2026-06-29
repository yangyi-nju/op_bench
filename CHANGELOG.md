# Changelog

This file records user-visible OpBench version milestones. Detailed design,
implementation, and experiment evidence remain in the versioned documents
under `docs/`.

## v0.4 - In Progress

Development started on 2026-06-21.

Implemented:

- Claude Code agent (`claude_code_action_bridge`) — validated 3/3 on v0.3 tasks.
- Remote GPU Docker executor via SSH (`src/op_bench/remote.py`) with rsync workspace sync and `--gpus all` flag injection.
- Two new runtime tiers: `cuda_python_overlay`, `cuda_kernel_build`.
- `inplace_build` source loading mode for full PyTorch source rebuilds (cuda_kernel_build tier).
- CUDA Docker images: `pytorch-cuda` (overlay) and `pytorch-cuda-devel` (with nvcc + ccache + cmake<4 + CMAKE_POLICY_VERSION_MINIMUM=3.5).
- Patch apply fuzz fallback (`patch -F 3`) for minor base-commit drift.
- `--no-public-tests` ablation flag.
- Preflight script (`scripts/preflight_task.py`) to verify task admission readiness offline: snapshot exists, patches apply, test names resolve.

v0.4 dataset (`datasets/pytorch_v0.4/dataset.json`):

- Verified (from v0.3): 10 tasks
- New CUDA candidates: 5 (132616, 132835, 141820, 143264, 139409) — 2 verified, 3 pending remote build
- Deprecated (incompatible with 2.6.0 stable wheel or non-admissible pattern): 147786, 131858, 133729
- Target task count: 15

Documents:

- `docs/v0.4/design.md`
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
