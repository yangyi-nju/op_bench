# Changelog

This file records user-visible OpBench version milestones. Detailed design,
implementation, and experiment evidence remain in the versioned documents
under `docs/`.

## v0.3 - Planned

Planning started on 2026-06-05.

Planned scope:

- Expand the PyTorch verified dataset to 10 tasks.
- Add public/hidden test separation.
- Add multi-file Python overlay support.
- Add patch scope validation.
- Standardize CPU Docker tooling for real agents.
- Attempt one CUDA task as a preview slice if the environment is tractable.

Documents:

- `docs/v0.3/design.md`

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
