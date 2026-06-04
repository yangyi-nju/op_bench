# Changelog

This file records user-visible OpBench version milestones. Detailed design,
implementation, and experiment evidence remain in the versioned documents
under `docs/`.

## v0.2 - In Development

Development started on 2026-06-04.

Planned scope:

- Expand the verified PyTorch operator dataset to 3-5 tasks.
- Add reusable environment and source snapshot registries.
- Add a formal task admission pipeline with stable replay evidence.
- Add evidence-aware dataset validation and environment lifecycle management.
- Continue using the v0.1 Codex action-bridge path for isolated agent scoring.

Documents:

- `docs/OpBench_v0.2_design.md`
- `docs/superpowers/plans/2026-06-04-opbench-v0.2-platform.md`

## v0.1 - Completed

OpBench v0.1 established the minimum isolated benchmark loop:

- Built task bundles from real PyTorch issues and PRs.
- Replayed fail-to-pass and pass-to-pass tests in task-specific Docker environments.
- Managed local full-repository source snapshots for reproducible workspaces.
- Ran a real Codex CLI agent through the OpBench action interface.
- Scored agent patches in fresh isolated workspaces and recorded experiment evidence.

Documents:

- `docs/OpBenchPRD v0.1.md`
- `docs/OpBench_v0.1_experiment_report.md`
- `docs/developer_guide.md`
- `docs/manual_validation.md`
