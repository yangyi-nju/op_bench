# OpBench v0.6 Release Notes

Date: 2026-07-18

Release target: `opbench-v0.6.0`

Decision: **Blocked — implementation complete, exact-Remote Must evidence incomplete**

## What v0.6 changes

v0.6 turns the existing real-Codex benchmark Demo into a versioned Agent
evaluation platform. It adds:

- strict v1 contracts and canonical identities for Manifest, TaskView, Action,
  Session, Evaluation, Result, Event, resources, and Integrity;
- a frozen Comparability Key, Cohort ID, Attempt ID, and complete expected
  task × agent × repeat matrix;
- an explicit FullTaskSpec → AgentTaskView information boundary;
- one Authoritative Workspace, one immutable canonical patch, and strict
  clean-source patch application;
- one server-authoritative Action Service shared by CLI and MCP;
- deterministic session/budget/termination semantics, append-only retry/resume,
  and hash-chained trajectory artifacts;
- Fresh Evaluation separated from the Agent Workspace, with independent
  validity, terminal, and evaluation-outcome axes;
- deterministic result/summary rebuild and a 14-check Integrity graph;
- versioned Local CPU, Remote CPU, CUDA Overlay, and CUDA Kernel Runtime
  Profiles with Attempt-owned resource ledgers and exact cleanup;
- Local/Scripted-Remote conformance, a frozen v0.5 17+17+51 replay inventory,
  and a process-isolated canonical Codex Adapter;
- a public synthetic v1 Scripted Demo, bilingual Quickstart, developer guide,
  support matrix, representative public Artifact index, and release review.

## Compatibility and migration

The v0.5 runner remains the default migration path. Existing commands omit
`--runtime-protocol` and continue to write the Legacy `results.jsonl` and
`summary.json` layout. The v1 Runtime must be selected explicitly with
`--runtime-protocol v1`, one `--runtime-profile`, and either
`scripted_canonical` or `codex_canonical`.

Legacy and v1 Cohorts never merge. v1-only flags on the Legacy path and
Legacy-only inputs on the v1 path fail before Runtime resources are created.
Historical v0.5 result and summary hashes were unchanged by M6 replay work.

## Evidence that passed

- 517/517 M6 full tests and 84/84 M6 focused tests;
- Local and Scripted-Remote CLI/MCP conformance;
- one valid real Codex local CPU Attempt through the canonical Adapter;
- a two-repeat real Codex local cohort covering infrastructure retry, resume,
  Attribution, Integrity, and exact cleanup;
- a synthetic local v1 Scripted Demo whose first run executes one Attempt and
  whose identical second run skips it byte-for-byte;
- RunManifest validation, deterministic rebuild, 14 Integrity checks, and
  exact resource ownership/cleanup verification;
- a complete, immutable inventory for 17 baseline + 17 gold + 51 historical
  final-patch replay cases.

The representative public synthetic Artifact index is
`configs/examples/v0.6_scripted_demo_artifact.example.json`. It contains only
path-independent public identities, axes, totals, actions, and hashes. It is a
controller/artifact demonstration, not a benchmark score.

## Blocked release gates

The only configured exact Remote target consistently timed out during the
workspace-create operation. Therefore:

- R-05: v0.5 17/17 Baseline Failure Replay is Blocked;
- R-06: v0.5 17/17 Gold Success Replay is Blocked;
- R-07: v0.5 51/51 Legacy Final Patch Replay is Blocked;
- R-08: per-Task/Environment/Protocol replay difference attribution is
  Blocked because execution evidence does not exist;
- R-10: representative Remote CPU, CUDA Overlay, and CUDA Kernel canaries are
  Blocked.

No ping, scan, discovery, broad process/container listing, or replacement
target search was used. M7 does not repeat the stable timeout. The exact target
must recover before these Must items can pass, so `opbench-v0.6.0` is not tagged
or declared Completed.

## Explicit non-claims

- The v0.5 37/51 result remains a v0.5 result; it is not a v0.6 score.
- The synthetic Scripted Demo and local Codex canaries are not a formal Agent
  ranking.
- v0.6 does not run the planned feedback-causality experiment.
- v0.6 does not establish cross-Agent rankings or population-level
  generalization.
- Dataset expansion and Boundary-slice work remain v0.7 scope.

See the [developer guide](developer_guide.md), [M6 verification](m6_verification.md),
[M7 verification](m7_verification.md), and [acceptance matrix](acceptance_matrix.md).
