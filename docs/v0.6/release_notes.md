# OpBench v0.6 Release Notes

Date: 2026-07-19

Release target: `opbench-v0.6.0`

Decision: **Completed — every Must gate passed**

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
Historical v0.5 result and summary hashes remained byte-identical through the
final exact replay.

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
- representative exact-target Remote CPU, CUDA Overlay, and CUDA Kernel
  canaries;
- 85/85 Passed for 17 baseline + 17 gold + 51 historical final-patch replay
  cases, with `failed=0`, `blocked=0`, and an empty difference report;
- 581/581 final full-suite tests with the 17-task verified Dataset, example
  Runtime contract, tracked JSON, and release-document checks passing.

The representative public synthetic Artifact index is
`configs/examples/v0.6_scripted_demo_artifact.example.json`. It contains only
path-independent public identities, axes, totals, actions, and hashes. It is a
controller/artifact demonstration, not a benchmark score.

## Closed release gates

The original M6/M7 freeze correctly recorded R-05–R-08 and R-10 as Blocked
after the one configured exact target returned `connection_timeout`. The same
target later recovered. The closing run used only that configured target and
performed no ping, scan, discovery, broad process/container listing, or
replacement-target search.

- R-05 Passed: all 17 baseline controls reproduced `f2p_failed`;
- R-06 Passed: all 17 strict Gold controls reproduced `resolved`;
- R-07 Passed: all 51 historical final patches reproduced their raw F2P/P2P
  outcomes;
- R-08 Passed: all expected and observed outcomes matched, so the canonical
  difference report is empty and historical score files remain unchanged;
- R-10 Passed: exact Remote CPU, CUDA Overlay, and CUDA Kernel canaries
  completed, with Attempt-owned cleanup evidence for the v1 canaries.

The replay's four persisted aggregate files prove outcome compatibility. Its
controller-private per-case ledgers are intentionally ephemeral and fail
closed on cleanup errors; persistent ownership/cleanup proof is supplied by
the v1 canary artifacts and backend fault-injection coverage.

The final public evidence roots are
`runs/v0.6_release_remote_cpu_canary`,
`runs/v0.6_release_cuda_overlay_canary`,
`runs/v0.6_release_cuda_kernel_canary`, and
`runs/v0.6_release3_legacy_replay_exact_complete`. The full Replay used the
same exact configured target with a private in-memory, commit-specific remote
workspace suffix. An incomplete run that encountered a stale deterministic
leaf from an earlier interruption correctly failed closed and is excluded from
release evidence; no target discovery, resource enumeration, or broad cleanup
was used to obtain the final result.

The published CPU/Overlay directories are redacted public subsets. The
complete controller-private roots passed all 14 Integrity checks before
publication; `private_evaluation.json` and `private_runtime_resources.json`
were then intentionally omitted. Those private files are necessary to rerun
the full Integrity graph, so the repository subsets do not claim standalone
reproduction of all 14 checks.

The final replay inventory hash is
`sha256:193ef08f68f50a50c67f22b41ca2a31043c78d6b2311d23f16c588a86b80daee`.
The manifest, results, empty differences, and summary file SHA-256 values are,
respectively,
`21f85f547b5efde922616a44390b5c07814aaf59c8db27a9863a37a61ac2b424`,
`3c14d5bd462a633b1c4b7b062d1447d6a575ed62244793d5b93154e43db8c9d1`,
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`,
and `1f5fa1515f2e93bbdec9a393e9fc07a3ccf4d121e6d33b175fdb7a1b09b03309`.

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
