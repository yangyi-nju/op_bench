# OpBench v0.6 Developer Guide

Date: 2026-07-19

Status: `opbench-v0.6.0` is Completed. M1–M7 and every Must gate in the
[acceptance matrix](acceptance_matrix.md) have Passed.

The later 51-Attempt real MCP full run is summarized in the
[v0.6 experiment report](experiment_report.md). Its frozen commands and
verification gates remain separate in the
[experiment guide](mcp_agent_experiment.md) and
[verification record](mcp_agent_experiment_verification.md).

## 1. Which execution path to use

`scripts/run_experiment.py` has two deliberately separate surfaces:

| Surface | Selection | Adapters | Result layout | Intended use |
| --- | --- | --- | --- | --- |
| v0.5 Legacy | Default; omit `--runtime-protocol` | `gold`, `codex_action_bridge` | Legacy `results.jsonl`/`summary.json` | Historical compatibility and v0.5 reproduction |
| v0.6 v1 | `--runtime-protocol v1` plus `--runtime-profile` | `scripted_canonical`, `codex_canonical`, `codex_mcp_canonical` | Cohort/Attempt/retry Artifact graph | Standard platform evaluation |

The parser rejects v1-only flags on the Legacy path and rejects Legacy-only
inputs on the v1 path. There is no automatic fallback between protocols. A v1
run accepts exactly one verified Dataset, one canonical Adapter, one Runtime
Profile, and a frozen task × agent × repeat matrix.

The Scripted Adapter is a deterministic no-edit controller smoke. The Codex
Adapter is the canonical real-Agent process boundary. Both call the same JSON
Action Client, Canonical Action Service, Authoritative Workspace, Fresh
Evaluator, ledger, Artifact store, and Integrity verifier.

The independent `codex_mcp_canonical` Adapter uses a real invocation-local
`mcp-stdio` server. For the frozen experiment, `--codex-model gpt-5.6-sol`
binds model `gpt-5.6-sol` and exact executable output
`codex-cli 0.145.0-alpha.27` into the Agent/model/adapter identities. It supplies
the server through per-invocation Codex configuration and never runs a global
MCP add/configuration command. Provider network is allowed only for the
host-side Codex call; Task network remains denied by the Runtime and Action
policies. Codex receives a separate read-only cwd. The launcher, trace, and
Action exchange remain in a controller-private sibling; a per-Attempt transport
token reaches the MCP server through a one-shot inherited pipe rather than the
Codex argv, denies direct Action-client calls, and is drained before Agent tool
execution. Controller executables are inode/mode/hash checked after exit.

## 2. Runtime support matrix

The tracked registry is `configs/runtime_profiles.v1.json`. Five registered
Profile IDs cover four Runtime classes:

| Runtime class | Profile ID | Backend | M6/M7 evidence |
| --- | --- | --- | --- |
| Local CPU process | `local-cpu-process-v1` | `local` | Passed: CLI/MCP conformance, Scripted Demo, real Codex canary/cohort, resume, Integrity, cleanup |
| Remote CPU overlay | `remote-cpu-pytorch-2.6-py311-v1` | `remote_docker` | Passed: exact v1 canary, Integrity, ownership, cleanup, and full replay |
| Remote CPU compile | `remote-cpu-compile-pytorch-2.6-py311-v1` | `remote_docker` | Passed: exact source-loading/replay coverage |
| CUDA Python overlay | `remote-cuda-overlay-pytorch-2.6-cu124-v1` | `remote_docker` | Passed: exact v1 canary, Integrity, ownership, cleanup, and full replay |
| CUDA kernel build | `remote-cuda-kernel-pytorch-2.6-cu124-v1` | `remote_docker` | Passed: exact inplace-build canary and full replay |

Scripted-Remote conformance Passed because it exercises the remote transport
semantics deterministically without claiming external hardware availability.
It is not a substitute for the exact Remote CPU/CUDA canaries.

Remote execution requires both `--target-config` and the selected remote
Profile. The private target file binds one exact backend, local workspace
parent, host alias, remote user/workspace parent, Docker executable, SSH/rsync
argv prefixes, optional GPU request, and an optional absolute ccache seed for
inplace builds. The seed is exact-copied into the Attempt workspace and is not
mutated or cleaned as a shared resource. The private binding is never copied
into a public Manifest or result. OpBench performs no target discovery, no
host/service scan, no ping, and no broad process/container enumeration. It
executes only the supplied target and manages only handles created for the
current Attempt.

The deterministic remote parent namespace may already exist, but the workspace
leaf is created exclusively. If that leaf exists, preparation fails before
rsync, ccache seeding, or Docker creation and does not delete the unowned leaf.
Git commands used to fingerprint controller workspace changes run under the
same narrow authority environment as source materialization, so ambient
`GIT_DIR`, worktree, index, object, or config variables cannot redirect sync.
The same isolation applies to legacy revision resolution, conformance HEAD
resolution, archive identities, local fresh-evaluation clone/apply commands,
and authoritative workspace Git operations.
The length-framed fingerprint includes every controller-created untracked file,
including paths matched by the frozen `.gitignore`, plus the workspace-root
mode, working-tree directory paths/modes, and tracked path types/modes.
Empty-directory creation, root/directory mode changes, staged hidden assets,
and tracked metadata changes therefore cannot be omitted from incremental
rsync.
Only a seeded `inplace_build` excludes the transfer-root `/.ccache/`; ordinary
CPU/Overlay sync excludes nothing. A frozen source-owned root `.ccache` fails
closed before any remote command instead of being silently replaced by seed
content.

## 3. Identity and comparability

Every wire contract uses schema version `v1` and canonical JSON SHA-256.
RunManifest freezes all identities before execution.

The **Comparability Key** hashes:

- `platform_version`, `action_protocol`, `evaluation_protocol`, and
  `scoring_protocol`;
- Evaluation, Dataset, FullTaskSpec, AgentTaskView, Agent/model/adapter/prompt,
  and Scoring identities;
- Capability, Budget, Runtime Profile, Retry, and Termination policies;
- repeat count.

The human timestamp `created_at` is intentionally excluded. The key becomes
the Cohort ID. Each Attempt ID then binds that Cohort to the exact Task, Agent,
and repeat index. A change that affects task visibility, policy, Runtime,
evaluation, scoring, or Agent configuration creates a different comparable
Cohort rather than silently joining old results.

## 4. Attempt lifecycle and actions

The standard flow is:

```text
frozen Manifest → Attempt-owned Runtime/Workspace → canonical JSON Actions
→ one frozen patch → fresh evaluation → append-only selection
→ results/summary rebuild → Integrity verification → exact cleanup evidence
```

The public action set is `workspace_list`, `workspace_search`,
`workspace_read`, `workspace_write`, `workspace_apply_patch`, `command_run`,
`test_run`, `vcs_diff`, and `session_finish`. Each request has a unique
`action_id`; duplicate IDs return the original Observation without repeating a
side effect. The service rechecks session state, capability, path, selector,
deadline, and budgets for every request.

The Adapter sees only AgentTaskView. Gold patch, hidden tests, admission
evidence, private outputs, credentials, and machine-local paths are excluded.
Fresh Evaluation starts after the Agent session is terminal, rebuilds the
verified Source independently, strictly applies only the frozen patch, then
injects evaluation-only tests.

Source materialization and controller-side patch staging do not inherit ambient
Git repository/index/object/config authority. Recursive inplace submodules are
tree-verified against each exact Gitlink commit before their temporary Git
metadata is removed.

## 5. Result axes and failure classification

Do not compress an Attempt into a single ambiguous status. Public results keep
three independent axes:

| Axis | Values | Meaning |
| --- | --- | --- |
| `attempt_validity` | `valid`, `infrastructure_invalid` | Whether the Attempt is eligible for the resolved denominator |
| `agent_terminal` | `finished`, `exited`, `timeout`, `budget`, `cancelled` | How Agent execution ended |
| `evaluation_outcome` | `resolved`, `f2p_failed`, `p2p_regression`, `invalid_patch`, `no_patch`, `evaluation_error`, `not_evaluated` | What fresh evaluation observed |

Action-level stable errors include `invalid_request`, `unsupported_action`,
`session_not_running`, `capability_denied`, `path_denied`, `selector_denied`,
`budget_exhausted`, `timeout`, `conflict`, `workspace_error`, `runtime_error`,
and `platform_error`.

At terminal attribution:

- malformed Agent action payloads are `invalid_request`, not infrastructure;
- capability/path/selector denial is an Agent-visible policy outcome;
- `budget_exhausted` remains a valid Agent terminal on the `budget` axis;
- Codex executable/provider failures become `provider_error`;
- action transport/protocol failures become `runtime_error`;
- Runtime provisioning or external target failures become `platform_error`;
- `provider_error`, `runtime_error`, and `platform_error` select an
  `infrastructure_invalid` retry and do not enter the resolved denominator;
- `invalid_patch`, `f2p_failed`, and `p2p_regression` are valid evaluated
  outcomes and do enter the denominator.

Public error text is sanitized. Exact command output, hidden selectors, and
private resource handles stay in the controller-only layer.

## 6. Artifact layout

A v1 run root uses this canonical layout:

```text
run_manifest.json
attempts.jsonl
results.jsonl
summary.json
integrity.json
attempts/<attempt-id>/
  integrity.json
  retries/retry-0001/
    agent_task_view.json
    events.jsonl
    session_result.json
    final.patch
    public_evaluation.json
    private_evaluation.json
    runtime_resources.jsonl
    private_runtime_resources.json
    runtime_cleanup.json
    runtime_conformance.json
    adapter_trace.json          # codex_mcp_canonical only
```

`private_evaluation.json` and `private_runtime_resources.json` are
controller-only even when the task is synthetic. Never publish raw real-task
private artifacts or target handles. `runtime_resources.jsonl` records public,
path-free state transitions; `runtime_cleanup.json` closes every exact handle.

`events.jsonl` is append-only, sequence-contiguous, and hash chained. Each
Action request has one Observation. Large public output is stored separately
and referenced by media type, byte size, and hash. `final.patch` is immutable
after Freeze; its bytes/hash must agree with Session, ledger, and Evaluation.

The root and per-Attempt `integrity.json` reports cover 14 checks: Manifest
identity, expected/observed matrix, retry audit, TaskView identity, event
chain, action pairing, lifecycle terminals, resource ownership, cleanup,
Session/Patch/Evaluation identity, public/private evaluation binding,
evaluation/scoring protocols, results rebuild, and summary rebuild.

For an MCP Attempt, `adapter_trace.json` adds the exact Adapter/model/CLI
identity, negotiated MCP version, initialize/list/call counters,
`protocol_error_count`, and sanitized server terminal state. Valid selected
MCP Attempts require one initialize, at least one list, exact tool-call/Event
pairing, and terminal agreement. Non-MCP Attempts must not contain this file.
An MCP startup or framing failure may be infrastructure-invalid; normal Action
denials remain Agent-visible results rather than MCP crashes.
Malformed/oversized framing and internal bridge corruption are terminal
`protocol_failed` conditions; unknown methods and invalid Agent parameters stay
recoverable. Provider and bridge stdout/stderr are drained with streaming hard
caps rather than accumulated without limit.

## 7. Resume, verification, and rebuild

Use the same Dataset, Adapter, repeat count, Runtime Profile, output directory,
and v1 flags to resume. The `retry_infrastructure` policy:

- skips a logical Attempt whose selected retry is valid;
- appends a new retry after an infrastructure-invalid selected retry;
- never overwrites an earlier retry or changes a frozen Attempt identity.

After selection, the orchestrator rebuilds `results.jsonl` and `summary.json`
from canonical Artifact bytes. Integrity compares the stored files with that
rebuild byte-for-byte. There is no separate mutable summary source.

Validate the frozen contract and resource evidence with read-only commands:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  runs/v0.6_m7_scripted_demo/run_manifest.json

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_runtime_resources.py \
  --run-root runs/v0.6_m7_scripted_demo
```

The resource verifier reads only the supplied run root. It does not inspect
host processes, Docker inventory, remote hosts, or network state.

If exact Codex process-group cleanup cannot be proven, the runner first records
an immutable infrastructure-invalid retry and then aborts the cohort. A private
0600 recovery marker retains only the recorded PGID. Resume uses no discovery:
it performs a zero-signal check on that exact PGID and remains blocked until
absence is proven; it never signals a possibly reused group from a later run.

Run the real stdio equivalence gate without a Runtime target:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_runtime_conformance.py \
  --transport mcp-stdio \
  --output-dir runs/v0.6_mcp_offline_conformance
```

The full real-Agent procedure and four fixed cohort commands are in
[mcp_agent_experiment.md](mcp_agent_experiment.md), with observed results and
gate evidence in
[mcp_agent_experiment_verification.md](mcp_agent_experiment_verification.md).
The report builder performs a pure read of the four completed roots, re-runs all
14 Integrity checks, and refuses incomplete trace/resource/evaluation identity
before producing its public index and summary. Its formal contract also freezes
the Dataset digest, platform version, four Runtime Profiles, exact 17-task
partition, and repeats 1/2/3; matching only the cohort sizes is insufficient.

Backend preparation and command registration are exact-resource transactions.
If private handle or ledger registration fails after an exclusive workspace,
container, or local process was created, the backend first removes that exact
owned resource, records its terminal transition, and only then reports the
failure. A cleanup failure is reported as `prepare_cleanup_failed` or
`runtime_cleanup_failed` rather than being hidden by the original error;
interrupts follow the same unwind path before propagation. Remote and Docker
process registration failures execute no container command.

For Legacy history, `scripts/run_legacy_replay.py` freezes the v0.5 17 baseline
+ 17 gold + 51 final-patch inventory. Without `--target-config`, all 85 cases
are explicitly Blocked. With it, only that exact target is used. The release
closure Passed all 85 exact cases with zero differences. `inplace_build`
materialization includes the exact recursive submodule Gitlink commits and
uses `setup.py build_ext --inplace`; a task whose historical Gold diff cannot
strictly apply to the frozen base may provide a hash-bound
`gold.replay-v1.patch` plus `gold.replay-v1.provenance.json` sidecar without
altering the original task asset.

Replay persists only its four aggregate compatibility files. The observer's
per-case resource ledger, private lease store, and cleanup report live in
controller-private temporary scratch and are removed when the observer closes;
cleanup still fails the case closed. Use completed v1 canary run roots and
`verify_runtime_resources.py`, together with backend failure-injection tests,
for persistent resource ownership and cleanup evidence rather than treating
the aggregate replay files as that evidence.

## 8. Known limits and release status

- The synthetic Scripted Demo is a platform lifecycle check, not a benchmark
  score and not evidence of coding quality.
- The local real-Codex M6 canary proves the canonical Adapter path, not a
  formal v0.6 ranking.
- v0.6 does not expand the 17-task v0.5 Dataset; expansion belongs to v0.7.
- v0.5's 37/51 result remains historical v0.5 evidence. It is not a v0.6
  score; the 85/85 replay is compatibility evidence, not a new Agent result.
- No feedback-causality, cross-Agent ranking, or population-generalization
  conclusion is claimed.
- The original M6/M7 `connection_timeout` remains in the historical record.
  After the same target recovered, exact Remote CPU/CUDA canaries and replay
  Passed; no replacement target was discovered or selected.
- Exact replay's per-case cleanup ledger is process-local temporary state. An
  abrupt controller death after remote-leaf creation can therefore leave no
  persisted ownership token; a later exclusive-leaf collision currently
  fail-closes and may be cached as target-global unavailability. Future
  hardening should persist protected per-case cleanup attestations and classify
  leaf collision separately. v0.6 never discovers, claims, or broadly cleans
  such a leaf.

See [M6 verification](m6_verification.md), [M7 verification](m7_verification.md),
and the [v0.6 release notes](release_notes.md) for frozen evidence and the
release decision.
