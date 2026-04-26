# op_bench MVP Design

## Context

`op_bench` is an operator-focused benchmark inspired by SWE-bench, but it cannot treat the repository snapshot and hidden tests as sufficient context. Operator bugs in PyTorch, TensorFlow, and related systems are frequently tied to dtype, shape, backend, kernel selection, dependency versions, hardware, driver versions, and numerical tolerance. The first MVP should therefore prove the full evaluation loop while keeping the execution surface intentionally small.

The first experimental slice is:

- framework: PyTorch
- tier: CPU-only deterministic tasks
- task count: 1 to 3 tasks
- task type: Python-level or lightweight operator-adjacent fixes
- goal: validate benchmark mechanics before expanding task count, frameworks, or hardware tiers

## Goals

The MVP must provide a reliable end-to-end loop:

1. Build or hand-curate a small task bundle from a real issue/PR pair.
2. Replay the task at the base repository state.
3. Confirm that the base state reproduces the failure.
4. Confirm that the gold patch resolves the failure.
5. Run at least a small set of agents against the same task contract.
6. Evaluate agent patches with the same hidden tests and environment evidence.
7. Produce machine-readable result summaries.

The MVP optimizes for correctness of the benchmark workflow, not dataset size.

## Non-Goals

The first version will not attempt:

- GPU execution
- distributed tests
- full PyTorch source builds
- TensorFlow support
- large-scale task mining
- performance-only benchmarks without functional pass/fail oracles
- production-grade container scheduling

Those capabilities should be reflected in the architecture but not implemented on the first critical path.

## Recommended Approach

Use a task-bundle plus environment-card plus runner design.

The existing builder remains the entry point for turning issue/PR metadata into draft tasks. The MVP adds the missing execution system around that draft:

- a task format that separates agent-visible and evaluator-only inputs
- a local executor that can later be replaced by Docker or hardware-specific executors
- a replay verifier that checks baseline failure and gold success
- an evaluator that applies agent patches and computes structured outcomes
- a reporter that aggregates per-agent and per-task results

This gives a fast local loop while preserving the project’s operator-specific direction.

## Task Bundle

Each task directory should contain:

- `task.json`: manifest and benchmark metadata
- `issue.md`: agent-visible problem statement
- `artifacts/gold.patch`: evaluator-only known-good patch
- `artifacts/test.patch`: evaluator-only hidden test patch
- `raw/`: original PR, issue, and file metadata
- optional `replay/`: stored verification logs and environment evidence

The manifest remains the source of truth for:

- source repository and commits
- operator metadata
- environment requirements
- setup commands
- fail-to-pass tests
- pass-to-pass tests
- artifact paths
- curation status

Before a task becomes benchmark-ready, it must move from `draft` to `verified`.

## Environment Model

Environment is part of the benchmark instance, not only the runner setup.

The MVP should define an `EnvironmentSpec` from manifest fields:

- tier
- image
- OS
- Python version
- build mode
- device requirements
- dependency list

The first executor can run locally, but every run must collect an environment evidence record:

- Python executable and version
- platform and OS details
- installed package versions relevant to the task
- detected device availability
- selected executor type
- command logs and exit codes

The evaluator should compare declared environment requirements with observed evidence and record mismatches as warnings or setup failures depending on severity.

## Executor Interface

The runner should depend on a small executor abstraction instead of hard-coding local shell execution everywhere.

Initial executor:

- `local`: runs commands in a workspace on the current machine

Future executors:

- `docker`: runs the same task in a pinned container image
- `gpu-docker`: adds device mounting and CUDA checks
- `remote`: schedules tasks onto a hardware pool

The interface should support:

- preparing a clean workspace
- running setup commands
- applying patches
- running evaluation commands
- collecting logs
- enforcing timeout
- collecting environment evidence

## Evaluation Protocol

Each verified task should support three evaluation modes:

- `baseline`: base repository plus hidden tests, expected to fail at least one fail-to-pass test
- `gold`: base repository plus gold patch plus hidden tests, expected to pass
- `agent`: base repository plus an agent-submitted patch plus hidden tests

An agent is resolved only if:

- its patch applies cleanly
- setup succeeds
- all fail-to-pass tests pass
- all selected pass-to-pass tests pass
- the task does not time out

The MVP should report both the final outcome and the failure reason.

## Failure Taxonomy

The result format should distinguish:

- `resolved`
- `setup_failed`
- `baseline_not_reproduced`
- `gold_failed`
- `patch_apply_failed`
- `fail_to_pass_failed`
- `pass_to_pass_regressed`
- `timeout`
- `environment_mismatch`
- `runner_error`

This is important for op_bench because environment and operator behavior failures are materially different from ordinary patch correctness failures.

## Agent Contract

For the MVP, agents can be represented by adapters.

Each adapter receives:

- task directory
- clean workspace at the base commit
- issue statement
- environment card
- allowed commands

Each adapter returns:

- unified diff patch path
- optional metadata such as runtime, command count, notes, or tool usage

Initial adapters can be simple:

- `gold`: returns the task gold patch, used as a sanity check
- `noop`: returns an empty patch, used to verify negative behavior
- `scripted`: runs a configured local command that produces a patch

This is enough to validate the system without committing to one agent framework.

## Reporter

The reporter should write:

- per-run JSON result
- JSONL records for multi-task experiments
- aggregate summary by agent
- aggregate summary by task
- counts by failure reason

Primary MVP metric:

- `resolved_rate`

Secondary MVP metrics:

- `fail_to_pass_rate`
- `pass_to_pass_rate`
- setup failure rate
- timeout rate
- median runtime

## First Experiment

The first experiment should run:

- 1 to 3 PyTorch CPU-only tasks
- `noop` agent
- `gold` agent
- optionally one real coding agent adapter if available locally

Expected validation:

- baseline fails for each verified task
- gold passes for each verified task
- noop fails for each non-trivial task
- reporter shows correct resolved rates and failure reasons

The first task may be a lightweight local smoke task if needed to prove the mechanics quickly. At least one follow-up task should come from a real PyTorch issue/PR before calling the MVP useful as a benchmark prototype.

## Data Expansion Path

After the smoke loop passes:

1. Add 3 verified PyTorch CPU tasks.
2. Add curation automation that flags candidate PRs by operator keywords, changed files, and test diffs.
3. Add Docker executor support for pinned images.
4. Add GPU tiers only after CPU replay is stable.
5. Add task-level environment compatibility checks before scheduling.

## Engineering Boundaries

The implementation should keep these modules separate:

- builder: create draft tasks from issue/PR metadata
- task model: load and validate manifests
- executor: run commands in a declared environment
- verifier: baseline and gold replay checks
- evaluator: evaluate submitted patches
- agents: produce patches through a stable adapter contract
- reporter: aggregate result records

Keeping these boundaries clear matters because environment management will grow into its own subsystem as the benchmark expands.

## Open Decisions

The MVP can proceed with these default choices:

- local executor first
- JSON/JSONL reports first
- fixture or lightweight task allowed for smoke testing
- Docker support deferred but reflected in interfaces

Decisions to revisit after the first closed-loop run:

- exact schema additions for environment evidence
- whether to store full command logs inside task directories or experiment directories
- whether hidden tests are always patches or can also be task-local test files
- how strict environment mismatch handling should be for CPU tasks
