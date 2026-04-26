# op_bench MVP Design

## 1. Goal

Build a minimal benchmark system that can:

1. store a small set of operator-related issue instances
2. replay them on a fixed repository snapshot and environment
3. run an agent against the task
4. execute evaluation tests automatically
5. report comparable metrics across agents

The MVP does not need a large dataset. It needs a stable workflow.

## 2. Core Difference From SWE-bench

SWE-bench is centered on repository-level issue resolution with hidden execution-based tests. `op_bench` should keep that spirit, but extend the task definition to include environment and hardware context.

For operator bugs, the real problem is often one of these:

- operator semantics differ across dtype or shape corner cases
- CPU and GPU behavior diverge
- error handling is inconsistent across backends
- numerical precision and tolerance cause regressions
- versioned dependencies or kernels affect reproductions

That means an operator benchmark instance is not just:

- issue text
- repo snapshot
- hidden tests

It is closer to:

- issue text
- repo snapshot
- environment card
- device tier
- hidden tests
- reproducibility constraints

## 3. Strong Recommendation For MVP Scope

Do not start with the full space of PyTorch and TensorFlow bugs.

The cheapest reliable MVP is:

- one framework first
- CPU-only first
- Python-level or lightweight source-edit tasks first
- tasks with deterministic unit tests
- fixes that do not require rebuilding giant native stacks

Practical scope for a first MVP:

- 3 to 5 tasks if the goal is only pipeline validation
- 8 to 15 tasks if the goal is a first internal comparison across agents

Recommended first slice:

- `pytorch` or `tensorflow`, but not both on day one
- operator-adjacent tasks in Python-facing modules
- issue/PR pairs where the merged fix already includes or updates tests

## 4. What To Exclude Initially

Exclude the following from the first release:

- multi-GPU or distributed execution
- tasks requiring building PyTorch or TensorFlow from scratch
- flaky tests
- performance-only regressions without a functional pass/fail oracle
- tasks whose issue description is too vague to solve without PR comments
- tasks that depend on proprietary hardware or unavailable drivers

These categories can become benchmark tiers later, but they will slow the first version dramatically.

## 5. Dataset Construction Strategy

### 5.1 Task Source

Mine issue/PR pairs from repositories where operator behavior is central.

Possible source categories:

- framework core repos such as `pytorch/pytorch` and `tensorflow/tensorflow`
- operator libraries or extension repos with lighter setup costs
- backend or runtime repos only after the CPU pipeline is stable

### 5.2 Selection Heuristics

A task should pass all of these checks:

- issue is linked to a merged PR
- base commit is recoverable
- fix commit is recoverable
- there is at least one deterministic `fail-to-pass` test
- there is a reasonable `pass-to-pass` regression set
- setup and test runtime fit within a predictable budget
- environment and device requirements can be described concretely

Useful ranking filters for mining:

- issue/PR title contains operator names or keywords such as `dtype`, `shape`, `broadcast`, `nan`, `inf`, `grad`, `cpu`, `cuda`, `op`, `kernel`
- changed files touch operator-relevant paths
- diff size is small to medium
- test patch is present and understandable

### 5.3 Human Verification

Even in MVP, add a light human check before admitting a task:

- Can the issue be understood from the issue text alone?
- Can the task be reproduced in the chosen environment?
- Is the expected behavior objectively testable?
- Is the task small enough that an agent can attempt it within budget?

This mirrors why SWE-bench later introduced Verified subsets.

## 6. Task Bundle Design

Each benchmark task should be packaged as a bundle with four parts:

### 6.1 Agent-visible inputs

- issue statement
- repository checkout at the base commit
- environment card
- allowed commands and resource budget

### 6.2 Evaluator-only metadata

- merged gold patch
- hidden evaluation tests
- standardized setup command
- canonical test command

### 6.3 Execution constraints

- hardware tier
- timeout
- memory floor
- build mode

### 6.4 Analysis metadata

- framework
- operator name
- bug type
- difficulty bucket
- runtime estimate

## 7. Minimal Task Schema

Each task manifest should record at least:

- `task_id`
- source repository and issue/PR references
- base and merge commits
- operator metadata
- environment card
- evaluator commands
- fail-to-pass and pass-to-pass test lists
- hidden artifact locations

The example manifest lives in [tasks/examples/sample_task.json](/Users/yy/dev/graduate/op_bench/tasks/examples/sample_task.json), and the schema lives in [schemas/task_manifest.schema.json](/Users/yy/dev/graduate/op_bench/schemas/task_manifest.schema.json).

## 8. Agent Evaluation Protocol

To compare agents fairly, the runner contract should be fixed:

### 8.1 Agent input

- task directory
- clean workspace cloned from the task's base commit
- environment card
- issue statement

### 8.2 Agent output

- a unified diff patch
- optional structured run metadata

### 8.3 Runner limits

- wall-clock timeout per task
- optional token budget
- optional command budget
- fixed hardware profile per benchmark tier

### 8.4 Evaluation

The evaluator applies the patch, runs setup if needed, and computes:

- `resolved`: all required fail-to-pass tests succeed and no selected regression tests fail
- `fail_to_pass_rate`
- `pass_to_pass_rate`
- `runtime_sec`
- optional cost metrics such as tokens or tool calls

## 9. Scoring Recommendation

Do not collapse everything into a single weighted score in the MVP.

Use:

- primary leaderboard metric: `resolved_rate`
- secondary metrics: `fail_to_pass_rate`, `pass_to_pass_rate`
- operational metrics: median runtime, timeout rate, setup failure rate

Report by bucket as well:

- by framework
- by operator problem type
- by hardware tier

This avoids hiding critical failure modes behind one scalar number.

## 10. Proposed System Architecture

The first implementation can be split into four modules.

### 10.1 Dataset Builder

Responsibilities:

- ingest curated issue/PR pairs
- materialize task manifests
- store hidden patches and test metadata

### 10.2 Task Runner

Responsibilities:

- prepare workspace at base commit
- expose visible task inputs to the agent
- capture patch output and run logs

### 10.3 Evaluator

Responsibilities:

- apply submitted patch
- execute setup and test commands
- compute task-level metrics

### 10.4 Reporter

Responsibilities:

- aggregate results across tasks
- produce per-agent summaries
- export machine-readable results

## 11. Concrete MVP Milestones

### Milestone A: Task Format

- finalize the task manifest schema
- hand-author 1 example task bundle
- build a validation utility

### Milestone B: Single-Task Replay

- create one real task from one repository
- confirm base commit reproduction
- confirm hidden evaluation passes on the gold patch

### Milestone C: Agent Harness

- define a standard way to launch an agent on a task
- capture patch outputs and logs
- run evaluation automatically

### Milestone D: Mini Benchmark

- expand to 3 to 5 tasks
- run at least two agents
- compare resolved rate and runtime

## 12. High-Leverage Product Decisions

If you want this project to scale, make these decisions early:

- separate `task metadata` from `agent-visible prompt files`
- separate `environment tier` from `framework name`
- keep hidden tests and gold patches outside agent-visible paths
- version the task schema from the beginning
- log setup failures separately from patch failures

Those choices will save a lot of pain when you add GPU tiers later.

## 13. My Suggested First Real Move

If we continue from here, the most effective next step is:

1. pick one framework for MVP
2. define one exact hardware tier
3. curate 3 candidate issue/PR pairs by hand
4. turn one of them into the first replayable task bundle

My default recommendation would be:

- framework: `pytorch`
- tier: `cpu-deterministic`
- dataset size target for first end-to-end run: `3 tasks`

That is narrow enough to finish and wide enough to expose workflow problems quickly.
