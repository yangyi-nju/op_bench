# op_bench

`op_bench` is an operator-focused benchmark for evaluating coding agents on real issue resolution tasks, inspired by SWE-bench but adapted for framework/operator repositories where environment and hardware matter.

## Why This Needs A Different Design

SWE-bench assumes that a repository snapshot plus hidden tests are often enough to judge whether a patch resolves a GitHub issue. Operator repositories add extra constraints:

- environment details are part of the problem, not just the setup
- device availability can change behavior and reproducibility
- numerical tolerance, dtype, backend, and kernel selection can all affect outcomes
- many upstream tasks are expensive to build or flaky to rerun

For that reason, `op_bench` treats the environment card and hardware tier as first-class task metadata.

## MVP Scope

The first version should optimize for a reliable end-to-end pipeline, not for benchmark size.

- focus on `CPU-only`, deterministic, unit-test-sized tasks
- prefer issues linked to merged PRs from operator-heavy repositories
- select tasks where the fix is small enough to replay and evaluate cheaply
- exclude distributed, multi-node, perf-only, and mandatory CUDA/C++ rebuild tasks

## Suggested Benchmark Tiers

- `cpu-deterministic`: default MVP tier, reproducible on standard CI machines
- `single-gpu`: later stage, pinned CUDA image and explicit device requirements
- `kernel-build`: later stage, tasks that require compiling native or CUDA kernels

## Repository Layout

- `docs/`: design notes and benchmark decisions
- `fixtures/`: local builder fixtures for offline development
- `schemas/`: task manifest schema
- `tasks/smoke/`: verified local smoke tasks for runner development
- `tasks/examples/`: example task bundle metadata
- `scripts/`: small utilities for validation and local workflow checks
- `src/op_bench/`: builder, task model, runner, evaluator, agent, and reporter code

## Python Environment

Create and use a local virtual environment:

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version
```

After the environment exists, run project commands with `python`:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

## Builder MVP

You can bootstrap draft benchmark tasks directly from GitHub pull requests with [scripts/build_task_from_pr.py](/Users/yy/dev/graduate/op_bench/scripts/build_task_from_pr.py).

- input: a PR URL, plus an optional issue URL override
- output: a draft task bundle with raw metadata, a draft manifest, and the merged patch

Builder details and example commands live in [docs/builder_workflow.md](/Users/yy/dev/graduate/op_bench/docs/builder_workflow.md).

## MVP Smoke Experiment

Run the local smoke benchmark:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --task tasks/smoke/expit_nan_cpu \
  --agent noop \
  --agent gold \
  --output-dir /tmp/op_bench_smoke
```

Expected result:

- `gold` resolves the task
- `noop` fails the fail-to-pass test
- `/tmp/op_bench_smoke/results.jsonl` contains per-run records
- `/tmp/op_bench_smoke/summary.json` contains aggregate resolved rates

## Real PyTorch Candidate Tasks

The first full-repository PyTorch candidates live under `tasks/pytorch/`.

- `tasks/pytorch/149693_lazylinear_init`: PR https://github.com/pytorch/pytorch/pull/149693
- `tasks/pytorch/160952_bilinear_lazy_check`: PR https://github.com/pytorch/pytorch/pull/160952

These tasks stay in `draft` status until replay evidence proves baseline failure and gold success on the declared environment.
If replay reports `environment_error`, the repository checkout and patches may be valid, but the declared runtime is not yet available on the machine running the benchmark.

Verify replay for one task:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

Run a real agent after replay is viable:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --agent noop \
  --agent gold \
  --agent codex \
  --output-dir runs/pytorch-mini-codex
```

## Recommended Build Order

1. Curate 3 to 5 hand-picked CPU tasks from one framework first.
2. Verify that each task can be replayed at the base commit and judged automatically.
3. Standardize the agent input/output contract.
4. Add runner automation and result aggregation.
5. Expand to a small cross-framework MVP only after the single-framework flow is stable.

More detail lives in [docs/mvp_design.md](/Users/yy/dev/graduate/op_bench/docs/mvp_design.md).
