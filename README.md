# OpBench

Language: English | [中文](README.zh-CN.md)

OpBench is an operator-focused benchmark for evaluating coding agents on real framework issues. It follows the SWE-bench idea of repairing real repository snapshots, but treats the runtime environment as part of each task because operator bugs often depend on framework version, Python package layout, device availability, numerical behavior, and backend selection.

v0.1 is intentionally small: one verified PyTorch CPU task, a Docker-backed replay/evaluation loop, and one real isolated agent path through Codex CLI.

## What v0.1 Contains

- A two-layer dataset model: `datasets/<slice>/dataset.json` points to task bundles under `tasks/`.
- Real PyTorch task bundles with PR/issue metadata, hidden test patches, gold patches, source snapshots, and environment declarations.
- A reusable CPU Docker environment under `environments/pytorch-cpu/`.
- A replay evaluator that checks baseline failure, gold success, agent patch success, and pass-to-pass regressions.
- A standardized workspace action interface for file operations, patch application, command execution, tests, and diff export.
- `codex_action_bridge`, the reference real-agent adapter. Codex runs on the host in a scratch workspace and can operate on the target repository only through OpBench actions.

Development-only experiment adapters have been removed from the public v0.1 surface. Future agents should integrate by implementing the same action-interface boundary used by `codex_action_bridge`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `datasets/` | Dataset manifests that select task bundles for experiments. |
| `tasks/` | Individual benchmark tasks with issue text, patches, environment metadata, and test lists. |
| `environments/` | Docker environment artifacts referenced by tasks. |
| `src/op_bench/` | Core implementation: task model, environment preparation, evaluator, actions, agent bridges, reporting. |
| `scripts/` | CLI entry points for validation, environment preparation, source snapshots, replay, and experiments. |
| `docs/developer_guide.md` | Module-level architecture and development guide. |
| `docs/manual_validation.md` | Manual workflow for promoting tasks from draft to verified. |
| `docs/OpBench_v0.1_experiment_report.md` | v0.1 experiment evidence and result analysis. |

## Quick Start

Create the project Python environment:

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version
```

Run unit tests:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

Validate the current mini dataset:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

Verify replay for the first admitted PyTorch task:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

Run a real isolated Codex CLI evaluation:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```

`scripts/run_experiment.py` prints timestamped progress logs to stderr. Add `--quiet` when only `results.jsonl` and `summary.json` are needed.

## Current Dataset

The first dataset slice is [datasets/pytorch_mini](datasets/pytorch_mini/README.md).

| Task | PR | Status |
| --- | --- | --- |
| `pytorch__149693__lazylinear_init` | https://github.com/pytorch/pytorch/pull/149693 | verified |
| `pytorch__160952__bilinear_lazy_check` | https://github.com/pytorch/pytorch/pull/160952 | draft |

Use `--verified-only` for benchmark runs that should count only admitted tasks.

## Runtime Boundary

Agent identity, model calls, and control logic run on the host. Each repair attempt gets a prepared workspace and a task-scoped Docker runtime. The agent must interact with the target repository through OpBench actions:

- `read_file`
- `write_file`
- `apply_patch`
- `run_command`
- `run_test`
- `git_diff`

For Docker tasks, preflight commands, setup commands, test commands, and action-interface command calls run in the same task container. This preserves setup state and avoids mixing host execution with benchmark execution.

`codex_action_bridge` is the reference implementation of this boundary. It gives Codex a scratch directory plus an `opbench_action.py` CLI. The target repository path is not exposed to Codex, and final scoring uses the patch exported through the action interface.

## Adding More Work

Read [docs/developer_guide.md](docs/developer_guide.md) before changing internals. The usual expansion path is:

1. Add or curate task bundles under `tasks/<framework>/`.
2. Add the task entries to a dataset manifest under `datasets/`.
3. Prepare or pin the task environment under `environments/`.
4. Verify baseline/gold replay with `scripts/verify_task_replay.py`.
5. Run `scripts/run_experiment.py` on `--verified-only`.
6. Add new real agents by implementing the action-interface boundary rather than giving them direct access to target workspaces.

## References

- [Developer guide](docs/developer_guide.md)
- [Manual validation workflow](docs/manual_validation.md)
- [v0.1 experiment report](docs/OpBench_v0.1_experiment_report.md)
- [Dataset builder workflow](docs/builder_workflow.md)
