# OpBench

Language: English | [中文](README.zh-CN.md)

OpBench is an operator-focused benchmark for evaluating coding agents on real framework issues. It follows the SWE-bench idea of repairing real repository snapshots, but treats the runtime environment as part of each task because operator bugs often depend on framework version, Python package layout, device availability, numerical behavior, and backend selection.

v0.1 established the isolated replay/evaluation loop. v0.2 completed the platform pieces needed to scale the dataset: asset registries, formal admission evidence, dataset curation, and container/cache management. v0.3 expanded the dataset to 10 verified tasks across 5 PyTorch subsystems, added patch scope enforcement, public/hidden test separation, multi-file overlay, and 3-repeat stability evaluation. The current `pytorch_v0.3` slice achieved 76.7% resolved rate with Codex CLI.

## What The Current Code Contains

- A two-layer dataset model: `datasets/<slice>/dataset.json` points to task bundles under `tasks/`.
- Real PyTorch task bundles with PR/issue metadata, hidden test patches, gold patches, source snapshots, and environment declarations.
- A reusable CPU Docker environment under `environments/pytorch-cpu/`.
- Environment and source registries under `environments/registry.json` and `sources/registry.json`.
- Formal admission evidence through `scripts/run_admission.py`.
- Dataset curation through `scripts/curate_dataset.py`.
- Asset and container inspection through `scripts/inspect_assets.py` and `scripts/manage_containers.py`.
- A replay evaluator that checks baseline failure, gold success, agent patch success, and pass-to-pass regressions.
- A standardized workspace action interface for file operations, patch application, command execution, tests, and diff export.
- `codex_action_bridge`, the reference real-agent adapter. Codex runs on the host in a scratch workspace and can operate on the target repository only through OpBench actions.

Development-only experiment adapters have been removed from the public v0.1 surface. Future agents should integrate by implementing the same action-interface boundary used by `codex_action_bridge`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `datasets/` | Dataset manifests that select task bundles for experiments. |
| `tasks/` | Individual benchmark tasks with issue text, patches, environment metadata, and test lists. |
| `environments/` | Docker environment artifacts and the environment registry. |
| `sources/` | Source snapshot registry metadata. |
| `src/op_bench/` | Core implementation: task model, environment preparation, evaluator, actions, agent bridges, reporting. |
| `scripts/` | CLI entry points for validation, environment preparation, source snapshots, replay, and experiments. |
| `docs/` | Versioned design docs, experiment reports, developer guides, and historical records. |
| `docs/v0.3/design.md` | v0.3 dataset expansion, multi-file overlay, public/hidden test split, and CUDA pilot design. |
| `docs/v0.2/developer_guide.md` | v0.2 registry, admission, curation, asset, and container workflow. |
| `docs/v0.1/developer_guide.md` | v0.1 module-level architecture and development guide. |

## Quick Start

Create the project Python environment:

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version
```

Rebuild source snapshots (required after fresh clone):

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/setup_sources.py
```

Run unit tests:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

Validate the current dataset:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.3/dataset.json
```

Run formal admission for the first admitted PyTorch task:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_admission.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output-dir runs/admission/pytorch__149693__lazylinear_init/manual \
  --write-task-evidence
```

Run the current dataset gold-loop check:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.3_gold
```

Inspect registered assets:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py
```

Create a verified-only dataset slice from the current mixed dataset:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/curate_dataset.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --output-dataset datasets/pytorch_mini_v0.2/dataset.json \
  --output-summary datasets/pytorch_mini_v0.2/summary.json \
  --verified-only \
  --dataset-id pytorch_mini_v0.2 \
  --version v0.2
```

Run a real isolated agent evaluation (Codex or Claude Code):

```bash
# Codex CLI
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_codex_r3

# Claude Code CLI
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CLAUDE_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent claude_code_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_claude_r3

# Multi-agent comparison
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --agent claude_code_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_multi_agent
```

Run only specific tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --filter-tasks autograd lbfgs \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_subset
```

`scripts/run_experiment.py` prints timestamped progress logs to stderr. Add `--quiet` when only `results.jsonl` and `summary.json` are needed.

## Current Dataset

The primary dataset is [datasets/pytorch_v0.3](datasets/pytorch_v0.3/dataset.json) (10 verified tasks).

| Task | PR | Component | Rate (3 repeats) |
| --- | --- | --- | --- |
| `pytorch__168295__autograd_create_graph` | #168295 | torch.autograd | 3/3 |
| `pytorch__150975__autograd_backward_inputs` | #150975 | torch.autograd | 3/3 |
| `pytorch__161488__lbfgs_wolfe` | #161488 | torch.optim | 3/3 |
| `pytorch__124385__load_state_dict_prefix` | #124385 | torch.nn.Module | 3/3 |
| `pytorch__149693__lazylinear_init` | #149693 | torch.nn.LazyLinear | 3/3 |
| `pytorch__147599__lazylinear_state_forward` | #147599 | torch.nn.LazyLinear | 3/3 |
| `pytorch__160952__bilinear_lazy_check` | #160952 | torch.nn.Bilinear | 3/3 |
| `pytorch__143455__set_submodule` | #143455 | torch.nn.Module | 2/3 |
| `pytorch__162340__nn_arg_length` | #162340 | torch.nn.conv/utils | 0/3 |
| `pytorch__163961__dataloader_subset` | #163961 | torch.utils.data | 0/3 |

Use `--verified-only` for benchmark runs that should count only admitted tasks. Use `--filter-tasks` to run a subset.

Every verified task has task-local stable admission evidence at its own `admission/evidence.json`.

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

Read [docs/v0.2/developer_guide.md](docs/v0.2/developer_guide.md) and [docs/v0.3/design.md](docs/v0.3/design.md) for the current expansion workflow. The usual path is:

1. Add or curate task bundles under `tasks/<framework>/`.
2. Register reusable environment/source assets under `environments/registry.json` and `sources/registry.json`.
3. Run `scripts/run_admission.py --write-task-evidence`.
4. Add only evidence-backed verified tasks to a verified dataset slice with `scripts/curate_dataset.py`.
5. Run `scripts/run_experiment.py` on `--verified-only`.
6. Add new real agents by implementing the action-interface boundary rather than giving them direct access to target workspaces.

## References

- [Docs index](docs/README.md)
- [v0.3 design](docs/v0.3/design.md)
- [v0.3 experiment report](docs/v0.3/experiment_report.md)
- [v0.2 developer guide](docs/v0.2/developer_guide.md)
- [v0.2 experiment report](docs/v0.2/experiment_report.md)
- [v0.1 developer guide](docs/v0.1/developer_guide.md)
- [v0.1 manual validation workflow](docs/v0.1/manual_validation.md)
