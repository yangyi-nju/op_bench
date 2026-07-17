# OpBench

Language: English | [中文](README.zh-CN.md)

OpBench is an operator-focused benchmark for evaluating coding agents on real framework issues. It follows the SWE-bench idea of repairing real repository snapshots, but treats the runtime environment as part of each task because operator bugs often depend on framework version, Python package layout, device availability, numerical behavior, and backend selection.

v0.1 established the isolated replay/evaluation loop. v0.2 added asset registries and formal admission. v0.3 expanded the dataset to 10 verified tasks and added 3-repeat stability evaluation. v0.4 added CUDA tiers and remote Docker. v0.5 is now complete: the verified cumulative dataset contains 17 tasks, including a 6-task precision slice, and its 51-attempt Codex run reached **72.5% resolved** (37/51) with eight-dimensional reporting and hard experiment-integrity checks.

The next release, v0.6, upgrades this working real-Codex benchmark demo into a standardized Agent evaluation platform. It unifies versioned task views, canonical actions with CLI/MCP adapters, attempt lifecycle and budgets, feedback trajectories, patch freezing, fresh evaluation, failure attribution, replay, and rebuildable artifacts. Boundary-task expansion follows in v0.7 after the platform contract is stable. See the [global project plan](docs/project_plan.md) and [current project state](docs/project_state.md).

## What The Current Code Contains

- A two-layer dataset model: `datasets/<slice>/dataset.json` points to task bundles under `tasks/`.
- Real PyTorch task bundles with PR/issue metadata, hidden test patches, gold patches, source snapshots, and environment declarations.
- Reusable Docker environments: `environments/pytorch-cpu/` (CPU), `environments/pytorch-cpu-compile/` (CPU + Inductor toolchain), `environments/pytorch-cuda/` (CUDA Python overlay), `environments/pytorch-cuda-devel/` (CUDA + nvcc/ccache/cmake for kernel builds).
- Environment and source registries under `environments/registry.json` and `sources/registry.json`.
- Three runtime tiers: `cpu_python_overlay`, `cuda_python_overlay`, `cuda_kernel_build` (last two use `remote_docker` backend over SSH).
- Formal admission evidence through `scripts/run_admission.py`, offline preflight through `scripts/preflight_task.py`.
- Dataset curation through `scripts/curate_dataset.py`.
- Asset and container inspection through `scripts/inspect_assets.py` and `scripts/manage_containers.py`.
- A replay evaluator that checks baseline failure, gold success, agent patch success, and pass-to-pass regressions; supports `python_overlay` and `inplace_build` source loading modes.
- A remote GPU Docker executor (`src/op_bench/remote.py`) that runs `docker` on an SSH host, rsyncs workspaces both ways, and persists an environment-scoped ccache across isolated workspaces.
- A standardized workspace action interface for file operations, patch application, command execution, tests, and diff export.
- `codex_action_bridge`, the reference real-agent adapter with automatic rate-limit-aware retry. Codex runs on the host in a scratch workspace and can operate on the target repository only through OpBench actions.

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
| `docs/project_plan.md` | Global mission, principles, roadmap, release gates, and research targets. |
| `docs/project_state.md` | Current baseline, active version, decisions, and next actions. |
| `docs/v0.6/` | v0.6 standardized Agent evaluation platform design, implementation plan, and acceptance matrix. |
| `docs/v0.7/design.md` | v0.7 Dataset Factory, Boundary Slice, and matched-runtime recovery design. |
| `docs/v0.5/design.md` | v0.5 dimension taxonomy and extended evaluation metrics. |
| `docs/v0.5/experiment_report.md` | v0.5 full 17-task, 51-attempt Codex evaluation and precision breakdown. |
| `docs/v0.4/design.md` | v0.4 CUDA tiers, remote GPU Docker executor over SSH, and `inplace_build` source loading. |
| `docs/v0.4/experiment_report.md` | v0.4 13-task 3-repeat Codex evaluation: 84.6% resolved. |
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
  datasets/pytorch_v0.5/dataset.json
```

Run offline preflight (patches apply, test names resolve, no docker/GPU needed):

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/preflight_task.py --all
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
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.5_gold
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

Run a real isolated Codex evaluation on CPU tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --filter-tasks pytorch__149693 pytorch__147599 pytorch__160952 pytorch__162340 \
                 pytorch__163961 pytorch__168295 pytorch__161488 pytorch__150975 \
                 pytorch__124385 pytorch__143455 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_codex_cpu
```

Run GPU tasks on a remote host over SSH (requires `configs/remote_hosts.json`, see [docs/v0.4/design.md](docs/v0.4/design.md#42-远程-gpu-docker-执行器)):

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src \
  OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
  OP_BENCH_CODEX_TIMEOUT_SEC=1200 \
  python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --filter-tasks pytorch__132835 pytorch__132616 pytorch__144009 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_codex_gpu
```

Codex CLI rate-limit auto-retry is enabled by default. Override via `OP_BENCH_CODEX_RATE_LIMIT_MAX_RETRIES` (default 3) and `OP_BENCH_CODEX_RATE_LIMIT_WAIT_SEC` (default 18300 = 5h5min).

Run only specific tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --filter-tasks autograd lbfgs \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_subset
```

`scripts/run_experiment.py` prints timestamped progress logs to stderr. Add `--quiet` when only `results.jsonl` and `summary.json` are needed.

## Current Dataset

The verified [pytorch_v0.5 manifest](datasets/pytorch_v0.5/dataset.json) contains all 13 v0.4 tasks plus four newly admitted precision tasks. Deprecated tasks #129154 and #144073 are excluded. The complete run scored **37/51 (72.5%)**; the reusable [precision slice](datasets/pytorch_v0.5_precision/dataset.json) scored **13/18 (72.2%)**:

| Task | PR | Subclass | Tier | Rate |
| --- | ---: | :---: | --- | ---: |
| `pytorch__140557__layer_norm_decomp_precision` | #140557 | P2 | cpu | 0/3 |
| `pytorch__139999__masked_mean_bool_upcast` | #139999 | P1 | cpu | 3/3 |
| `pytorch__129138__linear_add_bias_autocast` | #129138 | P3 | cpu | 3/3 |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | P3 | cuda_py | 1/3 |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | P5 | cuda_kernel | 3/3 |
| `pytorch__139372__histc_int8_cuda_bounds` | #139372 | P5 | cuda_kernel | 3/3 |

Full-run metrics: patch conciseness 1.000, pass-to-pass kept rate 94.1%, regression rate 0%, and tier-weighted score 76.8%. P4 remains N/A because no P4 task passed admission. See the [v0.5 experiment report](docs/v0.5/experiment_report.md) for integrity evidence, failure analysis, and metric definitions.

Tier codes: `cpu` = `cpu_python_overlay`, `cuda_py` = `cuda_python_overlay`, `cuda_kernel` = `cuda_kernel_build`.

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

Platform development should follow the [v0.6 design](docs/v0.6/design.md), [implementation plan](docs/v0.6/implementation_plan.md), and [acceptance matrix](docs/v0.6/acceptance_matrix.md). Dataset expansion follows the existing admission workflow below and the [v0.7 design](docs/v0.7/design.md):

1. Add or curate task bundles under `tasks/<framework>/`.
2. Register reusable environment/source assets under `environments/registry.json` and `sources/registry.json`.
3. Run `scripts/preflight_task.py` to catch snapshot/patch/test-name issues offline.
4. Run `scripts/run_admission.py --write-task-evidence`.
5. Add only evidence-backed verified tasks to a verified dataset slice with `scripts/curate_dataset.py`.
6. Run `scripts/run_experiment.py` on `--verified-only`.
7. Add new real agents by implementing the action-interface boundary rather than giving them direct access to target workspaces.

## References

- [Docs index](docs/README.md)
- [Global project plan](docs/project_plan.md)
- [Current project state](docs/project_state.md)
- [v0.6 platform design](docs/v0.6/design.md)
- [v0.6 implementation plan](docs/v0.6/implementation_plan.md)
- [v0.6 acceptance matrix](docs/v0.6/acceptance_matrix.md)
- [v0.7 Dataset Factory and Boundary design](docs/v0.7/design.md)
- [v0.5 design](docs/v0.5/design.md)
- [v0.5 experiment report](docs/v0.5/experiment_report.md)
- [v0.4 design](docs/v0.4/design.md)
- [v0.4 experiment report](docs/v0.4/experiment_report.md)
- [v0.3 design](docs/v0.3/design.md)
- [v0.3 experiment report](docs/v0.3/experiment_report.md)
- [v0.2 developer guide](docs/v0.2/developer_guide.md)
- [v0.2 experiment report](docs/v0.2/experiment_report.md)
- [v0.1 developer guide](docs/v0.1/developer_guide.md)
- [v0.1 manual validation workflow](docs/v0.1/manual_validation.md)
