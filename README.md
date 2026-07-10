# OpBench

Language: English | [中文](README.zh-CN.md)

OpBench is an operator-focused benchmark for evaluating coding agents on real framework issues. It follows the SWE-bench idea of repairing real repository snapshots, but treats the runtime environment as part of each task because operator bugs often depend on framework version, Python package layout, device availability, numerical behavior, and backend selection.

v0.1 established the isolated replay/evaluation loop. v0.2 completed the platform pieces needed to scale the dataset: asset registries, formal admission evidence, dataset curation, and container/cache management. v0.3 expanded the dataset to 10 verified tasks across 5 PyTorch subsystems, added patch scope enforcement, public/hidden test separation, multi-file overlay, and 3-repeat stability evaluation (76.7% resolved with Codex CLI). v0.4 added two CUDA runtime tiers (`cuda_python_overlay`, `cuda_kernel_build`), a remote-GPU Docker executor over SSH, and `inplace_build` source loading. The current `pytorch_v0.4` slice contains 13 verified tasks and reached **84.6% resolved rate** with Codex CLI (33/39). Multi-agent comparison with Claude Code is deferred to v0.5.

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
  datasets/pytorch_v0.4/dataset.json
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
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.4_gold
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

The primary dataset is [datasets/pytorch_v0.4](datasets/pytorch_v0.4/dataset.json) (13 verified tasks, Codex CLI 3-repeat resolved rate **84.6%**, 33/39).

| Task | PR | Tier | Component | Rate |
| --- | --- | --- | --- | --- |
| `pytorch__168295__autograd_create_graph` | #168295 | cpu | torch.autograd | 3/3 |
| `pytorch__150975__autograd_backward_inputs` | #150975 | cpu | torch.autograd | 3/3 |
| `pytorch__161488__lbfgs_wolfe` | #161488 | cpu | torch.optim | 3/3 |
| `pytorch__124385__load_state_dict_prefix` | #124385 | cpu | torch.nn.Module | 3/3 |
| `pytorch__149693__lazylinear_init` | #149693 | cpu | torch.nn.LazyLinear | 3/3 |
| `pytorch__147599__lazylinear_state_forward` | #147599 | cpu | torch.nn.LazyLinear | 3/3 |
| `pytorch__160952__bilinear_lazy_check` | #160952 | cpu | torch.nn.Bilinear | 3/3 |
| `pytorch__143455__set_submodule` | #143455 | cpu | torch.nn.Module | 3/3 |
| `pytorch__162340__nn_arg_length` | #162340 | cpu | torch.nn.conv/utils | 0/3 |
| `pytorch__163961__dataloader_subset` | #163961 | cpu | torch.utils.data | 0/3 |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | cuda_py | torch.nested._internal.sdpa | 3/3 |
| `pytorch__132616__cuda_mem_get_info` | #132616 | cuda_py | torch.cuda.memory | 3/3 |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | cuda_kernel | aten.native.cuda.softmax | 3/3 |

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

Read [docs/v0.2/developer_guide.md](docs/v0.2/developer_guide.md), [docs/v0.3/design.md](docs/v0.3/design.md), and [docs/v0.4/design.md](docs/v0.4/design.md) for the current expansion workflow. The usual path is:

1. Add or curate task bundles under `tasks/<framework>/`.
2. Register reusable environment/source assets under `environments/registry.json` and `sources/registry.json`.
3. Run `scripts/preflight_task.py` to catch snapshot/patch/test-name issues offline.
4. Run `scripts/run_admission.py --write-task-evidence`.
5. Add only evidence-backed verified tasks to a verified dataset slice with `scripts/curate_dataset.py`.
6. Run `scripts/run_experiment.py` on `--verified-only`.
7. Add new real agents by implementing the action-interface boundary rather than giving them direct access to target workspaces.

## References

- [Docs index](docs/README.md)
- [v0.4 design](docs/v0.4/design.md)
- [v0.4 experiment report](docs/v0.4/experiment_report.md)
- [v0.3 design](docs/v0.3/design.md)
- [v0.3 experiment report](docs/v0.3/experiment_report.md)
- [v0.2 developer guide](docs/v0.2/developer_guide.md)
- [v0.2 experiment report](docs/v0.2/experiment_report.md)
- [v0.1 developer guide](docs/v0.1/developer_guide.md)
- [v0.1 manual validation workflow](docs/v0.1/manual_validation.md)
