# OpBench

语言：[English](README.md) | 中文

OpBench 是一个面向算子问题的 benchmark，用于评测 coding agent 解决真实框架 issue 的能力。它借鉴 SWE-bench 的真实仓库修复思路，但把运行环境作为每条任务的一部分，因为算子问题经常依赖框架版本、Python 包布局、设备可用性、数值行为和后端选择。

v0.1 已经建立隔离 replay/evaluation 闭环。v0.2 已完成数据集扩展所需的平台能力：资产 registry、正式 admission evidence、dataset curation，以及容器和 cache 管理。v0.3 将数据集扩展到 10 条 verified task，覆盖 5 个 PyTorch 子系统，新增 patch scope enforcement、public/hidden test 分层、multi-file overlay 和 3 次重复稳定性评测。当前 `pytorch_v0.3` 切片 Codex CLI resolved rate 为 76.7%。

## 当前代码包含什么

- 两层数据集模型：`datasets/<slice>/dataset.json` 指向 `tasks/` 下的 task bundle。
- 真实 PyTorch task bundle，包含 PR/issue 元数据、hidden test patch、gold patch、source snapshot 和环境声明。
- 位于 `environments/pytorch-cpu/` 的可复用 CPU Docker 环境。
- 位于 `environments/registry.json` 和 `sources/registry.json` 的环境与源码资产 registry。
- 通过 `scripts/run_admission.py` 生成正式 admission evidence。
- 通过 `scripts/curate_dataset.py` 生成 verified-only 数据集切片。
- 通过 `scripts/inspect_assets.py` 和 `scripts/manage_containers.py` 检查资产和管理容器。
- Replay evaluator，用于检查 baseline failure、gold success、agent patch success 和 pass-to-pass 回归。
- 标准 workspace action interface，覆盖文件操作、patch 应用、命令执行、测试执行和 diff 导出。
- `codex_action_bridge` 作为参考真实 agent adapter。Codex 在 host 侧 scratch workspace 中运行，只能通过 OpBench actions 操作目标仓库。

开发阶段的临时模型直连 adapter 已从 v0.1 公开表面移除。后续接入新 agent 时，应复用 `codex_action_bridge` 所验证的 action-interface 边界。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `datasets/` | 数据集 manifest，用于选择实验 task bundle。 |
| `tasks/` | 单条 benchmark 任务，包含 issue 文本、patch、环境元数据和测试列表。 |
| `environments/` | task 引用的 Docker 环境 artifact 和环境 registry。 |
| `sources/` | source snapshot registry 元数据。 |
| `src/op_bench/` | 核心实现：task model、环境准备、evaluator、actions、agent bridge、reporting。 |
| `scripts/` | 校验、环境准备、source snapshot、replay、实验运行等 CLI 入口。 |
| `docs/` | 按版本归档的设计、实验报告、开发指南和历史记录。 |
| `docs/v0.3/design.md` | v0.3 数据扩展、multi-file overlay、public/hidden test 分层和 CUDA 试点设计。 |
| `docs/v0.2/developer_guide.md` | v0.2 registry、admission、curation、资产和容器管理流程。 |
| `docs/v0.1/developer_guide.md` | v0.1 模块级架构和开发指南。 |

## 快速开始

创建项目 Python 环境：

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version
```

重建 source snapshot（新设备 clone 后必须执行）：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/setup_sources.py
```

运行单元测试：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

校验当前数据集：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.3/dataset.json
```

运行第一条已准入 PyTorch 任务的正式 admission：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_admission.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output-dir runs/admission/pytorch__149693__lazylinear_init/manual \
  --write-task-evidence
```

运行当前数据集的 gold 闭环检查：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.3_gold
```

检查已登记资产：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py
```

从当前混合数据集生成 verified-only 数据集切片：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/curate_dataset.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --output-dataset datasets/pytorch_mini_v0.2/dataset.json \
  --output-summary datasets/pytorch_mini_v0.2/summary.json \
  --verified-only \
  --dataset-id pytorch_mini_v0.2 \
  --version v0.2
```

运行真实隔离 Codex CLI 评测：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_codex_r3
```

只运行指定 task：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --filter-tasks autograd lbfgs \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_subset
```

`scripts/run_experiment.py` 默认向 stderr 输出带时间戳的进度日志。若只需要 `results.jsonl` 和 `summary.json`，可添加 `--quiet`。

## 当前数据集

主数据集是 [datasets/pytorch_v0.3](datasets/pytorch_v0.3/dataset.json)（10 条 verified task）。

| Task | PR | 组件 | 通过率 (3 次重复) |
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

正式 benchmark 运行请使用 `--verified-only`，只统计已准入任务。使用 `--filter-tasks` 可只运行子集。

每条 verified task 都有 task-local stable admission evidence，位于对应 task 目录的 `admission/evidence.json`。

## 运行边界

Agent 身份、模型调用和控制逻辑运行在 host 侧。每次修复 attempt 会获得一个准备好的 workspace 和 task-scoped Docker runtime。Agent 必须通过 OpBench actions 操作目标仓库：

- `read_file`
- `write_file`
- `apply_patch`
- `run_command`
- `run_test`
- `git_diff`

对于 Docker task，preflight、setup、测试命令和 action-interface 命令都会在同一个 task container 中执行。这样可以保留 setup 状态，并避免把 host 执行和 benchmark 执行混在一起。

`codex_action_bridge` 是该边界的参考实现。它只给 Codex 一个 scratch directory 和 `opbench_action.py` CLI，不暴露目标仓库路径；最终评分使用通过 action interface 导出的 patch。

## 继续扩展

修改内部实现前先阅读 [docs/v0.2/developer_guide.md](docs/v0.2/developer_guide.md) 和 [docs/v0.3/design.md](docs/v0.3/design.md)。通常扩展路径是：

1. 在 `tasks/<framework>/` 下新增或整理 task bundle。
2. 在 `environments/registry.json` 和 `sources/registry.json` 中登记可复用环境和源码资产。
3. 运行 `scripts/run_admission.py --write-task-evidence`。
4. 用 `scripts/curate_dataset.py` 只把有 evidence 的 verified task 写入正式切片。
5. 用 `scripts/run_experiment.py --verified-only` 运行实验。
6. 新增真实 agent 时，实现 action-interface 边界，而不是让 agent 直接访问目标 workspace。

## 参考文档

- [文档索引](docs/README.zh-CN.md)
- [v0.4 设计方案](docs/v0.4/design.md)
- [v0.3 设计方案](docs/v0.3/design.md)
- [v0.3 实验报告](docs/v0.3/experiment_report.md)
- [v0.2 开发者指南](docs/v0.2/developer_guide.md)
- [v0.2 实验报告](docs/v0.2/experiment_report.md)
- [v0.1 开发者指南](docs/v0.1/developer_guide.md)
- [v0.1 手动验证流程](docs/v0.1/manual_validation.md)
