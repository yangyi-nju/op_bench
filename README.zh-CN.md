# OpBench

语言：[English](README.md) | 中文

OpBench 是一个面向算子问题的 benchmark，用于评测 coding agent 解决真实框架 issue 的能力。它借鉴 SWE-bench 的真实仓库修复思路，但把运行环境作为每条任务的一部分，因为算子问题经常依赖框架版本、Python 包布局、设备可用性、数值行为和后端选择。

v0.1 刻意保持小规模：一条已验证的 PyTorch CPU 任务、一套 Docker-backed replay/evaluation 闭环，以及一条通过 Codex CLI 跑通的真实隔离 agent 路径。

## v0.1 包含什么

- 两层数据集模型：`datasets/<slice>/dataset.json` 指向 `tasks/` 下的 task bundle。
- 真实 PyTorch task bundle，包含 PR/issue 元数据、hidden test patch、gold patch、source snapshot 和环境声明。
- 位于 `environments/pytorch-cpu/` 的可复用 CPU Docker 环境。
- Replay evaluator，用于检查 baseline failure、gold success、agent patch success 和 pass-to-pass 回归。
- 标准 workspace action interface，覆盖文件操作、patch 应用、命令执行、测试执行和 diff 导出。
- `codex_action_bridge` 作为参考真实 agent adapter。Codex 在 host 侧 scratch workspace 中运行，只能通过 OpBench actions 操作目标仓库。

开发阶段的临时模型直连 adapter 已从 v0.1 公开表面移除。后续接入新 agent 时，应复用 `codex_action_bridge` 所验证的 action-interface 边界。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `datasets/` | 数据集 manifest，用于选择实验 task bundle。 |
| `tasks/` | 单条 benchmark 任务，包含 issue 文本、patch、环境元数据和测试列表。 |
| `environments/` | task 引用的 Docker 环境 artifact。 |
| `src/op_bench/` | 核心实现：task model、环境准备、evaluator、actions、agent bridge、reporting。 |
| `scripts/` | 校验、环境准备、source snapshot、replay、实验运行等 CLI 入口。 |
| `docs/developer_guide.md` | 模块级架构和开发指南。 |
| `docs/manual_validation.md` | 将 task 从 draft 晋升为 verified 的手动验证流程。 |
| `docs/OpBench_v0.1_experiment_report.md` | v0.1 实验证据和结果分析。 |

## 快速开始

创建项目 Python 环境：

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version
```

运行单元测试：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

校验当前 mini dataset：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

验证第一条已准入 PyTorch 任务的 replay：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

运行真实隔离 Codex CLI 评测：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```

`scripts/run_experiment.py` 默认向 stderr 输出带时间戳的进度日志。若只需要 `results.jsonl` 和 `summary.json`，可添加 `--quiet`。

## 当前数据集

第一个数据集切片是 [datasets/pytorch_mini](datasets/pytorch_mini/README.zh-CN.md)。

| Task | PR | 状态 |
| --- | --- | --- |
| `pytorch__149693__lazylinear_init` | https://github.com/pytorch/pytorch/pull/149693 | verified |
| `pytorch__160952__bilinear_lazy_check` | https://github.com/pytorch/pytorch/pull/160952 | draft |

正式 benchmark 运行请使用 `--verified-only`，只统计已准入任务。

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

修改内部实现前先阅读 [docs/developer_guide.md](docs/developer_guide.md)。通常扩展路径是：

1. 在 `tasks/<framework>/` 下新增或整理 task bundle。
2. 在 `datasets/` 下的数据集 manifest 中登记 task。
3. 在 `environments/` 下准备或 pin 任务环境。
4. 用 `scripts/verify_task_replay.py` 验证 baseline/gold replay。
5. 用 `scripts/run_experiment.py --verified-only` 运行实验。
6. 新增真实 agent 时，实现 action-interface 边界，而不是让 agent 直接访问目标 workspace。

## 参考文档

- [开发者指南](docs/developer_guide.md)
- [手动验证流程](docs/manual_validation.md)
- [v0.1 实验报告](docs/OpBench_v0.1_experiment_report.md)
- [数据构建流程](docs/builder_workflow.md)
