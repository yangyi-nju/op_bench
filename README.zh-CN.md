# OpBench

语言：[English](README.md) | 中文

OpBench 是一个面向算子问题的 benchmark，用于评测 coding agent 解决真实框架 issue 的能力。它借鉴 SWE-bench 的真实仓库修复思路，但把运行环境作为每条任务的一部分，因为算子问题经常依赖框架版本、Python 包布局、设备可用性、数值行为和后端选择。

v0.1 建立隔离 replay/evaluation 闭环，v0.2 加入资产 registry 和正式 admission，v0.3 扩展到 10 条 verified task 并加入 3-repeat，v0.4 加入 CUDA tier 和 remote Docker。v0.5 现已完成：verified 累计数据集包含 17 条 task，其中 precision slice 为 6 条；51-attempt Codex 全量实验达到 **72.5% resolved**（37/51），并落地 8 维指标与实验完整性硬校验。

v0.6 平台实现已在本地完成 M1～M7：严格版本化合同、唯一 Authoritative Workspace 与不可变 Patch、服务端权威 CLI/MCP Action Service、确定性的 Attempt/Trajectory/Evaluation/Artifact 语义、版本化 Runtime Profile、精确 Attempt-owned Local/Docker/Remote 资源、Conformance 与 Legacy Replay、支持 Resume/Integrity 的进程隔离 Canonical Codex Adapter，以及可执行的公开 Demo 和文档入口均已落地。真实 Codex 本地 CPU canary 已通过；正式 `opbench-v0.6.0` 发布仍是 **Blocked** 而不是 Completed，因为已配置目标持续 `connection_timeout`，精确 Remote Replay 与 CUDA Must 证据尚不可得。v0.6 发布门关闭后才进入 v0.7 Boundary Task 扩充。详见[全局项目方案](docs/project_plan.md)、[当前项目状态](docs/project_state.md)和[v0.6 发布说明](docs/v0.6/release_notes.md)。

## 当前代码包含什么

- 两层数据集模型：`datasets/<slice>/dataset.json` 指向 `tasks/` 下的 task bundle。
- 真实 PyTorch task bundle，包含 PR/issue 元数据、hidden test patch、gold patch、source snapshot 和环境声明。
- 可复用 Docker 环境：`environments/pytorch-cpu/`（CPU）、`environments/pytorch-cuda/`（CUDA Python overlay）、`environments/pytorch-cuda-devel/`（CUDA + nvcc/ccache/cmake，用于 kernel_build）。
- 位于 `environments/registry.json` 和 `sources/registry.json` 的环境与源码资产 registry。
- 三种 runtime tier：`cpu_python_overlay`、`cuda_python_overlay`、`cuda_kernel_build`（后两者走 `remote_docker` backend，通过 SSH）。
- 通过 `scripts/run_admission.py` 生成正式 admission evidence，`scripts/preflight_task.py` 做离线预检。
- 通过 `scripts/curate_dataset.py` 生成 verified-only 数据集切片。
- 通过 `scripts/inspect_assets.py` 和 `scripts/manage_containers.py` 检查资产和管理容器。
- Replay evaluator，用于检查 baseline failure、gold success、agent patch success 和 pass-to-pass 回归；支持 `python_overlay` 和 `inplace_build` 两种 source loading 模式。
- 远程 GPU Docker 执行器（`src/op_bench/remote.py`）：在 SSH 主机上运行 `docker`，rsync 双向同步 workspace，跨 attempt 复用 ccache/build 缓存。
- 标准 workspace action interface，覆盖文件操作、patch 应用、命令执行、测试执行和 diff 导出。
- `codex_action_bridge` 作为参考真实 agent adapter，包含 rate-limit 自动重试。Codex 在 host 侧 scratch workspace 中运行，只能通过 OpBench actions 操作目标仓库。
- `src/op_bench/runtime/` 下的严格 v0.6 Runtime Contract 和 Canonical SHA-256 身份，包括确定性 RunManifest、Cohort ID、Attempt ID 与冻结的 task × agent × repeat 矩阵。
- 显式 FullTaskSpec → AgentTaskView 公开白名单，递归拒绝答案来源、凭据、私有输出和本机路径；每个投影视图进入 Manifest 与 Attempt Identity。
- 与本机路径无关的 Authoritative Workspace Identity、受限 regular-file 原子写入、symlink/特殊文件拒绝、确定性 add/modify/delete/empty patch、并发 Freeze 收敛和干净 base 严格应用校验。
- Canonical Action Service 覆盖 list/search/read/write/apply-patch、policy-bound command、registry-bound test、diff 与 finish；CLI/MCP 共用唯一执行权威，标准 Adapter 只得到已扫描 LaunchInput 和 JSON-only action client。
- AttemptSession 状态机统一执行服务端 deadline/resource budget、固定终止优先级、在途 Action/事件发布屏障、唯一 Patch Freeze 与唯一 Terminal SessionResult。
- Canonical append-only EventJournal 提供原子 Action 事件批次、连续 hash chain、Public Artifact spill 和严格描述符绑定持久化；Evaluation-aware AttemptLedger 以最终 Evaluation Result 和 append-only retry 历史提供确定性 resume 决策。
- Fresh Evaluation 从校验过的本地 Source 副本执行，严格应用 patch，在 Session 终止后注入 evaluation-only tests，并记录 F2P/P2P 与 validity/terminal/outcome 三轴证据。
- 描述符绑定的 public/private attempt Artifact、只读 12 项引用图完整性检查、篡改检测，以及 byte-exact `results.jsonl`/`summary.json` 确定性重建。
- 零依赖独立 JSON Schema 校验器、`schemas/` 下的严格 Schema，以及不会启动 Agent 或连接 Runtime 的离线构建/校验 CLI。

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
| `schemas/` | 严格的 v0.6 Runtime wire-contract JSON Schema。 |
| `configs/examples/` | 公开合成的 v0.6 配置和 Manifest 示例。 |
| `docs/` | 按版本归档的设计、实验报告、开发指南和历史记录。 |
| `docs/project_plan.md` | 全局使命、原则、路线图、发布门和研究目标。 |
| `docs/project_state.md` | 当前基线、开发版本、已确认决策和下一动作。 |
| `docs/v0.6/` | v0.6 规范 Agent 评测平台设计、实施计划和验收矩阵。 |
| `docs/v0.7/design.md` | v0.7 Dataset Factory、Boundary Slice 与 matched-runtime 恢复设计。 |
| `docs/v0.5/design.md` | v0.5 问题维度分类和扩展评测指标。 |
| `docs/v0.5/experiment_report.md` | v0.5 全量 17-task、51-attempt Codex 评测和 precision 拆解。 |
| `docs/v0.4/design.md` | v0.4 CUDA tier、远程 GPU Docker SSH 执行器、`inplace_build` 源码加载。 |
| `docs/v0.4/experiment_report.md` | v0.4 13 task × 3 repeat Codex 评测：84.6% resolved。 |
| `docs/v0.3/design.md` | v0.3 数据扩展、multi-file overlay、public/hidden test 分层和 CUDA 试点设计。 |
| `docs/v0.2/developer_guide.md` | v0.2 registry、admission、curation、资产和容器管理流程。 |
| `docs/v0.1/developer_guide.md` | v0.1 模块级架构和开发指南。 |

## 快速开始

OpBench v0.6 没有第三方 Python 依赖。先创建干净环境，运行全量测试并校验冻结的
v0.5 Dataset：

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version

PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover \
  -s tests -p 'test_*.py'

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json --require-verified
```

离线构建并校验 v0.6 RunManifest；这不会启动 Agent 或连接 Runtime：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/build_run_manifest.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --output /tmp/opbench-v0.6-manifest.json \
  --agent example-agent \
  --model example-model \
  --adapter canonical-cli-v1 \
  --repeat 1 \
  --created-at 2026-07-18T00:00:00Z

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  /tmp/opbench-v0.6-manifest.json
```

### 离线 v1 Scripted Demo

先生成 synthetic 本地输入，再通过正式 v1 Orchestrator 和
`local-cpu-process-v1` Runtime Profile 运行：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_v0_6_demo.py \
  --output-dir runs/v0.6_m7_demo_input

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m7_demo_input/dataset/dataset.json \
  --verified-only \
  --agent scripted_canonical \
  --agent-repeat 1 \
  --output-dir runs/v0.6_m7_scripted_demo \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1
```

原样再次运行 `run_experiment.py` 命令即可验证 resume。第一次输出
`ran=1, skipped=0`，第二次输出 `ran=0, skipped=1`，且不会改写已选中的
Artifact。随后校验冻结合同和当前 Attempt 精确持有资源的清理状态：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  runs/v0.6_m7_scripted_demo/run_manifest.json

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_runtime_resources.py \
  --run-root runs/v0.6_m7_scripted_demo
```

每次成功的 v1 运行都会执行 Integrity 校验，并从 canonical Artifact 确定性重建
`results.jsonl` 与 `summary.json`。这个 synthetic Demo 只验证控制器和 Artifact
生命周期，不是 benchmark score，也不衡量修复能力。

### 可选的真实 Codex 本地 canary

Canonical Codex Adapter 使用同一个 Demo Dataset、Runtime、Action Service、Evaluator
和 Artifact 路径。下面的可选命令会调用本地已配置的 Codex CLI，并可能使用其正常的
OpenAI 网络访问：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m7_demo_input/dataset/dataset.json \
  --verified-only \
  --agent codex_canonical \
  --agent-repeat 1 \
  --output-dir runs/v0.6_m7_codex_demo \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1 \
  --enable-external-canary
```

这是 Runtime/Adapter canary，不是 benchmark score。M6 已记录一个有效的真实 Codex
本地 Attempt 和一个双重复 resume Cohort。精确 Remote CPU、CUDA Overlay、CUDA Kernel
以及 17+17+51 Replay 证据仍因已配置目标稳定 `connection_timeout` 保持 **Blocked**；
OpBench 不探测或发现替代目标。

### Legacy v0.5 兼容路径

迁移期 `Legacy` 仍是 `default` protocol；下面的 Legacy 命令刻意 `omit`
`--runtime-protocol`，v1 专属参数若误用于 Legacy 会被拒绝而不是静默忽略。新 clone
在运行真实 v0.5 task 前需要重建 source snapshot：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/setup_sources.py

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.5_gold
```

Runtime 支持状态、Artifact layout、失败归因、Comparability Key、Resume、Replay 和精确
目标配置详见 [v0.6 开发者指南](docs/v0.6/developer_guide.md)。

## 当前数据集

正式 [pytorch_v0.5 manifest](datasets/pytorch_v0.5/dataset.json) 已 verified：v0.4 的 13 条全部保留，另加入 4 条新 precision task；deprecated 的 #129154 和 #144073 不进入清单。全量结果为 **37/51（72.5%）**，可复用的 [precision slice](datasets/pytorch_v0.5_precision/dataset.json) 为 **13/18（72.2%）**：

| Task | PR | 子类 | Tier | 通过率 |
| --- | ---: | :---: | --- | ---: |
| `pytorch__140557__layer_norm_decomp_precision` | #140557 | P2 | cpu | 0/3 |
| `pytorch__139999__masked_mean_bool_upcast` | #139999 | P1 | cpu | 3/3 |
| `pytorch__129138__linear_add_bias_autocast` | #129138 | P3 | cpu | 3/3 |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | P3 | cuda_py | 1/3 |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | P5 | cuda_kernel | 3/3 |
| `pytorch__139372__histc_int8_cuda_bounds` | #139372 | P5 | cuda_kernel | 3/3 |

全量指标：patch conciseness 1.000，pass-to-pass kept rate 94.1%，regression rate 0%，tier-weighted score 76.8%。P4 尚无通过 admission 的 task，因此保持 N/A。完整性证据、失败分析和指标定义见 [v0.5 实验报告](docs/v0.5/experiment_report.md)。

Tier 简写：`cpu` = `cpu_python_overlay`，`cuda_py` = `cuda_python_overlay`，`cuda_kernel` = `cuda_kernel_build`。

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

平台开发按 [v0.6 设计](docs/v0.6/design.md)、[实施计划](docs/v0.6/implementation_plan.md)和[验收矩阵](docs/v0.6/acceptance_matrix.md)推进。数据扩充继续遵循下列 Admission 流程和 [v0.7 设计](docs/v0.7/design.md)：

1. 在 `tasks/<framework>/` 下新增或整理 task bundle。
2. 在 `environments/registry.json` 和 `sources/registry.json` 中登记可复用环境和源码资产。
3. 运行 `scripts/preflight_task.py` 做离线预检（快速排查 snapshot/patch/test 名称问题）。
4. 运行 `scripts/run_admission.py --write-task-evidence`。
5. 用 `scripts/curate_dataset.py` 只把有 evidence 的 verified task 写入正式切片。
6. 用 `scripts/run_experiment.py --verified-only` 运行实验。
7. 新增真实 agent 时，实现 action-interface 边界，而不是让 agent 直接访问目标 workspace。

## 参考文档

- [文档索引](docs/README.zh-CN.md)
- [全局项目方案](docs/project_plan.md)
- [当前项目状态](docs/project_state.md)
- [v0.6 平台设计](docs/v0.6/design.md)
- [v0.6 实施计划](docs/v0.6/implementation_plan.md)
- [v0.6 验收矩阵](docs/v0.6/acceptance_matrix.md)
- [v0.7 Dataset Factory 与 Boundary 设计](docs/v0.7/design.md)
- [v0.5 设计方案](docs/v0.5/design.md)
- [v0.5 实验报告](docs/v0.5/experiment_report.md)
- [v0.4 设计方案](docs/v0.4/design.md)
- [v0.4 实验报告](docs/v0.4/experiment_report.md)
- [v0.3 设计方案](docs/v0.3/design.md)
- [v0.3 实验报告](docs/v0.3/experiment_report.md)
- [v0.2 开发者指南](docs/v0.2/developer_guide.md)
- [v0.2 实验报告](docs/v0.2/experiment_report.md)
- [v0.1 开发者指南](docs/v0.1/developer_guide.md)
- [v0.1 手动验证流程](docs/v0.1/manual_validation.md)
