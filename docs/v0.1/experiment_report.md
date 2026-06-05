# OpBench v0.1 实验报告

日期：2026-05-23

## 1. 实验目标

v0.1 的目标不是扩大数据集规模，而是验证一个完整、可复现的算子 benchmark 闭环：

1. 使用真实社区 PR 构建小型数据集。
2. 将运行环境作为 task manifest 的一部分，而不是依赖开发者口头安装依赖。
3. 在声明的 Docker 环境中复现 fail-to-pass，并用 gold patch 证明任务可判分。
4. 让真实 agent 通过 OpBench action interface 修改隔离容器内的任务仓库。
5. 输出结构化结果、patch、action log，方便复查和后续扩展。

## 2. 当前系统组成

核心模块：

- `src/op_bench/task.py`：读取单条任务 manifest，解析源码、环境、测试、patch 等字段。
- `src/op_bench/dataset.py`：读取 dataset manifest，支持两层数据集结构：dataset 汇总多条 task bundle。
- `src/op_bench/environment.py`：准备任务环境。Docker task 会检查镜像、启动 task-scoped 容器、运行 preflight，并在结束后清理容器。
- `src/op_bench/evaluator.py`：执行 baseline、gold、agent patch 判分。判分标准为 fail-to-pass 修复且 pass-to-pass 不回归。
- `src/op_bench/actions.py`：标准 action interface，支持读取文件、写入文件、应用 patch、运行命令、运行测试和导出 diff。
- `src/op_bench/action_bridge.py`：给 Codex CLI 使用的文件队列 action bridge。Codex 不直接进入目标仓库，只能通过 `opbench_action.py` 请求 action。
- `src/op_bench/agents.py`：agent adapter，包括 `gold` 和 `codex_action_bridge`。其中 `codex_action_bridge` 是 v0.1 的正式真实 agent 路径。
- `scripts/run_experiment.py`：实验入口，负责 baseline gating、agent 多次尝试、patch 收集、结果聚合。
- `scripts/validate_dataset.py`、`scripts/validate_task.py`、`scripts/verify_task_replay.py`：数据集和任务验证工具。

v0.1 的正式真实 agent 闭环使用 `codex_action_bridge`。旧的 host-workspace 直连 Codex 探索路径已清理，不作为正式评测入口。

## 3. 数据集

数据集 manifest：`datasets/pytorch_mini/dataset.json`

当前包含 2 个真实 PyTorch PR：

| task_id | PR | Issue | 状态 |
| --- | --- | --- | --- |
| `pytorch__149693__lazylinear_init` | https://github.com/pytorch/pytorch/pull/149693 | https://github.com/pytorch/pytorch/issues/149691 | verified |
| `pytorch__160952__bilinear_lazy_check` | https://github.com/pytorch/pytorch/pull/160952 | https://github.com/pytorch/pytorch/issues/160407 | draft |

本次 v0.1 实验只对 `--verified-only` 子集运行，因此只评测 `pytorch__149693__lazylinear_init`。

`pytorch__149693__lazylinear_init` 的环境与源码策略：

- Docker image：`op-bench/pytorch-cpu:torch2.6.0-py311`
- source snapshot：本地缓存的 PyTorch full repo snapshot，避免每次实验依赖 GitHub 大仓库在线拉取。
- source loading：`python_overlay`，只将 agent 修改过的 Python 文件同步到容器内已安装的 PyTorch wheel overlay 路径。
- overlay 文件：`torch/nn/modules/linear.py`
- fail-to-pass：`TestLazyModules.test_lazy_linear_reset_uses_inferred_in_features`
- pass-to-pass：`TestLazyModules.test_linear_state`

这个设计解决了 PyTorch full repo 的两个问题：agent 可以按真实仓库上下文读 PyTorch superproject 中的相关 Python 源码；测试运行又不需要从源码完整构建 PyTorch。

需要注意的是，v0.1 的 source snapshot 不保证包含完整 git submodule 内容，部分 `third_party/` 目录可能只是 submodule 占位目录。该限制不影响当前 verified task，因为它只覆盖 Python-level overlay 文件 `torch/nn/modules/linear.py`，运行时依赖 Docker image 中已安装的 PyTorch wheel。对于 C++/CUDA/build/submodule 相关任务，当前 snapshot 策略不能直接视为完整源码环境，必须额外准备完整 submodule 或采用 full source build 路线。

## 4. 实验过程

### 4.1 数据集结构验证

命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

结果：dataset 结构有效，共 2 条 task entry，其中 1 条 verified。

### 4.2 fail-to-pass / gold replay

命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693_overlay.json
```

结果：

| mode | status | fail-to-pass | pass-to-pass | duration |
| --- | --- | --- | --- | --- |
| baseline | `baseline_reproduced` | 0/1 | 1/1 | 42.10s |
| gold | `resolved` | 1/1 | 1/1 | 42.42s |

解释：

- baseline 能稳定复现目标失败，说明 task admissible。
- gold patch 能修复 fail-to-pass 且不破坏 pass-to-pass，说明该任务可以自动判分。

### 4.3 真实 agent 评测

命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge-timeout1200
```

结果：

| agent | task total | resolved | resolved rate | fail-to-pass | pass-to-pass |
| --- | ---: | ---: | ---: | --- | --- |
| `codex_action_bridge` | 1 | 1 | 1.0 | 1/1 | 1/1 |

关键 metadata：

- `runtime_boundary = op_bench_action_interface_file_cli`
- `shell_boundary = workspace_write_scratch_only`
- `action_count = 6`
- `integrity_status = clean`
- `timed_out = false`
- `timeout_sec = 1200`

Codex 提交的 patch：

```diff
diff --git a/torch/nn/modules/linear.py b/torch/nn/modules/linear.py
@@ -285,6 +285,7 @@ class LazyLinear(LazyModuleMixin, Linear):
     def initialize_parameters(self, input) -> None:  # type: ignore[override]
         if self.has_uninitialized_params():
             with torch.no_grad():
+                self.in_features = input.shape[-1]
                 self.weight.materialize((self.out_features, input.shape[-1]))
                 if self.bias is not None:
                     self.bias.materialize((self.out_features,))
```

该 patch 与问题语义一致：LazyLinear 在 materialize weight 前先从输入 shape 设置 `in_features`，避免 reset_parameters 仍看到 `in_features == 0`。

## 5. 结果分析

v0.1 闭环已经验证以下能力：

- 环境是 task 的一部分：任务声明 Docker image、preflight、source loading 策略和硬件层级。
- full repo 可用：agent 看到的是 PyTorch full repo snapshot，而不是手写最小样例。
- 判分隔离：测试和命令通过 task-scoped Docker 容器执行。
- 真实 agent 可跑通：Codex CLI 没有直接编辑目标仓库，而是通过 OpBench action bridge 操作容器内工作区。
- 证据可审计：结果包含 baseline/gold 记录、agent patch、action log、环境命令和完整 summary。

这次实验也暴露并修正了几个关键工程点：

- 直接在线 clone PyTorch 大仓库不稳定，因此 v0.1 使用 source snapshot，后续需要标准化 snapshot 构建和缓存管理。
- PyTorch preflight 不能在未构建的源码仓库根目录执行 `import torch`，否则会错误导入 source tree；v0.1 已支持 `environment.preflight_workdir`，默认在 `/tmp` 检查镜像内 wheel。
- 普通 Codex CLI 不能直接作为 Docker task 的正式分数，因为它会在 host workspace 中运行；v0.1 的正式路径改为 `codex_action_bridge`。
- 短 timeout 容易误判真实 agent，当前 Codex 默认 timeout 调整为 1200 秒。

## 6. 问题总结

v0.1 开发和验证过程中，主要问题集中在环境可复现、源码加载、agent 隔离和项目可理解性四类。下面按问题、原因、解决办法和效果归纳。

| 问题 | 原因 | 解决办法 | 效果 |
| --- | --- | --- | --- |
| PyTorch 大仓库在线 clone/checkout 不稳定 | PyTorch 仓库体积大，网络抖动会导致 fetch 或 checkout 失败；如果每次 replay 都依赖在线 clone，会把网络问题混入 benchmark 结果 | 引入 source snapshot，把 full repo 预先准备到 `.op_bench_cache/sources/...`，replay 时复制本地 snapshot | replay 热路径不再依赖 GitHub 网络稳定性，baseline/gold/agent 评测更可复现 |
| source snapshot 不包含完整 submodule | PyTorch 的 `third_party/` 依赖大量 git submodule；当前 v0.1 snapshot 主要服务 Python-level overlay，没有初始化所有 submodule | 将 v0.1 verified task 明确限制为不依赖 submodule 的 Python-level 修改；文档中标明 C++/CUDA/build 任务必须另行准备完整 submodule 或 full source build | 当前 `pytorch__149693__lazylinear_init` 结论不受影响；后续扩展不会误把该 snapshot 当作完整 build-ready 源码树 |
| PyTorch preflight 错误导入未构建源码树 | 如果在 mounted repo root 下执行 `import torch`，Python 会优先看到源码目录，而不是 Docker image 中的 PyTorch wheel | 增加 `environment.preflight_workdir`，preflight 默认从 `/tmp` 执行 | preflight 检查的是镜像内真实运行环境，避免把未构建源码误判为环境不可用 |
| full repo 上下文与低成本测试存在冲突 | agent 需要完整 PyTorch 源码上下文；但从源码完整构建 PyTorch 成本过高，不适合作为 v0.1 热路径 | 使用 `python_overlay` source loading：agent 修改 full repo 中的 Python 文件，测试前只把指定 overlay 文件同步到容器内 wheel overlay 路径 | 保留 full repo 修复上下文，同时避免完整源码构建；`pytorch__149693__lazylinear_init` 能稳定完成 baseline/gold replay |
| 早期 Codex 直连方式绕过正式隔离边界 | Codex CLI 如果直接在 prepared host workspace 中运行，命令和探索过程不一定通过 Docker runtime，也不能代表正式隔离分数 | 移除 host-workspace 直连路径，正式保留 `codex_action_bridge`：Codex 只拿到 scratch workspace 和 `opbench_action.py` | Codex 的文件读取、修改、命令执行、测试和 diff 导出都经过 OpBench action interface；结果中记录 action log 和 integrity status |
| action bridge 通信方式需要避免额外服务依赖 | 本地服务或复杂 tool 暴露方式会增加端口、权限和 agent 工具集成不确定性 | 采用文件队列式 action bridge：Codex CLI 调用本地 `opbench_action.py`，host 侧 bridge 轮询 request/response 文件 | 不依赖网络端口或额外服务；action 调用可记录为 JSONL，便于审计 |
| Docker 容器状态需要在同一次 attempt 内保持一致 | setup、preflight、测试如果每次都启动独立容器，会丢失 setup 状态，也难以保证 runtime 一致 | `EnvironmentManager` 为每个 replay/agent workspace 创建 task-scoped persistent container，attempt 结束后清理 | preflight、setup、test、agent command action 运行在同一个容器中，环境状态一致且可清理 |
| timeout 过短导致真实 agent 被误判失败 | 真实 agent 首次读取、定位、测试和修复大型仓库任务时耗时明显高于普通单元测试 | Codex CLI 默认 timeout 调整为 1200 秒，并保留 `OP_BENCH_CODEX_TIMEOUT_SEC` 覆盖 | 当前 Codex action bridge 实验未因 timeout 中断，最终 `resolved` |
| 手动实验时终端长时间无输出 | replay 和 agent 修复阶段可能持续数分钟，只有最终 summary 会让使用者误以为进程卡住 | 增加默认 progress log：输出 task 阶段、环境准备、命令退出码、agent attempt、action bridge 调用和结果写入；保留 `--quiet` | 实验室手动验证时可以直接从终端观察进度和卡点，机器可读结果仍写入 `results.jsonl` 和 `summary.json` |
| 开发阶段临时 agent loop 容易混淆正式能力 | 早期为了验证模型调用，存在手写模型 action loop 和具体模型配置；Codex action bridge 跑通后，这些不应作为 v0.1 正式入口 | 从公开代码、配置、测试和 README 中移除临时模型直连入口；文档明确新 agent 应复用 action-interface boundary | 项目边界更清晰：v0.1 的正式真实 agent 是 `codex_action_bridge`，后续 Claude Code、OpenHands 等应按同一隔离边界接入 |
| 项目入口文档混乱 | README 同时承载设计背景、历史实验、临时方案和操作命令，新开发者难以判断阅读顺序 | 重写 README 为入口文档，新增 `docs/v0.1/developer_guide.md` 解释模块职责、数据流、状态含义和扩展方式，新增 `docs/README.md` 作为文档索引 | 开发者可以按 README -> developer guide -> manual validation -> experiment report 的顺序理解和复现项目 |

这些问题的共同结论是：OpBench 不能简单复制 SWE-bench 的“仓库 + 测试”模型。算子 benchmark 的可靠性来自三件事同时成立：环境 artifact 可执行、源码加载策略可复现、agent 操作边界可审计。v0.1 的最终实现围绕这三点进行了收敛。

## 7. 开发者复现步骤

前置条件：

- 已创建 `.venv`。
- 所有命令使用 `PATH=.venv/bin:$PATH PYTHONPATH=src python`。
- 本机可用 Docker。
- 已准备 `op-bench/pytorch-cpu:torch2.6.0-py311` 镜像。
- 已准备 PyTorch source snapshot；可通过 `scripts/prepare_source_snapshot.py` 生成或复用 `.op_bench_cache/sources/...`。
- 运行 `codex_action_bridge` 时，需要本机可执行 `codex` CLI，并已完成对应模型认证。

推荐验证顺序：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693_manual.json

PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/manual-v0.1-codex-action-bridge
```

开发者应检查：

- `runs/replay/pytorch_149693_manual.json` 中 baseline 为 `baseline_reproduced`，gold 为 `resolved`。
- `runs/manual-v0.1-codex-action-bridge/summary.json` 中 `codex_action_bridge.resolved_rate` 是否为 `1.0`。
- agent record 中 `agent_metadata.integrity_status` 是否为 `clean`。
- patch 是否只修改 production source，不依赖修改测试。

## 8. v0.1 边界

v0.1 已完成一个真实 PyTorch task 的 verified agent 闭环，但还不是大规模 benchmark：

- `pytorch__160952__bilinear_lazy_check` 仍为 draft，需要完成 replay 验证后才能进入 `--verified-only`。
- 当前验证集中只有 CPU + Python-level overlay；CUDA、C++ rebuild、native kernel build 需要后续 tier 设计。
- Docker image 目前是本地 tag，后续需要镜像 registry、digest pinning 和环境 artifact 生命周期管理。
- 新 agent 接入应复用 action-interface runtime boundary，而不是新增临时模型直连 loop。

结论：v0.1 的核心方法已经跑通。下一阶段应优先扩充 verified task 数量，并把 source snapshot、Docker image 和 task admission evidence 做成可自动审计的资产管理流程。
