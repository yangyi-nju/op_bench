# OpBench v0.2 实验报告

日期：2026-06-05

## 0. v0.2 完成度判断

从当前定义的 v0.2 目标看，本版本已经完成。v0.2 的重点不是扩大 leaderboard，而是把数据集扩展、环境资产管理、任务准入和真实 agent 隔离评测这条工程链路固定下来。

| v0.2 目标 | 当前状态 | 证据 |
| --- | --- | --- |
| 数据集从 v0.1 单任务扩展到 3-5 条真实任务 | 已完成，当前 3 条 verified PyTorch task | `datasets/pytorch_mini/dataset.json` |
| 为每条任务固定可复现环境和源码资产 | 已完成，使用 environment/source registry | `environments/registry.json`, `sources/registry.json` |
| 建立正式 admission 机制 | 已完成，每条 verified task 有 task-local evidence | `tasks/pytorch/*/admission/evidence.json` |
| 校验 baseline fail-to-pass 和 pass-to-pass | 已完成，三条任务 baseline 均为 fail-to-pass 0/1, pass-to-pass 1/1 | admission evidence |
| 校验 gold patch 可修复 | 已完成，三条任务 gold 均为 fail-to-pass 1/1, pass-to-pass 1/1 | admission evidence |
| 跑通真实 agent 隔离评测 | 已完成，Codex CLI 通过 `codex_action_bridge` 评测 3/3 resolved | `runs/experiments/pytorch_mini_codex_action_bridge_v0.2/` |
| 记录实验产物、问题和边界 | 已完成 | 本报告 |

因此，v0.2 可以视为“已完成并可提交”的版本。后续工作不应继续扩大 v0.2 范围，而应进入 v0.3 或后续版本，包括 hidden/public test 分层、多文件 overlay、更复杂 runtime tier 和更多 task 扩展。

## 1. 实验目标

v0.2 实验验证两个层面的目标：

1. 数据集构建与准入：确认 OpBench 可以从 v0.1 的单任务 MVP 扩展到 3 条真实 PyTorch operator task，并为每条 task 固化环境、源码、hidden tests 和 admission evidence。
2. 真实 agent 评测：确认真实 Codex CLI 可以在不直接访问目标 workspace 的前提下，通过 OpBench action interface 在 Docker runtime 中完成修复，并由独立 scorer 给出结果。

本轮具体检查项：

1. 将 PyTorch verified task 从 1 条扩展到 3 条。
2. 使用 environment/source registry 管理可复用资产。
3. 对每条 task 执行 baseline/gold admission。
4. 生成 task-local stable admission evidence。
5. 用 `gold` agent 跑通 3-task 评测闭环。
6. 用真实 `codex_action_bridge` 跑通 3-task 隔离评测。
7. 检查 Docker image、source snapshot 和容器生命周期状态。

## 2. 当前数据集

数据集文件：

```text
datasets/pytorch_mini/dataset.json
```

当前状态：`verified`

| Task | PR | Issue | Runtime tier | Admission |
| --- | --- | --- | --- | --- |
| `pytorch__149693__lazylinear_init` | https://github.com/pytorch/pytorch/pull/149693 | https://github.com/pytorch/pytorch/issues/149691 | `cpu_python_overlay` | `verified` |
| `pytorch__160952__bilinear_lazy_check` | https://github.com/pytorch/pytorch/pull/160952 | https://github.com/pytorch/pytorch/issues/160407 | `cpu_python_overlay` | `verified` |
| `pytorch__147599__lazylinear_state_forward` | https://github.com/pytorch/pytorch/pull/147599 | https://github.com/pytorch/pytorch/issues/147389 | `cpu_python_overlay` | `verified` |

三条任务都来自真实 PyTorch issue/PR，且都属于 Python-level `torch.nn` 算子或模块行为问题。

每条任务在进入正式评测前必须满足两个条件：

1. 未修复 baseline 必须能复现问题，即 fail-to-pass 测试失败。
2. 上游修复对应的 gold patch 必须能修复问题，且不破坏 pass-to-pass 测试。

这一步是 OpBench 与普通“跑一个测试看看”的核心区别。数据集中的每条样本不是单独的代码片段，而是“真实 issue + source snapshot + Docker runtime + hidden tests + admission evidence”的组合。

## 3. 环境与源码资产

环境 registry：

```text
environments/registry.json
```

源码 registry：

```text
sources/registry.json
```

当前 verified slice 使用同一个 Docker runtime：

```text
op-bench/pytorch-cpu:torch2.6.0-py311
sha256:9507757065fd73fa7a958cd13f01de163a3fa46d826c09142f9522eda0f1e400
```

本机检查结果：

```text
1/1 environment ready
3/3 source snapshots ready
0 OpBench managed container residue
```

当前 source snapshot 是 PyTorch superproject snapshot，但 submodule 未初始化。该限制不影响当前 3 条 verified task，因为它们都使用 `python_overlay`，只让 `torch/nn/modules/linear.py` 在测试 runtime 中生效。需要 C++/CUDA/build/submodule 的 task 不能直接沿用该 runtime tier。

## 4. Admission 结果

每条 task 都执行了：

1. baseline workspace 准备。
2. Docker runtime preflight。
3. apply hidden `test.patch`。
4. baseline fail-to-pass/pass-to-pass。
5. gold workspace 准备。
6. apply hidden `test.patch` 和 `gold.patch`。
7. gold fail-to-pass/pass-to-pass。
8. 写入 stable evidence。

准入条件：

- baseline: fail-to-pass 必须失败，pass-to-pass 必须通过。
- gold: fail-to-pass 必须通过，pass-to-pass 必须通过。

Admission 结果：

| Task | Baseline | Gold | Evidence |
| --- | --- | --- | --- |
| `pytorch__149693__lazylinear_init` | fail-to-pass 0/1, pass-to-pass 1/1 | fail-to-pass 1/1, pass-to-pass 1/1 | `tasks/pytorch/149693_lazylinear_init/admission/evidence.json` |
| `pytorch__160952__bilinear_lazy_check` | fail-to-pass 0/1, pass-to-pass 1/1 | fail-to-pass 1/1, pass-to-pass 1/1 | `tasks/pytorch/160952_bilinear_lazy_check/admission/evidence.json` |
| `pytorch__147599__lazylinear_state_forward` | fail-to-pass 0/1, pass-to-pass 1/1 | fail-to-pass 1/1, pass-to-pass 1/1 | `tasks/pytorch/147599_lazylinear_state_forward/admission/evidence.json` |

严格数据集校验命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json --require-verified
```

结果：

```text
dataset looks valid (3 tasks)
```

## 5. Gold Agent 闭环

本轮先使用 `gold` agent 跑通系统闭环。`gold` agent 不代表真实 agent 能力，只用于证明数据集、环境、patch replay 和评分系统可运行。

命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --agent gold \
  --output runs/experiments/pytorch_mini_gold_v0.2.json
```

结果：

| Agent | Resolved | Total | Resolved rate | Median runtime |
| --- | ---: | ---: | ---: | ---: |
| `gold` | 3 | 3 | 1.0 | 约 34.7 秒 |

结果文件：

```text
runs/experiments/pytorch_mini_gold_v0.2.json/summary.json
```

## 6. 真实 Agent 实验

真实 agent 使用 Codex CLI：

```text
codex-cli 0.125.0
```

运行命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/experiments/pytorch_mini_codex_action_bridge_v0.2
```

输出目录：

```text
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/
```

关键文件：

```text
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/summary.json
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/results.jsonl
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/patches/codex_action_bridge/attempt_001/
```

## 7. 真实 Agent 评测流程

一次 `codex_action_bridge` 评测包含三个阶段。

### 7.1 Baseline 复现阶段

系统为每条 task 准备 fresh workspace，并启动 task-scoped Docker container。然后执行：

1. 拷贝 source snapshot 到 workspace。
2. 检查 Docker image digest 和 preflight command。
3. 应用 hidden `test.patch`。
4. 使用 baseline 代码运行 fail-to-pass 和 pass-to-pass。

只有当 baseline 满足“fail-to-pass 失败、pass-to-pass 通过”时，才继续执行 agent。否则该 task 不进入 agent 修复阶段。

### 7.2 Agent 修复阶段

Codex CLI 运行在 host 侧 scratch workspace。它不能直接访问目标 PyTorch workspace，只能调用 `opbench_action.py`，由 action interface 转发到目标 workspace/container。

本次允许的 action 包括：

```text
read_file
write_file
apply_patch
run_command
run_test
git_diff
```

所有 action 都记录在 `*_actions.jsonl` 中。修复阶段结束后，系统只接受 action interface 导出的 `git_diff` 作为 agent patch。

### 7.3 最终评分阶段

最终评分不会复用 agent 修复阶段的容器状态。系统重新准备 fresh workspace 和 task-scoped Docker container，然后执行：

1. 应用 hidden `test.patch`。
2. 应用 agent patch。
3. 运行 fail-to-pass tests。
4. 运行 pass-to-pass tests。
5. 根据测试结果给出 `resolved` 或失败原因。

因此，agent 在修复阶段自己运行过什么命令不会直接决定分数；分数只由独立评分阶段决定。

## 8. Codex CLI 实验结果

总体结果：

| Agent | Resolved | Total | Resolved rate | Median runtime |
| --- | ---: | ---: | ---: | ---: |
| `codex_action_bridge` | 3 | 3 | 1.0 | 34.92 秒 |

逐任务结果：

| Task | Status | Fail-to-pass | Pass-to-pass | Runtime | Actions | Integrity | Timeout |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `pytorch__149693__lazylinear_init` | `resolved` | 1/1 | 1/1 | 42.79 秒 | 11 | `clean` | false |
| `pytorch__160952__bilinear_lazy_check` | `resolved` | 1/1 | 1/1 | 34.92 秒 | 12 | `clean` | false |
| `pytorch__147599__lazylinear_state_forward` | `resolved` | 1/1 | 1/1 | 34.16 秒 | 10 | `clean` | false |

Codex 生成的最终 patch：

```text
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/patches/codex_action_bridge/attempt_001/pytorch__149693__lazylinear_init__codex_action_bridge.patch
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/patches/codex_action_bridge/attempt_001/pytorch__160952__bilinear_lazy_check__codex_action_bridge.patch
runs/experiments/pytorch_mini_codex_action_bridge_v0.2/patches/codex_action_bridge/attempt_001/pytorch__147599__lazylinear_state_forward__codex_action_bridge.patch
```

三条 patch 都只修改 `torch/nn/modules/linear.py`，符合当前 `cpu_python_overlay` runtime tier 的评分边界。

## 9. 逐任务复盘

### 9.1 `pytorch__149693__lazylinear_init`

问题类型：`LazyLinear` 初始化后 `in_features` 状态没有正确更新，影响 lazy module 状态一致性。

Codex 修复过程：

1. 读取 `test/nn/test_lazy_modules.py`、`torch/nn/modules/lazy.py` 和 `torch/nn/modules/linear.py`。
2. 用 Python probe 复现 `LazyLinear` forward 后的状态。
3. 前两次 `apply_patch` 因 patch 格式问题返回 `exit=128`。
4. Codex 根据失败反馈读取目标文件片段，重新生成 patch。
5. 运行 `TestLazyModules.test_linear_state` 通过。
6. 导出最终 diff。

最终 patch 行为：

```text
在 materialize weight 前设置 self.in_features = input.shape[-1]
再使用 self.in_features 作为 weight shape 的输入维度
```

最终评分：

```text
fail-to-pass 1/1
pass-to-pass 1/1
status resolved
```

### 9.2 `pytorch__160952__bilinear_lazy_check`

问题类型：`Bilinear` 在构造阶段过早拒绝 `in1_features <= 0`，导致 Lazy Bilinear 类场景无法延迟初始化。

Codex 修复过程：

1. 读取 `torch/nn/modules/linear.py` 和 hidden test 文件路径。
2. 尝试使用 `rg` 搜索，容器返回 `rg: command not found`。
3. 使用 `grep` fallback 搜索 `Bilinear` 和错误检查逻辑。
4. 读取已有测试片段，构造 Lazy Bilinear probe。
5. 将 `in1_features <= 0` 检查从 `__init__` 移动到 `reset_parameters` 中基于实际 weight size 的阶段。
6. `run_test` 尝试运行 hidden pass-to-pass 时 exit 2，因为 hidden test 文件在 agent 修复阶段不可见。
7. 使用 `py_compile` 做语法检查，导出最终 diff。

最终 patch 行为：

```text
允许 __init__ 阶段传入 in1_features=0
在 reset_parameters 时根据 self.weight.size(1) 检查真实输入维度
```

最终评分：

```text
fail-to-pass 1/1
pass-to-pass 1/1
status resolved
```

### 9.3 `pytorch__147599__lazylinear_state_forward`

问题类型：`LazyLinear` 加载已初始化 `Linear` 的 state_dict 后，forward 时没有把 `in_features` 更新为权重实际输入维度。

Codex 修复过程：

1. 读取 `torch/nn/modules/linear.py`。
2. 尝试读取 hidden test 文件路径。
3. 第一次 Python here-doc probe 因 shell quoting 问题 exit 2。
4. 读取 `torch/nn/modules/lazy.py` 理解 lazy module 状态转换。
5. 应用 patch：当参数已经初始化时，从 `self.weight.shape[-1]` 回填 `self.in_features`。
6. 使用 Python one-liner probe 验证加载 state_dict 后 forward 行为。
7. `run_test` 尝试运行 hidden pass-to-pass 时 exit 2，因为 hidden test 文件在 agent 修复阶段不可见。
8. 导出最终 diff。

最终 patch 行为：

```text
如果 LazyLinear 已经没有 uninitialized params，则根据 weight shape 更新 self.in_features
```

最终评分：

```text
fail-to-pass 1/1
pass-to-pass 1/1
status resolved
```

## 10. 过程观察

### 10.1 Action interface 隔离生效

`codex_action_bridge` 给 Codex CLI 的是 host 侧 scratch workspace 和 `opbench_action.py` CLI。目标 PyTorch workspace 没有直接暴露给 Codex。Codex 的读文件、运行命令、应用 patch、运行测试和导出 diff 都通过 action 日志记录。

最终评分没有复用 Codex 修复阶段的容器状态，而是使用导出的 patch，在 fresh workspace 和 task-scoped Docker container 中重新执行 fail-to-pass/pass-to-pass。因此本次 3/3 resolved 代表隔离评分闭环通过，而不是只依赖 agent 自己运行过的命令。

### 10.2 源码上下文对当前任务足够

当前三条 task 都是 Python-level `torch.nn` 行为问题。Codex 在修复过程中主要读取：

```text
torch/nn/modules/linear.py
torch/nn/modules/lazy.py
test/nn/test_lazy_modules.py
```

对这类任务，当前 PyTorch superproject snapshot 加 `cpu_python_overlay` 足以提供必要源码上下文。需要 C++/CUDA/build/submodule 的问题不应直接复用这个 runtime tier。

### 10.3 Hidden test 可见性存在改进空间

`pytorch__160952__bilinear_lazy_check` 和 `pytorch__147599__lazylinear_state_forward` 的 pass-to-pass 测试来自 hidden `test.patch`。在 agent 修复阶段，hidden test 文件不在 workspace 中，因此 Codex 调用这些测试时出现 exit 2：

```text
python: can't open file '/workspace/test/nn/test_bilinear_lazy_op_bench.py'
python: can't open file '/workspace/test/nn/test_lazylinear_state_op_bench.py'
```

这没有影响最终独立评分，因为 scorer 会在 fresh workspace 中应用 hidden `test.patch` 后再评分。但 prompt 当前把这些测试描述成 visible regression tests，容易误导 agent。后续应区分：

- hidden scoring tests：只用于最终评分。
- public/visible tests：允许 agent 在修复阶段主动运行。

### 10.4 容器内缺少 `rg`

Codex 在 `pytorch__160952__bilinear_lazy_check` 上尝试使用 `rg`，容器返回：

```text
bash: line 1: rg: command not found
```

随后 Codex 使用 `grep` fallback 完成定位，最终评分通过。这说明 action interface 能承载真实 agent 的自适应过程，但环境基础工具仍应更明确。后续可以选择：

1. 在 Docker image 中安装 `ripgrep`。
2. 在 agent prompt 中声明可用工具集合。
3. 在 action interface 提供受控 search API，减少对容器系统工具的依赖。

### 10.5 Patch 应用反馈有效

`pytorch__149693__lazylinear_init` 的 action 日志中出现过两次 corrupt patch，Codex 根据 `apply_patch` 反馈重新读取文件片段并生成可应用 patch。该过程验证了 action interface 对 agent 修复循环的最小反馈能力。

## 11. 本轮问题与处理

| 问题 | 影响 | 本次处理 | 效果 | 后续建议 |
| --- | --- | --- | --- | --- |
| `pytorch__160952__bilinear_lazy_check` 初始 hidden test 没有真正执行 | baseline 被误判为未复现 | 为 test patch 增加 `unittest.main()`，重设 fail-to-pass/pass-to-pass | 任务从 draft 晋升 verified | 新任务准入时检查测试文件是否实际运行用例 |
| 手写 test patch hunk 行数错误 | `git apply` 报 corrupt patch 或测试文件截断 | 用临时 git 仓库先校验 patch，再修 hunk header | hidden tests 能稳定应用 | 建议后续用工具生成 patch，减少手写 |
| 一次 gold experiment 出现 exit 139 | 可能误判 task flaky | 单独 stability check，复跑恢复预期结果 | 暂不标记 blocked | runner 增加非预期信号退出分类和有限重试 |
| hidden pass-to-pass 被描述为 visible test | agent 会尝试运行不存在的隐藏测试文件 | 保持最终 scorer 隔离评分，记录为报告问题 | 不影响本次 3/3 resolved | 增加 `public_tests` 字段或调整 prompt |
| Docker image 缺少 `rg` | agent 的搜索命令可能失败 | agent 自动 fallback 到 `grep` | 不影响最终结果 | image 安装 `ripgrep` 或提供 search action |
| 当前 source snapshot 未初始化 submodule | 对 C++/CUDA/build task 上下文不足 | 本次只纳入 Python-level verified task | 不影响当前数据集 | 新增 runtime tier 前必须重新设计 source policy |
| 当前 overlay 只同步指定 Python 文件 | 多文件修复能力受限 | 三条任务均只需 `torch/nn/modules/linear.py` | 当前评分边界清晰 | v0.3 可扩展多文件 overlay admission |

## 12. 正确性边界

本次 3/3 resolved 只能说明 v0.2 当前 slice 的评测闭环成立，不应被过度解释为 OpBench 已经覆盖所有算子问题类型。

当前成立的边界：

1. 任务都是 PyTorch Python-level `torch.nn` 行为问题。
2. runtime tier 是 `cpu_python_overlay`。
3. agent 修改的文件通过 overlay 同步到 installed PyTorch wheel 后运行测试。
4. source snapshot 的 submodule 未初始化，但当前任务不依赖 submodule。
5. 真实 agent 只有 Codex CLI 一个，尚未做多 agent 对比。

当前不覆盖的范围：

1. C++/CUDA kernel 修改。
2. 需要 full PyTorch source build 的任务。
3. 依赖 GPU、特定驱动、特定 BLAS/cuDNN/cuBLAS 后端的问题。
4. 多文件、大规模重构或跨模块行为变更。
5. 多 agent leaderboard。

这些边界不是 v0.2 的缺陷，而是当前版本有意收敛后的工程范围。后续扩展数据集时，必须为新的问题类型新增 runtime tier 和 admission policy，而不是默认复用 `cpu_python_overlay`。

## 13. 可复现实验步骤

从干净工作区复核本次实验时，建议按以下顺序执行。

检查数据集：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json --require-verified
```

预期结果：

```text
dataset looks valid (3 tasks)
```

检查 Docker 和 source assets：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py --check-docker
```

预期结果：

```text
1/1 environment ready
3/3 source snapshots ready
```

检查容器残留：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/manage_containers.py list
```

预期结果：

```text
[]
```

运行真实 Codex CLI 评测：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/experiments/pytorch_mini_codex_action_bridge_v0.2
```

预期结果：

```text
codex_action_bridge resolved 3/3
```

运行单元测试：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

当前验证结果：

```text
Ran 63 tests in 4.518s
OK
```

## 14. 收尾验证

本轮收尾验证命令：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py --check-docker
```

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json --require-verified
```

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/manage_containers.py list
```

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover tests -v
```

结果：

- assets: 1/1 environment ready, 3/3 sources ready
- dataset: valid, 3 tasks
- containers: `[]`
- unit tests: 63 tests OK

## 15. 结论

本次实验完成了 v0.2 的小规模真实闭环：

- 3 条 verified PyTorch CPU operator task 全部 baseline 可复现。
- 每条 task 都有 stable admission evidence。
- `gold` agent 在 3-task slice 上全部 resolved。
- Codex CLI 通过 OpBench action interface 完成修复。
- Codex CLI 的 3 条任务最终独立评分全部 resolved。
- 实验保留了 summary、results、patch、action logs，可供复核。

因此，v0.2 当前已经具备“小规模 verified 数据集 + 环境资产管理 + 真实 agent 隔离评测”的基本工程闭环。

后续版本应优先解决 visible/hidden test 分层、容器基础工具标准化、多文件 overlay，以及更复杂 runtime tier 的准入策略。
