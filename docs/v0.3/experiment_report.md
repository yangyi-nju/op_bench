# OpBench v0.3 实验报告

日期：2026-06-21

## 0. v0.3 完成度判断

v0.3 的重点是将数据集从 v0.2 的 3 条同质 LazyLinear task 扩展为多组件、多 bug 类型的数据集，并引入 patch scope 和 hidden/public test 分层机制。本版本已完成。

| v0.3 目标 | 当前状态 | 证据 |
| --- | --- | --- |
| 数据集扩展到 10+ 条 verified task | 10 条 verified | `datasets/pytorch_v0.3/dataset.json` |
| 覆盖多个 PyTorch 组件 | autograd, optim, utils.data, nn.Module, nn.conv | task manifest component 字段 |
| 包含 multi-file overlay task | 1 条 (conv.py + utils.py) | pytorch__162340 |
| 实现 patch scope enforced 模式 | 已实现，30 次 agent 运行无 scope 违规 | `src/op_bench/patch_scope.py` |
| 实现 public/hidden test 分层 | 平台代码就绪 | evaluator hidden_test_patch + public_test_patch |
| 3 次重复的真实 agent 评测 | 已完成 | `runs/v0.3_codex_r3/`, `runs/v0.3_legacy_3/` |

后续工作应进入 v0.4，包括多 agent 对比、public test 效果实验和 GPU runtime tier。

## 1. 实验目标

1. 确认 10 条 verified task 的评测闭环在 Docker runtime 中端到端工作。
2. 用 3 次重复区分 agent 的稳定能力和偶然成功。
3. 验证 patch scope enforced 模式在真实评测中正确拦截越界修改。

## 2. 当前数据集

数据集文件：

```text
datasets/pytorch_v0.3/dataset.json
```

当前状态：10 条 verified，3 条 draft

| Task | PR | Component | Patch size | Multi-file | Admission |
| --- | --- | --- | ---: | --- | --- |
| `pytorch__168295__autograd_create_graph` | #168295 | torch.autograd | 2 行 | No | verified |
| `pytorch__150975__autograd_backward_inputs` | #150975 | torch.autograd | 21 行 | No | verified |
| `pytorch__161488__lbfgs_wolfe` | #161488 | torch.optim.LBFGS | 9 行 | No | verified |
| `pytorch__124385__load_state_dict_prefix` | #124385 | torch.nn.Module | 11 行 | No | verified |
| `pytorch__143455__set_submodule` | #143455 | torch.nn.Module | 76 行 | No | verified |
| `pytorch__162340__nn_arg_length` | #162340 | torch.nn.modules.conv/utils | 17 行 | Yes | verified |
| `pytorch__163961__dataloader_subset` | #163961 | torch.utils.data | 30 行 | No | verified |
| `pytorch__149693__lazylinear_init` | #149693 | torch.nn.LazyLinear | 3 行 | No | verified |
| `pytorch__147599__lazylinear_state_forward` | #147599 | torch.nn.LazyLinear | 3 行 | No | verified |
| `pytorch__160952__bilinear_lazy_check` | #160952 | torch.nn.Bilinear | 5 行 | No | verified |

## 3. 实验配置

```text
Agent: codex_action_bridge (codex-cli 0.134.0)
Dataset: datasets/pytorch_v0.3/dataset.json (--verified-only)
Repeat: 3
Output: runs/v0.3_codex_r3/ (7 new tasks), runs/v0.3_legacy_3/ (3 v0.2 tasks)
```

运行命令：

```bash
# v0.3 新 task（首轮，v0.2 task snapshot 缺失时跑的）
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_codex_r3

# v0.2 旧 task（snapshot 恢复后补跑）
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --filter-tasks lazylinear bilinear \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_legacy_3
```

## 4. 实验结果

### 4.1 总体（合并两次运行）

| Metric | Value |
| --- | --- |
| 评测 task 数 | 10 / 10 |
| 总 agent 运行 | 30 |
| Resolved | 23 |
| Resolved rate | **76.7%** |
| Median runtime (resolved) | 14.0s |

### 4.2 逐 Task 结果

| Task | A1 | A2 | A3 | Rate |
| --- | --- | --- | --- | --- |
| `pytorch__168295__autograd_create_graph` | resolved | resolved | resolved | **3/3** |
| `pytorch__150975__autograd_backward_inputs` | resolved | resolved | resolved | **3/3** |
| `pytorch__161488__lbfgs_wolfe` | resolved | resolved | resolved | **3/3** |
| `pytorch__124385__load_state_dict_prefix` | resolved | resolved | resolved | **3/3** |
| `pytorch__149693__lazylinear_init` | resolved | resolved | resolved | **3/3** |
| `pytorch__147599__lazylinear_state_forward` | resolved | resolved | resolved | **3/3** |
| `pytorch__160952__bilinear_lazy_check` | resolved | resolved | resolved | **3/3** |
| `pytorch__143455__set_submodule` | fail | resolved | resolved | 2/3 |
| `pytorch__162340__nn_arg_length` | fail | fail | fail | 0/3 |
| `pytorch__163961__dataloader_subset` | fail | fail | fail | 0/3 |

### 4.3 稳定性分类

- **稳定 resolved (3/3)**：7 条。共性：gold patch ≤ 21 行，单文件，bug 语义明确。
- **不稳定 (2/3)**：1 条。`set_submodule` gold patch 76 行，第一次修复不完整。
- **稳定失败 (0/3)**：2 条。multi-file 参数验证 + DataLoader dispatch 逻辑。

## 5. 失败分析

### `pytorch__162340__nn_arg_length` (0/3)

Multi-file task，需要同时修改 `conv.py` 和 `utils.py`。Agent 生成了 patch 但未通过 hidden test。推测 agent 只修了其中一个文件的逻辑。

### `pytorch__163961__dataloader_subset` (0/3)

涉及 Subset 子类 `__getitem__` 的 dispatch 逻辑和 Python MRO。issue 描述可能不够具体，导致修复方向偏离。

### `pytorch__143455__set_submodule` (2/3)

Gold patch 76 行，第一次尝试遗漏边界条件。后续两次成功说明 agent 有能力但需要更多探索。

## 6. 过程观察

### 6.1 Patch Scope

30 次 agent 运行均未触发 `patch_out_of_scope`。Prompt 中的 scope 提示有效引导了修改范围。

## 7. v0.2 → v0.3 对比

| 维度 | v0.2 | v0.3 |
| --- | --- | --- |
| Verified task 数 | 3 | 10 |
| 组件覆盖 | torch.nn.modules.linear | 5 个子系统 |
| Bug 类型 | 1 种 | 6 种 |
| Multi-file task | 无 | 有 |
| Patch scope | 无 | enforced |
| Agent repeat | 1 | 3 |
| Resolved rate | 100% (3/3) | 76.7% (23/30) |

Resolved rate 下降是预期的：v0.2 的 3 条 task 全是同一个文件的简单 lazy init 问题，v0.3 引入了真正多样化的问题。v0.2 原有的 3 条 task 在 v0.3 中仍保持 100% (9/9)。

## 8. 可复现实验步骤

```bash
PYTHONPATH=src python3 scripts/validate_dataset.py datasets/pytorch_v0.3/dataset.json

PYTHONPATH=src python3 -m unittest discover tests -v

PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.3_codex_r3_reproduce
```

## 9. 结论

v0.3 完成了从"最小闭环"到"多组件数据集"的扩展：

- 10 条 verified task 全部参与评测，覆盖 5 个 PyTorch 子系统和 6 种 bug 类型。
- Patch scope enforced 模式确保评分边界清晰。
- 3 次重复揭示了 agent 稳定性：7/10 稳定 resolved，1/10 不稳定，2/10 稳定失败。
- 整体 resolved rate 76.7% (23/30)。
- 平台代码新增 patch_scope、hidden/public test 分层、multi-file overlay、`--filter-tasks` 增量评测，69 个单元测试通过。

后续建议：
1. 分析 2 条稳定失败 task 的 agent 行为日志，改进 issue 描述或引入 public test 作为修复引导。
2. 引入第二个 agent（Claude Code）做多 agent 对比。
3. 为 3 条 draft task 排查 admission 失败原因并修复，扩展到 13 条。
4. 设计 public test 对 agent 能力的影响实验（ablation study）。
