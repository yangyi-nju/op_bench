# OpBench v0.5 实验报告

日期：2026-07-11

## 0. 版本结论

v0.5 已完成。正式累计数据集包含 **17 条 verified PyTorch task**，使用同一 Codex 版本和 Linux remote execution policy 完成 **17 task x 3 repeat = 51 个有效 attempt**。总体结果为 **37/51 resolved（72.5%）**；其中 6 条 precision task 的结果为 **13/18（72.2%）**。

本版本的核心产物不是把 P1-P5 配额填满，而是建立一个可发布、可复算的精度问题切片，并把实验完整性纳入评分流程。P4 没有候选通过 admission，因此保持 N/A；两个因 base commit 与 wheel API 错配而 deprecated 的 task 不进入正式数据集。

| Release contract | Result |
| --- | --- |
| Cumulative dataset | 17 verified tasks |
| Precision slice | 6 verified tasks（P1/P2/P3/P5；P4 N/A） |
| Admission replay | 17/17 baseline reproduced + gold resolved |
| Codex experiment | 51/51 logical attempts complete |
| Environment integrity | 0 logical transient；2 条原始 rsync transient 已补跑覆盖 |
| Dataset status | `verified` |

## 1. 数据集

正式清单为 `datasets/pytorch_v0.5/dataset.json`：v0.4 的 13 条 task 全部保留，新增 #140557、#139999、#129138、#139372。独立精度切片为 `datasets/pytorch_v0.5_precision/dataset.json`，包含两条 v0.4 锚点和四条新增 task。

| Task | PR | Subclass | Tier |
| --- | ---: | :---: | --- |
| `140557_layer_norm_decomp_precision` | #140557 | P2 | `cpu_python_overlay` |
| `139999_masked_mean_bool_upcast` | #139999 | P1 | `cpu_python_overlay` |
| `129138_linear_add_bias_autocast` | #129138 | P3 | `cpu_python_overlay` |
| `132835_njt_sdpa_autocast` | #132835 | P3 | `cuda_python_overlay` |
| `144009_softmax_ilpreduce_size` | #144009 | P5 | `cuda_kernel_build` |
| `139372_histc_int8_cuda_bounds` | #139372 | P5 | `cuda_kernel_build` |

#129154 和 #144073 已 deprecated。两者都要求 matched wheel 或 source build 才能把 base commit 的 Python API 与 runtime 对齐，不能用 Python overlay 构造可信的 fail-to-pass。

## 2. 实验配置

```text
Agent:       codex_action_bridge
Codex CLI:   0.144.0-alpha.4
Model:       gpt-5.6-terra, reasoning effort low
Dataset:     datasets/pytorch_v0.5/dataset.json
Repeat:      3
Runtime:     remote Linux Docker, 4 x Tesla V100 SXM2 32GB
CPU output:  runs/v0.5_codex_legacy_cpu/
CUDA output: runs/v0.5_codex_legacy_cuda/
Precision:   runs/v0.5_precision_codex_cpu/ + runs/v0.5_precision_codex_gpu/
Summary:     runs/v0.5_codex/summary.json
```

CPU、CUDA overlay 和 kernel build 分批运行，但都通过同一 action bridge 与 remote evaluator 评分。聚合使用 dataset x agent x repeat 完整性校验；原始 JSONL append-only，同一 attempt 的环境失败补跑后只取最新逻辑记录。

## 3. 总体结果

### 3.1 八维指标

| Metric | Result |
| --- | ---: |
| Resolved rate | **72.5%（37/51）** |
| Patch conciseness | **1.000** |
| Pass-to-pass kept rate | **94.1%（48/51）** |
| Strict resolved rate | **72.5%（37/51）** |
| Regression rate | **0.0%（0/51）** |
| Tier-weighted score | **76.8%** |
| Per-problem resolved rate | precision 72.2%；历史未分类 task 72.7% |
| Median evaluator runtime | **101.0s** |

`regression_rate` 只统计“F2P 已修复但 P2P 被破坏”。#140557 的三次补丁同时未修复 F2P 且破坏 import/P2P，因此属于 `fail_to_pass_failed`，不是 regression。v0.5 同时修正了 evaluator 的状态判定顺序，并对旧 JSONL 按实际测试计数归一化。

### 3.2 逐 task 结果

| Task | Subclass | Tier | A1 | A2 | A3 | Rate | Median evaluator (s) |
| --- | :---: | --- | --- | --- | --- | ---: | ---: |
| `149693__lazylinear_init` | - | cpu | resolved | resolved | resolved | 3/3 | 96.9 |
| `147599__lazylinear_state_forward` | - | cpu | resolved | resolved | resolved | 3/3 | 172.7 |
| `160952__bilinear_lazy_check` | - | cpu | resolved | resolved | resolved | 3/3 | 210.7 |
| `162340__nn_arg_length` | - | cpu | fail | fail | fail | 0/3 | 68.6 |
| `163961__dataloader_subset` | - | cpu | fail | fail | fail | 0/3 | 119.6 |
| `168295__autograd_create_graph` | - | cpu | resolved | resolved | resolved | 3/3 | 83.4 |
| `161488__lbfgs_wolfe` | - | cpu | resolved | resolved | resolved | 3/3 | 77.6 |
| `150975__autograd_backward_inputs` | - | cpu | resolved | resolved | resolved | 3/3 | 64.1 |
| `124385__load_state_dict_prefix` | - | cpu | resolved | resolved | resolved | 3/3 | 271.1 |
| `143455__set_submodule` | - | cpu | fail | fail | fail | 0/3 | 61.5 |
| `132835__njt_sdpa_autocast` | P3 | cuda_py | resolved | fail | fail | 1/3 | 88.5 |
| `132616__cuda_mem_get_info` | - | cuda_py | resolved | resolved | resolved | 3/3 | 118.5 |
| `144009__softmax_ilpreduce_size` | P5 | cuda_kernel | resolved | resolved | resolved | 3/3 | 579.6 |
| `140557__layer_norm_decomp_precision` | P2 | cpu | fail | fail | fail | 0/3 | 101.0 |
| `139999__masked_mean_bool_upcast` | P1 | cpu | resolved | resolved | resolved | 3/3 | 90.3 |
| `129138__linear_add_bias_autocast` | P3 | cpu | resolved | resolved | resolved | 3/3 | 122.4 |
| `139372__histc_int8_cuda_bounds` | P5 | cuda_kernel | resolved | resolved | resolved | 3/3 | 607.2 |

稳定性：12 条稳定 resolved，4 条稳定失败，1 条不稳定。四条新增 precision task 中三条稳定 resolved，一条稳定失败。

## 4. 分组结果

### 4.1 按 tier

| Tier | Tasks | Attempts | Resolved | Rate |
| --- | ---: | ---: | ---: | ---: |
| `cpu_python_overlay` | 13 | 39 | 27 | 69.2% |
| `cuda_python_overlay` | 2 | 6 | 4 | 66.7% |
| `cuda_kernel_build` | 2 | 6 | 6 | 100.0% |

kernel tier 的两条 task 都是 3/3，但样本量仍不足以推断 kernel bug 普遍更容易。该结果只说明当前 source build、ccache 和 V100 evaluator 通路稳定。

### 4.2 Precision subclass

| Subclass | Tasks | Attempts | Resolved | Rate |
| :---: | ---: | ---: | ---: | ---: |
| P1 数值累积误差 | 1 | 3 | 3 | 100.0% |
| P2 dtype/分解转换损失 | 1 | 3 | 0 | 0.0% |
| P3 混合精度不一致 | 2 | 6 | 4 | 66.7% |
| P4 数值不稳定 | 0 | 0 | N/A | N/A |
| P5 CUDA kernel 精度 | 2 | 6 | 6 | 100.0% |

P4 保持 N/A 是 admission 质量约束的结果，不按 0% 计分，也不使用非精度或版本错配 task 填充。

## 5. 失败分析

### `#140557`（P2，0/3）

三次都识别到 `native_layer_norm` 需要 fake/decomposition 路径，但使用了不适用的 `torch.library.register_fake` 形式，分别触发重复 Meta kernel、非法 qualname 或非法 overload。Gold 使用内部 `torch._subclasses.fake_impls.register_op_impl`。该 task 稳定区分“知道要注册 fake”与“掌握 PyTorch 内部 dispatch API”。

### `#132835`（P3，1/3）

三次都尝试按 autocast dtype 转换 query/key/value。唯一 resolved 的补丁在 `_validate_sdpa_input` 之前转换；另两次放在校验之后，mixed dtype 输入已先被拒绝。差异来自控制流位置，适合作为中等难度不稳定样本。

### 历史 hard cases

`#162340` 和 `#163961` 继续保持 0/3，分别卡在精确异常语义/多文件范围，以及构造期 fail-fast 与运行时 fallback 的设计差异。这与 v0.4 一致。

`#143455` 从 v0.4 的 3/3 变为 0/3。三次补丁都只实现了问题的一部分：单段路径 parent 修复、`rpartition` 修复或强制 existing-child 检查；都没有同时实现 gold 的 `strict` 参数、非 Module 校验、创建/替换语义。hidden test 对这组 API contract 做了完整断言，因此失败是有效的 agent 行为，不是平台异常。

## 6. v0.4 对比

| Metric | v0.4 full | v0.5 full | v0.5 precision slice |
| --- | ---: | ---: | ---: |
| Dataset tasks | 13 | 17 | 6 |
| Attempts | 39 | 51 | 18 |
| Resolved rate | 84.6%（33/39） | **72.5%（37/51）** | 72.2%（13/18） |
| Pass-to-pass kept | 100.0% | 94.1% | 83.3% |
| Regression rate | 0.0% | 0.0% | 0.0% |
| Tier-weighted score | 88.2% | 76.8% | 78.8% |

v0.5 中继承的 13 条 task 为 **28/39（71.8%）**，四条新增 task 为 **9/12（75.0%）**。不能把 v0.4 到 v0.5 的 -12.1pp 解释为数据集单方面变难：Codex CLI/model 版本、CPU 执行后端和采样都发生变化。主要波动来自 #143455（3/3 -> 0/3）和 #132835（3/3 -> 1/3）；两个长期 hard case 仍为 0/3，其余继承 task 保持稳定 resolved。

Patch conciseness 在 v0.5 首次有完整 patch artifact 支撑。v0.4 的历史 patch path 不能完整复算，因此不比较该指标。

## 7. 运行完整性改进

1. 17 条 task 全部统一到 Linux `remote_docker` policy，消除 v0.4 CPU 的 macOS/QEMU 差异；10 条历史 CPU task 已重新 admission。
2. run state 现在包含 task replay hash，task 内容变化后不能错误 resume。
3. `environment_unavailable/environment_error` 不占完成 key；原始记录 append-only，补跑后 summary 按 task/agent/attempt 取最新记录。
4. rsync 对连接错误和 mutable Git pack 导致的 exit 23 做最多三次有界重试。
5. baseline 和每个 attempt 落盘后立即刷新 summary；聚合器可用 `--expected-repeat 3 --require-complete` 硬校验数据集完整性。
6. evaluator 先判断 F2P，再判断 P2P regression；报告器兼容归一化旧状态。

本轮首条 CPU task 并发同步时产生两条 exit-23 `environment_unavailable`。修复后以同一幂等 key 串行补跑并 resolved。最终 summary 为 51 个逻辑 attempt，原始 53 条 agent record 保留两条 transient 供审计。

## 8. 复现

```bash
PYTHONPATH=src python3 scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json

PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'

PYTHONPATH=src python3 scripts/aggregate_experiments.py \
  runs/v0.5_precision_codex_cpu \
  runs/v0.5_precision_codex_gpu \
  runs/v0.5_codex_legacy_cpu \
  runs/v0.5_codex_legacy_cuda \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --expected-repeat 3 \
  --require-complete \
  --output runs/v0.5_codex/summary.json
```

完整运行配置见 `docs/v0.5/setup_remote_agent.md`。本地 `configs/remote_hosts.json` 不进入 git。

## 9. 后续版本

- v0.6：边界问题维度，并为 #129154/#144073 提供 matched-wheel 或 source-build runtime 后重新评估。
- v0.7：设备/API 兼容维度。
- P4 numerical-instability 候选继续进入 backlog；只有 admission 稳定后才加入后续正式数据集。
- 多 agent 对比继续后移，避免在问题维度和执行口径仍扩展时制造不可比结果。
