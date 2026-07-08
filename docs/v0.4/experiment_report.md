# OpBench v0.4 实验报告

日期：2026-07-08

## 0. v0.4 完成度判断

v0.4 的重点是把 benchmark 从单 tier CPU-only 扩展到 CUDA / kernel_build，并在实验中量化 tier 的区分度。本版本已完成实验。多 agent 对比（Claude Code）因外部条件限制推迟到 v0.5。

| v0.4 目标 | 当前状态 | 证据 |
| --- | --- | --- |
| Remote GPU Docker 执行器 | 已实现 | `src/op_bench/remote.py`、`RemoteDockerExecutor` |
| CUDA 环境镜像 | 已实现 | `environments/pytorch-cuda/`、`environments/pytorch-cuda-devel/` |
| `cuda_python_overlay` tier | 2 条 verified | 132616、132835 |
| `cuda_kernel_build` tier | 1 条 verified | 144009 |
| 数据集规模 | 13 条 verified | `datasets/pytorch_v0.4/dataset.json` |
| Codex CLI 全量评测 | 39 attempts 完成 | `runs/v0.4_codex_cpu/`、`runs/v0.4_codex_gpu/` |
| 多 agent 对比 | 推迟到 v0.5 | 记录在 [下一步](#8-下一步) |
| Public test ablation | 放弃 | 无 task 内含 `public_test.patch`，投入产出比不足 |

## 1. 实验目标

1. 验证 remote_docker backend 端到端可运行：workspace rsync、GPU container 启动、SSH 长连接、build 缓存复用。
2. 量化 `cuda_python_overlay` 和 `cuda_kernel_build` tier 相对 `cpu_python_overlay` 的区分度。
3. 验证 Codex Agent 在 kernel 级 C++/CUDA 修改任务上的可用性。
4. 建立 v0.5 引入多 agent 对比时的 v0.4 基线数据。

## 2. 当前数据集

数据集文件：

```text
datasets/pytorch_v0.4/dataset.json
```

当前状态：13 条 verified，0 条 draft

| Task | PR | Tier | Component | Patch 行数 |
| --- | --- | --- | --- | ---: |
| `pytorch__168295__autograd_create_graph` | #168295 | cpu_python_overlay | torch.autograd | 13 |
| `pytorch__150975__autograd_backward_inputs` | #150975 | cpu_python_overlay | torch.autograd | 46 |
| `pytorch__161488__lbfgs_wolfe` | #161488 | cpu_python_overlay | torch.optim | 20 |
| `pytorch__124385__load_state_dict_prefix` | #124385 | cpu_python_overlay | torch.nn.modules.module | 36 |
| `pytorch__143455__set_submodule` | #143455 | cpu_python_overlay | torch.nn.modules.module | 127 |
| `pytorch__162340__nn_arg_length` | #162340 | cpu_python_overlay | torch.nn.modules.conv | 44 |
| `pytorch__163961__dataloader_subset` | #163961 | cpu_python_overlay | torch.utils.data | 48 |
| `pytorch__149693__lazylinear_init` | #149693 | cpu_python_overlay | torch.nn | 14 |
| `pytorch__147599__lazylinear_state_forward` | #147599 | cpu_python_overlay | torch.nn | 24 |
| `pytorch__160952__bilinear_lazy_check` | #160952 | cpu_python_overlay | torch.nn | 24 |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | cuda_python_overlay | torch.nested._internal.sdpa | 64 |
| `pytorch__132616__cuda_mem_get_info` | #132616 | cuda_python_overlay | torch.cuda.memory | 26 |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | cuda_kernel_build | aten.native.cuda.softmax | 22 |

## 3. 实验配置

```text
Agent:   codex_action_bridge (codex-cli 0.134.0)
Dataset: datasets/pytorch_v0.4/dataset.json (--verified-only)
Repeat:  3
Output:  runs/v0.4_codex_cpu/ (10 CPU tasks)
         runs/v0.4_codex_gpu/ (3 GPU tasks, remote_docker on gpu-a10 / 4× V100)
```

分组：

- Batch A — CPU 10 条，本地 Colima Docker (darwin/amd64)，并发
- Batch B — GPU 3 条，远程 4× Tesla V100 SXM2 32GB，串行（`cuda_kernel_build` 占满 20 核 CPU 编译）

运行命令：

```bash
# Batch A：CPU
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --filter-tasks \
    pytorch__149693 pytorch__147599 pytorch__160952 pytorch__162340 \
    pytorch__163961 pytorch__168295 pytorch__161488 pytorch__150975 \
    pytorch__124385 pytorch__143455 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_codex_cpu

# Batch B：GPU（远程）
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only \
  --filter-tasks \
    pytorch__132835 pytorch__132616 pytorch__144009 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_codex_gpu
```

## 4. 实验结果

### 4.1 总体

| Metric | CPU (Batch A) | GPU (Batch B) | 合计 |
| --- | ---: | ---: | ---: |
| Task 数 | 10 | 3 | 13 |
| Agent 运行数 | 30 | 9 | 39 |
| Resolved | 24 | 9 | **33** |
| Resolved rate | 80.0% | 100.0% | **84.6%** |
| Median runtime | 45.9s | 82.2s | — |

### 4.2 逐 Task 结果

| Task | Tier | A1 | A2 | A3 | Rate | Median (s) |
| --- | --- | --- | --- | --- | :---: | ---: |
| `pytorch__149693__lazylinear_init` | cpu | resolved | resolved | resolved | 3/3 | 46 |
| `pytorch__147599__lazylinear_state_forward` | cpu | resolved | resolved | resolved | 3/3 | 37 |
| `pytorch__160952__bilinear_lazy_check` | cpu | resolved | resolved | resolved | 3/3 | 37 |
| `pytorch__168295__autograd_create_graph` | cpu | resolved | resolved | resolved | 3/3 | 48 |
| `pytorch__161488__lbfgs_wolfe` | cpu | resolved | resolved | resolved | 3/3 | 50 |
| `pytorch__150975__autograd_backward_inputs` | cpu | resolved | resolved | resolved | 3/3 | 48 |
| `pytorch__124385__load_state_dict_prefix` | cpu | resolved | resolved | resolved | 3/3 | 46 |
| `pytorch__143455__set_submodule` | cpu | resolved | resolved | resolved | 3/3 | 45 |
| `pytorch__162340__nn_arg_length` | cpu | fail | fail | fail | 0/3 | 46 |
| `pytorch__163961__dataloader_subset` | cpu | fail | fail | fail | 0/3 | 44 |
| `pytorch__132835__njt_sdpa_autocast` | cuda_py | resolved | resolved | resolved | 3/3 | 82 |
| `pytorch__132616__cuda_mem_get_info` | cuda_py | resolved | resolved | resolved | 3/3 | 74 |
| `pytorch__144009__softmax_ilpreduce_size` | cuda_kernel | resolved | resolved | resolved | 3/3 | 5449 |

### 4.3 分 tier 汇总

| Tier | Task 数 | Attempts | Resolved | Rate |
| --- | ---: | ---: | ---: | ---: |
| `cpu_python_overlay` | 10 | 30 | 24 | 80.0% |
| `cuda_python_overlay` | 2 | 6 | 6 | 100.0% |
| `cuda_kernel_build` | 1 | 3 | 3 | 100.0% |

Batch B 全 resolved，说明 remote_docker + inplace_build 通路稳定，但样本量偏小（3 条 task），单条 kernel_build 只能证明 tier 可运行、不能判断 tier 难度。

### 4.4 稳定性分类

- **稳定 resolved (3/3)**：11 条。CPU 8 条 + GPU 3 条。
- **稳定失败 (0/3)**：2 条，均为 CPU：`162340_nn_arg_length`、`163961_dataloader_subset`。
- **不稳定 (1-2 resolved)**：0 条。

相比 v0.3 (7 稳定 resolved、1 不稳定、2 稳定失败)，v0.4 消除了不稳定层。 `143455_set_submodule` 从 v0.3 的 2/3 提升到 v0.4 的 3/3。

## 5. 失败分析

### 5.1 `pytorch__162340__nn_arg_length` — 0/3（沿袭 v0.3）

Gold patch 涉及 2 个文件（`torch/nn/modules/utils.py` + `torch/nn/modules/conv.py`），核心是把非法 iterable 参数从 `ValueError` 改为带特定消息的 `AssertionError`，并处理 length-1 iterable 自动扩展。

Codex 的 patch 只改了 `utils.py`，抛的是 `ValueError`，缺 length-1 扩展和 `conv.py` hunk。隐藏测试 `test_fractional_max_pool2d_invalid_output_ratio` 断言 `assertRaisesRegex(AssertionError, "Expected an iterable of length 2, but got length 3")`，因此 `fail_to_pass_failed`。

失败根因：agent 理解了问题方向（校验 iterable 长度），但选错了异常类型和实现范围。

### 5.2 `pytorch__163961__dataloader_subset` — 0/3（沿袭 v0.3）

Gold patch 在 `Subset.__init__` 里做**实例化时检查**：若子类覆盖了 `__getitem__` 但没覆盖 `__getitems__`，直接抛 `NotImplementedError`（fail-fast 语义）。

Codex 的 patch 是在 `Subset.__getitems__` 内做**运行时检查**：若子类覆盖了 `__getitem__`，走 `self[idx]` fallback（silent fallback 语义）。

隐藏测试 `test_subset_override_getitem_requires_getitems` 明确期望 `IncompleteSubset(dataset, [...])` **在构造时**抛 `NotImplementedError`。Codex 的运行时 fallback 无法在实例化时触发异常，因此 `fail_to_pass_failed`。

失败根因：agent 与 gold 的 API 设计哲学不同（early-fail vs silent fallback），非知识盲区。

### 5.3 结论

两条失败均为**合法难负样本**：
- 修复方向正确
- 修复范围或异常语义与 gold 有偏差
- Hidden test 用精确断言把 gold 的设计选择固化

这两条 task 不需要修改，保留作为 v0.4 的区分度证据。

## 6. 过程观察

### 6.1 Remote executor 稳定性

- 3 次 SSH 长连接触发 keep-alive（`ServerAliveInterval=30`）；未见断线。
- rsync 排除 `.ccache/` 和 `build/` 后，`144009` 三次 attempt 的第二次和第三次 build 时间显著低于首次（ccache 命中）。
- 一次 subprocess timeout 后 `_kill_remote_container_processes` 成功清理远程僵尸进程（未污染后续 attempt）。

### 6.2 Codex rate limit 自动等待

- 本次 Codex 全量运行触发 2 次 rate limit，均在 CPU Batch A 中。
- 每次自动 sleep 18300s（5h5min）后重试，均在下一 attempt 成功。
- Batch B 未触发。

CPU wall clock 14h37m 中 10h10m 用于 rate-limit sleep。有效运行 4h27m。

### 6.3 本地 Colima QEMU 偶发 exit 139

- 3 次 attempt 中出现 1 次 agent 自测 subprocess `exit=139`（segfault）。
- 该 attempt 后续在 official evaluation 中通过（agent 自测不影响 official test）。
- 是 amd64 emulation 已知偶发，未定位到 root cause，暂不阻塞。
- 建议 v0.5 起把 CPU batch 也搬到 Linux GPU 主机跑，规避 QEMU 层。

### 6.4 CUDA kernel_build 时间构成

`144009_softmax_ilpreduce_size` 三次 attempt runtime：

```text
attempt 1: ~90 min (cold build + agent + evaluation build)
attempt 2: ~90 min (ccache 命中，仍需 inplace_build agent workspace)
attempt 3: ~90 min
median:    ~91 min
```

单条 task 编译占约 60min，agent 探索约 15-20min，evaluation 又需完整 inplace_build 再编译一次。ccache 减小 incremental compile 时长但完整 link 无法省略。

## 7. v0.3 → v0.4 对比

| 维度 | v0.3 | v0.4 |
| --- | --- | --- |
| Verified task 数 | 10 | 13 |
| Tier 数 | 1 (cpu_python_overlay) | 3 (+cuda_python_overlay, +cuda_kernel_build) |
| 组件覆盖 | 5 个 CPU 子系统 | 5 CPU + 3 CUDA 子系统 |
| Runtime 后端 | local docker | local docker + remote_docker (SSH) |
| Multi-file task | 1 条 | 1 条 |
| Agent | codex_action_bridge | codex_action_bridge |
| Resolved rate | 76.7% (23/30) | **84.6% (33/39)** |
| 稳定 resolved | 7 / 10 | 11 / 13 |
| 稳定失败 | 2 / 10 | 2 / 13 |
| 不稳定 | 1 / 10 | 0 / 13 |

Resolved rate 从 76.7% 升到 84.6%，主要贡献：
- `143455_set_submodule` 从 v0.3 的 2/3 变 3/3。
- 3 条 CUDA task 全 resolved（9/9）。

不稳定层消除，v0.3 的两条稳定失败在 v0.4 中依然稳定失败——说明其失败是设计而非采样偏差。

## 8. 下一步

`v0.5` 建议内容：

1. **多 Agent 对比**：接入 Claude Code agent 并复用 v0.4 的 13 条 task 做双 agent 3-repeat 对比。目标是量化 Codex vs Claude Code 的能力差和稳定性差。
2. **CPU 跑机 Linux 化**：把 CPU batch 从 macOS Colima/QEMU 迁到 Linux 主机，消除 amd64 emulation exit 139 偶发。
3. **扩 kernel_build tier**：新增 1-2 条 `cuda_kernel_build` task，样本量提到 ≥ 3 才能判断 tier 相对难度。
4. **失败样本再评估**：如果 `162340`/`163961` 在 Claude Code 下仍稳定失败，说明是 benchmark 层面的 hard case；否则说明是 agent 层面的 blind spot，两种结论都有价值。
5. **Public test ablation**：v0.4 无 task 携带 `public_test.patch`，机制未启用。v0.5 决定是否投入内容制作还是直接移除机制。

## 9. 可复现实验步骤

```bash
# 校验环境
PYTHONPATH=src python3 -m unittest discover tests   # 92 tests, all pass
PYTHONPATH=src python3 scripts/validate_dataset.py datasets/pytorch_v0.4/dataset.json

# CPU batch
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json --verified-only \
  --filter-tasks pytorch__149693 pytorch__147599 pytorch__160952 pytorch__162340 \
                 pytorch__163961 pytorch__168295 pytorch__161488 pytorch__150975 \
                 pytorch__124385 pytorch__143455 \
  --agent codex_action_bridge --agent-repeat 3 \
  --output-dir runs/v0.4_codex_cpu_reproduce

# GPU batch（需要 configs/remote_hosts.json 指向配好 nvidia-docker 的 host）
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json --verified-only \
  --filter-tasks pytorch__132835 pytorch__132616 pytorch__144009 \
  --agent codex_action_bridge --agent-repeat 3 \
  --output-dir runs/v0.4_codex_gpu_reproduce
```

## 10. 结论

v0.4 完成了 tier 扩展和 remote GPU 通路建设：

- 13 条 verified task，覆盖 3 个 runtime tier。
- Remote SSH Docker 执行器在 3 条 GPU task × 3 repeat 上稳定运行；ccache/rsync exclude/`_kill_remote_container_processes` 三项工程改造均在真实数据上被验证。
- Codex CLI 全量 resolved rate 84.6%（33/39），比 v0.3 的 76.7% 提升 8pp。
- `cuda_kernel_build` tier 首次跑通：144009 三次 attempt 全 resolved，单次约 90min。
- 两条稳定失败沿袭自 v0.3，属于合法难负样本，保留作 benchmark 区分度证据。
- Claude Code 多 agent 对比因外部条件推迟，实验设计和 baseline 已就绪，v0.5 可直接接入。
