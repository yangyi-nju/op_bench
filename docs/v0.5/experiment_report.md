# OpBench v0.5 Precision 阶段实验报告

日期：2026-07-11

## 0. 阶段结论

v0.5 当前完成的是 precision 维度阶段，不是最终冻结版本。累计数据集 manifest 已扩展为 17 条 verified task（v0.4 的 13 条加 4 条新 precision task）；本轮只重跑其中 6 条 precision task，共 18 次 Codex attempt。边界、兼容等后续维度加入后，再对冻结后的 v0.5 manifest 做 17+ task 全量重跑。

本轮 Codex resolved rate 为 **72.2%（13/18）**，tier-weighted score 为 **78.8%**。P2 和 P5 均为 100%，P1 为 0%，P3 为 66.7%；P4 尚无 admitted task，记为 N/A 而不是 0%。结果说明 precision slice 已经提供稳定成功、稳定失败和不稳定三种区分度。

## 1. 数据集与实验范围

数据集文件：

```text
datasets/pytorch_v0.5/dataset.json
```

manifest 状态为 `draft`，包含 17 条 verified task：13 条继承自 v0.4，新增 #140557、#139999、#129138、#139372。已 deprecated 的 #129154 和 #144073 不进入正式清单。

本轮 precision slice 包含 6 条：

| Task | PR | Subclass | Tier | Difficulty | Component |
| --- | ---: | :---: | --- | --- | --- |
| `pytorch__140557__layer_norm_decomp_precision` | #140557 | P1 | `cpu_python_overlay` | easy | `torch._refs.native_layer_norm` |
| `pytorch__139999__masked_mean_bool_upcast` | #139999 | P2 | `cpu_python_overlay` | easy | `torch.masked._ops.mean` |
| `pytorch__129138__linear_add_bias_autocast` | #129138 | P3 | `cpu_python_overlay` | easy | `torch._inductor.fx_passes.mkldnn_fusion` |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | P3 | `cuda_python_overlay` | medium | `torch.nested._internal.sdpa` |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | P5 | `cuda_kernel_build` | hard | `aten.native.cuda.softmax` |
| `pytorch__139372__histc_int8_cuda_bounds` | #139372 | P5 | `cuda_kernel_build` | hard | `aten.native.cuda.histc` |

选择累计 precision slice 而不是只跑 4 条新增 task，是为了保留 v0.4 的 P3/P5 锚点，同时避免在后续维度尚未冻结时重复支付 13 条历史 task 的全量成本。

## 2. 实验配置

```text
Agent:   codex_action_bridge (codex-cli 0.144.0-alpha.4)
Dataset: datasets/pytorch_v0.5/dataset.json (--verified-only)
Scope:   problem_dimension=precision, 6 tasks
Repeat:  3
Output:  runs/v0.5_precision_codex_cpu/
         runs/v0.5_precision_codex_gpu/
Summary: runs/v0.5_precision_codex/summary.json
Host:    remote Linux, 4 x Tesla V100 SXM2 32GB
```

CPU 和 GPU 分批运行；GPU batch 使用 `max-parallel=1`，避免两个 kernel build 争用编译资源。正式汇总命令：

```bash
PYTHONPATH=src python3 scripts/aggregate_experiments.py \
  runs/v0.5_precision_codex_cpu \
  runs/v0.5_precision_codex_gpu \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --output runs/v0.5_precision_codex/summary.json
```

## 3. 总体结果

### 3.1 八维指标

| Metric | Result | Interpretation |
| --- | ---: | --- |
| Resolved rate | **72.2% (13/18)** | 主指标 |
| Patch conciseness | **1.000** | resolved attempt 的中位数；比 gold 更短时 clamp 为 1 |
| Pass-to-pass kept rate | **83.3% (15/18)** | P1 的 3 次 import failure 同时破坏 P2P |
| Fail-to-pass-only / strict resolved rate | **72.2% (13/18)** | F2P 与 P2P 同时通过，等于 resolved rate |
| Regression rate | **0.0% (0/18)** | 没有“F2P 已修复但 P2P 被破坏”的 attempt |
| Tier-weighted score | **78.8%** | cpu=1、cuda_py=2、cuda_kernel=3 |
| Per-problem resolved rate | 见 3.3 | 按 P1-P5 和具体 `problem_type` 拆分 |
| Median evaluator runtime | **107.6s** | 仅 official evaluator；不含 agent 探索和远程源码传输 |

`pass_to_pass_regressed` 是 evaluator 的宽泛状态标签：P1 三次实际都是 F2P=0/1、P2P=0/1，并不满足扩展指标对 regression 的严格定义。因此报告以 `regression_rate=0` 为准，同时保留原始状态供追溯。

### 3.2 逐 task 结果

| Task | Subclass | A1 | A2 | A3 | Rate | Median evaluator (s) |
| --- | :---: | --- | --- | --- | ---: | ---: |
| `140557_layer_norm_decomp_precision` | P1 | fail | fail | fail | 0/3 | 101.0 |
| `139999_masked_mean_bool_upcast` | P2 | resolved | resolved | resolved | 3/3 | 90.3 |
| `129138_linear_add_bias_autocast` | P3 | resolved | resolved | resolved | 3/3 | 122.4 |
| `132835_njt_sdpa_autocast` | P3 | resolved | fail | fail | 1/3 | 88.5 |
| `144009_softmax_ilpreduce_size` | P5 | resolved | resolved | resolved | 3/3 | 579.6 |
| `139372_histc_int8_cuda_bounds` | P5 | resolved | resolved | resolved | 3/3 | 607.2 |

稳定性分类：4 条稳定 resolved，1 条稳定失败（P1），1 条不稳定（P3 CUDA autocast）。

### 3.3 按 subclass

| Subclass | Tasks | Attempts | Resolved | Rate |
| :---: | ---: | ---: | ---: | ---: |
| P1 数值累积/分解精度 | 1 | 3 | 0 | 0.0% |
| P2 dtype 转换损失 | 1 | 3 | 3 | 100.0% |
| P3 混合精度不一致 | 2 | 6 | 4 | 66.7% |
| P4 数值不稳定 | 0 | 0 | N/A | N/A |
| P5 Kernel 精度 bug | 2 | 6 | 6 | 100.0% |

按具体 `problem_type`：`decomposition-dtype-parity` 0/3，`dtype-accumulation-error` 3/3，`autocast-dtype-mismatch` 3/3，`dtype-precision` 1/3，`kernel-index-underflow` 3/3，`integer-bounds-precision` 3/3。

### 3.4 按 tier

| Tier | Tasks | Attempts | Resolved | Rate |
| --- | ---: | ---: | ---: | ---: |
| `cpu_python_overlay` | 3 | 9 | 6 | 66.7% |
| `cuda_python_overlay` | 1 | 3 | 1 | 33.3% |
| `cuda_kernel_build` | 2 | 6 | 6 | 100.0% |

kernel tier 的 100% 说明当前两条 kernel task 对实现定位清晰，但样本仍只有 2 条，不能推出 kernel bug 普遍更容易。

## 4. 失败分析

### 4.1 P1 `#140557`：0/3

三次补丁都识别到需要为 `aten.native_layer_norm.default` 补 fake 路径，但使用了 `torch.library.register_fake`。三种写法分别触发重复 Meta kernel、非法 qualname、非法 overload name，导致 `test_decomp.py` 在 import 阶段失败，F2P 和 P2P 都未执行到断言。

Gold 使用的是内部 `torch._subclasses.fake_impls.register_op_impl`，并把实现委托给已有 decomposition。该 task 稳定区分了“理解需要 fake registration”和“掌握对应 PyTorch 内部 dispatch API”两种能力。

### 4.2 P3 `#132835`：1/3

三次都尝试在 jagged SDPA 路径中按 autocast dtype 转换 query/key/value。唯一 resolved 的补丁在 `_validate_sdpa_input` **之前**转换；另外两次把转换放在校验之后，因此 mixed dtype 输入先被 `_validate_sdpa_input` 拒绝，F2P 仍报 dtype mismatch。两次失败的 P2P 都通过，没有引入回归。

该 task 的不稳定性来自控制流位置，而不是 API 方向错误，适合作为 P3 的中等难度样本保留。

## 5. v0.4 对比

| Metric | v0.4 full | v0.5 precision phase |
| --- | ---: | ---: |
| Dataset tasks | 13 | 17 cumulative / 6 executed |
| Attempts | 39 | 18 |
| Resolved rate | 84.6% (33/39) | 72.2% (13/18) |
| Pass-to-pass kept | 100.0% | 83.3% |
| Regression rate | 0.0% | 0.0% |
| Tier-weighted score | 88.2% | 78.8% |

两组 scope 不同，resolved rate 的 -12.4pp 不能解释为 agent 整体退化。v0.4 的 13 条覆盖多种非精度问题，而本轮刻意集中于 precision。两条重叠 precision task 在 v0.4 是 6/6，本轮为 4/6：`#144009` 仍为 3/3，`#132835` 从 3/3 变为 1/3，表明 3-repeat 结果会受 agent 版本与采样波动影响。

v0.4 历史 patch path 已不能完整复算 patch conciseness，因此不把聚合脚本给出的 0 当作可比较观测；v0.5 的 1.000 是该指标首次有完整 artifact 支撑的数据。

## 6. 运行与复现观察

1. `op-bench/pytorch-cpu:torch2.6.0-py311` 的实际 image ID 已校正为 `sha256:721dd55c...`，并重新 admission #140557、#139999，避免 registry 声明与服务器 tag 漂移。
2. #144009 增加 `BUILD_TEST=0` 和 `TORCH_CUDA_ARCH_LIST=7.0` 后重新 admission verified。其 baseline 冷构建约 27 分钟，随后 ccache 增量构建约 3 分钟；本轮 official evaluator 中位数约 9.7 分钟，而 v0.4 约 90 分钟。
3. GPU 首次批次出现一次 SSH/rsync 瞬时断链。该批次被整体隔离到 `/tmp`，服务器磁盘、inode、Docker 均正常；独立 rsync 探针通过后，从干净输出目录重跑。最终 18 次统计不含 `environment_unavailable`。
4. kernel agent 自测两次以 exit 137 结束，但 official evaluator 均 resolved。自测发生在未完成 source load 的 agent 容器中，不作为评分依据。

## 7. 下一步

1. 为 P4 admission 真实数值不稳定 task；当前缺口必须保持 N/A，不能用非精度或版本错配 task 凑数。
2. 扩充边界、兼容等维度后冻结 `pytorch_v0.5`，再运行累计 17+ task 的 3-repeat 全量实验。
3. 冻结前修正 evaluator 的宽泛 `pass_to_pass_regressed` 状态命名，使其与严格 regression metric 一致；同时为 remote rsync 增加有界重试。
4. 全量重跑时保留本轮 8 维指标和 P1-P5 breakdown，避免只比较总体 resolved rate。

## 8. 复现命令

```bash
PYTHONPATH=src python3 scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json

OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
OP_BENCH_CODEX_TIMEOUT_SEC=1200 \
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --only-tasks \
    pytorch__132835__njt_sdpa_autocast \
    pytorch__144009__softmax_ilpreduce_size \
    pytorch__140557__layer_norm_decomp_precision \
    pytorch__139999__masked_mean_bool_upcast \
    pytorch__129138__linear_add_bias_autocast \
    pytorch__139372__histc_int8_cuda_bounds \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --max-parallel 1 \
  --output-dir runs/v0.5_precision_codex_reproduce
```
