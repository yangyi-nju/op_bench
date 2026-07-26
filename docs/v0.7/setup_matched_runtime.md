# OpBench v0.7 Matched Runtime 操作指南

日期：2026-07-27

状态：P2 Passed

## 1. 目的与判定边界

Matched Runtime 用于判断冻结 Base Commit 的 Python 源码能否在一个可复现
运行时中被真实加载和测试。策略顺序固定为：

1. 官方 matched wheel；
2. 从 Base Commit 构建的 wheel；
3. 完整 source build。

只有前一策略得到带稳定 failure code 的不兼容结论时才进入下一策略。本轮两个
任务都在第一策略通过，因此没有执行 source-built wheel 或 full source build，
也没有使用预留的 ccache key/build flags。

Compatibility 不是 F2P。它只验证源码身份、运行时身份、目标文件来源、目标模块
导入、F2P/P2P selector 收集和非 F2P 最小操作。Baseline Failure、Gold Success
和 P2P 保持必须由之后的 Fresh Admission 在独立 workspace 中证明。

## 2. 冻结运行时资产

### 2.1 PyTorch #129154

- 环境：`pytorch-matched-ff89ebc-torch2.4.0-py311-cu124`
- 镜像：`op-bench/pytorch-matched-ff89ebc:torch2.4.0-cu124-py311`
- 实测 local image ID：
  `sha256:f7fdabf3d4d9fc01c8d0f67961986968b06eb49d3724361c7ce64c1564f865c7`
- wheel：
  `https://download-r2.pytorch.org/whl/cu124/torch-2.4.0%2Bcu124-cp311-cp311-linux_x86_64.whl`
- wheel SHA-256：
  `81397ff1c84a3f2c666d2627144ecac268665325726267e092a80113385ad3e8`
- Python ABI：`cp311-cp311`
- CUDA build/runtime：`12.4` / `12.4`
- 实测设备能力：`7.0`

构建：

```bash
docker build --pull=false \
  --tag op-bench/pytorch-matched-ff89ebc:torch2.4.0-cu124-py311 \
  environments/pytorch-matched-ff89ebc-cu124
```

### 2.2 PyTorch #144073

- 环境：`pytorch-matched-06e9dea-torch2.7.0-py311-cpu`
- 镜像：`op-bench/pytorch-matched-06e9dea:torch2.7.0-cpu-py311`
- 实测 local image ID：
  `sha256:ccd5eb7b2703b9b2ac7c7a9e47cd56ffe77135e3a62b29be4d821833e318056f`
- torch wheel：
  `https://download-r2.pytorch.org/whl/cpu/torch-2.7.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl`
- torch wheel SHA-256：
  `6b7edcbf8bb0b9ac2e6c001434797c5ec3f25394f91eb0ed7aeeeeed9ad4500f`
- companion torchvision wheel：
  `https://download-r2.pytorch.org/whl/cpu/torchvision-0.22.0%2Bcpu-cp311-cp311-manylinux_2_28_x86_64.whl`
- companion SHA-256：
  `670082705cfb51a35ae35090b5a0e66ec09e6d9c3845e16417399adec7a17ff2`
- Python ABI：`cp311-cp311`
- source overlay：
  `torch/_refs/linalg/__init__.py` 和
  `torch/testing/_internal/inductor_utils.py`

第二个 overlay 是冻结 selector 所需的同 Base-Commit 测试辅助模块，不扩大
Agent Patch Scope。Agent 仍只能修改 `torch/_refs/linalg/__init__.py`。

构建：

```bash
docker build --pull=false \
  --tag op-bench/pytorch-matched-06e9dea:torch2.7.0-cpu-py311 \
  environments/pytorch-matched-06e9dea-cpu
```

## 3. Probe 与证据验证

真实远端连接参数保存在被 Git 忽略的 `configs/remote_hosts.json`。以下命令不把
主机地址、密钥路径或原始命令输出写入公开 evidence。

任务 #129154：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/probe_matched_runtime.py \
  --task tasks/pytorch/129154_exp_decomp_numerics \
  --strategy matched_wheel \
  --artifact-kind official_wheel \
  --artifact-id 'torch-2.4.0+cu124-cp311-cp311-linux_x86_64.whl' \
  --artifact-digest sha256:81397ff1c84a3f2c666d2627144ecac268665325726267e092a80113385ad3e8 \
  --artifact-digest-kind wheel_sha256 \
  --output runs/v0.7_matched_runtime/pytorch__129154__exp_decomp_numerics/matched_wheel/evidence.json
```

任务 #144073：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/probe_matched_runtime.py \
  --task tasks/pytorch/144073_vector_norm_scalar_overflow \
  --strategy matched_wheel \
  --artifact-kind official_wheel \
  --artifact-id 'torch-2.7.0+cpu-cp311-cp311-manylinux_2_28_x86_64.whl' \
  --artifact-digest sha256:6b7edcbf8bb0b9ac2e6c001434797c5ec3f25394f91eb0ed7aeeeeed9ad4500f \
  --artifact-digest-kind wheel_sha256 \
  --output runs/v0.7_matched_runtime/pytorch__144073__vector_norm_scalar_overflow/matched_wheel/evidence.json
```

输出文件是 append-by-new-path 的审计资产；CLI 拒绝覆盖已有 evidence。重跑时使用
新的时间戳目录或文件名，验证通过后再把选定证据固化为任务的
`compatibility/evidence.json`。

双重验证：

```bash
PYTHONPATH=src python scripts/validate_matched_runtime_evidence.py \
  tasks/pytorch/129154_exp_decomp_numerics/compatibility/evidence.json \
  --schema schemas/matched_runtime_compatibility.schema.json

PYTHONPATH=src python scripts/validate_matched_runtime_evidence.py \
  tasks/pytorch/144073_vector_norm_scalar_overflow/compatibility/evidence.json \
  --schema schemas/matched_runtime_compatibility.schema.json
```

## 4. Admission 与 Promotion

先运行 preflight：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/preflight_task.py tasks/pytorch/129154_exp_decomp_numerics

OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/preflight_task.py tasks/pytorch/144073_vector_norm_scalar_overflow
```

Admission：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/run_admission.py \
  --task tasks/pytorch/129154_exp_decomp_numerics \
  --output-dir runs/v0.7_matched_runtime/pytorch__129154__exp_decomp_numerics/admission \
  --write-task-evidence

OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json PYTHONPATH=src \
python scripts/run_admission.py \
  --task tasks/pytorch/144073_vector_norm_scalar_overflow \
  --output-dir runs/v0.7_matched_runtime/pytorch__144073__vector_norm_scalar_overflow/admission \
  --write-task-evidence
```

只有 compatibility=`compatible`、Admission=`verified`、replay hash 一致且每个
F2P/P2P 都真实执行时，才运行 promotion：

```bash
PYTHONPATH=src python scripts/promote_matched_runtime_task.py \
  --task tasks/pytorch/129154_exp_decomp_numerics/task.json \
  --compatibility-evidence tasks/pytorch/129154_exp_decomp_numerics/compatibility/evidence.json \
  --admission-evidence tasks/pytorch/129154_exp_decomp_numerics/admission/evidence.json \
  --verified-at 2026-07-26T16:08:26Z

PYTHONPATH=src python scripts/promote_matched_runtime_task.py \
  --task tasks/pytorch/144073_vector_norm_scalar_overflow/task.json \
  --compatibility-evidence tasks/pytorch/144073_vector_norm_scalar_overflow/compatibility/evidence.json \
  --admission-evidence tasks/pytorch/144073_vector_norm_scalar_overflow/admission/evidence.json \
  --verified-at 2026-07-26T16:30:16Z
```

Promotion 失败时 `task.json` 字节保持不变。

## 5. 本轮结论

| Task | Compatibility | Baseline | Gold | 结论 |
| --- | --- | --- | --- | --- |
| #129154 | 6/6 checks passed | F2P 0/1，P2P 1/1 | F2P 1/1，P2P 1/1 | verified |
| #144073 | 6/6 checks passed | F2P 0/1，P2P 1/1 | F2P 1/1，P2P 1/1 | verified |

本轮目标中的 2/2 条任务恢复为 verified；仓库另有 7 条历史 deprecated Task，
不属于 P2 的恢复范围。正式 Precision/Cumulative Dataset 纳入留到 P4 Freeze。

公开 compatibility content hash：

- #129154：
  `sha256:8b444b60b6c61c9f1261dc8d8306ece8db13915208efef5ea9fbf5fd4abf018f`
- #144073：
  `sha256:eb25e02ac1ed39e11911e32c1e0bd97585e4e83ba6df27ce3735628c18817171`

诊断中遇到的稳定失败包括 probe 使用不兼容私有 CUDA API、workspace 源码阴影、
未应用 hidden test patch、旧 torchvision companion、缺少同 commit
`RUN_CPU` 测试辅助符号，以及 Gold Patch 超出 Agent Scope。它们均在平台、
运行时或任务资产层修复后重新完整执行；没有把失败结果改写成通过。

## 6. 清理、重跑与测试不变性

- 完整日志只保存在被忽略的 `runs/v0.7_matched_runtime/`，公开 evidence 只含
  有界摘要和哈希。
- 正常 probe/Admission 会通过 EnvironmentManager 删除本次精确容器与远端
  workspace。异常时先用运行日志中的 exact container identity 检查，不使用
  广泛 Docker 清理。
- 构建镜像是注册资产，不随一次运行自动删除；需要重建时使用相同 tag，重新
  `docker image inspect` 并更新 registry/task 中的实测 ID。
- 重跑不得覆盖 canonical task evidence；先写新 run 路径，再通过契约、Schema、
  replay hash 和 promotion gate 选择结论。
- hidden F2P/P2P 内容没有为 compatibility 或 Admission 放宽、跳过或改写。
  #144073 的 Gold Patch 仅删除了任务 scope 外、且不参与冻结 F2P/P2P 的 OpInfo
  expected-failure 元数据 hunk，以保持 Gold 与 Agent 的单文件权限一致。

最终验证通过 47 项 matched-runtime focused tests 与 793 项全仓测试
（unittest 报告耗时 1150.968 秒），并通过 compileall、全部 tracked JSON
解析、17-task v0.5 verified Dataset 校验、两条 Task/Compatibility 校验、
Admission replay hash 对比和 `git diff --check`。
