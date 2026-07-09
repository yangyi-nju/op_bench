# v0.5 Task Admission — Prompt Template for Remote Agent

给远程 agent 用于跑新 task admission 的模板。首条 pilot 是 `#140557`；后续 5 条批量走同一流程，只是 `--task` 目标不同。

---

## 首条 pilot（#140557）

你是 OpBench 项目的远程执行 agent。任务：对新加入的 pilot task
`pytorch__140557__layer_norm_decomp_precision` 跑 admission，验证 v0.5 首条
precision 任务能走通全流程。

### 前置

```bash
cd <op_bench_root>
git pull --ff-only origin main   # 拉到 ebcaff0 或更新
```

必读：
- `tasks/pytorch/140557_layer_norm_decomp_precision/task.json` — 任务定义
- `tasks/pytorch/140557_layer_norm_decomp_precision/issue.md` — 复现说明
- `docs/v0.5/setup_remote_agent.md` — 远程环境说明

前置资源：
- `configs/remote_hosts.json` 配置好指向服务器
- 服务器有 CPU docker image `op-bench/pytorch-cpu:torch2.6.0-py311`

### 步骤 1：拉源码 snapshot

```bash
PYTHONPATH=src python3 scripts/setup_sources.py
```

期望：`sources/registry.json` 里新加的 `pytorch-240aa77-python-overlay`
会被拉到
`.op_bench_cache/sources/pytorch/pytorch/240aa77ad01c4f0cd9b2417748272f2f617c112f/source/`
（sparse checkout `torch/` + `test/`，几分钟）。

### 步骤 2：离线预检

```bash
PYTHONPATH=src python3 scripts/preflight_task.py \
  tasks/pytorch/140557_layer_norm_decomp_precision
```

期望：
- snapshot 存在
- `hidden_test.patch` 可以 `git apply`
- `gold.patch` 可以 `git apply`（在 `hidden_test.patch` 之上）
- 两个测试名 `DecompOneOffTestsCPU.test_native_layer_norm_cpu_decomp_cpu`
  和 `DecompOneOffTestsCPU.test_contiguous_softmax_cpu` 可解析
- `PREFLIGHT OK`

如果 preflight 失败，把完整输出贴回来，暂停不推进。

### 步骤 3：正式 admission

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_admission.py \
  --task tasks/pytorch/140557_layer_norm_decomp_precision \
  --output-dir runs/admission/pytorch__140557__layer_norm_decomp_precision/v1 \
  --write-task-evidence
```

期望的 JSON 输出（最后一行）：

```json
{"baseline": "baseline_reproduced", "gold": "resolved", "decision": "verified", ...}
```

同时 `task.json` 里 `admission.status` 会从 `draft` 变 `verified`，
`tasks/pytorch/140557_layer_norm_decomp_precision/admission/evidence.json` 生成。

### 步骤 4：报告

把下面几项贴回来：

1. `setup_sources.py` 最后一段输出（确认 snapshot 拉到）
2. `preflight_task.py` 完整输出
3. `run_admission.py` 最后 20 行 stderr + 最后的 JSON 判定
4. 生成的 `admission/evidence.json` 的 `decision` 字段
5. 如果 decision 是 verified：`git diff --stat` 确认哪些文件变化
   （应该是 `task.json` + `evidence.json`）
6. 如果 decision 不是 verified：baseline 或 gold 阶段的完整 stderr 摘要，
   **不要自作主张改任务定义**

### 硬性约束

1. **不要提前推 commit**。所有报告先发回来，维护者确认后再决定推 main。
2. **不要修改任何 task.json、gold.patch、hidden_test.patch**。这些是 pilot
   数据，问题要暴露出来而不是被静默 patch 掉。
3. 出错先看 preflight。preflight 通过但 admission 失败往往说明 baseline
   或 gold 有真实内容问题，非平台问题。

### 时间预期

`setup_sources` ~3 min（浅 clone + sparse checkout）。preflight <30s。
admission ~10 min（CPU python_overlay，无 build）。总共不到 20 分钟。

---

## 后续 5 条批量（等 pilot verified 后）

Pilot verified 后，维护者会推 5 条新 task bundle：

- `#139999` masked.mean bool（P2, CPU）
- `#129138` linear_add_bias（P3, CPU）
- `#129154` exp decomp（P4, CPU）
- `#144073` vector_norm overflow（P4, CPU）
- `#139372` histc int8（P5, CUDA kernel_build ~90min）

流程同上。三点差异：

1. `setup_sources.py` 一次跑，会把 5 条对应的 snapshot 都拉下来
   （5 × ~3min 顺序完成）。
2. 每条独立跑 `preflight_task.py` + `run_admission.py`。
3. `#139372` 是 `cuda_kernel_build` tier，单次 admission 约 90 分钟
   （首次 build 60min + 增量 build + baseline/gold 各跑一次）。安排最后跑，
   前 4 条 CPU 完成后再启动。

对每条 task 用同样的四步报告：`setup_sources` 输出、preflight、admission
JSON、evidence decision、涉及的 git diff。

### 批量执行建议

CPU 4 条串行跑，`--max-parallel` 保持 1（每条只有 baseline + gold，共 2 次
容器启动，并发意义不大）：

```bash
for task in 139999_masked_mean_bool_upcast \
            129138_linear_add_bias_autocast \
            129154_exp_decomp_numerics \
            144073_vector_norm_scalar_overflow; do
  OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
  PYTHONPATH=src python3 scripts/run_admission.py \
    --task tasks/pytorch/$task \
    --output-dir runs/admission/pytorch__$task/v1 \
    --write-task-evidence 2>&1 | tee /tmp/admission_$task.log
done
```

最后单独跑 `#139372`：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_admission.py \
  --task tasks/pytorch/139372_histc_int8_cuda_bounds \
  --output-dir runs/admission/pytorch__139372_histc_int8_cuda_bounds/v1 \
  --write-task-evidence 2>&1 | tee /tmp/admission_139372.log
```

### 后续如果某条 admission 失败

- 记录失败 task 的完整 preflight + admission log
- **不改任务定义**
- 报告回来，维护者会静态分析（patch 是否需要重新提取、test 名是否有隐藏
  decorator、base commit 是否有 wheel 不兼容问题）
- Passed 的其余 task 正常写 evidence 到 main，failed 的挂在 draft 状态即可
