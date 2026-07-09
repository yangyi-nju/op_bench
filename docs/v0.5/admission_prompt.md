# v0.5 Task Admission — Prompt Template for Remote Agent

给远程 agent 用于跑新 task admission 的模板。首条 pilot 是 `#140557`；后续 5 条批量走同一流程，只是 `--task` 目标不同。

---

## 后续 5 条批量

你是 OpBench 项目的远程执行 agent。Pilot task `#140557` 已 verified。
现在对其余 5 条 v0.5 precision task 跑批量 admission。

### 前置检查

```bash
cd <op_bench_root>
git pull --ff-only origin main   # 拉到 fe5b4ab 或更新
```

**⚠️ 第一步：确认 configs/remote_hosts.json 存在**

这个文件不在 git 里（含服务器 IP 等敏感信息）。如果不存在，先创建：

```bash
cat > configs/remote_hosts.json << 'EOF'
{
  "hosts": {
    "gpu-a10": {
      "user": "<your-username>",
      "hostname": "<server-ip-or-hostname>",
      "port": 22,
      "identity_file": "~/.ssh/<your_key>",
      "remote_workspace_root": "/data/op_bench_workspaces"
    }
  }
}
EOF
```

字段说明：与之前 v0.4 用的同一份配置文件一致，照旧就好。

**⚠️ 第二步：确认 Docker 镜像已构建**

运行 admission 前检查服务器上对应镜像是否存在：

```bash
# CPU 镜像（4 条 cpu_python_overlay task 用）
ssh <server> "docker image inspect op-bench/pytorch-cpu:torch2.6.0-py311 2>&1 | head -3"

# CUDA overlay 镜像（#129154 exp_decomp 用）
ssh <server> "docker image inspect op-bench/pytorch-cuda:torch2.6.0-cu124-py311 2>&1 | head -3"

# CUDA devel 镜像（#139372 histc kernel_build 用）
ssh <server> "docker image inspect op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311 2>&1 | head -3"
```

如果镜像不存在，先在服务器上构建（以 CPU 为例）：

```bash
# 本地触发远程构建
ssh <server> "docker build \
  -t op-bench/pytorch-cpu:torch2.6.0-py311 \
  -f /path/to/op_bench/environments/pytorch-cpu/Dockerfile \
  /path/to/op_bench/environments/pytorch-cpu"

# 或者本地构建后 docker save | ssh | docker load
docker build -t op-bench/pytorch-cpu:torch2.6.0-py311 \
  -f environments/pytorch-cpu/Dockerfile environments/pytorch-cpu
docker save op-bench/pytorch-cpu:torch2.6.0-py311 | \
  ssh <server> "docker load"
```

CUDA overlay 和 CUDA devel 镜像同理，Dockerfile 分别在
`environments/pytorch-cuda/` 和 `environments/pytorch-cuda-devel/`。

### 步骤 1：拉源码 snapshot（一次跑完 5 条）

```bash
PYTHONPATH=src python3 scripts/setup_sources.py
```

5 个新的 base commit 都在 sources/registry.json 里。顺序拉，每个约 3 分钟，
共约 15 分钟。期望：`Done: N ok, 0 failed`。

### 步骤 2：离线预检（每条独立跑）

```bash
for task in 139999_masked_mean_bool_upcast \
            129138_linear_add_bias_autocast \
            129154_exp_decomp_numerics \
            144073_vector_norm_scalar_overflow \
            139372_histc_int8_cuda_bounds; do
  echo "=== preflight: $task ==="
  PYTHONPATH=src python3 scripts/preflight_task.py tasks/pytorch/$task
done
```

期望每条都输出 `PREFLIGHT OK`。如有任何一条 `PREFLIGHT FAILED`，把
完整输出贴回来，暂停该条不推进，其他条可继续。

### 步骤 3：admission — CPU 4 条串行

```bash
for task in 139999_masked_mean_bool_upcast \
            129138_linear_add_bias_autocast \
            144073_vector_norm_scalar_overflow; do
  echo "=== admitting $task ==="
  OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
  PYTHONPATH=src python3 scripts/run_admission.py \
    --task tasks/pytorch/$task \
    --output-dir runs/admission/pytorch__${task}/v1 \
    --write-task-evidence 2>&1 | tee /tmp/admission_${task}.log
done
```

注意 `#129154` exp_decomp 是 `cuda_python_overlay`（需要 GPU），单独放到 GPU 队列：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_admission.py \
  --task tasks/pytorch/129154_exp_decomp_numerics \
  --output-dir runs/admission/pytorch__129154_exp_decomp_numerics/v1 \
  --write-task-evidence 2>&1 | tee /tmp/admission_129154.log
```

### 步骤 4：admission — P5 CUDA kernel_build（最后单独跑）

约 90 分钟，建议 tmux 包住：

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_admission.py \
  --task tasks/pytorch/139372_histc_int8_cuda_bounds \
  --output-dir runs/admission/pytorch__139372_histc_int8_cuda_bounds/v1 \
  --write-task-evidence 2>&1 | tee /tmp/admission_139372.log
```

### 步骤 5：报告

对每条 task 报告：

1. 最后的 JSON 判定（`decision` 字段）
2. 如果 verified：`cat tasks/pytorch/<task>/admission/evidence.json | python3 -m json.tool | head -10`
3. 如果不是 verified：关键失败阶段的 stderr 摘要

**不要改任何 task.json / gold.patch / hidden_test.patch。**
**不要提前 push。** 维护者看到报告后更新 task.json 状态并统一 push。

### 如果 admission 失败

- `baseline_not_reproduced`：说明 base commit 上 gold 前测试就过了，
  task 的 baseline 假设不成立，报回来由维护者判断。
- `gold_not_resolved`：gold.patch 没修好测试，报回来由维护者重新提取 patch。
- `environment_unavailable`：镜像缺失或 SSH 连不到，先检查步骤 0 的镜像
  和 configs/remote_hosts.json。

---

## 附：如何确认 configs/remote_hosts.json 格式

参见 v0.4 实验时用过的格式（`src/op_bench/remote.py` RemoteHost 文档）：

```json
{
  "hosts": {
    "gpu-a10": {
      "user": "ubuntu",
      "hostname": "10.0.0.X",
      "port": 22,
      "identity_file": "~/.ssh/KeyPair-02-openssh",
      "remote_workspace_root": "/data/op_bench_workspaces"
    }
  }
}
```

`OP_BENCH_REMOTE_HOSTS_PATH` 指向该文件路径（支持绝对路径或相对路径）。
如果已经有一份 v0.4 用的 hosts 配置，直接复用即可。
