# 远程执行环境配置

本文档记录如何让本地 OpBench 控制端通过 SSH 驱动远程 Docker。Codex CLI 在本地控制端运行，只能通过 action bridge 读写和测试目标 workspace；PyTorch runtime 与 official evaluator 在远程容器中运行。

## 前置条件

- 远程服务器已配置 SSH 访问（key 已在 `configs/remote_hosts.json` 注册）
- Docker + nvidia-container-toolkit 已安装（v0.4 admission 期间已验证）
- 本地控制端有 Python 3.11+、Codex CLI、SSH 和 rsync

服务器需要预先构建 task registry 引用的镜像。除了基础 CPU/CUDA 镜像，
调用 `torch.compile`/Inductor 的 CPU task 使用带 C++ 编译器和 Python headers 的
`op-bench/pytorch-cpu-compile:torch2.6.0-py311`：

```bash
docker build \
  -t op-bench/pytorch-cpu-compile:torch2.6.0-py311 \
  environments/pytorch-cpu-compile
```

## 在本地控制端安装 Codex CLI

```bash
# 通过 npm 安装（需要 node 18+）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g @openai/codex

# 验证
codex --version
```

## 在本地控制端配置 Codex 账号

Codex 使用已登录账号或 OpenAI API key。若使用 API key，在本地控制端：

```bash
# 写入 .env（codex 启动时会自动读取）
echo "OPENAI_API_KEY=<your-key>" >> ~/.env

# 或者直接 export（不持久）
export OPENAI_API_KEY=<your-key>
```

**注意**：并发 attempt 共享本地 Codex 账号 quota，也共享 SSH/rsync 链路和远端磁盘。CPU tier 从 `--max-parallel 1-3` 起步；只有在传输与 quota 稳定时再提高。

## 配置 op_bench 本地客户端指向服务器

`configs/remote_hosts.json`（不纳入 git，只在本机维护）：

```json
{
  "hosts": {
    "gpu-a10": {
      "user": "ubuntu",
      "hostname": "<server-ip-or-hostname>",
      "port": 22,
      "identity_file": "~/.ssh/your_key",
      "remote_workspace_root": "/data/op_bench_workspaces"
    }
  }
}
```

所有 tier（包括 CPU）的环境 registry 条目都声明了 `"host": "gpu-a10"`，因此本地发起的实验全部通过 SSH 路由到服务器。

## 运行实验

### CPU task（在服务器上远程执行）

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --filter-tasks pytorch__149693 pytorch__147599 pytorch__160952 \
                 pytorch__162340 pytorch__163961 pytorch__168295 \
                 pytorch__161488 pytorch__150975 pytorch__124385 pytorch__143455 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --max-parallel 3 \
  --output-dir runs/v0.5_codex_cpu
```

### GPU task（串行，避免 GPU 竞争）

```bash
OP_BENCH_REMOTE_HOSTS_PATH=configs/remote_hosts.json \
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --filter-tasks pytorch__132835 pytorch__132616 pytorch__144009 \
  --agent codex_action_bridge \
  --agent-repeat 3 \
  --max-parallel 1 \
  --output-dir runs/v0.5_codex_gpu
```

`--max-parallel 1` 对 GPU tier 是强烈建议的：`--gpus all` 分配不支持同时多 container，内核 build 本身也已吃满 20 核 CPU。

### 中断后续跑

直接重新执行相同命令即可。默认 resume 行为：已完成的 attempt 不重跑。

```bash
# 强制从头开始
--fresh

# 只重跑特定 task
--only-tasks pytorch__144009

# 调试时强制走本地 Docker（不走 SSH）
OP_BENCH_FORCE_LOCAL_DOCKER=1 ...
```

## 验证服务器环境

```bash
# SSH 通达
ssh -i ~/.ssh/your_key ubuntu@<server-ip> "echo ok"

# Docker + GPU
ssh -i ~/.ssh/your_key ubuntu@<server-ip> "docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"

# CPU image 可用
ssh -i ~/.ssh/your_key ubuntu@<server-ip> "docker image inspect op-bench/pytorch-cpu:torch2.6.0-py311 | jq '.[0].Id'"

# torch.compile CPU image 可用
ssh -i ~/.ssh/your_key ubuntu@<server-ip> "docker image inspect op-bench/pytorch-cpu-compile:torch2.6.0-py311 | jq '.[0].Id'"

# Codex CLI 可用
ssh -i ~/.ssh/your_key ubuntu@<server-ip> "codex --version"
```

## 注意事项

- `OP_BENCH_FORCE_LOCAL_DOCKER=1`：临时调试用，强制所有任务走本地 Docker（需要本地有 Colima/Docker）。
- 服务器磁盘：每个 workspace rsync 约 500MB（python_overlay），kernel_build 约 3GB。建议 `remote_workspace_root` 指向空间充裕的数据盘。
- kernel build 的 ccache 持久化在 `<remote_workspace_root>/_cache/ccache/<environment-id>`；随机 workspace 清理不会删除它。需要强制冷编译时再手动清理对应环境目录。
- rate-limit 行为：本地控制端 Codex 触发 rate limit 后，`_run_codex` 自动 sleep `OP_BENCH_CODEX_RATE_LIMIT_WAIT_SEC`（默认 18300s）。长期 sleep 建议用 `tmux` 包住整个实验命令，避免控制端会话中断。
