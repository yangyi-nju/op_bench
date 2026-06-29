# OpBench v0.4 设计方案

日期：2026-06-21

## 1. 背景

v0.3 完成了 10 条 verified PyTorch CPU task 的多组件评测，Codex CLI resolved rate 76.7% (23/30)。但存在几个明显局限：

1. 只有一个 agent (Codex CLI)，无法对比不同 agent 的能力差异。
2. 全部是 CPU-only Python 问题，不涉及 GPU/CUDA 环境。深度学习算子中大量 bug 与硬件相关（精度、device dispatch、kernel 行为差异）。
3. 7/10 task 稳定 resolved (3/3)，说明现有 task 偏简单，缺乏区分度。
4. Public test 机制已实现但未验证其对 agent 的实际帮助。

## 2. v0.4 目标

1. **多 Agent 对比**：引入 Claude Code 作为第二 agent，建立多 agent 对比基线。
2. **远程 GPU 环境**：支持通过 SSH 连接云 GPU 实例的 Docker，运行 CUDA 相关 task。
3. **更难的 task**：新增 5-10 条 task（含 CUDA 精度/device dispatch bug），gold patch 50-150 行，目标拉低 resolved rate 到 40-60%。
4. **Public test ablation**：验证 public test 对 agent 能力的影响，无价值则移除简化平台。
5. **数据集规模**：15-20 条 verified task。

## 3. 非目标

1. 不引入 PyTorch 以外的框架。
2. 不要求 C++/CUDA kernel 源码修改（gold patch 仍为 Python-level）。
3. 不做完整 leaderboard UI，只产出对比 summary.json。
4. 不默认支持 full PyTorch source build。

## 4. 技术设计

### 4.1 Claude Code Agent

复用 `ActionBridgeServer` + `opbench_action.py` 的 IPC 模式（和 Codex 完全一样）。差异仅在 CLI 调用方式。

**新增类**：`ClaudeCodeActionBridgeAgent`

```python
class ClaudeCodeActionBridgeAgent:
    name = "claude_code_action_bridge"
    requires_workspace = True
    requires_actions = True

    def produce_patch(self, task, output_dir, workspace, actions):
        # 1. 创建 scratch_dir
        # 2. 启动 ActionBridgeServer
        # 3. 生成 opbench_action.py
        # 4. 构建 prompt（共享 _build_bridge_prompt）
        # 5. 执行: claude --print --dangerously-skip-permissions -p "<prompt>"
        # 6. 收集 git_diff
```

**配置**：
- 环境变量 `OP_BENCH_CLAUDE_TIMEOUT_SEC`（默认 1200s）
- `agent_by_name("claude_code_action_bridge")` 注册

**改动文件**：
- `src/op_bench/agents.py` — 新增 class + 更新工厂函数

### 4.2 远程 GPU Docker 执行器

本地 macOS 无 GPU，通过 SSH 连接云 GPU 实例执行 Docker 命令。

**新文件**：`src/op_bench/remote.py`

```python
@dataclass(frozen=True)
class RemoteHost:
    user: str
    hostname: str
    port: int = 22
    identity_file: str | None = None
    remote_workspace_root: str = "/tmp/op_bench_workspaces"

class RemoteDockerExecutor:
    """所有 docker 命令前加 ssh user@host 前缀"""

    def sync_to_remote(self, local_workspace: Path) -> CommandResult
    def sync_from_remote(self, local_workspace: Path) -> CommandResult
    def start(self, cwd, timeout_sec) -> CommandResult   # docker run --gpus all
    def run(self, command, cwd, timeout_sec) -> CommandResult  # docker exec
    def close(self, timeout_sec) -> CommandResult  # docker rm -f + cleanup
```

**工作流**：

```
Local (macOS)                        Remote (GPU cloud)
─────────────                        ──────────────────
prepare():
  rsync workspace → remote      →   /tmp/op_bench_workspaces/<task>/
  ssh docker run --gpus all     →   container started with GPU
  ssh docker exec preflight     →   torch.cuda.is_available() = True

agent repair:
  action.run_command()          →   ssh docker exec <cmd>
  action.run_test()             →   ssh docker exec python test...

evaluate:
  rsync workspace ← remote      ←   git diff result
  score locally
```

**改动文件**：
- `src/op_bench/environment.py` — `prepare()` 新增 `remote_docker` backend
- `src/op_bench/task.py` — 新增 `requires_gpu`、`environment_host` 属性
- `environments/registry.json` — 新增 CUDA 环境条目
- `schemas/task_manifest.schema.json` — backend enum 添加 `"remote_docker"`

**主机配置** (`configs/remote_hosts.json`)：

```json
{
  "hosts": {
    "gpu-a100": {
      "user": "ubuntu",
      "hostname": "10.0.0.42",
      "port": 22,
      "identity_file": "~/.ssh/gpu_key",
      "remote_workspace_root": "/data/op_bench"
    }
  }
}
```

通过 `OP_BENCH_REMOTE_HOSTS_PATH` 环境变量指定路径。

### 4.3 CUDA Docker 环境

**新文件**：`environments/pytorch-cuda/Dockerfile`

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
RUN pip install --no-cache-dir pytest hypothesis expecttest
```

**环境 registry 条目**：

```json
{
  "id": "pytorch-cuda-torch2.6.0-py311-cu124",
  "framework": "pytorch",
  "runtime_tier": "cuda_python_overlay",
  "docker": {
    "image": "op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
    "dockerfile": "pytorch-cuda/Dockerfile",
    "build_context": "pytorch-cuda"
  },
  "hardware": {
    "requires_gpu": true,
    "cuda": "12.4",
    "device": "cuda",
    "min_memory_gb": 16
  },
  "host": "gpu-a100",
  "source_loading_modes": ["python_overlay"],
  "preflight": {
    "commands": [
      "python --version",
      "python -c \"import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))\""
    ]
  }
}
```

### 4.4 CUDA Task 类型与 Runtime Tier

v0.4 引入两个新 runtime tier：

**`cuda_python_overlay`** — Python-only CUDA bug（约 70% 数据）
- 修复仅涉及 Python 文件
- gold patch 50-150 行
- 复用 `python_overlay` source loading 模式
- 示例：dtype 精度、device dispatch、CUDA autograd 行为差异

**`cuda_kernel_build`** — C++/CUDA kernel 级 bug（约 20-30% 数据）
- 允许修改 `.cpp` / `.cu` / `.h` 文件
- 需要在远程 GPU 实例上做 PyTorch in-place rebuild
- 使用新增的 `inplace_build` source loading 模式
- gold patch 可以包含 kernel 代码
- 单次评测时间 5-60 分钟（首次 build 慢，ccache 后增量编译 2-5 分钟）

| 类型 | Tier | 示例 |
| --- | --- | --- |
| 精度累积 | cuda_python_overlay | float16 EmbeddingBag sum 精度丢失 |
| Device dispatch | cuda_python_overlay | 某 functional 漏转 intermediate tensor 到 CUDA |
| dtype 推断 | cuda_python_overlay | autocast 场景下 output dtype 不一致 |
| CUDA-specific autograd | cuda_python_overlay | create_graph 在 CUDA 上行为与 CPU 不同 |
| Kernel 边界条件 | cuda_kernel_build | scatter_add CUDA kernel 越界 |
| Kernel 算法错误 | cuda_kernel_build | reduction kernel 精度损失 |

**inplace_build 实现**：

```python
# source_loading.build_command 默认值
"cd {workspace_dir} && python setup.py develop --no-deps 2>&1 | tail -50"
```

- 在 cuda-devel image 内运行（含 nvcc + build toolchain）
- 用 ccache 加速增量编译（首次构建后 cache 在 /workspace/.ccache）
- 每个 task 可在 `source_loading.build_command` 覆盖默认命令

**Docker 镜像**：
- `op-bench/pytorch-cuda:torch2.6.0-cu124-py311` — for cuda_python_overlay
- `op-bench/pytorch-cuda-devel:torch2.6.0-cu124-py311` — for cuda_kernel_build（含 nvcc/cmake/ccache，~12GB）

### 4.5 Public Test Ablation

**实验设计**：

1. 为 5-6 条现有 task 创建 `artifacts/public_test.patch`
2. Public test 是 hidden test 的子集（同一个 test class，更少用例）
3. 跑两组实验：

```bash
# Group A: with public tests
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only --agent codex_action_bridge --agent claude_code_action_bridge \
  --agent-repeat 3 --output-dir runs/v0.4_with_public

# Group B: without public tests
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.4/dataset.json \
  --verified-only --no-public-tests \
  --agent codex_action_bridge --agent claude_code_action_bridge \
  --agent-repeat 3 --output-dir runs/v0.4_without_public
```

4. 比较两组 resolved rate。如果差异 < 5%，移除 public test 机制。

**改动**：`scripts/run_experiment.py` 新增 `--no-public-tests` flag。

## 5. 实现顺序

```
Phase 1（并行，无依赖）:
  ├── A: Claude Code agent          [1-2 天]
  └── D: Public test 素材准备        [1 天]

Phase 2（依赖 Phase 1-A 可测试）:
  └── B: Remote executor 基础设施    [3-4 天]

Phase 3（依赖 Phase 2-B）:
  └── C: CUDA task 构建 + admission  [3-5 天]

Phase 4（依赖全部）:
  └── 全量实验 + ablation + 报告
```

## 6. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| rsync 大 workspace 慢 | 远程保留 snapshot cache，只同步 overlay 增量 |
| nvidia-container-toolkit 未装 | preflight 先跑 `ssh host nvidia-smi` |
| Claude CLI 接口变化 | 抽象调用参数为常量，log 精确命令 |
| CUDA 测试非确定性 | 选 tolerance 断言，用 `use_deterministic_algorithms` |
| 远程 container 泄漏 | TTL label + cleanup 脚本 |
| SSH 连接不稳定 | 重试机制 + connection keep-alive |

## 7. 文件变更清单

### 新文件

| 文件 | 用途 |
| --- | --- |
| `src/op_bench/remote.py` | RemoteHost、RemoteDockerExecutor |
| `environments/pytorch-cuda/Dockerfile` | CUDA Docker 镜像 |
| `configs/remote_hosts.example.json` | 远程主机配置示例 |
| `datasets/pytorch_v0.4/dataset.json` | v0.4 数据集 |
| `tests/test_remote_executor.py` | 远程执行器单元测试 |
| `tests/test_claude_code_agent.py` | Claude Code agent 测试 |
| `docs/v0.4/design.md` | 本文档 |
| `docs/v0.4/experiment_report.md` | 实验报告（完成后） |

### 修改文件

| 文件 | 变更 |
| --- | --- |
| `src/op_bench/agents.py` | 新增 ClaudeCodeActionBridgeAgent + 工厂更新 |
| `src/op_bench/environment.py` | 新增 remote_docker backend 分支 |
| `src/op_bench/task.py` | 新增 requires_gpu、environment_host 属性 |
| `scripts/run_experiment.py` | 新增 --no-public-tests flag |
| `environments/registry.json` | 新增 CUDA 环境条目 |
| `sources/registry.json` | 新增 CUDA task source 条目 |
| `schemas/task_manifest.schema.json` | backend enum + host 字段 |
| `CHANGELOG.md` | v0.4 条目 |
| `README.md` | 更新 Current Dataset 和 Quick Start |

## 8. 验证方式

1. `PYTHONPATH=src python -m unittest discover tests -v` — 全部通过
2. `PYTHONPATH=src python scripts/preflight_task.py --all` — 所有 task 离线预检通过
3. Claude Code 在现有 CPU task 上跑通评测闭环
4. Remote executor preflight: `ssh gpu-host nvidia-smi` + `docker run --gpus all`
5. CUDA task admission: baseline fail + gold pass
6. 多 agent 对比实验产出 summary.json
7. Ablation 对比：with/without public tests 的 resolved rate

## 9. 运维注意事项 (Operational Gotchas)

实际跑实验时遇到并解决了以下问题，记录下来避免后续踩坑。

### 9.1 PyTorch 版本与 wheel 兼容性

Python overlay 模式下，base commit 必须是 PyTorch 2.6.0 release 附近的 commit。**post-2.6.0 nightly commit 不可用**，因为它们会依赖 wheel 里不存在的符号（如 `torch.float8_e8m0fnu`、`FileLike`、`torch.utils.serialization`、`torch._dynamo` 内部重组）。一旦 overlay 后触发 import 错误，往往要追加越来越多文件到 overlay_paths 才能解决，最终放弃更划算。

放弃判断标准：如果连续 2 次补 overlay 后还是新的 ImportError，直接 deprecate 这条 task。

### 9.2 `instantiate_device_type_tests` 测试命名约定

PyTorch 用 `instantiate_device_type_tests(MyTest, globals())` 装饰的测试类会在 import 时被重命名：

- 类名加 device 后缀：`TestFoo` → `TestFooCPU` / `TestFooCUDA` / `TestFooXPU`
- 方法名加 device 后缀：`test_bar(self, device)` → `test_bar_cpu` / `test_bar_cuda`
- 如果还有 `@dtypes(...)` 装饰：`test_bar(self, device, dtype)` → `test_bar_cuda_float32` / `test_bar_cuda_complex64`

`fail_to_pass` 和 `pass_to_pass` 必须用 **重命名后** 的名字。preflight 脚本会本地验证名字解析。

### 9.3 inplace_build 工程坑

- **CC=\"ccache gcc\" 会导致 CMake CheckAbi 失败**：CMake 在某些路径上把复合 CC 错误展开，导致用源文件名当 compiler 执行。改用 `/usr/lib/ccache` symlink 方式（已在 Dockerfile 修复）。
- **CMake 4.0+ 移除了对 `cmake_minimum_required < 3.5` 的兼容**：PyTorch 2.6 vendored protobuf 还是用 < 3.5 的语法。已通过 pin `cmake<4` + `CMAKE_POLICY_VERSION_MINIMUM=3.5` 解决。
- **`/workspace/.ccache` 必须在 rsync 时 exclude**：否则每次 workspace 同步都会清掉 build cache，每条 task 都要 30-60 分钟冷编译。已在 `_rsync_command` 加入 `--exclude=.ccache/ --exclude=build/`。

### 9.4 Source snapshot 完整性

- **python_overlay snapshot**：只需 sparse-checkout `torch/` + `test/`
- **kernel_build snapshot**：必须 sparse-checkout `torch/` + `aten/` + `c10/` + `caffe2/` + `tools/` + `cmake/` + `setup.py` + `CMakeLists.txt` + `version.txt` + `requirements.txt` + `.gitmodules`，并 init 必要的 third_party submodules（cutlass, eigen, fmt, pybind11, onnx）

如果 inplace_build 跑出 `setup.py not found`，第一时间检查 snapshot 完整性（preflight 脚本会查 setup.py 但 kernel_build 模式专属字段尚未单独校验）。

### 9.5 SSH 长连接

cuda_kernel_build 单条 task 可能跑 30-60 分钟。SSH 配置已包含 `ServerAliveInterval=30`，`_run_local` 现在会在 SSH exit 255 时自动重试一次。如果仍频繁断开，考虑用 `tmux` 在远程跑 admission/experiment 命令本身。
