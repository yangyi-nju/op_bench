# 在 GPU 主机上运行新核心

这里的远端执行是把 OpBench 控制器安装在计算主机上，并在该主机调用 CLI。
当前 CLI 没有 SSH 主机参数、远端任务分发器或旧环境注册表；终端连接和文件传输
使用部署方已有的受信任通道。以下命令在该主机的 OpBench 仓库根目录执行，
`/srv/opbench/runs` 是可替换的控制器输出目录。

## 控制器与输入准备

控制器需要 Python 3.12+、Git、可访问的 Docker daemon；GPU 主机还需兼容的
NVIDIA 驱动和 NVIDIA Container Toolkit/runtime，使 Docker 能按任务声明选用 GPU。
宿主 Python 与镜像内的任务 Python 是不同环境。先安装当前核心：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install .
source .venv/bin/activate
opbench doctor
```

`doctor` 检查基本能力；退出码 0 只表示 Python/Git 可用，还需检查
`container_ready`，它也不证明 GPU、完整框架或题目已可执行。核对任务的镜像、
CPU/内存/PID 限制、GPU 选择及构建超时，给源码、临时副本、Docker volume 和构建
产物预留磁盘空间。更改设备或构建条件须记录新的环境条件，不能沿用不相容的基线。

完整任务包、上游 Git checkout、grader、控制补丁和结果保留在可信控制器侧。
Agent 只获得会话工作区、公开输入及声明的依赖；不挂载整个仓库、任务包、宿主
凭据或 Docker socket。导出的源码没有上游 Git 历史，求解 Git 只有一个初始提交。
私有 grader 不进入基线构建或候选构建；评分时按驱动需要进入可信评分侧。

以现有 CUDA 候选为例，先按执行主机的实际调度调整 `environment.gpus` 并记录
设备身份；当前 `device=2` 是原部署的选择，不表示任意主机的可用 GPU。此任务
固定 SM 7.0；改用其他架构需另行修订环境和验证。随后准备依赖镜像和精确源码：

```bash
docker build -t opbench/pytorch-cuda-source:cu124-sm70-r1 \
  -f datasets/v0.8_candidates/log_softmax_cuda/environment/Dockerfile \
  datasets/v0.8_candidates/log_softmax_cuda/environment
opbench prepare --fetch --fetch-timeout 7200 \
  --task datasets/v0.8_candidates/log_softmax_cuda/task.json \
  --output /srv/opbench/runs/v0.8/validation/gpu-smoke-r1/cuda-source-export
```

镜像依赖下载与显式源码准备可以联网；Docker 任务的 prepare/build、固定工具求解
和评分保持断网。Local 后端仅供开发，不提供同样的网络或私有文件隔离。
依赖镜像须预先存在于执行主机的 Docker daemon；运行器不会隐式 pull 镜像。
`--fetch-timeout` 是此次源码准备的总秒数，不是框架构建超时。`--fetch` 在
`task.source.path` 缺失时准备指定提交及递归 gitlink 对应的子模块；已有 checkout
只读校验，缺对象或子模块时不会自动重置它，应改用新的受管源码路径。

任务内的 `source.path`、`grader_dir` 相对于 `task.json`；命令行任务、补丁和输出
路径相对于当前目录。`prepare` 输出公开 `workspace/`、来源及输入记录，不执行
`environment.prepare/build`，也不生成可替代原任务包的私有 `task.json`。
后续命令继续使用原任务文件，其声明的源码已在控制器准备好。

## 完整构建与控制组

```bash
opbench build-baseline \
  --task datasets/v0.8_candidates/log_softmax_cuda/task.json \
  --output /srv/opbench/runs/cache/cuda-baseline-r1
opbench check-task \
  --task datasets/v0.8_candidates/log_softmax_cuda/task.json \
  --reference datasets/v0.8_candidates/log_softmax_cuda/controls/reference.patch \
  --alternative datasets/v0.8_candidates/log_softmax_cuda/controls/alternative.patch \
  --mutation datasets/v0.8_candidates/log_softmax_cuda/controls/mutation.patch \
  --mutation datasets/v0.8_candidates/log_softmax_cuda/controls/bridge_only.patch \
  --baseline-artifact /srv/opbench/runs/cache/cuda-baseline-r1 \
  --output /srv/opbench/runs/v0.8/validation/gpu-smoke-r1/cuda-controls
```

只有 `baseline.json` 为 `ready` 才能复用。它来自无补丁、无私有 grader、无模型
凭据的可信构建；每个控制仍复制独立工作区、应用自己的补丁并执行正常构建。
所有必需评分变体都需相应产物。兼容检查比较源码和完整环境声明；它不是内容
认证，也不接受 Agent 产生的缓存。[基线合同](task_format.md#显式复用基线构建)
说明增量依赖、runtime 与合法源码边界。

`check-task` 保存完整 `control_plan`，准备失败或中断保留已执行证据与剩余
`not_run`。该命令没有 `--resume`；重做控制使用新输出目录，原失败记录保留。
控制执行成功仍需逐项检查真实失败原因、加载路径和语义覆盖，不自动完成准入。

## 固定工具模型实验

先按[接入说明](agent_integration.md)配置控制器侧模型客户端。容器镜像提供任务的
构建和公开测试依赖，不安装自主 Agent，也不接收模型凭据。HTTP 服务或本地 Codex
在控制器中产生同一协议的模型响应，工作区工具由 OpBench 管理。

以[实验配置格式](agent_integration.md#实验配置与运行)为起点，替换数据集
路径、实际模型和预算即可。数据集路径相对于实验 JSON；任务索引相对于数据集
JSON，且必须在其目录范围内。单题联调在实验配置中用 `task_ids` 选择题目。
运行前准备所有选中任务的源码、求解镜像及必需评分变体镜像；上面的 CUDA 准备
命令不代表主候选清单中的 CPU/compile 环境也已准备。

```bash
opbench run --experiment gpu.experiment.json --output /srv/opbench/runs/v0.8/validation/gpu-smoke-r1/gpu-model
opbench run --experiment gpu.experiment.json --output /srv/opbench/runs/v0.8/validation/gpu-smoke-r1/gpu-model --resume
opbench report --run /srv/opbench/runs/v0.8/validation/gpu-smoke-r1/gpu-model
```

可信基线可以在计划内复用，各模型仍获得独立副本。基线准备在求解前完成，模型
触发的构建和公开测试计入预算；最终评分独立重建。恢复继续原计划，只启动尚无
尝试记录的任务，已有尝试不重跑。模型服务能联网不代表任务容器可以联网；任务工作区保持
Docker `network: "none"`，没有模型网关或 relay。

## 留存与当前证据

大构建产物与可下载的执行证据分别管理：

| 内容 | 留存方式 |
| --- | --- |
| 显式基线 `source/`、`workspace/`，实验输入快照和共享基线 | 留在计算主机的足量存储；复用产物不能只保留元数据。 |
| 求解/评分临时副本及 `reuse_baseline: false` 的单次基线工作区 | 执行期间占用存储，结束后清理；单次基线保留构建记录与日志，不能仅凭这些记录复用。 |
| `baseline.json`、构建 `logs/`，`controls.json`、控制补丁及各控制的结果、评分日志/观测 | 可通过部署方文件传输工具单独下载，保留原相对路径；不必复制整个构建工作区。 |
| 实验 `plan.json`、`attempt.json`、`patch.diff`、评分证据、模型与工具轨迹及汇总报告 | 保留失败、中断和未知消耗；下载副本不能替代缺失的复评或环境重建资产。 |

评分日志可能含私有 oracle；下载后仍按控制器资产管理，不回流给求解 Agent。
`logs/` 与构建工作区的复制是两件事，不能用编译日志或镜像存在宣称控制通过。

当前 [CUDA 候选](../../datasets/v0.8_candidates/log_softmax_cuda/README.md)固定
SM 7.0；2026-09-09 的控制记录来自共享 V100，不能推及任意 GPU 架构或当前
设备占用，也不提供性能结论。
[CPU 候选](../../datasets/v0.8_candidates/layer_norm_cpu/README.md)的 r2 完整构建
因 NNPACK 在线依赖和 FBGEMM 编译错误失败，[失败记录](../../datasets/v0.8_candidates/layer_norm_cpu/validation/r2_full_build_failure.json)
已保留；环境 r3 显式关闭 NNPACK/FBGEMM，依赖镜像仍为 r2。CPU r3 与 CUDA 完整
配置框架及 compile 候选均已有历史控制证据；这些构建和控制不证明当前固定 MCP
模型循环在三题上通过，也不代表本机已有对应镜像，新部署仍需准备对应环境。
三题尚未准入；实际构建、控制结果及模型验证状态以[验证记录](validation_report.md)为准。
