# PyTorch CPU 环境

语言：[English](README.md) | 中文

这是第一个可复用的 PyTorch CPU-only Docker 环境，用于 PyTorch 算子任务。
它基于 `intel/intel-extension-for-pytorch:2.6.0-pip-base`，在 Python 3.11 上提供 PyTorch `2.6.0+cpu`。

该镜像是可执行的数据集 artifact。task 会通过 `environment.dockerfile` 和 `environment.build_context` 引用当前目录；在 baseline、gold 或 agent 评测前，replay 会先运行 `environment.preflight_commands` 验证环境。

当前镜像面向 CPU deterministic、Python-level 的 PyTorch 任务。任务只有在该环境中证明 baseline failure 且 gold success 后，才会从 `draft` 晋升为 `verified`。

Replay 时，runner 会为每次 task attempt 从该镜像启动一个隔离容器，将 task workspace 挂载到 `/workspace`，在容器内运行 preflight/setup/test 命令，并在 attempt 结束后清理容器。

## 使用方式

通过 task manifest 构建或检查环境：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_environment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output runs/env/pytorch_149693.json
```

如果当前机器没有 Docker，OpBench 应在源码 checkout 前直接返回 `environment_unavailable`，避免把环境调度失败误算为 agent 失败。
