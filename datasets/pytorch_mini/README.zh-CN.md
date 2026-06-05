# PyTorch Mini 数据集

语言：[English](README.md) | 中文

这是 `op_bench` 第一组来自真实 PR 的 PyTorch CPU 算子数据集切片。当前数据集已经通过 v0.2 replay admission，可作为 verified 小规模实验集使用。

当前包含三条 PyTorch CPU 算子任务：

- `pytorch__149693__lazylinear_init`
  - PR: https://github.com/pytorch/pytorch/pull/149693
  - Issue: https://github.com/pytorch/pytorch/issues/149691
- `pytorch__160952__bilinear_lazy_check`
  - PR: https://github.com/pytorch/pytorch/pull/160952
  - Issue: https://github.com/pytorch/pytorch/issues/160407
- `pytorch__147599__lazylinear_state_forward`
  - PR: https://github.com/pytorch/pytorch/pull/147599
  - Issue: https://github.com/pytorch/pytorch/issues/147389

数据集整体标记为 `verified`。每条 task 都在声明的 Docker 环境中完成 baseline/gold replay：

- `pytorch__149693__lazylinear_init` 当前为 `verified`。
- `pytorch__160952__bilinear_lazy_check` 当前为 `verified`。
- `pytorch__147599__lazylinear_state_forward` 当前为 `verified`。

三条任务都使用 `op-bench/pytorch-cpu:torch2.6.0-py311`，通过 `cpu_python_overlay` 将 source snapshot 中的 Python 文件同步到容器内已安装 PyTorch wheel 的 runtime overlay。只有 baseline 能复现 fail-to-pass、pass-to-pass 不回归、gold 能修复成功的任务才进入本数据集。

v0.2 增加了 registry-backed assets 和 stable admission evidence：

- 环境 registry：`environments/registry.json`
- 源码 registry：`sources/registry.json`
- `pytorch__149693__lazylinear_init` 的 stable evidence：`tasks/pytorch/149693_lazylinear_init/admission/evidence.json`
- `pytorch__160952__bilinear_lazy_check` 的 stable evidence：`tasks/pytorch/160952_bilinear_lazy_check/admission/evidence.json`
- `pytorch__147599__lazylinear_state_forward` 的 stable evidence：`tasks/pytorch/147599_lazylinear_state_forward/admission/evidence.json`

每条任务仍可声明 `.op_bench_cache/sources` 下的本地 source snapshot 路径。snapshot 内容由本地生成，不提交到 git；registry 元数据记录 snapshot 身份和 submodule policy。

## 校验

校验数据集 manifest 以及其引用的所有 task manifest：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

准备所有声明的 task 环境：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_environment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --task tasks/pytorch/147599_lazylinear_state_forward \
  --output runs/env/pytorch_mini.json
```

准备 source snapshots：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_source_snapshot.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --task tasks/pytorch/147599_lazylinear_state_forward \
  --output runs/sources/pytorch_mini.json
```

正式 admission 仍然是 task 的准入门槛：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_admission.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output-dir runs/admission/pytorch__149693__lazylinear_init/manual \
  --write-task-evidence
```

检查已登记资产的 cache 状态：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py
```

当前 replay 状态：

- `pytorch__149693__lazylinear_init`: `verified`
- `pytorch__160952__bilinear_lazy_check`: `verified`
- `pytorch__147599__lazylinear_state_forward`: `verified`

运行 gold agent 闭环检查：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --agent gold \
  --output runs/experiments/pytorch_mini_gold_v0.2.json
```

如果实验只应统计已准入任务，仍可使用 `--verified-only`：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```
