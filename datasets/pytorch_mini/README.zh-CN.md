# PyTorch Mini 数据集

语言：[English](README.md) | 中文

这是 `op_bench` 第一组来自真实 PR 的数据集切片。

当前包含两条 PyTorch CPU 算子任务：

- `pytorch__149693__lazylinear_init`
  - PR: https://github.com/pytorch/pytorch/pull/149693
  - Issue: https://github.com/pytorch/pytorch/issues/149691
- `pytorch__160952__bilinear_lazy_check`
  - PR: https://github.com/pytorch/pytorch/pull/160952
  - Issue: https://github.com/pytorch/pytorch/issues/160407

数据集整体仍标记为 `draft`，因为并非所有 task 都已经完成 replay admission。每条 task 的实际状态记录在 `dataset.json` 中。

- `pytorch__149693__lazylinear_init` 当前为 `verified`。
- `pytorch__160952__bilinear_lazy_check` 当前为 `draft`。

两条任务都声明了可运行的 Docker 环境元数据，并且可以通过 `op-bench/pytorch-cpu:torch2.6.0-py311` 的 preflight。只有当某条任务在声明环境中完成 baseline/gold replay，且 baseline 能复现失败、gold 能修复成功后，该任务才会晋升为 `verified`。

每条任务还会声明 `.op_bench_cache/sources` 下的本地 source snapshot 路径。snapshot 由本地生成，不提交到 git。

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
  --output runs/env/pytorch_mini.json
```

准备 source snapshots：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_source_snapshot.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --output runs/sources/pytorch_mini.json
```

完整 replay 仍然是 task admission 的准入门槛：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

当前 replay 状态：

- `pytorch__149693__lazylinear_init`: `verified`
- `pytorch__160952__bilinear_lazy_check`: `pending`

如果实验只应统计已准入任务，请使用 `--verified-only`：

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```
