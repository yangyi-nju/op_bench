# PyTorch Mini Dataset

Language: English | [中文](README.zh-CN.md)

This is the first real-PR `op_bench` dataset slice.

It contains two PyTorch CPU operator tasks:

- `pytorch__149693__lazylinear_init`
  - PR: https://github.com/pytorch/pytorch/pull/149693
  - Issue: https://github.com/pytorch/pytorch/issues/149691
- `pytorch__160952__bilinear_lazy_check`
  - PR: https://github.com/pytorch/pytorch/pull/160952
  - Issue: https://github.com/pytorch/pytorch/issues/160407

The dataset is intentionally marked `draft` at the dataset level because not every task has completed replay admission. Each task records its own status in `dataset.json`.

- `pytorch__149693__lazylinear_init` is currently `verified`.
- `pytorch__160952__bilinear_lazy_check` is currently `draft`.

Both tasks have runnable Docker environment metadata and pass preflight against `op-bench/pytorch-cpu:torch2.6.0-py311`. A task is promoted to `verified` only after baseline/gold replay succeeds on the declared environment.

Each task also declares a local source snapshot path under `.op_bench_cache/sources`. Snapshots are generated locally and are not committed.

## Validation

Validate the dataset manifest and all referenced task manifests:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_mini/dataset.json
```

Prepare all declared task environments:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_environment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --output runs/env/pytorch_mini.json
```

Prepare source snapshots:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_source_snapshot.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --output runs/sources/pytorch_mini.json
```

Full replay remains the admission gate:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

Current replay status:

- `pytorch__149693__lazylinear_init`: `verified`
- `pytorch__160952__bilinear_lazy_check`: `pending`

Use `--verified-only` when running experiments that should count only admitted tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```
