# PyTorch Mini Dataset

Language: English | [中文](README.zh-CN.md)

This is the first real-PR PyTorch CPU operator dataset slice for `op_bench`.
It has completed v0.2 replay admission and can be used as a verified small
experiment set.

It contains three PyTorch CPU operator tasks:

- `pytorch__149693__lazylinear_init`
  - PR: https://github.com/pytorch/pytorch/pull/149693
  - Issue: https://github.com/pytorch/pytorch/issues/149691
- `pytorch__160952__bilinear_lazy_check`
  - PR: https://github.com/pytorch/pytorch/pull/160952
  - Issue: https://github.com/pytorch/pytorch/issues/160407
- `pytorch__147599__lazylinear_state_forward`
  - PR: https://github.com/pytorch/pytorch/pull/147599
  - Issue: https://github.com/pytorch/pytorch/issues/147389

The dataset is marked `verified`. Every task completed baseline/gold replay on
the declared Docker environment:

- `pytorch__149693__lazylinear_init` is currently `verified`.
- `pytorch__160952__bilinear_lazy_check` is currently `verified`.
- `pytorch__147599__lazylinear_state_forward` is currently `verified`.

All three tasks use `op-bench/pytorch-cpu:torch2.6.0-py311` and the
`cpu_python_overlay` runtime path: the agent edits a full source snapshot, and
the selected Python files are synced into the installed PyTorch wheel at test
time. A task enters this dataset only after baseline reproduces fail-to-pass,
pass-to-pass remains green, and gold resolves the failure.

v0.2 adds registry-backed assets and stable admission evidence:

- Environment registry: `environments/registry.json`
- Source registry: `sources/registry.json`
- Stable evidence for `pytorch__149693__lazylinear_init`: `tasks/pytorch/149693_lazylinear_init/admission/evidence.json`
- Stable evidence for `pytorch__160952__bilinear_lazy_check`: `tasks/pytorch/160952_bilinear_lazy_check/admission/evidence.json`
- Stable evidence for `pytorch__147599__lazylinear_state_forward`: `tasks/pytorch/147599_lazylinear_state_forward/admission/evidence.json`

Each task can still declare local source snapshot paths under `.op_bench_cache/sources`. Snapshot contents are generated locally and are not committed; registry metadata records the snapshot identity and submodule policy.

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
  --task tasks/pytorch/147599_lazylinear_state_forward \
  --output runs/env/pytorch_mini.json
```

Prepare source snapshots:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_source_snapshot.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --task tasks/pytorch/160952_bilinear_lazy_check \
  --task tasks/pytorch/147599_lazylinear_state_forward \
  --output runs/sources/pytorch_mini.json
```

Formal admission remains the gate:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_admission.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output-dir runs/admission/pytorch__149693__lazylinear_init/manual \
  --write-task-evidence
```

Inspect registered asset cache state:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/inspect_assets.py
```

Current replay status:

- `pytorch__149693__lazylinear_init`: `verified`
- `pytorch__160952__bilinear_lazy_check`: `verified`
- `pytorch__147599__lazylinear_state_forward`: `verified`

Run the gold-agent loop check:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --agent gold \
  --output runs/experiments/pytorch_mini_gold_v0.2.json
```

Use `--verified-only` when running experiments that should count only admitted tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```
