# PyTorch Mini Candidate Dataset

This file records the first real-PR candidates for the full-repo op_bench path.

## Included Drafts

- `pytorch__149693__lazylinear_init`
  - PR: https://github.com/pytorch/pytorch/pull/149693
  - Issue: https://github.com/pytorch/pytorch/issues/149691
  - Rationale: small Python-level `torch.nn.LazyLinear` regression with an upstream regression test.
  - Status: draft until baseline fail and gold pass are replayed locally.

- `pytorch__160952__bilinear_lazy_check`
  - PR: https://github.com/pytorch/pytorch/pull/160952
  - Issue: https://github.com/pytorch/pytorch/issues/160407
  - Rationale: small Python-level `torch.nn.Bilinear` lazy-module behavior candidate.
  - Status: draft; hidden test is hand-authored from the issue and must be replay-verified.

## Deferred Candidates

- https://github.com/pytorch/pytorch/pull/151959
- https://github.com/pytorch/pytorch/pull/147918

These are FlexAttention / Inductor tasks with CUDA-sensitive behavior and should move to a later hardware-aware tier.

## Replay Gate

Use:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

Only change `metadata.curation_status` from `draft` to `verified` after the replay evidence reports `baseline_reproduced` and `resolved`.
