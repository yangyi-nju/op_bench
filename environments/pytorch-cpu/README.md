# PyTorch CPU Environment

Language: English | [中文](README.zh-CN.md)

This is the first reusable CPU-only Docker environment for PyTorch operator tasks.
It is based on `intel/intel-extension-for-pytorch:2.6.0-pip-base`, which provides PyTorch `2.6.0+cpu` on Python 3.11.

The image is an executable dataset artifact. Tasks reference this directory through `environment.dockerfile` and `environment.build_context`, then replay runs `environment.preflight_commands` before baseline, gold, or agent evaluation.

The current image is intended for CPU-deterministic Python-level PyTorch tasks. Tasks remain `draft` until replay evidence proves baseline failure and gold success inside this environment.

During replay, the runner starts one isolated container from this image for each task attempt, mounts the task workspace at `/workspace`, runs preflight/setup/test commands inside that container, and removes it after the attempt.

## Usage

Build or inspect the environment through the task manifest:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_environment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output runs/env/pytorch_149693.json
```

If Docker is unavailable, OpBench should report `environment_unavailable` before spending time on source checkout.
