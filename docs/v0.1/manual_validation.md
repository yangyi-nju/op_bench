# Manual Validation Workflow

Use this workflow before promoting any real task from `draft` to `verified`.

All commands assume the project virtual environment is active through `PATH` and the package is loaded from `src`:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python --version
```

## 1. Validate Manifests

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_task.py \
  tasks/pytorch/149693_lazylinear_init/task.json
```

The validator checks that the task bundle is structurally complete and that Docker-backed tasks declare an executable image/preflight environment.

## 2. Prepare The Declared Environment

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_environment.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output runs/env/pytorch_149693.json
```

This step checks Docker availability, builds the image if the task declares a Dockerfile, starts one isolated container, runs the task preflight commands, then removes the container.

For Docker tasks, preflight runs from `environment.preflight_workdir` rather than the mounted source checkout. This avoids importing an unbuilt framework source tree when the preflight command is meant to validate the image.

Expected outcomes:

- `ready`: the environment artifact exists and preflight passed.
- `environment_unavailable`: Docker is missing, the image could not be built/found, or preflight failed.

`environment_unavailable` is not an agent failure. It means the dataset item cannot be scheduled on the current machine.

## 3. Prepare Source Snapshot

For large repositories, prepare the source snapshot before replay:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_source_snapshot.py \
  --task tasks/pytorch/149693_lazylinear_init \
  --output runs/sources/pytorch_149693.json
```

Source snapshots keep replay independent from live upstream clone stability. They are local artifacts under `.op_bench_cache/sources` and are not committed.

## 4. Verify Baseline And Gold Replay

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_task_replay.py \
  tasks/pytorch/149693_lazylinear_init \
  --output runs/replay/pytorch_149693.json
```

A task can be promoted only when:

- baseline status is `baseline_reproduced`
- gold status is `resolved`

Keep `metadata.curation_status` and the dataset entry `admission_status` as `draft` until both conditions are true on the declared environment.

## 5. Run Reference Agent Experiments

Run the gold upper-bound check:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/pytorch-mini-verified-gold
```

Run the reference real-agent path:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src OP_BENCH_CODEX_TIMEOUT_SEC=1200 python scripts/run_experiment.py \
  --dataset datasets/pytorch_mini/dataset.json \
  --verified-only \
  --agent codex_action_bridge \
  --output-dir runs/pytorch-mini-codex-action-bridge
```

`codex_action_bridge` runs Codex in a scratch workspace and exposes the target repository only through `opbench_action.py`, a local CLI backed by OpBench's action interface. For Docker tasks, command and test actions run in the task container. Check `agent_metadata.integrity_status`; `clean` means workspace changes were accounted for by the action log.

Codex CLI attempts default to a 1200 second timeout. Set `OP_BENCH_CODEX_TIMEOUT_SEC` higher for slower models or larger repositories.

The experiment runner prints progress logs to stderr by default. Use `--quiet` for unattended runs where only `results.jsonl` and `summary.json` are needed.

Read:

- `runs/<experiment>/results.jsonl` for per-task command evidence
- `runs/<experiment>/summary.json` for aggregate agent status
- `runs/<experiment>/patches/` for submitted patches and action logs

## Status Interpretation

- `resolved`: fail-to-pass passed and pass-to-pass did not regress
- `baseline_reproduced`: the original bug was reproduced before applying a fix
- `baseline_not_reproduced`: the task is not admissible yet
- `fail_to_pass_failed`: patch did not fix the targeted behavior
- `pass_to_pass_regressed`: patch broke existing coverage
- `environment_unavailable`: task cannot run on this host/environment
- `environment_error`: tests failed due to missing imports, shared libraries, CUDA, or similar runtime environment errors
- `agent_runtime_unsupported`: the selected agent cannot honor the task runtime boundary
