# OpBench

**OpBench evaluates models on operator repair tasks.** A model receives buggy code and a problem statement, repairs it using a shared set of tools, and submits a patch for independent evaluation. Tasks focus on operator numerics, behavior, and the compiler or runtime paths that directly affect them.

The first phase asks: **which models repair operator defects best under the same tasks, tools, and budget?** v0.8 provides the evaluation workflow; dataset expansion and admission are planned for v0.9, followed by multi-model experiments in v0.10.

[中文](README.zh-CN.md) · [Architecture guide (中文)](docs/architecture.md) · [Documentation](docs/README.md) · [Version history](docs/history/README.md)

## How one task is evaluated

For a softmax overflow bug, a task supplies source code, the expected behavior, an environment, and public tests. The model can inspect and edit files, build, and run those public tests. Private tests and reference patches are withheld from the model.

1. **Prepare:** load the task and experiment configuration; prepare source and the execution environment.
2. **Repair:** OpBench calls the model, executes its choices from the fixed MCP tools, and returns public feedback.
3. **Evaluate:** stop solving, save the patch, then build and test it in an independent workspace.
4. **Report:** retain the trajectory, patch, and test results; compute each model's `resolved_rate`.

`resolved_rate = resolved tasks / planned tasks`. Passing public tests is feedback, not a final score: the build and required grading tests must pass. Missing results and execution errors are reported separately; an incomplete experiment report shows progress.

## Run an evaluation

Use Python 3.12+, Git, Docker, and Linux/macOS. From the repository root:

```bash
python -m pip install .
opbench doctor
opbench datasets validate --dataset datasets/v0.8_candidates/dataset.json
```

The current [candidate collection](datasets/v0.8_candidates/README.md) contains upstream operator tasks awaiting admission. Validation checks the manifest; running those tasks also requires their source checkouts, dependency images, and CPU/GPU resources. Prepare them using the [execution guide](docs/v0.8/remote_execution.md).

Then create an experiment configuration selecting the dataset, models, and common budget. The [model integration guide](docs/v0.8/agent_integration.md#实验配置与运行) provides the configuration and `run`, `run --resume`, and `report` commands. Compatible Chat Completions services use per-model Base URLs and API-key environment variables; local Codex CLI is also available for model-only integration checks. Controlled evaluation uses offline Docker workspaces; the controller calls the model and executes its choices from OpBench's tools.

For a quick installation or engine check without a full framework build, see [Contributing](CONTRIBUTING.md). Its small test fixtures are internal checks, not benchmark data.

## Repository map

| Directory | Contents |
| --- | --- |
| [`src/op_bench/`](src/op_bench/) | Implementation organized into data, runner, runtime, evaluation, and results; start with the [architecture guide](docs/architecture.md) |
| [`tasks/`](tasks/README.md) | Historical task materials in the legacy schema, awaiting migration and admission |
| [`datasets/`](datasets/) | Dataset manifests, candidate tasks, and data notes |
| [`tests/`](tests/) | Behavior tests in `core/`; small integration task bundles in [fixtures/](tests/fixtures/README.md) |
| [`runs/`](runs/README.md) | Local run evidence organized by version and batch, with separate validation, experiments, caches, and archives |
| [`docs/`](docs/README.md) | Current guides and [records of earlier iterations](docs/history/README.md) |

Adding a task primarily means creating a task bundle. Switching models primarily means editing an experiment configuration. Grading changes belong in `evaluation`. See the [architecture guide](docs/architecture.md) for module responsibilities, dependencies, and a reading order.

## Current scope

v0.8's engineering implementation and local acceptance checks are complete; the package remains the unreleased development version `0.8.0.dev1`. Current candidate tasks and retained historical data await the revised admission process. v0.9 will introduce the category and admission workflow, with at least ten independent tasks per category. v0.10 will run the fixed-tool multi-model experiments; each target service first needs a real API smoke run under the chosen conditions. External Agent framework comparisons and behavior analysis come later.

See the [CHANGELOG](CHANGELOG.md), [current status](docs/v0.8/development_status.md), and [validation records](docs/v0.8/validation_report.md) for implementation details and evidence. Historical documents retain the design and experiment context of their release; they are not current usage instructions.

For development setup and testing, see [Contributing](CONTRIBUTING.md). Code licensing awaits author and school confirmation; upstream code and patches retain their own attribution and licensing requirements.
