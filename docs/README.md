# Documentation

Start with the [project README](../README.md), then choose a guide. Most detailed guides are currently written in Chinese.

## Understand and use the current project

| Goal | Start here |
| --- | --- |
| Understand modules and the execution flow | [Architecture guide](architecture.md) |
| Run a model, inspect results, or resume an experiment | [Experiment configuration and model integration](v0.8/agent_integration.md#实验配置与运行) |
| Create a task and check correct or incorrect patches | [Task format](v0.8/task_format.md), [candidate task bundles](../datasets/v0.8_candidates/README.md) |
| Understand the tools available to models | [Model and tool integration](v0.8/agent_integration.md) |
| Understand public inputs and private grading | [Information boundary](v0.8/information_boundary.md) |
| Interpret resolved_rate and incomplete runs | [Scoring rules](v0.8/statistical_protocol.md) |
| Prepare a CPU/GPU execution host | [Execution guide](v0.8/remote_execution.md) |
| Change code and run tests | [Contributing](../CONTRIBUTING.md) |
| Store runs, distinguish batches, or find older evidence | [Run records](../runs/README.md) |

## Current design and progress

v0.8's engineering implementation and local acceptance checks are complete; it is not yet a published release. v0.9 updates the dataset and admission process; v0.10 runs the multi-model experiments after target-service smoke checks.

- [v0.8 design](v0.8/design.md): scope and design decisions.
- [Status](v0.8/development_status.md), [implementation plan](v0.8/implementation_plan.md), and [acceptance checklist](v0.8/acceptance_matrix.md): completed and remaining work.
- [Validation records](v0.8/validation_report.md): actual tests, runs, and limitations.
- [Candidate tasks](../datasets/v0.8_candidates/README.md) and [historical task review](v0.8/task_disposition.md): inputs to the next dataset revision.

## Earlier iterations

The [version history archive](history/README.md) preserves earlier documents and experiment reports in their original form. Interpret them in the context of their release; use the guides above for current installation, commands, and architecture. See the [CHANGELOG](../CHANGELOG.md) for a summary of changes.
