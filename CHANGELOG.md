# Changelog

## v0.8 — Fixed model evaluation framework (unreleased)

Comments and pre-PR documentation review, 2026-09-23:

- Explain model-client, attempt, recovery, budget and grading boundaries beside
  the implementation. Clarify current model API integration and later Agent
  framework integration; runtime behavior and experiment protocols are unchanged.
- Align the English/Chinese entry points, current status, task/runtime contracts
  and CLI examples. Date historical validation claims and separate earlier real
  model smoke runs from current scripted coverage; preserve original evidence.

Internal simplification and run organization, 2026-09-22:

- Consolidate engineering tasks under `tests/fixtures/`; remove the separate
  `examples/` tree, redundant guides and unused dataset/model configurations.
  Keep one fixture guide and one Docker model smoke configuration. User guides
  now start from actual candidate tasks and the shared experiment workflow.
- Give the standard-library numeric fixture its own Python/Git Docker image
  (environment revision 2), and align Docker CI and tests with it. Retain the
  separate C++ and PyTorch checks for their compilation and framework behavior.
  Remove the unused empty baseline patch; baseline evaluation is a CLI operation.
- Store each grading result once. Variant parents and controls use schema 2
  references to child `result.json` files, retaining statuses and interruption
  evidence without duplicate case lists or full inline results. Share the
  control execution loop and remove derivable counters.
- Normalize legacy model labels and terminal aliases at the record boundary;
  report protocol 5 consumes the normalized records without editing history.
  Replay requires an explicit frozen-submission declaration.
- Validate `HarnessSpec` at construction, share source identity and local process
  environment rules, remove unused baseline return fields, and keep Codex stdin
  in its temporary directory instead of persisting a duplicate request.
- Organize existing runs by version, purpose and batch; separate caches and
  historical archives. Preserve failed/interrupted evidence and historical
  reports. Add [storage guidance](runs/README.md) and a
  [relocation index](runs/archive/README.md); raw runs remain ignored by Git.

Detailed architecture review, 2026-09-22:

- Separate experiment scheduling/recovery, single-attempt execution and input
  preparation within `runner`. Remove unused repeat configuration, duplicate
  workspace task files and inline harness results; attempts reference the saved
  harness result and evaluation.
- Make baseline relocation a runtime responsibility. Preserve its original build
  path so moving an artifact does not leave local runtime variables pointing at
  the old directory; orchestration no longer rewrites baseline records.
- Share one subprocess lifecycle for commands and numeric observations, and one
  installed-image ID resolver. Remove the unused Docker network choice and an
  unused workspace-initialization parameter.
- Share pure grading judgments between execution and verification. Verification
  no longer imports the evaluator; reports consume evaluation rules. Reject
  variant revision/environment mismatches and handle damaged patch encoding or
  overflowing duration values as invalid evidence.
- Rename grading-control output to `controls.json` and remove constant admission
  declarations. Dataset selection/admission remains a v0.9 responsibility.
- Validate output paths before writing any artifacts. One task-level rule keeps
  evaluation, attempts, controls, replays and preparation outputs out of source
  and private grader directories, including symlink aliases.
- Record the harness protocol in plans and attempts, and reject pending work
  under changed tools or harnesses. At that review stage the harness became
  `opbench-fixed-tools-3`, tools remained `controlled-tools-2`, and the report
  protocol was 4 (superseded by 5 in the later internal simplification). Completed older
  records retain their original conditions.
- Repair stale commands, distinguish historical task materials from executable
  bundles, and make CI run an installed-package evaluation and verification away
  from the checkout. Architecture documentation describes the current structure;
  changes and validation evidence have their own records.

Simplification review, 2026-09-22:

- Keep the five responsibility packages within one Python application; remove
  the parallel batch prediction-import experiment flow. `evaluate` now handles
  one task and patch or baseline; `run` owns model experiment plans and recovery.
- Save complete scores only in `evaluation/result.json`. Attempt and replay
  records reference that file; reads verify its evidence and frozen submission.
- Share one report calculation between `run` and `report`; remove `report --reason`,
  report history directories and version chains. Original attempts, evaluations
  and historical version documents remain intact.
  Summarize once at completion or interruption, avoiding repeated verification
  of all prior scores after every attempt; manual progress reports remain available.
- Share fixed tool argument validation and remove the unused MCP stdio entry
  point. Advance tool/harness protocols to version 2 for the unified validation
  behavior, without relabeling earlier experiments.
  Reject new model executions when resuming an unfinished plan with another
  tool protocol, while completed plans remain readable.
- Close HTTP responses on rejected or interrupted model calls, including status,
  content-type, response-size and deadline failures.

Architecture and documentation organization, 2026-09-21:

- Organize the implementation into `data`, `runner`, `runtime`, `evaluation` and
  `results`, with one CLI entry point. Extract workspace management and patch
  capture from evaluation and experiment orchestration.
- Rename the task-control module to `evaluation/controls.py`; it validates
  grading behavior and does not implement dataset admission.
- Restore v0.1–v0.7 design, implementation and experiment documents, with a
  [history index](docs/history/README.md). Add a [source map](docs/architecture.md)
  and rewrite the reading and quickstart paths.

The scope was revised on 2026-09-14: compare models with the same controller loop,
MCP workspace tools, prompts and budgets, then independently grade frozen patches.
This replaces the earlier v0.8 autonomous-Agent and inference-gateway proposal.

- Separate task loading, model clients, fixed execution, independent evaluation
  and read-only reporting. Retain one implementation of workspace tools.
- Keep offline workspace isolation, clean baseline source, private-grader-free
  builds, frozen submissions, actual candidate builds and independent scoring.
- Use one main metric, `resolved_rate`, with the complete planned denominator and
  explicit missing/error status. Dataset admission does not gate development rates.
- Remove arbitrary Agent command launching, model gateways/relays and their
  examples, along with retired multi-repeat and comparison statistics.
- Retain Task schema 2, grading controls, execution provenance, recovery and
  frozen-patch re-evaluation. Historical evidence keeps its original protocol.
- Reserve classification, ten independent tasks per category and a reusable
  selection/admission process for v0.9; fixed-framework model experiments for v0.10.
  Agent-framework comparison and behavior analysis follow later.

The final September 22 internal simplification passed 224 core tests with Docker
enabled and a fresh installed-package scripted HTTP run, resume, report, verify,
replay and four grading controls. Output destinations inside task inputs were
also rejected before writing any artifacts.
Archived September 14 and September 22 runs still passed read-only report and
grading verification after directory organization.
Local Codex gpt-5.5/high completed a controlled fixture on September 14;
resume, reporting and independent verification passed. Evidence is maintained in the
[acceptance matrix](docs/v0.8/acceptance_matrix.md) and
[validation report](docs/v0.8/validation_report.md). Earlier successful native
model runs do not by themselves validate the new fixed loop.

See the [design](docs/v0.8/design.md) and [task guide](docs/v0.8/task_format.md).
Code licensing awaits author/school confirmation; this update selects no license.

## Earlier releases

The [version documents](docs/history/README.md),
[complete historical changelog](docs/history/CHANGELOG-through-v0.7.md) and
[v0.7 source](https://github.com/yangyi-nju/op_bench/tree/82c8e064bf30fead2d941d11cf94053ce3fc7e14)
retain earlier designs, implementation and reported results. Verification labels and
scores belong to those protocols; selected task materials remain for v0.9 review.
