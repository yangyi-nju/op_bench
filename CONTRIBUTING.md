# Contributing

Start with the [architecture and source map](docs/architecture.md). The main path
is `data → runner → evaluation → results`; `runtime` provides shared workspaces
and process execution. The CLI entry point is `src/op_bench/cli.py`.

## Develop and validate

Use Python 3.12+ and Git, with Docker for controlled task execution:

```bash
python -m pip install -e .
python -m unittest discover -s tests/core -v
python scripts/check_markdown_links.py
```

Docker-specific unit tests are opt-in. To include them locally, build the small
numeric image and enable the test environment:

```bash
docker build -t opbench/numeric-fixture:py3.12 tests/fixtures/numeric_softmax
OPBENCH_TEST_DOCKER=1 OPBENCH_TEST_DOCKER_IMAGE=opbench/numeric-fixture:py3.12 \
  python -m unittest discover -s tests/core -v
```

Test changed behavior and consequential failures. Execution changes need relevant
container checks; a CPU fixture does not validate an untested GPU or framework.
Do not pin document text, installed CLI versions, file counts or generated hashes.
Source and image identities are provenance, not proof of semantic quality.
Comments should explain non-obvious contracts, ordering and failure handling;
keep model-facing prompts and recorded protocol strings separate from commentary.

## Test fixtures

Small task bundles live only in [tests/fixtures/](tests/fixtures/README.md). They
exercise numeric isolation, compiled-library loading, and PyTorch integration;
they are not benchmark members. CI builds each required image and runs baseline,
reference, alternative, and mutation controls. The numeric image contains only
Python and Git dependencies; PyTorch is required only by the PyTorch fixture.

The fixture guide includes a short local installation check, Docker controls,
and optional model smoke testing. Model configuration and experiment commands
are documented once in the [integration guide](docs/v0.8/agent_integration.md).
The only maintenance script, `scripts/check_markdown_links.py`, checks local
links in current guides; it is not part of evaluation.

## Keep one path per responsibility

- `data/` defines task specifications and datasets, independent of model clients.
- `runner/` owns the fixed model loop, budgets and tool dispatch. Model
  adapters translate the same protocol; they do not implement their own workspace tools.
- MCP exposes one workspace-tool implementation. Task code executes in the declared
  isolated environment; tool inputs cannot become arbitrary controller commands.
- `runtime/` owns process sessions, source snapshots, baseline builds and patch
  capture. It does not import the runner, evaluator or reporter.
- `evaluation/` consumes a frozen patch and task. Model changes do not
  require another evaluator or access to hidden grading feedback.
- `results/` reads the fixed plan and verified records to compute `resolved_rate`; it does not
  repair records or make task admission decisions during reads.

Remove abandoned paths and their dedicated tests rather than retain parallel
runtimes, gateways or compatibility layers without current users. Historical
results retain their original protocol and are not promoted by new defaults.
Keep trajectories for later analysis; do not add an empty analysis platform.

Preserve version design, implementation and experiment documents. Index them in
[version history](docs/history/README.md); keep their original conclusions and
protocols rather than rewriting them as current results. The Markdown checker
checks current guides and history navigation by default. `--include-history`
also inspects original records against today's tree; references to retired assets
must be interpreted in the historical commit documented in the history index.

## Tasks and version scope

[The design](docs/v0.8/design.md) assigns core reliability and simplification to
v0.8, category coverage and selection/admission workflows to v0.9, and fixed-MCP
multi-model experiments to v0.10. Later work compares Agent frameworks and analyzes
behavior. Those later stages do not block this software iteration.

New task bundles use schema 2. Follow the [task guide](docs/v0.8/task_format.md),
including public behavior, source provenance, actual build/load paths, independent
grading and meaningful controls. `check-task` is a grading-development check,
not automatic admission. Retained historical materials need review and rebuilding;
the old JSON is not a current executable task bundle.

Preserve original failures and identify which protocol and inputs new evidence
actually validates. Fixes to scoring or environments may require patch re-evaluation;
new model attempts belong to a new experiment. Upstream attribution and licensing
remain separate from engineering test results. The retired
[v0.7 implementation](https://github.com/yangyi-nju/op_bench/tree/55c8ef2bd7d0bf78fc26bfd25498d9252e70781f)
remains available in Git history.

Code licensing awaits author and school confirmation. This guide does not grant
a license or resolve upstream rights.
