# OpBench

Language: English | [中文](README.zh-CN.md)

OpBench is an operator-focused benchmark for evaluating coding agents on real framework issues. It follows the SWE-bench idea of repairing real repository snapshots, but treats the runtime environment as part of each task because operator bugs often depend on framework version, Python package layout, device availability, numerical behavior, and backend selection.

v0.1 established the isolated replay/evaluation loop. v0.2 added asset registries and formal admission. v0.3 expanded the dataset to 10 verified tasks and added 3-repeat stability evaluation. v0.4 added CUDA tiers and remote Docker. v0.5 is now complete: the verified cumulative dataset contains 17 tasks, including a 6-task precision slice, and its 51-attempt Codex run reached **72.5% resolved** (37/51) with eight-dimensional reporting and hard experiment-integrity checks.

The v0.6 platform is **Completed** across M1–M7: strict versioned contracts, one authoritative workspace and immutable patch, a server-authoritative CLI/MCP action service, deterministic Attempt/trajectory/evaluation/artifact semantics, versioned Runtime Profiles, exact Attempt-owned Local/Docker/Remote resources, conformance and legacy replay, a process-isolated canonical Codex Adapter with resume and Integrity verification, and an executable public Demo/documentation surface. The recovered exact target passed representative Remote CPU, CUDA Overlay, and CUDA Kernel canaries. The complete frozen replay passed all 17 baseline + 17 gold + 51 historical final-patch cases with zero failures, blocks, or differences, closing the formal `opbench-v0.6.0` release gate. Boundary-task expansion follows in v0.7. See the [global project plan](docs/project_plan.md), [current project state](docs/project_state.md), and [v0.6 release notes](docs/v0.6/release_notes.md).

## What The Current Code Contains

- A two-layer dataset model: `datasets/<slice>/dataset.json` points to task bundles under `tasks/`.
- Real PyTorch task bundles with PR/issue metadata, hidden test patches, gold patches, source snapshots, and environment declarations.
- Reusable Docker environments: `environments/pytorch-cpu/` (CPU), `environments/pytorch-cpu-compile/` (CPU + Inductor toolchain), `environments/pytorch-cuda/` (CUDA Python overlay), `environments/pytorch-cuda-devel/` (CUDA + nvcc/ccache/cmake for kernel builds).
- Environment and source registries under `environments/registry.json` and `sources/registry.json`.
- Three runtime tiers: `cpu_python_overlay`, `cuda_python_overlay`, `cuda_kernel_build` (last two use `remote_docker` backend over SSH).
- Formal admission evidence through `scripts/run_admission.py`, offline preflight through `scripts/preflight_task.py`.
- Dataset curation through `scripts/curate_dataset.py`.
- Asset and container inspection through `scripts/inspect_assets.py` and `scripts/manage_containers.py`.
- A replay evaluator that checks baseline failure, gold success, agent patch success, and pass-to-pass regressions; supports `python_overlay` and `inplace_build` source loading modes.
- A remote GPU Docker executor (`src/op_bench/remote.py`) that runs `docker` on an SSH host, rsyncs workspaces both ways, and persists an environment-scoped ccache across isolated workspaces.
- A standardized workspace action interface for file operations, patch application, command execution, tests, and diff export.
- `codex_action_bridge`, the reference real-agent adapter with automatic rate-limit-aware retry. Codex runs on the host in a scratch workspace and can operate on the target repository only through OpBench actions.
- Strict v0.6 runtime contracts and canonical SHA-256 identities under `src/op_bench/runtime/`, including deterministic RunManifest, Cohort ID, Attempt ID, and frozen task × agent × repeat matrices.
- An explicit FullTaskSpec → AgentTaskView public whitelist with recursive answer-source, credential, private-output, and machine-local-path rejection; each projected view is frozen into Manifest and Attempt identity.
- A path-independent Authoritative Workspace identity with bounded regular-file access, atomic scoped writes, symlink/special-file rejection, deterministic add/modify/delete/empty patches, concurrent Freeze convergence, and strict clean-base `git apply --check --index` verification.
- A Canonical Action Service for list/search/read/write/apply-patch, policy-bound commands, registry-bound tests, diff, and finish; CLI and MCP share the same execution authority, and the standard Adapter receives only a scanned launch view plus a JSON-only action client.
- An AttemptSession state machine with server-owned deadline/resource budgets, deterministic termination priority, in-flight Action/publication barriers, one patch freeze, and one terminal SessionResult.
- A canonical append-only EventJournal with atomic Action event batches, continuous hash chaining, public Artifact spill, strict descriptor-bound persistence, and an Evaluation-aware AttemptLedger for deterministic retry/resume decisions.
- Fresh evaluation from a verified local Source copy with strict patch apply, post-session evaluation-only test injection, F2P/P2P evidence, and independent validity/terminal/outcome axes.
- Descriptor-bound public/private attempt artifacts, a read-only 14-check integrity graph, tamper detection, and byte-exact deterministic `results.jsonl`/`summary.json` rebuilds.
- An independent zero-dependency JSON Schema validator, a strict schema artifact under `schemas/`, and offline build/validation CLIs that do not launch an Agent or contact a runtime.

Development-only experiment adapters have been removed from the public v0.1 surface. Future agents should integrate by implementing the same action-interface boundary used by `codex_action_bridge`.

## Repository Map

| Path | Purpose |
| --- | --- |
| `datasets/` | Dataset manifests that select task bundles for experiments. |
| `tasks/` | Individual benchmark tasks with issue text, patches, environment metadata, and test lists. |
| `environments/` | Docker environment artifacts and the environment registry. |
| `sources/` | Source snapshot registry metadata. |
| `src/op_bench/` | Core implementation: task model, environment preparation, evaluator, actions, agent bridges, reporting. |
| `scripts/` | CLI entry points for validation, environment preparation, source snapshots, replay, and experiments. |
| `schemas/` | Strict v0.6 runtime wire-contract JSON Schema. |
| `configs/examples/` | Public synthetic v0.6 configuration and manifest examples. |
| `docs/` | Versioned design docs, experiment reports, developer guides, and historical records. |
| `docs/project_plan.md` | Global mission, principles, roadmap, release gates, and research targets. |
| `docs/project_state.md` | Current baseline, active version, decisions, and next actions. |
| `docs/v0.6/` | v0.6 standardized Agent evaluation platform design, implementation plan, and acceptance matrix. |
| `docs/v0.7/design.md` | v0.7 Dataset Factory, Boundary Slice, and matched-runtime recovery design. |
| `docs/v0.5/design.md` | v0.5 dimension taxonomy and extended evaluation metrics. |
| `docs/v0.5/experiment_report.md` | v0.5 full 17-task, 51-attempt Codex evaluation and precision breakdown. |
| `docs/v0.4/design.md` | v0.4 CUDA tiers, remote GPU Docker executor over SSH, and `inplace_build` source loading. |
| `docs/v0.4/experiment_report.md` | v0.4 13-task 3-repeat Codex evaluation: 84.6% resolved. |
| `docs/v0.3/design.md` | v0.3 dataset expansion, multi-file overlay, public/hidden test split, and CUDA pilot design. |
| `docs/v0.2/developer_guide.md` | v0.2 registry, admission, curation, asset, and container workflow. |
| `docs/v0.1/developer_guide.md` | v0.1 module-level architecture and development guide. |

## Quick Start

OpBench v0.6 has no third-party Python dependency. Create a clean environment,
run the full suite, and validate the frozen v0.5 Dataset:

```bash
python3 -m venv .venv
PATH=.venv/bin:$PATH python --version

PATH=.venv/bin:$PATH PYTHONPATH=src python -m unittest discover \
  -s tests -p 'test_*.py'

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_dataset.py \
  datasets/pytorch_v0.5/dataset.json --require-verified
```

Build and validate an offline v0.6 RunManifest. This command does not launch an
Agent or contact a Runtime:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/build_run_manifest.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --output /tmp/opbench-v0.6-manifest.json \
  --agent example-agent \
  --model example-model \
  --adapter canonical-cli-v1 \
  --repeat 1 \
  --created-at 2026-07-18T00:00:00Z

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  /tmp/opbench-v0.6-manifest.json
```

### Offline v1 Scripted Demo

Prepare the synthetic local input, then run it through the production v1
orchestrator and `local-cpu-process-v1` Runtime Profile:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/prepare_v0_6_demo.py \
  --output-dir runs/v0.6_m7_demo_input

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m7_demo_input/dataset/dataset.json \
  --verified-only \
  --agent scripted_canonical \
  --agent-repeat 1 \
  --output-dir runs/v0.6_m7_scripted_demo \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1
```

Run the exact `run_experiment.py` command again to exercise resume. The first
run reports `ran=1, skipped=0`; the second reports `ran=0, skipped=1` and does
not change the selected artifacts. Validate the frozen contract and exact
Attempt-owned resource cleanup:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/validate_runtime_contract.py \
  runs/v0.6_m7_scripted_demo/run_manifest.json

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_runtime_resources.py \
  --run-root runs/v0.6_m7_scripted_demo
```

Integrity verification and deterministic rebuilding of `results.jsonl` and
`summary.json` are part of every successful v1 run. This synthetic Demo checks
the controller and artifact lifecycle; it is not a benchmark score and does
not measure repair quality.

### Optional real Codex local canary

The canonical Codex Adapter uses the same Demo Dataset, Runtime, action service,
evaluator, and artifact path. This optional command invokes the locally
configured Codex CLI and may use its normal OpenAI network access:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m7_demo_input/dataset/dataset.json \
  --verified-only \
  --agent codex_canonical \
  --agent-repeat 1 \
  --output-dir runs/v0.6_m7_codex_demo \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1 \
  --enable-external-canary
```

This is a Runtime/Adapter canary, not a benchmark score. M6 already recorded a
valid real-Codex local Attempt and a two-repeat resume cohort. After the exact
configured target recovered, representative Remote CPU, CUDA Overlay, and CUDA
Kernel canaries passed, followed by an 85/85 exact replay. OpBench used only the
configured target; it did not probe or discover replacement targets.

### Real Codex MCP Adapter

`codex_mcp_canonical` is the independent real-Agent MCP path. It starts one
invocation-local `mcp-stdio` server, passes that server to a single ephemeral
Codex invocation without changing global Codex configuration, and binds the
exact model and CLI version into the Agent identity. The frozen v0.6 experiment
uses `gpt-5.6-sol` and `codex-cli 0.145.0-alpha.18`:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m6_local_codex_input/dataset/dataset.json \
  --verified-only \
  --agent codex_mcp_canonical \
  --codex-model gpt-5.6-sol \
  --agent-repeat 1 \
  --output-dir runs/v0.6_mcp_local_canary_r6 \
  --runtime-protocol v1 \
  --runtime-profile local-cpu-process-v1 \
  --enable-external-canary
```

Provider network access is allowed for the host-side Codex invocation. Task network
access remains denied, and the Agent reaches the task only through the
nine canonical MCP Actions. Codex runs in a separate read-only working directory;
the controller-private launcher, token-bound Action client, and trace are outside
that directory and are identity-checked after exit. Each selected retry records public-safe
`adapter_trace.json` initialize/list/call counters; the existing 14-check
Integrity graph binds those counters to Action events, model/CLI identity,
terminal state, evaluation, and exact resource cleanup. See the
[real MCP Agent experiment guide](docs/v0.6/mcp_agent_experiment.md) for the four
cohorts, resume rules, evidence split, hard output bounds, exact formal-matrix
contract, report builder, and safety boundary.

### Legacy v0.5 compatibility

Legacy remains the default protocol during migration. Legacy commands
intentionally omit `--runtime-protocol`; v1-only flags are rejected instead of
being silently ignored. Rebuild source snapshots after a fresh clone before
running real v0.5 tasks:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/setup_sources.py

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json \
  --verified-only \
  --agent gold \
  --output-dir runs/experiments/pytorch_v0.5_gold
```

See the [v0.6 developer guide](docs/v0.6/developer_guide.md) for Runtime support,
artifact layout, failure attribution, Comparability Key, resume, replay, and
exact-target configuration.

## Current Dataset

The verified [pytorch_v0.5 manifest](datasets/pytorch_v0.5/dataset.json) contains all 13 v0.4 tasks plus four newly admitted precision tasks. Deprecated tasks #129154 and #144073 are excluded. The complete run scored **37/51 (72.5%)**; the reusable [precision slice](datasets/pytorch_v0.5_precision/dataset.json) scored **13/18 (72.2%)**:

| Task | PR | Subclass | Tier | Rate |
| --- | ---: | :---: | --- | ---: |
| `pytorch__140557__layer_norm_decomp_precision` | #140557 | P2 | cpu | 0/3 |
| `pytorch__139999__masked_mean_bool_upcast` | #139999 | P1 | cpu | 3/3 |
| `pytorch__129138__linear_add_bias_autocast` | #129138 | P3 | cpu | 3/3 |
| `pytorch__132835__njt_sdpa_autocast` | #132835 | P3 | cuda_py | 1/3 |
| `pytorch__144009__softmax_ilpreduce_size` | #144009 | P5 | cuda_kernel | 3/3 |
| `pytorch__139372__histc_int8_cuda_bounds` | #139372 | P5 | cuda_kernel | 3/3 |

Full-run metrics: patch conciseness 1.000, pass-to-pass kept rate 94.1%, regression rate 0%, and tier-weighted score 76.8%. P4 remains N/A because no P4 task passed admission. See the [v0.5 experiment report](docs/v0.5/experiment_report.md) for integrity evidence, failure analysis, and metric definitions.

Tier codes: `cpu` = `cpu_python_overlay`, `cuda_py` = `cuda_python_overlay`, `cuda_kernel` = `cuda_kernel_build`.

Use `--verified-only` for benchmark runs that should count only admitted tasks. Use `--filter-tasks` to run a subset.

Every verified task has task-local stable admission evidence at its own `admission/evidence.json`.

## Runtime Boundary

Agent identity, model calls, and control logic run on the host. Each repair attempt gets a prepared workspace and a task-scoped Docker runtime. The agent must interact with the target repository through OpBench actions:

- `read_file`
- `write_file`
- `apply_patch`
- `run_command`
- `run_test`
- `git_diff`

For Docker tasks, preflight commands, setup commands, test commands, and action-interface command calls run in the same task container. This preserves setup state and avoids mixing host execution with benchmark execution.

`codex_action_bridge` is the reference implementation of this boundary. It gives Codex a scratch directory plus an `opbench_action.py` CLI. The target repository path is not exposed to Codex, and final scoring uses the patch exported through the action interface.

## Adding More Work

Platform development should follow the [v0.6 design](docs/v0.6/design.md), [implementation plan](docs/v0.6/implementation_plan.md), and [acceptance matrix](docs/v0.6/acceptance_matrix.md). Dataset expansion follows the existing admission workflow below and the [v0.7 design](docs/v0.7/design.md):

1. Add or curate task bundles under `tasks/<framework>/`.
2. Register reusable environment/source assets under `environments/registry.json` and `sources/registry.json`.
3. Run `scripts/preflight_task.py` to catch snapshot/patch/test-name issues offline.
4. Run `scripts/run_admission.py --write-task-evidence`.
5. Add only evidence-backed verified tasks to a verified dataset slice with `scripts/curate_dataset.py`.
6. Run `scripts/run_experiment.py` on `--verified-only`.
7. Add new real agents by implementing the action-interface boundary rather than giving them direct access to target workspaces.

## References

- [Docs index](docs/README.md)
- [Global project plan](docs/project_plan.md)
- [Current project state](docs/project_state.md)
- [v0.6 platform design](docs/v0.6/design.md)
- [v0.6 implementation plan](docs/v0.6/implementation_plan.md)
- [v0.6 acceptance matrix](docs/v0.6/acceptance_matrix.md)
- [v0.6 real MCP Agent experiment](docs/v0.6/mcp_agent_experiment.md)
- [v0.7 Dataset Factory and Boundary design](docs/v0.7/design.md)
- [v0.5 design](docs/v0.5/design.md)
- [v0.5 experiment report](docs/v0.5/experiment_report.md)
- [v0.4 design](docs/v0.4/design.md)
- [v0.4 experiment report](docs/v0.4/experiment_report.md)
- [v0.3 design](docs/v0.3/design.md)
- [v0.3 experiment report](docs/v0.3/experiment_report.md)
- [v0.2 developer guide](docs/v0.2/developer_guide.md)
- [v0.2 experiment report](docs/v0.2/experiment_report.md)
- [v0.1 developer guide](docs/v0.1/developer_guide.md)
- [v0.1 manual validation workflow](docs/v0.1/manual_validation.md)
