# OpBench v0.6 Real MCP Agent Experiment

Date: 2026-07-20

Status: execution protocol frozen; record observed results only after all four
cohorts pass verification.

## 1. Purpose and frozen identities

This experiment validates the platform through the same interface intended for
real Agent evaluation. It runs all 17 verified PyTorch tasks three times, for
51 logical Attempts, through the independent `codex_mcp_canonical` Adapter.

The immutable identities are:

- Adapter: `codex_mcp_canonical`
- transport: invocation-local `mcp-stdio`
- model: `gpt-5.6-sol`
- CLI: `codex-cli 0.145.0-alpha.18`
- Action protocol: `action-v1`
- platform: `opbench-v0.6.0`
- resume policy: `retry_infrastructure`

Use `--codex-model gpt-5.6-sol` on every real MCP command. The CLI version is
detected from the exact local executable before the Manifest or Runtime output
is created; a version mismatch requires a new frozen platform identity and a
fresh offline gate.

## 2. MCP and network boundary

Each Attempt creates one private scratch directory, one descriptor-bound Action
exchange, and one zero-dependency stdio MCP server. The Adapter passes that
server to one ephemeral Codex invocation with invocation-local configuration.
It does not read or modify global Codex MCP configuration. MCP tools are only
the nine canonical OpBench Actions, and all Action semantics remain owned by
the server-side Canonical Action Service.

Codex itself receives a separate read-only cwd. The controller-private sibling
contains the launcher, trace, and Action exchange. A random per-Attempt
transport token reaches the MCP server through a one-shot inherited pipe, never
the Codex argv; the server drains it before Agent tool execution and passes it
only to the generated Action client. Direct client invocation is denied.
Launcher/client inode, mode, and content hash are rechecked after Codex exits.
Both Provider output and bridge output use hard byte limits while being drained
incrementally.

Provider network is permitted for the host-side Codex invocation. Task network
is denied. Remote Runtime traffic may address only the exact binding in
`configs/remote_hosts.json`. Safety rules are explicit: no ping, no host/service scan,
no target discovery, no process/container enumeration, and no search for
alternate hosts or services. Cleanup targets only the exact Attempt-owned
workspace, container, and process-group handles recorded during creation.

## 3. Offline proof and canaries

Before any real Provider call, run the real stdio server as a subprocess and
compare it with the canonical CLI path:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_runtime_conformance.py \
  --transport mcp-stdio \
  --output-dir runs/v0.6_mcp_offline_conformance_r3
```

The comparison covers Action observations, Budget deltas/totals, Event
request/observation pairing, workspace tree, frozen patch bytes, finish count,
and terminal reason. MCP initialize/list counters are validated independently.

Run the local canary first, then the exact Remote CPU canary:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset runs/v0.6_m6_local_codex_input/dataset/dataset.json \
  --verified-only \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 1 \
  --output-dir runs/v0.6_mcp_local_canary_r3 \
  --runtime-protocol v1 --runtime-profile local-cpu-process-v1 \
  --enable-external-canary

PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json --verified-only \
  --only-tasks pytorch__149693__lazylinear_init \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 1 \
  --output-dir runs/v0.6_mcp_remote_cpu_canary_r3 \
  --runtime-protocol v1 \
  --runtime-profile remote-cpu-pytorch-2.6-py311-v1 \
  --target-config configs/remote_hosts.json --enable-external-canary
```

A valid Agent failure is final and must not be retried. Only an
`infrastructure_invalid` record receives the next retry index under
`retry_infrastructure`; every prior retry stays immutable.

Malformed/oversized JSON-RPC and internal bridge failures terminate the MCP
server as `protocol_failed`; a later message cannot turn that retry valid.
Unknown methods and invalid Agent parameters remain recoverable protocol
responses. Standard request `_meta`, including `progressToken`, is removed at
the MCP boundary before Action validation. The invocation-local server is
required, and its exact nine tools are pre-approved for this non-interactive
run, so startup failures abort before Agent work while `approval=never` cannot
silently cancel Action calls. This approval applies only to the isolated
OpBench server; shell/sandbox policy is unchanged. Its frozen value is part of
the Agent adapter/config identity, so a changed approval mode cannot resume an
existing run root. If exact Codex PGID cleanup is unproven, the invalid retry is made
durable and the cohort aborts. A private 0600 marker blocks resume until a
zero-signal check proves that exact recorded PGID absent; no process listing,
name lookup, or later signal is used.

The first two local canary roots and the focused diagnostic roots are retained
as immutable infrastructure-invalid evidence from the pre-fix platforms. The
`r3` roots above are the clean canaries for the final platform identity.

## 4. Four formal cohorts

Run cohorts serially so the exact target and Provider are not oversubscribed.

### Remote CPU: 12 tasks × 3

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json --verified-only \
  --only-tasks \
    pytorch__149693__lazylinear_init \
    pytorch__147599__lazylinear_state_forward \
    pytorch__160952__bilinear_lazy_check \
    pytorch__162340__nn_arg_length \
    pytorch__163961__dataloader_subset \
    pytorch__168295__autograd_create_graph \
    pytorch__161488__lbfgs_wolfe \
    pytorch__150975__autograd_backward_inputs \
    pytorch__124385__load_state_dict_prefix \
    pytorch__143455__set_submodule \
    pytorch__140557__layer_norm_decomp_precision \
    pytorch__139999__masked_mean_bool_upcast \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 3 \
  --output-dir runs/v0.6_mcp_full_20260720_remote_cpu \
  --runtime-protocol v1 \
  --runtime-profile remote-cpu-pytorch-2.6-py311-v1 \
  --target-config configs/remote_hosts.json --enable-external-canary
```

### Remote CPU Compile: 1 task × 3

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json --verified-only \
  --only-tasks pytorch__129138__linear_add_bias_autocast \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 3 \
  --output-dir runs/v0.6_mcp_full_20260720_remote_cpu_compile \
  --runtime-protocol v1 \
  --runtime-profile remote-cpu-compile-pytorch-2.6-py311-v1 \
  --target-config configs/remote_hosts.json --enable-external-canary
```

### CUDA Overlay: 2 tasks × 3

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json --verified-only \
  --only-tasks pytorch__132835__njt_sdpa_autocast pytorch__132616__cuda_mem_get_info \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 3 \
  --output-dir runs/v0.6_mcp_full_20260720_cuda_overlay \
  --runtime-protocol v1 \
  --runtime-profile remote-cuda-overlay-pytorch-2.6-cu124-v1 \
  --target-config configs/remote_hosts.json --enable-external-canary
```

### CUDA Kernel: 2 tasks × 3

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.5/dataset.json --verified-only \
  --only-tasks pytorch__144009__softmax_ilpreduce_size pytorch__139372__histc_int8_cuda_bounds \
  --agent codex_mcp_canonical --codex-model gpt-5.6-sol \
  --agent-repeat 3 \
  --output-dir runs/v0.6_mcp_full_20260720_cuda_kernel \
  --runtime-protocol v1 \
  --runtime-profile remote-cuda-kernel-pytorch-2.6-cu124-v1 \
  --target-config configs/remote_hosts.json --enable-external-canary
```

## 5. Evidence and verification

Every selected MCP retry contains `adapter_trace.json`. It binds Adapter,
model, CLI, negotiated protocol, initialize/list/call counts,
`protocol_error_count`, and sanitized server terminal status. The public
`events.jsonl`, `session_result.json`, `final.patch`, `public_evaluation.json`,
`runtime_resources.jsonl`, and `runtime_cleanup.json` support review without
revealing target handles.

`private_evaluation.json` and `private_runtime_resources.json` remain
controller-only. They must not be published or staged with the platform commit.
The root and per-Attempt Integrity reports retain exactly 14 checks, including
Action pairing, MCP identity/trace binding, selected retry attribution,
evaluation identity, deterministic result rebuild, exact ownership, and exact
cleanup.

Verify each root with:

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/verify_runtime_resources.py \
  --run-root RUN_ROOT
```

Proceed to reporting only when selected counts are 36, 3, 6, and 6; all 51 are
valid, unblocked, and trace-complete; and every fresh Integrity/resource check
passes.

## 6. Deterministic four-cohort report

```bash
PATH=.venv/bin:$PATH PYTHONPATH=src python scripts/summarize_mcp_experiment.py \
  --run-root runs/v0.6_mcp_full_20260720_remote_cpu \
  --run-root runs/v0.6_mcp_full_20260720_remote_cpu_compile \
  --run-root runs/v0.6_mcp_full_20260720_cuda_overlay \
  --run-root runs/v0.6_mcp_full_20260720_cuda_kernel \
  --output-dir runs/v0.6_mcp_full_20260720_report \
  --expected-model gpt-5.6-sol \
  --expected-cli-version 'codex-cli 0.145.0-alpha.18'
```

The builder is read-only with respect to all four run roots. It re-runs
Integrity and produces only `experiment_index.json`, `experiment_summary.json`,
and `experiment_report.md` in the separate report directory. An identical
rebuild is byte-identical; an existing nonmatching report is refused. The CLI
also enforces the frozen Dataset digest, `opbench-v0.6.0`, the four exact Runtime
Profiles, the 17-task partition above, and repeats 1/2/3. Four unrelated roots
with the same 36/3/6/6 sizes are rejected.
