# OpBench v0.6 Real MCP Experiment Verification

Date: 2026-07-23

Status: Passed. The formal matrix contains 51 distinct valid Attempts, 51
complete MCP traces, zero infrastructure-invalid results, and zero logical
retries.

## 1. Frozen experiment identity

- Dataset: `pytorch_v0.5`
- Dataset digest: `sha256:ff9d0c2999d1175a45165b387e0731dcaa211a190d994b176441ce81a0382abc`
- Platform: `opbench-v0.6.0`
- Adapter: `codex_mcp_canonical`
- Transport: invocation-local `mcp-stdio`
- Model: `gpt-5.6-sol`
- Codex CLI: `codex-cli 0.145.0-alpha.27`
- Action protocol: `action-v1`
- Evaluation: `evaluation-v1`
- Scoring: `legacy-v0.5-resolved-v1`
- Resume policy: `retry_infrastructure`

The four Runtime Profiles have separate Comparability Keys. Aggregation keeps
them as four cohorts and does not imply that their raw outcome rates are
interchangeable.

## 2. Formal cohort gates

| Runtime Profile | Expected/observed/valid | Outcome distribution | Root Integrity | Attempt Integrity | Trace | Cleanup |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `remote-cpu-pytorch-2.6-py311-v1` | 36/36/36 | 21 resolved, 14 F2P failed, 1 P2P regression | 14/14 | 36 × 10/10 | 36/36 | 36/36 |
| `remote-cpu-compile-pytorch-2.6-py311-v1` | 3/3/3 | 2 resolved, 1 F2P failed | 14/14 | 3 × 10/10 | 3/3 | 3/3 |
| `remote-cuda-overlay-pytorch-2.6-cu124-v1` | 6/6/6 | 6 resolved | 14/14 | 6 × 10/10 | 6/6 | 6/6 |
| `remote-cuda-kernel-pytorch-2.6-cu124-v1` | 6/6/6 | 6 resolved | 14/14 | 6 × 10/10 | 6/6 | 6/6 |
| **Matrix** | **51/51/51** | **35 resolved, 15 F2P failed, 1 P2P regression** | **4 × 14/14** | **51 × 10/10** | **51/51** | **51/51** |

All 51 Attempt IDs are distinct. Every terminal reason is `agent_finished` and
every public result is attributed to the Agent; MCP, Provider, Runtime, and
infrastructure attribution counts are zero. Legal low-score results were not
rerun.

The read-only resource verifier returned `runtime_resource_ownership=passed`
and `runtime_cleanup=passed` for every cohort. It reads only recorded handles;
no process/container enumeration, target discovery, port scan, ping, or broad
cleanup was used.

## 3. Per-task observations

Each row contains three independent Agent repeats.

| Task | Runtime Profile | Outcomes |
| --- | --- | --- |
| `pytorch__124385__load_state_dict_prefix` | Remote CPU | 3 resolved |
| `pytorch__129138__linear_add_bias_autocast` | Remote CPU Compile | 2 resolved, 1 F2P failed |
| `pytorch__132616__cuda_mem_get_info` | CUDA Overlay | 3 resolved |
| `pytorch__132835__njt_sdpa_autocast` | CUDA Overlay | 3 resolved |
| `pytorch__139372__histc_int8_cuda_bounds` | CUDA Kernel | 3 resolved |
| `pytorch__139999__masked_mean_bool_upcast` | Remote CPU | 3 resolved |
| `pytorch__140557__layer_norm_decomp_precision` | Remote CPU | 2 resolved, 1 F2P failed |
| `pytorch__143455__set_submodule` | Remote CPU | 1 resolved, 2 F2P failed |
| `pytorch__144009__softmax_ilpreduce_size` | CUDA Kernel | 3 resolved |
| `pytorch__147599__lazylinear_state_forward` | Remote CPU | 3 resolved |
| `pytorch__149693__lazylinear_init` | Remote CPU | 3 resolved |
| `pytorch__150975__autograd_backward_inputs` | Remote CPU | 2 F2P failed, 1 P2P regression |
| `pytorch__160952__bilinear_lazy_check` | Remote CPU | 3 resolved |
| `pytorch__161488__lbfgs_wolfe` | Remote CPU | 3 F2P failed |
| `pytorch__162340__nn_arg_length` | Remote CPU | 3 F2P failed |
| `pytorch__163961__dataloader_subset` | Remote CPU | 3 F2P failed |
| `pytorch__168295__autograd_create_graph` | Remote CPU | 3 resolved |

The descriptive resolved proportion is 35/51 (68.6%). It is not a causal MCP
comparison and must not be compared as if it were the same Agent/protocol as
the v0.5 experiment.

## 4. MCP and Action evidence

The 51 immutable `adapter_trace.json` records contain:

- initialize: 51/51;
- tool list: 51/51;
- tool calls: 747;
- protocol errors: 0;
- server terminal: `client_closed` for 51/51;
- model/CLI identity match: 51/51.

Read, edit, diff, and finish coverage is 51/51. Agent-side registered test
coverage is 0/51; official F2P/P2P tests still ran in the independent Fresh
Evaluator for all 51 Attempts. The public trace preserves 80
`capability_denied`, 131 `invalid_request`, and 151 `path_denied` observations.
They are valid recoverable Action results, not infrastructure timeouts.

All 51 patches changed one file. Patch sizes total 39,817 bytes, range from 532
to 2,358 bytes, and are bound across SessionResult, FrozenPatch, evaluation,
results, and report identities.

## 5. Execution diagnostics and platform fixes

Interrupted and invalid diagnostic roots are excluded from all counts. During
the Kernel cohort bring-up, three infrastructure defects were isolated before
the final clean root:

1. the 1.7 GB recursive PyTorch snapshot needed bounded `rsync --partial`
   continuation in the same exact Attempt-owned remote workspace;
2. the exact post-sync ccache seed copy needed the same bounded idempotent retry
   discipline after a transient new SSH connection failure;
3. the public absolute-path scanner mistook the valid C++ source comment
   `/*is_cuda=*/true` for a machine-local path.

Each fix was introduced with a failing regression test, focused verification,
and a real CUDA Kernel canary. The final Kernel root then completed 6/6 valid
resolved Attempts with no retry. The transfer and scanner fixes do not change
the frozen contracts, task matrix, scoring, or already completed successful
artifacts in the first three cohorts; they affect only transient continuation
and a source-comment false positive. This controller maintenance boundary is
recorded here rather than hidden as a single source commit claim.

## 6. Deterministic report

The public report builder was run twice against the same four immutable roots.
The two output directories were byte-identical:

| Public file | SHA-256 |
| --- | --- |
| [experiment_report.md](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_report.md) | `3a3ea200a58713bc7f5060a70a098e3e542bce6cb4c6965257baf273f4b77caf` |
| [experiment_index.json](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_index.json) | `4ab5a112291b7c8c4ba6df319765a1bf1fb225fcd1030a33078f3efacad114ff` |
| [experiment_summary.json](../../runs/v0.6_mcp_full_20260722_event_redaction_r5_report/experiment_summary.json) | `8c03361f088704d56730f8adc9a2e01b7a6d5115a78f2c49d1807e6ff76d2443` |

The public Artifact whitelist plus every Integrity report passed recursive
safety scanning. Complete controller manifests and private evaluation/resource
evidence were validated separately and are not public report inputs.

## 7. Publication and cleanup boundary

The three report files above are the only retained experiment output intended
for version control. Full cohort roots were required temporarily for 14/14
Integrity and exact cleanup verification, then removed from the workspace along
with canaries, failed attempts, and diagnostic roots. Cleanup uses explicit
known local paths only and does not touch remote resources or perform discovery.

The public report excludes private evaluation evidence, private resource
handles, raw Provider output, credentials/authentication state, target
host/user/key/root, PID/PGID, raw container names, remote workspace paths, and
controller absolute paths. Because the full roots are not published, 14/14
cannot be independently recomputed from the redacted report alone.
