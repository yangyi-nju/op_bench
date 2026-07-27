# OpBench pytorch_v0.7_boundary MCP Validation (opbench-v0.6.0)

- Adapter: `codex_mcp_canonical`
- Model: `gpt-5.6-sol`
- Codex CLI: `codex-cli 0.146.0-alpha.3.1`
- Cohorts: 5
- Selected Attempts: 18
- Complete MCP traces: 18
- Retries: 0

## Evaluation outcomes

- `f2p_failed`: 3
- `no_patch`: 1
- `resolved`: 14

## Cohorts

- `cohort:v1:2aa586581e3460fc7c9abc3bd3e064452ce1c1c3299e237678dcc7415b05a63c`: 3 Attempts; profiles `remote-cpu-boundary-torch2.2-py311-v1`
- `cohort:v1:63a0072d7aae61790edb09230ceb8538e58dab275330f744f000ea11a2334b9e`: 3 Attempts; profiles `remote-cpu-boundary-torch2.4-py311-v1`
- `cohort:v1:6eb3f928238d14127969f7e6035e8a71ea0c1380714e657013d9e407c77e9e8a`: 6 Attempts; profiles `remote-cpu-source-boundary-py311-v1`
- `cohort:v1:89a6e398a24a1ff60e46d6537650c4e665e1ea6797dd5dc86acdb038a4c72a55`: 3 Attempts; profiles `remote-cpu-boundary-torch2.3-py311-v1`
- `cohort:v1:ca1a4130a8dc0c773275874ee05668b783b3109d68582901bdec27312bfc310a`: 3 Attempts; profiles `remote-cuda-boundary-torch2.6-cu124-v1`
