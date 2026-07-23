# OpBench v0.6 MCP Agent Experiment

- Adapter: `codex_mcp_canonical`
- Model: `gpt-5.6-sol`
- Codex CLI: `codex-cli 0.145.0-alpha.27`
- Cohorts: 4
- Selected Attempts: 51
- Complete MCP traces: 51
- Retries: 0

## Evaluation outcomes

- `f2p_failed`: 15
- `p2p_regression`: 1
- `resolved`: 35

## Cohorts

- `cohort:v1:30d7c886bb3f20792e90f40ca1b6a0dbcda18d3811f8989b5ac3b0174efbc050`: 3 Attempts; profiles `remote-cpu-compile-pytorch-2.6-py311-v1`
- `cohort:v1:7eb74f136f871afcc688e05864ff7e408273a1ec1a9a92c19da55c841ec4e497`: 6 Attempts; profiles `remote-cuda-overlay-pytorch-2.6-cu124-v1`
- `cohort:v1:8a1fc9b877a8c134ae50f3e4bc37a283c5ac05b3b5a43f6f227b51b4a71b91e3`: 6 Attempts; profiles `remote-cuda-kernel-pytorch-2.6-cu124-v1`
- `cohort:v1:d9562328d82679ed4bc4df489cdb2057405115f7bffc9a5f9c74d433ae045b0e`: 36 Attempts; profiles `remote-cpu-pytorch-2.6-py311-v1`
