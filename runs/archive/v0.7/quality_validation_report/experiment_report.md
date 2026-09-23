# OpBench pytorch_v0.7 MCP Validation (opbench-v0.6.0)

- Adapter: `codex_mcp_canonical`
- Model: `gpt-5.6-sol`
- Codex CLI: `codex-cli 0.147.0-alpha.1.2`
- Cohorts: 17
- Selected Attempts: 122
- Complete MCP traces: 122
- Retries: 8

## Evaluation outcomes

- `f2p_failed`: 52
- `invalid_patch`: 28
- `resolved`: 42

## Cohorts

- `cohort:v1:010b0e263d2b4c93c39471c3952f2c05b0923b3c33dc0837c2a68a3d5b980416`: 3 Attempts; profiles `remote-cuda-expansion-nightly-torch2.13.0dev20260417-cu126-devel-py311-v1`
- `cohort:v1:0a58ed2f5d2c9df0bce09eeed9de7cdc1bc6963b41422643cd460b55ea967f4f`: 1 Attempts; profiles `remote-cpu-matched-torch2.7-py311-v1`
- `cohort:v1:106af1849403f978cecfcc3579bd593dd10c12ffee63c6469eb59583f35b1d74`: 3 Attempts; profiles `remote-cuda-expansion-nightly-torch2.14.0dev20260612-cu126-devel-py311-v1`
- `cohort:v1:21f3d3e32c61e6ebd42eb3ba51cd137461cc5604bb3b68331129e1e6601969fe`: 15 Attempts; profiles `remote-cpu-expansion-nightly-torch2.14.0dev20260612-py311-v1`
- `cohort:v1:4198d443af15a05f0dbc00e956675d028086ca1fcf47474b9c0031227d24cb5b`: 3 Attempts; profiles `remote-cuda-expansion-nightly-torch2.14.0dev20260612-cu126-py311-v1`
- `cohort:v1:437bfa0969002063611aa9dd5e259f90f0a7eddd63285462a6186b645cbb30aa`: 21 Attempts; profiles `remote-cuda-expansion-nightly-torch2.12.0dev20260407-cu126-py311-v1`
- `cohort:v1:5a85bb51703a63d5da373672adbaf9ea2fb19f3ac113cb5ca048678bf1165e03`: 15 Attempts; profiles `remote-cpu-expansion-nightly-torch2.13.0dev20260423-py311-v1`
- `cohort:v1:6d357c8ed2c1951d5f1286fbec4112cd444214401b8ab25d22c9d5924cd43232`: 9 Attempts; profiles `remote-cpu-expansion-nightly-torch2.14.0dev20260707-py311-v1`
- `cohort:v1:8367482e5d8a01bf65f6c71234b8e0c3306d91858c212e63bb4d390a62c78bb0`: 3 Attempts; profiles `remote-cpu-expansion-nightly-torch2.14.0dev20260710-py311-v1`
- `cohort:v1:8a4ece2ce2b12ea99552b4c6708059e46fce25d0b842fee3ea023ee0c4c1dd0a`: 1 Attempts; profiles `remote-cuda-matched-torch2.4-cu124-py311-v1`
- `cohort:v1:8b44417cf5d44361c824dacd8ae9c26c3cfd52132d45b897e77afe4d0cc45ef7`: 33 Attempts; profiles `remote-cpu-expansion-nightly-torch2.12.0dev20260407-py311-v1`
- `cohort:v1:9c6dca717bf5dd49b82e4611f4095db7b886cb04f26f48882bc88ed4e57fa487`: 7 Attempts; profiles `remote-cpu-pytorch-2.6-py311-v1`
- `cohort:v1:a4a4bfd4b6ce8d1cf001ee5890bd59a5ad769d9200c39593b2c8267214f98366`: 2 Attempts; profiles `remote-cuda-kernel-pytorch-2.6-cu124-v1`
- `cohort:v1:b8646b5e53d82bb88ce6aead757e9ec752649ce1c63dfb79edaceca8b887b50c`: 1 Attempts; profiles `remote-cpu-compile-pytorch-2.6-py311-v1`
- `cohort:v1:c0ce598126c8605a6e2b0595fd455f75710f301b8cc133e43952af4b8f60f83f`: 1 Attempts; profiles `remote-cuda-overlay-pytorch-2.6-cu124-v1`
- `cohort:v1:c98e0e3e3bd013879f1aeaa068123de45eb8a930a696728308240e6cce7b9a73`: 3 Attempts; profiles `remote-cuda-expansion-nightly-torch2.14.0dev20260710-cu126-py311-v1`
- `cohort:v1:ff3941129a3069a21112b8f197f152a3ec948379fb85b3ef49c51dcd3a9b187c`: 1 Attempts; profiles `remote-cpu-source-boundary-py311-v1`

## Quality-axis coverage

- origin: `new_or_replacement` 36 Tasks/108 Attempts, `retained_historical` 14 Tasks/14 Attempts
- difficulty: `hard` 46 Tasks/116 Attempts, `medium` 4 Tasks/6 Attempts
- contract_family: `api_behavior` 27 Tasks/69 Attempts, `efficiency_safety` 6 Tasks/16 Attempts, `gradient` 2 Tasks/6 Attempts, `mutation_state` 5 Tasks/13 Attempts, `result` 6 Tasks/8 Attempts, `tensor_metadata` 4 Tasks/10 Attempts
- failure_type: `missing_error` 5 Tasks/5 Attempts, `performance_regression` 2 Tasks/6 Attempts, `unexpected_error` 31 Tasks/87 Attempts, `wrong_result` 12 Tasks/24 Attempts
- devices: `cpu` 35 Tasks/85 Attempts, `cuda` 15 Tasks/37 Attempts
- modes: `compile` 35 Tasks/99 Attempts, `eager` 21 Tasks/35 Attempts
- phases: `backward` 4 Tasks/8 Attempts, `forward` 49 Tasks/119 Attempts
- derived_slices: `boundary` 31 Tasks/77 Attempts, `device` 15 Tasks/37 Attempts, `precision` 5 Tasks/5 Attempts

## Failure attribution

- `task`: 0
- `agent`: 80
- `evaluator`: 0
- `runtime`: 7
- `infrastructure_retries`: 8
- Agent counts describe selected valid outcomes; Runtime and infrastructure-retry counts describe prior invalid retry history and are not additive with the Agent denominator.

## Ceiling/floor observations

- All observed Attempts resolved: 15 Tasks
- No observed Attempt resolved: 29 Tasks
- These are descriptive observations for the frozen current configuration, not leaderboard claims.
