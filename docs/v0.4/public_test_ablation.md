# Public Test Patches — Design Notes

For the v0.4 ablation experiment, we add `artifacts/public_test.patch` to a subset of
existing v0.3 tasks. The public test should:

1. Be a **simplified version** of the hidden test (sanity check)
2. Pass at gold AND fail at baseline (so it gives the agent a signal)
3. Not leak the hidden test's exact assertion

## Tasks selected for ablation

| Task | Hidden test | Public test design |
|------|-------------|--------------------|
| `pytorch__168295__autograd_create_graph` | asserts `g[0].requires_grad == False` when `create_graph=False` | Same API call but only check 1 input |
| `pytorch__124385__load_state_dict_prefix` | tests unexpected key with prefix-match | Basic state dict load (smoke test) |
| `pytorch__161488__lbfgs_wolfe` | tests max-iter behavior | Smoke test: `step()` doesn't crash |
| `pytorch__149693__lazylinear_init` | tests `reset_parameters` call count | Forward pass shape sanity |

## Patch file format

Each task's `artifacts/public_test.patch` adds a test that:
- Is in a separate file (e.g., `test/nn/test_op_bench_public.py`)
- Has a clear name like `test_<task>_smoke`
- Imports torch and basic deps only

## Ablation experiment

After adding patches, run:

```bash
# With public tests
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json --verified-only \
  --filter-tasks autograd_create_graph load_state_dict_prefix lbfgs_wolfe lazylinear_init \
  --agent codex_action_bridge --agent claude_code_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_ablation_with_public

# Without public tests (same tasks, --no-public-tests)
PYTHONPATH=src python3 scripts/run_experiment.py \
  --dataset datasets/pytorch_v0.3/dataset.json --verified-only \
  --filter-tasks autograd_create_graph load_state_dict_prefix lbfgs_wolfe lazylinear_init \
  --no-public-tests \
  --agent codex_action_bridge --agent claude_code_action_bridge \
  --agent-repeat 3 \
  --output-dir runs/v0.4_ablation_no_public
```

## Decision criteria

If `with_public` resolved rate > `no_public` by ≥ 10 percentage points → public tests
help, keep the mechanism.

If diff < 5 percentage points → public tests don't significantly help, simplify by
removing the mechanism in v0.5.

## TODO

- [ ] Write public_test.patch for 4 selected tasks
- [ ] Verify patch applies to source snapshot
- [ ] Verify public test passes after gold patch
- [ ] Update task.json `evaluation.public_tests` to declare test names
- [ ] Run ablation experiment
