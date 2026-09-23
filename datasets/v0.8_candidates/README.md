# Upstream migration candidates

The [candidate manifest](dataset.json), version **0.8.0-candidate.2**, contains
three upstream-derived tasks. **All three remain unadmitted.** This index is a
development collection, not an admitted benchmark release or an Agent ranking.
The earlier `0.8.0-dev` manifest contained only the LayerNorm candidate; its
execution plans retain that earlier dataset identity and membership.

Version `0.8.0-candidate.2` records the compiled vector_norm task's
`candidate.2` prompt clarification: strict float32 singleton tolerances apply
to `ord=-41/+41`; the existing ordinary regression tolerances remain unchanged.
Its scoring revision stays `candidate.1`, with identical runtime inputs and
environment. Earlier controls and plans retain their original task and dataset
identities; this wording update creates no new physical execution or Agent evidence.

| Candidate | Operator behavior | Evidence and review |
| --- | --- | --- |
| [CPU LayerNorm](layer_norm_cpu/task.json) | Eager/FakeTensor dtype and shape, with numerical and gradient regressions | [Contract](layer_norm_cpu/README.md), [review](layer_norm_cpu/migration_review.md), [versioned validation](layer_norm_cpu/validation/) |
| [CUDA LogSoftmax](log_softmax_cuda/task.json) | Source-built CUDA reduction values and gradient regressions | [Contract](log_softmax_cuda/README.md), [review](log_softmax_cuda/migration_review.md), [versioned validation](log_softmax_cuda/validation/) |
| [CPU compiled vector_norm](vector_norm_compile_cpu/task.json) | Actual Inductor execution of singleton and ordinary norm reductions | [Contract](vector_norm_compile_cpu/README.md), [relocation review](vector_norm_compile_cpu/migration_review.md), [versioned validation](vector_norm_compile_cpu/validation/) |

Execution progress belongs in each task's validation records, with the relevant
source, task, environment and scoring revisions. Preserve failures and older
results; do not copy changing runtime status into task metadata or interpret a
successful control matrix as formal admission.

```bash
opbench datasets validate --dataset datasets/v0.8_candidates/dataset.json
```

The manifest can load before all source checkouts or images are prepared.
Schema validation establishes structure and declared identities, not runnable
environments or semantic admission. Each candidate uses a complete configured
PyTorch source build; every patch still participates in normal building even
when a compatible trusted baseline is reused. The two CPU tasks deliberately
share a source/build environment, while retaining separate logical identities.

See the [execution guide](../../docs/v0.8/remote_execution.md) for controller-side
preparation and controls. Task packages, private graders and control patches
stay with the controller. Historical `verified` labels do not transfer to this
collection; independent review and release admission remain required.
