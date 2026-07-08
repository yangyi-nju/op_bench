# v0.5 Precision PR Screening Summary

## Candidate Pool

| # | PR | Subclass | Component | Patch Lines | Files | Bug Pattern |
|---:|---:|:---:|---|---:|---:|---|
| 1 | [#108559](https://github.com/pytorch/pytorch/pull/108559) | P1 | `aten/src/ATen/native/ReduceOps.cpp` | 77 | 2 | Permuted bf16/fp16 CPU reductions truncate intermediate sums when TensorIterator reduces more than two logical dimensions. |
| 2 | [#141052](https://github.com/pytorch/pytorch/pull/141052) | P1 | `torch/_inductor/codegen/triton.py` | 71 | 2 | Generated Triton fp16/bf16 math reductions accumulate in low precision instead of promoting reducer inputs to fp32. |
| 3 | [#147203](https://github.com/pytorch/pytorch/pull/147203) | P1 | `aten/src/ATen/native/layer_norm.cpp` | 36 | 3 | RMSNorm downcasts the normalized value before the weight multiply, causing extra fp16/bf16 truncation. |
| 4 | [#140557](https://github.com/pytorch/pytorch/pull/140557) | P1 | `torch/_refs/__init__.py` | 30 | 2 | CPU native_layer_norm decomposition misses CPU-specific low-precision upcast/return-dtype behavior under fake tensor dispatch. |
| 5 | [#128953](https://github.com/pytorch/pytorch/pull/128953) | P2 | `torch/ao/quantization/fx/convert.py` | 67 | 3 | PT2E decomposed dequantize conversion drops the intended non-float output dtype such as bf16. |
| 6 | [#139999](https://github.com/pytorch/pytorch/pull/139999) | P2 | `torch/masked/_ops.py` | 29 | 3 | torch.masked.mean on bool tensors infers bool for the internal sum, clamping totals to True and losing count precision. |
| 7 | [#144646](https://github.com/pytorch/pytorch/pull/144646) | P2 | `torch/_inductor/codegen/cpp.py` | 164 | 3 | Inductor low-precision bitcast legalization upcasts values before dtype-sensitive sink operations such as bitwise ops. |
| 8 | [#148686](https://github.com/pytorch/pytorch/pull/148686) | P2 | `aten/src/ATen/native/mps/operations/BitwiseOps.mm` | 59 | 2 | MPS scalar-to-tensor bitshifts cast the scalar after the operation, diverging from CPU low-width integer dtype semantics. |
| 9 | [#151598](https://github.com/pytorch/pytorch/pull/151598) | P2 | `torch/_inductor/codegen/triton.py` | 52 | 2 | Inductor uint view-copy codegen undoes an upcast before dtype bitcast using imprecise dtype logic. |
| 10 | [#129138](https://github.com/pytorch/pytorch/pull/129138) | P3 | `torch/_inductor/fx_passes/mkldnn_fusion.py` | 30 | 2 | linear_add_bias fusion assumes bf16 bias under autocast and mishandles fp32 bias with bf16 weight. |
| 11 | [#132835](https://github.com/pytorch/pytorch/pull/132835) | P3 | `torch/nested/_internal/sdpa.py` | 125 | 2 | Nested jagged tensor SDPA intercepts before dispatcher autocast, so SDPA inputs are not cast to the expected autocast dtype. |
| 12 | [#133938](https://github.com/pytorch/pytorch/pull/133938) | P3 | `aten/src/ATen/ThreadLocalState.cpp` | 49 | 3 | CUDA AMP autocast dtype is thread-local, causing backward side threads to use fp16 when forward selected bf16. |
| 13 | [#137495](https://github.com/pytorch/pytorch/pull/137495) | P3 | `torch/amp/autocast_mode.py` | 63 | 3 | Non-strict export drops autocast enter/exit nodes, losing mixed-precision context in the exported graph. |
| 14 | [#141065](https://github.com/pytorch/pytorch/pull/141065) | P3 | `torch/_export/passes/replace_autocast_with_hop_pass.py` | 60 | 3 | Autocast higher-order-op replacement pass mishandles nested autocast regions. |
| 15 | [#119898](https://github.com/pytorch/pytorch/pull/119898) | P4 | `aten/src/ATen/native/vulkan/ops/Softmax.cpp` | 60 | 2 | Vulkan log_softmax can take log(0) after fp16 softmax underflows tiny probabilities to zero, producing -inf/NaN. |
| 16 | [#121381](https://github.com/pytorch/pytorch/pull/121381) | P4 | `aten/src/ATen/native/mps/operations/TensorCompare.mm` | 50 | 2 | MPS torch.clamp uses min/max forms that do not propagate NaN like PyTorch clamp semantics require. |
| 17 | [#129154](https://github.com/pytorch/pytorch/pull/129154) | P4 | `torch/_refs/__init__.py` | 24 | 2 | exp decomposition misses eager numeric guards, allowing sampled/log-transformed values to produce inf. |
| 18 | [#129352](https://github.com/pytorch/pytorch/pull/129352) | P4 | `aten/src/ATen/native/cpu/ReduceOpsKernel.cpp` | 55 | 2 | Large CPU norm reductions overflow fp32 range when reduced directly, producing incorrect results. |
| 19 | [#144073](https://github.com/pytorch/pytorch/pull/144073) | P4 | `torch/_refs/linalg/__init__.py` | 34 | 3 | vector_norm for scalar input computes power then root and can overflow where eager avoids the overflow. |
| 20 | [#137529](https://github.com/pytorch/pytorch/pull/137529) | P5 | `torch/csrc/distributed/c10d/CUDASymmetricMemoryOps.cu` | 99 | 3 | CUDA multimem.ld_reduce accumulates bfloat16 in bf16 precision instead of using fp32 accumulator precision. |
| 21 | [#139372](https://github.com/pytorch/pytorch/pull/139372) | P5 | `aten/src/ATen/native/cuda/SummaryOps.cu` | 32 | 3 | CUDA histc stores min/max bounds in low-precision input_t for int8, making min > max checks compare the wrong values. |

## 筛选说明

检索使用 `docs/v0.5/candidate_search.md` 修订后的路径 A：PyTorch main `git log` 反查 `Pull Request resolved` commit，并用 commit author date 填入 screener 的 `mergedAt` 字段。P1 的 4 条候选按维护者确认直接纳入最终池。

- P1: 9 fallback candidates checked; 7 passed hard filter; 3 soft rejected or reclassified; 4 accepted by maintainer.
- P2: 61 candidates checked; 13 passed hard filter; 8 soft rejected/not duplicated; 5 final.
- P3: 88 candidates checked; 18 passed hard filter; 13 soft rejected; 5 final.
- P4: 340 candidates checked; 66 passed hard filter; 61 soft rejected; 5 final.
- P5: 16 candidates checked; 4 passed hard filter; 2 soft rejected; 2 final. P5 is intentionally below 3 because strict CUDA-kernel precision candidates were sparse; #144009 and #140259 were plausible but failed hard filters.

P5 说明：最终只保留 2 条，是因为真实 CUDA kernel 精度 bug 在窗口内非常稀缺。未为了凑数纳入 H100/FP8、纯性能优化、无测试或低于 20 行硬过滤的 PR。
