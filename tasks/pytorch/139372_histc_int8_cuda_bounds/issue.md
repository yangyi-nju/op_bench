# CUDA histc does not check min > max for int8 tensors

Source: [pytorch/pytorch#139372](https://github.com/pytorch/pytorch/pull/139372)
Issue: [pytorch/pytorch#139360](https://github.com/pytorch/pytorch/issues/139360)

## Symptom

`torch.histc` on CUDA with `int8` input and out-of-range bounds should raise
`RuntimeError: max must be larger than min`. Instead the `min` value wraps to
int8 and the invalid range is silently accepted:

```python
import torch
t = torch.tensor([1., 2, 1], dtype=torch.int8, device='cuda')
# 256 wraps to 0 when stored in int8, so base incorrectly sees 0 < 1.
torch.histc(t, bins=4, min=256, max=1)
```

The root cause is in `_histc_cuda_template` in `SummaryOps.cu`: the `minvalue`
and `maxvalue` variables are declared as `input_t` (i.e. `int8`). When `min=5`
and `max=1` are passed, the int8 cast wraps/clips them silently, making the
`min == max` branch take a different path without triggering the error check.

## Fix approach

`aten/src/ATen/native/cuda/SummaryOps.cu` is in scope. Change `input_t minvalue`
and `input_t maxvalue` to `at::acc_type<input_t, true> minvalue / maxvalue` so the
bounds comparison uses a wider type. You may also modify
`torch/testing/_internal/common_methods_invocations.py` to add `torch.uint8` to
`dtypesIfCUDA` for `histc` so the test exercises the fix.
