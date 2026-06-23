# torch.lerp raises RuntimeError when weight is CPU scalar and input/end are CUDA

## Bug Description

`torch.lerp(input, end, weight)` should accept a CPU scalar tensor as weight when `input` and `end` are on CUDA, matching the behavior of other binary operators. Currently it raises a RuntimeError because the device check rejects the mixed case. The fix relaxes the check in the CPU dispatch path and adds proper CUDA kernel handling for CPU scalar weight.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/141820
