# nll_loss2d on CUDA accumulates into uninitialized total_weight buffer

## Bug Description

The CUDA implementation of nll_loss2d_forward, on the reduction=None code path, resizes total_weight to a scalar without zeroing it, then accumulates into it. Because the buffer contains uninitialized GPU memory, the accumulated result is nondeterministic and typically wrong. The fix is a single total_weight.zero_() before the accumulation loop.

Reference: https://github.com/pytorch/pytorch/pull/182082
