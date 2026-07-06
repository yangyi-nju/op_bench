# SoftMax CUDA ilpReduce underflows size counter, produces wrong output for double

## Bug Description

In the CUDA SoftMax `ilpReduce` and `WriteFpropResultsVectorized` inner loops, the tail iteration unconditionally decrements the loop counter with `size -= blockDim.x`. When the remaining elements are fewer than blockDim.x, this makes `size` underflow (large unsigned value) and the subsequent tail work touches invalid indices. Result: log_softmax + exp does not sum to 1 for certain shapes (e.g. (5, 513) in float64).

Fix (2 lines): clamp the decrement to at most `size` itself, so the counter never underflows.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/144009
- Issue: https://github.com/pytorch/pytorch/issues/143644
