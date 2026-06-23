# torch.addcmul does not support CPU scalar tensor1/tensor2 with CUDA self

## Bug Description

`torch.addcmul(self, tensor1, tensor2, value=alpha)` should support CPU scalar tensors for `tensor1` or `tensor2` when `self` is on CUDA, matching general binary operator semantics. The fix updates the dispatch and CUDA kernel to accept CPU scalars and broadcast them properly.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/143264
