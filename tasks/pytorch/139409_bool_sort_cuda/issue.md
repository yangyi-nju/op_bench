# torch.sort does not support torch.bool dtype on CUDA

## Bug Description

torch.sort raises RuntimeError on CUDA when input dtype is torch.bool, while CPU works. The CUDA kernel dispatch table is missing bool dtype handling. Fix adds bool to the CUDA Sort dispatch.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/139409
