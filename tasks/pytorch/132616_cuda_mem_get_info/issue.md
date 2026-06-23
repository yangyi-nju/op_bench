# torch.cuda.memory.mem_get_info rejects device str without explicit index

## Bug Description

`torch.cuda.memory.mem_get_info` raises an error when called with a device string like `"cuda"` (no explicit index), but should default to the current device, similar to other torch.cuda APIs.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/132616
