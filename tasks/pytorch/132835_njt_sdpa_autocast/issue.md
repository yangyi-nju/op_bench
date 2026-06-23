# NJT SDPA missing manual autocast for nested tensor inputs

## Bug Description

When using `scaled_dot_product_attention` (SDPA) with nested tensors under autocast, the dtype is not correctly propagated, causing precision/dtype mismatches between the autocast cast and the operator inputs. The fix manually casts the nested tensor inputs to the autocast dtype.

## Reference

- PR: https://github.com/pytorch/pytorch/pull/132835
