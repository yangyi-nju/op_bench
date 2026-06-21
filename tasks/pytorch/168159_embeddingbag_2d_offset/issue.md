# EmbeddingBag produces wrong results with 2D input and include_last_offset=True

## Bug Description

When using `nn.EmbeddingBag` with a 2D input tensor and `include_last_offset=True`, the offsets are computed incorrectly, leading to wrong aggregation results.

## Steps to Reproduce

```python
import torch
import torch.nn as nn

embedding_bag = nn.EmbeddingBag(10, 3, mode='sum', include_last_offset=True)

# 2D input: each row is a separate bag
input_2d = torch.tensor([[1, 2, 4], [4, 3, 2]])

# This should work correctly but produces wrong results
output = embedding_bag(input_2d)
print(output.shape)  # Expected: (2, 3)
```

## Expected Behavior

`EmbeddingBag` should correctly handle 2D input with `include_last_offset=True`, producing the same results as when the offsets are manually specified.

## Actual Behavior

The offset computation in `torch.nn.functional.embedding_bag` does not account for `include_last_offset` when constructing offsets from a 2D input, causing incorrect aggregation.

## Reference

- Issue: https://github.com/pytorch/pytorch/issues/167974
- Fix PR: https://github.com/pytorch/pytorch/pull/168159
