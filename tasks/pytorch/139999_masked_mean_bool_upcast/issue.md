# torch.masked.mean on bool tensor silently produces wrong output

Source: [pytorch/pytorch#139999](https://github.com/pytorch/pytorch/pull/139999)

## Symptom

`torch.masked.mean` with a `torch.bool` input infers `dtype=torch.bool` for the internal sum, which clamps any count ≥ 2 to `True` (1). The result is silently wrong instead of raising an error.

```python
import torch
x = torch.tensor([True, True, True, True])
print(torch.masked.mean(x, 0))  # outputs tensor(True) instead of raising
```

## Fix approach

Only `torch/masked/_ops.py` is in scope. After inferring `dtype`, check that `dtype` is floating-point or complex; raise `ValueError` otherwise. This matches the contract of `torch.mean` itself.
