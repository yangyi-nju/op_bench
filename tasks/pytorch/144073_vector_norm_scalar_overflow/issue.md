# torch.linalg.vector_norm overflows for scalar inputs with large ord

Source: [pytorch/pytorch#144073](https://github.com/pytorch/pytorch/pull/144073)
Issue: [pytorch/pytorch#143960](https://github.com/pytorch/pytorch/issues/143960)

## Symptom

`torch._refs.linalg.vector_norm` computes `pow(sum(pow(x, ord)), 1/ord)`.
For a single-element tensor with large `|ord|`, `pow(x, ord)` overflows to inf even though the correct answer is simply `abs(x)`.

```python
import torch
x = torch.tensor([2.0])
# large negative ord: should be abs(x) = 2.0
result = torch.compile(lambda a: torch.linalg.vector_norm(a, ord=-41.0))(x)
assert not result.isinf()  # fails on base commit
```

Eager `torch.linalg.vector_norm` handles this correctly; the Inductor path via `torch._refs` does not.

## Fix approach

Only `torch/_refs/linalg/__init__.py` is in scope. Add an early-return for scalar
(single-element) tensors before the pow-then-root reduction: return `abs(x)` directly.
