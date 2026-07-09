# torch._refs.exponential produces Inf due to log(0) when uniform sample = 1.0

Source: [pytorch/pytorch#129154](https://github.com/pytorch/pytorch/pull/129154)
Issue: [pytorch/pytorch#127749](https://github.com/pytorch/pytorch/issues/127749)

## Symptom

`torch._refs.exponential` (the Python decomp of `Tensor.exponential_`) uses:

```python
return -1 / rate * torch.log1p(-torch.rand_like(self))
```

`curand_uniform` samples in `(0, 1]` — it can return exactly `1.0`. When `uniform = 1.0`, `log1p(-1.0) = log(0) = -inf`, and the result is `+inf`.

```python
import torch
inp = torch.empty((4, 400, 256), device='cuda')
out = torch._refs.exponential(inp)
assert not out.isinf().any()  # fails on base commit
```

## Fix approach

Only `torch/_refs/__init__.py` is in scope. Add an epsilon guard: when `uniform_val >= 1 - eps/2`, substitute `-eps/2` instead of calling `log`, matching CUDA's numerical handling in `curand`.
