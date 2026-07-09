# CPU native_layer_norm decomposition returns wrong dtype under fake tensor dispatch

Source: [pytorch/pytorch#140557](https://github.com/pytorch/pytorch/pull/140557).

## Symptom

When `torch.ops.aten.native_layer_norm.default` is invoked under `FakeTensorMode`
with the Python dispatcher enabled, the decomposition in
`torch/_refs/__init__.py` misses the CPU-specific low-precision upcast /
return-dtype handling that eager CPU `native_layer_norm` performs.

Repro (bfloat16 inputs, CPU):

```python
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

def f(x, w, b):
    return torch.ops.aten.native_layer_norm.default(x, [1, 2, 3], w, b, eps=0.5)

x = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
w = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
b = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
out_ref = f(x, w, b)

with torch._subclasses.fake_impls.enable_python_dispatcher(), FakeTensorMode():
    x_fake = torch.randn(1, 2, 3, dtype=torch.bfloat16, device="cpu")
    w_fake = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
    b_fake = torch.randn(1, 2, 3, dtype=torch.bfloat16, requires_grad=True, device="cpu")
    out_fake = f(x_fake, w_fake, b_fake)

for r, fk in zip(out_ref, out_fake):
    assert r.dtype == fk.dtype, f"{r.dtype} vs {fk.dtype}"
```

The final `assert` fires because the mean/rstd tensors from the fake dispatch
path lose the CPU-specific dtype semantics.

## Fix approach

Only `torch/_refs/__init__.py` is in scope. Register a fake impl for
`aten.native_layer_norm.default` that routes back through the real
`native_layer_norm` decomposition so the CPU-specific dtype path is preserved
under fake tensor mode.
