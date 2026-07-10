# Inductor linear_add_bias fusion folds fp32 bias under autocast bf16

Source: [pytorch/pytorch#129138](https://github.com/pytorch/pytorch/pull/129138)

## Symptom

Under `torch.autocast('cpu', dtype=torch.bfloat16)`, the `linear_add_bias` MKLDNN fusion pass in Inductor incorrectly folds a fp32 bias when the weight is bf16. The pass checks that bias has the same dtype as the fused linear output but not that it matches the weight dtype, so a fp32 bias gets folded into a bf16-typed kernel path. On the pinned runtime this produces a `LoweringException` for the invalid mixed-dtype `mkldnn._linear_pointwise` graph; the compiled call never reaches a valid output comparison.

```python
import torch
from torch._inductor import config

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 64, bias=False)
        self.bias = torch.randn(64)  # fp32 bias, weight will be bf16 under autocast
    def forward(self, x):
        return self.linear(x) + self.bias

with config.patch({"freezing": True}), torch.no_grad():
    m = M().eval()
    x = torch.randn(2, 10)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected = m(x)
        actual = torch.compile(m)(x)
    torch.testing.assert_close(actual, expected)
```

The bug is in Inductor's freezing pass. The hidden test contains a narrow
compatibility shim for the `Match.erase_nodes` signature that changed between
the PR base commit and the PyTorch 2.6 runtime wheel; no other Inductor source
dependency is replaced.

## Fix approach

Only `torch/_inductor/fx_passes/mkldnn_fusion.py` is in scope. In `is_linear_add_bias`, add a check `if bias_meta.dtype != weight_meta.dtype: return False` before allowing the fusion.
