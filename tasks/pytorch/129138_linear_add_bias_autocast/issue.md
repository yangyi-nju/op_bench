# Inductor linear_add_bias fusion folds fp32 bias under autocast bf16

Source: [pytorch/pytorch#129138](https://github.com/pytorch/pytorch/pull/129138)

## Symptom

Under `torch.autocast('cpu', dtype=torch.bfloat16)`, the `linear_add_bias` MKLDNN fusion pass in Inductor incorrectly folds a fp32 bias when the weight is bf16. The pass checks that bias has the same dtype as the fused linear output but not that it matches the weight dtype, so a fp32 bias gets folded into a bf16-typed kernel path.

```python
import torch
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 64, bias=False)
        self.bias = torch.randn(64)  # fp32 bias, weight will be bf16 under autocast
    def forward(self, x):
        return self.linear(x) + self.bias

with torch.autocast('cpu', dtype=torch.bfloat16):
    m = torch.compile(M().eval())
    m(torch.randn(2, 10))  # wrong result due to bias dtype mismatch
```

## Fix approach

Only `torch/_inductor/fx_passes/mkldnn_fusion.py` is in scope. In `is_linear_add_bias`, add a check `if bias_meta.dtype != weight_meta.dtype: return False` before allowing the fusion.
