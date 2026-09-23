"""Public CPU bfloat16 eager/FakeTensor reproduction, with no repair hints."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_torch import load

torch = load(sys.argv[1])
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode

torch.manual_seed(1)
x = torch.randn(1, 2, 3, dtype=torch.bfloat16)
w = torch.ones_like(x, requires_grad=True)
b = torch.zeros_like(x, requires_grad=True)
eager = torch.ops.aten.native_layer_norm.default(x, [1, 2, 3], w, b, eps=0.5)
with enable_python_dispatcher(), FakeTensorMode() as mode:
    fake = torch.ops.aten.native_layer_norm.default(mode.from_tensor(x), [1, 2, 3], mode.from_tensor(w), mode.from_tensor(b), eps=0.5)
observed = {"eager_dtypes": [str(t.dtype) for t in eager], "fake_dtypes": [str(t.dtype) for t in fake],
            "eager_shapes": [list(t.shape) for t in eager], "fake_shapes": [list(t.shape) for t in fake]}
print(json.dumps(observed, indent=2))
assert observed["eager_dtypes"] == observed["fake_dtypes"], "CPU bfloat16 eager/fake dtype mismatch"
assert observed["eager_shapes"] == observed["fake_shapes"], "CPU eager/fake shape mismatch"
