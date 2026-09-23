"""Public, deterministic numerical reproducer; private oracle is not imported."""
import math
from pathlib import Path
import sys

# Python isolated mode does not prepend the script directory automatically.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from numeric_worker import load_source_torch


torch = load_source_torch(Path(__file__).resolve().parent.parent)
import torch.nn.functional as functional

rows, width = 5, 513
values = [[(((column * 37 + row * 13) % 113) - 56) / 8 for column in range(width)]
          for row in range(rows)]
tensor = torch.tensor(values, dtype=torch.float64, device="cuda:0")
actual = functional.log_softmax(tensor, dim=1)
assert actual.shape == tensor.shape and actual.dtype == tensor.dtype and actual.is_cuda
torch.cuda.synchronize(0)
maximum = 0.0
for row, outputs in zip(values, actual.cpu().tolist()):
    assert all(math.isfinite(value) for value in outputs), "CUDA log_softmax returned nonfinite values"
    anchor = max(row)
    normalizer = math.log(math.fsum(math.exp(value - anchor) for value in row))
    expected = [value - anchor - normalizer for value in row]
    maximum = max(maximum, max(abs(left - right) for left, right in zip(outputs, expected)))
print({"shape": [rows, width], "dtype": "float64", "device": "cuda:0", "max_abs_error": maximum})
assert math.isfinite(maximum) and maximum <= 1e-10, "CUDA log_softmax disagrees with the independent scalar reference"
