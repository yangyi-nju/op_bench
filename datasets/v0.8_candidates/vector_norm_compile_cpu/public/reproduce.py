"""Public deterministic reproducer; no private oracle is read."""
from numeric_worker import load_source_torch
import os
from pathlib import Path
import sys
import tempfile

workspace = Path(sys.argv[1]).resolve()
os.environ['TORCHINDUCTOR_CACHE_DIR'] = tempfile.mkdtemp(prefix='opbench-reproduce-', dir=os.environ['TMPDIR'])
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
torch = load_source_torch(workspace)

def operation(value):
    return torch.linalg.vector_norm(value, ord=-41.0, dim=1)

value = torch.tensor([-16.0, -0.0625, 16.0, 0.0625] * 4, dtype=torch.float32).reshape(4, 1, 4)
compiled = torch.compile(operation, fullgraph=True, backend='inductor', dynamic=False)
actual = compiled(value)
expected = value.abs().squeeze(1)
print({'compiled': actual.tolist(), 'expected_singleton_abs': expected.tolist()})
if not torch.isfinite(actual).all() or not torch.allclose(actual, expected, atol=1e-6, rtol=1e-6):
    raise AssertionError('Compiled singleton vector_norm does not preserve the finite absolute value')
