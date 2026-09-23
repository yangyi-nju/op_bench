"""Public CPU Inductor transport. Contains no oracle or numerical verdicts."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def mapped_libraries():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if '/' in line and '.so' in line.split()[-1]})


def load_source_torch(workspace):
    package = workspace / 'build/opbench-package'
    if not (package / 'torch/__init__.py').is_file() or not list((package / 'torch').glob('_C*.so')):
        raise RuntimeError('Build the complete configured source into build/opbench-package first')
    if 'torch' in sys.modules:
        raise RuntimeError('A fresh source-loading process is required')
    sys.path.insert(0, str(package))
    import torch
    for path in (torch.__file__, torch._C.__file__):
        if not Path(path).resolve().is_relative_to(package.resolve()):
            raise RuntimeError('Torch package/native extension is outside the complete candidate build')
    libraries = [path for path in mapped_libraries()
                 if Path(path).name.startswith(('libtorch', 'libc10'))]
    required = {'libtorch_cpu.so', 'libtorch_python.so', 'libc10.so'}
    if not required <= {Path(path).name for path in libraries}:
        raise RuntimeError('Required source-built framework libraries are not loaded')
    if any(not Path(path).resolve().is_relative_to(workspace) for path in libraries):
        raise RuntimeError('Framework library came from outside the candidate workspace')
    print(json.dumps({'event': 'source_cpu_runtime', 'package': torch.__file__,
                      'native_extension': torch._C.__file__, 'libraries': libraries}), file=sys.stderr)
    return torch


def main():
    workspace = Path(sys.argv[1]).resolve()
    request = json.load(sys.stdin)
    runtime = Path(os.environ['TMPDIR']).resolve()
    if not runtime.is_relative_to(workspace):
        raise RuntimeError('Compiler cache must stay within the declared workspace runtime')
    cache = Path(tempfile.mkdtemp(prefix='opbench-inductor-', dir=runtime))
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
    os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '0'
    torch = load_source_torch(workspace)
    from torch._inductor import config, metrics
    torch.set_num_threads(1)
    torch._dynamo.reset()
    metrics.reset()
    observed = []
    with torch._dynamo.config.patch(suppress_errors=False), config.patch(
            cpu_backend='cpp', fx_graph_cache=False, force_disable_caches=True, compile_threads=1):
        for item in request['operations']:
            dtype = {'float32': torch.float32, 'float64': torch.float64}[item['dtype']]
            value = torch.tensor(item['values'], dtype=dtype, device='cpu').reshape(item['shape'])
            if item.get('layout') == 'noncontiguous':
                value = value.transpose(0, -1).contiguous().transpose(0, -1)
                if value.is_contiguous():
                    raise RuntimeError('Requested noncontiguous input was not constructed')
            dim = None if item['dim'] is None else tuple(item['dim'])
            order, keepdim = float(item['ord']), item['keepdim']

            def operation(argument):
                return torch.linalg.vector_norm(argument, ord=order, dim=dim, keepdim=keepdim)

            compiled = torch.compile(operation, fullgraph=True, backend='inductor', dynamic=False)
            result = compiled(value)
            axes = set(range(value.ndim)) if dim is None else set(dim)
            shape = [1 if axis in axes else size for axis, size in enumerate(value.shape)] if keepdim else [
                size for axis, size in enumerate(value.shape) if axis not in axes]
            if list(result.shape) != shape or result.dtype != dtype or result.device.type != 'cpu':
                raise RuntimeError('Compiled vector_norm changed declared shape, dtype or device')
            numbers = result.detach().reshape(-1).tolist()
            # Nonfinite results remain valid transport observations. The private
            # controller decides whether finite flags and numbers are correct.
            for number in numbers:
                finite = math.isfinite(number)
                observed.extend((1.0 if finite else 0.0, number if finite else 0.0))
            print(json.dumps({'event': 'compiled_observation', 'name': item['name'],
                              'shape': list(result.shape), 'dtype': str(result.dtype),
                              'input_stride': list(value.stride()), 'returned_values': len(numbers)}), file=sys.stderr)
    compiled_libraries = [name for name in mapped_libraries() if Path(name).resolve().is_relative_to(cache)]
    if metrics.generated_kernel_count < 1 or not compiled_libraries:
        raise RuntimeError('No generated CPU kernel and loaded compiler-cache library were observed')
    print(json.dumps({'event': 'cpu_compilation_witness', 'generated_kernels': metrics.generated_kernel_count,
                      'cache': str(cache), 'loaded_libraries': compiled_libraries,
                      'limit': 'Ordinary compilation/loading diagnostics, not adversarial attestation of candidate code'}), file=sys.stderr)
    print(json.dumps({'shape': [len(observed)], 'values': observed}, allow_nan=False))


if __name__ == '__main__':
    main()
