"""Public numerical transport for the complete source-built CUDA framework.

No expected values, tolerance decisions, private grader imports, or test verdicts
are present here. The controller compares the returned observations separately.
"""
import json
from pathlib import Path
import sys


def load_source_torch(workspace):
    builds = [path.resolve() for path in (workspace / "build").glob("lib.*")
              if (path / "torch/__init__.py").is_file()
              and list((path / "torch").glob("_C*.so"))]
    if len(builds) != 1:
        raise RuntimeError("Build the complete candidate using python3 setup.py build; expected one build/lib.* package")
    if "torch" in sys.modules:
        raise RuntimeError("Source loader requires a fresh Python process")
    sys.path.insert(0, str(builds[0]))
    import torch
    if not Path(torch.__file__).resolve().is_relative_to(builds[0]):
        raise RuntimeError("Loaded torch Python package is outside the candidate build")
    if not Path(torch._C.__file__).resolve().is_relative_to(builds[0]):
        raise RuntimeError("Loaded torch native extension is outside the candidate build")
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise RuntimeError("A working source-built CUDA runtime and a selected GPU are required")
    torch.cuda.set_device(0)
    if torch.cuda.get_device_capability(0) != (7, 0):
        raise RuntimeError("This environment revision requires a selected compute-capability 7.0 device")
    torch.set_num_threads(1)
    libraries = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                        if "/" in line and ("/libtorch" in line or "/libc10" in line)})
    if not any(Path(name).name == "libtorch_cuda.so" for name in libraries):
        raise RuntimeError("The CUDA framework library was not loaded")
    if any(not Path(name).resolve().is_relative_to(workspace) for name in libraries):
        raise RuntimeError("A framework shared library was loaded from outside the candidate workspace")
    print(json.dumps({"event": "source_cuda_runtime", "python_package": torch.__file__,
        "native_extension": torch._C.__file__, "framework_libraries": libraries,
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0), "compute_capability": [7, 0],
        "device_count": torch.cuda.device_count(),
        "witness_limit": "Normal runtime diagnostics, not proof that arbitrary candidate code cannot spoof device or import metadata"}), file=sys.stderr)
    return torch


def execute_batch(torch, request):
    import torch.nn.functional as functional
    observed = []
    for index, item in enumerate(request["operations"]):
        dtype = {"float64": torch.float64, "float32": torch.float32}[item["dtype"]]
        shape = item["shape"]
        value = torch.tensor(item["values"], dtype=dtype, device="cuda:0").reshape(shape)
        if item.get("layout", "contiguous") == "noncontiguous":
            value = value.t().contiguous().t()
            if value.is_contiguous():
                raise RuntimeError("Requested noncontiguous layout was not constructed")
        gradient = item["operation"].startswith("grad_")
        value = value.detach().requires_grad_(gradient)
        function = functional.log_softmax if item["operation"].endswith("log_softmax") else functional.softmax
        output = function(value, dim=item["dim"])
        if gradient:
            weights = torch.tensor(item["gradient"], dtype=dtype, device="cuda:0").reshape(shape)
            output = torch.autograd.grad(output, value, grad_outputs=weights)[0]
        if output.shape != value.shape or output.dtype != dtype or not output.is_cuda:
            raise RuntimeError("CUDA operator output must preserve the requested shape, dtype and device")
        torch.cuda.synchronize(0)
        print(json.dumps({"event": "cuda_observation", "index": index, "name": item["name"],
            "operation": item["operation"], "shape": list(value.shape), "stride": list(value.stride()),
            "dim": item["dim"], "dtype": str(output.dtype), "device": str(output.device)}), file=sys.stderr)
        observed.extend(output.detach().cpu().reshape(-1).tolist())
    return {"shape": [len(observed)], "values": observed}


def main():
    workspace = Path(__file__).resolve().parent.parent
    torch = load_source_torch(workspace)
    request = json.load(sys.stdin)
    observation = execute_batch(torch, request)
    print(json.dumps(observation, allow_nan=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
