"""Public source-package loader and actual native-library location evidence."""
import inspect
import json
from pathlib import Path
import sys


def load(workspace):
    workspace = Path(workspace).resolve()
    package = (workspace / "build/opbench-package").resolve()
    if not package.is_relative_to(workspace) or not (package / "torch/__init__.py").is_file():
        raise RuntimeError("Run the full source build; build/opbench-package is missing")
    if "torch" in sys.modules:
        raise RuntimeError("Source loading requires a fresh interpreter without an imported Torch")
    sys.path.insert(0, str(package))
    import torch
    import torch._refs
    from torch._subclasses.fake_tensor import FakeTensorMode

    loaded = {"torch": Path(torch.__file__).resolve(), "torch._C": Path(torch._C.__file__).resolve(),
              "torch._refs": Path(torch._refs.__file__).resolve(),
              "fake_tensor": Path(inspect.getfile(FakeTensorMode)).resolve()}
    if not all(path.is_relative_to(package) for path in loaded.values()):
        raise RuntimeError("Torch Python/native extension was not loaded from the rebuilt package")
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError("This environment contract requires a CPU-only build")
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("This candidate's source-loading evidence requires Linux /proc/self/maps")
    libraries = {}
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[-1].startswith("/"):
            path = Path(fields[-1]).resolve()
            if path.name in {"libtorch_cpu.so", "libtorch_python.so", "libc10.so"}:
                if not path.is_relative_to(package):
                    raise RuntimeError("A loaded Torch native library came from outside the rebuilt package")
                libraries[path.name] = str(path)
    if libraries.keys() != {"libtorch_cpu.so", "libtorch_python.so", "libc10.so"}:
        raise RuntimeError("Required CPU Torch native-library loading was not observed")
    print(json.dumps({"source_loading": {name: str(path) for name, path in loaded.items()},
                      "native_library_loading": libraries, "torch_version": torch.__version__}, sort_keys=True), flush=True)
    return torch


if __name__ == "__main__":
    load(sys.argv[1])
