"""Rebuild the entire selected CPU Torch package through its upstream build.

The packaging output is recreated so stale Python copies cannot conceal edits.
Native CMake/Ninja dependencies still perform their normal full-source build or
incremental rebuild. This helper never installs or overlays an existing wheel.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    required = ("setup.py", "aten/src/ATen/native/layer_norm.cpp", "torch/_refs/__init__.py", "CMakeLists.txt")
    missing = [name for name in required if not (workspace / name).is_file()]
    missing += [name for name in ("git", "cmake", "ninja", "gcc", "g++") if shutil.which(name) is None]
    missing += [name for name in ("numpy", "yaml", "astunparse", "setuptools", "sympy", "optree") if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Missing full-source build inputs/prerequisites: " + ", ".join(missing))
    # Upstream setup.py additionally validates required submodule contents. The
    # preparation phase must supply all recursive pinned submodules offline.
    print(json.dumps({"workspace": str(workspace), "python": sys.version,
                      "phase": "prerequisites" if args.check else "full_source_build"}), flush=True)
    if args.check:
        return 0
    build_parent = workspace / "build"
    if build_parent.is_symlink() or (build_parent.exists() and not build_parent.resolve().is_relative_to(workspace)):
        raise SystemExit("build must stay inside the workspace")
    package = build_parent / "opbench-package"
    if package.is_symlink():
        package.unlink()
    elif package.exists():
        shutil.rmtree(package)
    # Retain all ordinary source and native build rules. Do not extract or
    # specially substitute native_layer_norm, and do not require Gold paths.
    subprocess.run([sys.executable, "setup.py", "build", "--build-lib", str(package)], cwd=workspace, check=True)
    if not (package / "torch/__init__.py").is_file() or not list((package / "torch").glob("_C*.so")):
        raise SystemExit("Upstream build did not produce the complete Torch package")
    # A separate interpreter verifies actual loading after native linking.
    probe = Path(__file__).with_name("source_torch.py")
    subprocess.run([sys.executable, "-I", str(probe), str(workspace)], cwd=workspace, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
