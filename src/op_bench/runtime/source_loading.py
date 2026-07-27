from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import PurePosixPath
import shlex

from op_bench.runtime.contracts import RuntimeProfile
from op_bench.runtime.validation import ContractError, require_int, require_str


_PYTHON_OVERLAY_PROGRAM = r"""
import importlib
import json
import os
import pathlib
import shutil
import sys

cfg = json.loads(sys.argv[1])
workspace = pathlib.Path(".").resolve()
runtime_site = pathlib.Path("/tmp/op_bench_runtime/site-packages")
package = cfg["package"]
paths = cfg["paths"]
runtime_site.mkdir(parents=True, exist_ok=True)
os.chdir("/tmp")
installed = pathlib.Path(importlib.import_module(package).__file__).resolve().parent
destination_package = runtime_site / package
if not destination_package.exists():
    shutil.copytree(installed, destination_package, symlinks=True)
libs = installed.parent / f"{package}.libs"
if libs.exists() and not (runtime_site / libs.name).exists():
    shutil.copytree(libs, runtime_site / libs.name, symlinks=True)
for relative in paths:
    pure = pathlib.PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != package:
        raise ValueError("invalid overlay path")
    source = workspace.joinpath(*pure.parts)
    target = runtime_site.joinpath(*pure.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
print(json.dumps({"mode": "python_overlay", "overlay_count": len(paths)}, sort_keys=True))
""".strip()
_CPU_INPLACE_BUILD_ENVIRONMENT = (
    ("BUILD_TEST", "0"),
    ("CC", "gcc"),
    ("CCACHE_BASEDIR", "/workspace"),
    ("CCACHE_DIR", "/workspace/.ccache"),
    ("CCACHE_NOHASHDIR", "true"),
    ("CMAKE_BUILD_TYPE", "Release"),
    ("CMAKE_CXX_COMPILER_LAUNCHER", "ccache"),
    ("CMAKE_C_COMPILER_LAUNCHER", "ccache"),
    ("CXX", "g++"),
    ("MAX_JOBS", "8"),
    ("USE_CUDA", "0"),
    ("USE_DISTRIBUTED", "0"),
    ("USE_FBGEMM", "0"),
    ("USE_FLASH_ATTENTION", "0"),
    ("USE_GLOO", "0"),
    ("USE_KINETO", "0"),
    ("USE_KLEIDIAI", "0"),
    ("USE_MEM_EFF_ATTENTION", "0"),
    ("USE_MKLDNN", "0"),
    ("USE_NCCL", "0"),
    ("USE_NNPACK", "0"),
    ("USE_QNNPACK", "0"),
    ("USE_ROCM", "0"),
    ("USE_TENSORPIPE", "0"),
    ("USE_XNNPACK", "0"),
)
_CPU_INPLACE_BUILD_EXPORTS = " ".join(
    f"export {key}={shlex.quote(value)};"
    for key, value in _CPU_INPLACE_BUILD_ENVIRONMENT
)
_INPLACE_BUILD_PREFIX = (
    "set -o pipefail; "
    "unset PYTHONPATH; "
    "test -f setup.py || { echo 'setup.py missing' >&2; exit 2; }; "
)
_CPU_INPLACE_BUILD_COMMAND = (
    _INPLACE_BUILD_PREFIX
    + "mkdir -p -- third_party/nccl; "
    f"{_CPU_INPLACE_BUILD_EXPORTS} "
    "export TORCH_CUDA_ARCH_LIST=7.0; "
    "{python} setup.py build_ext --inplace"
)
_CUDA_INPLACE_BUILD_COMMAND = (
    _INPLACE_BUILD_PREFIX
    + "export BUILD_TEST=0; "
    "export TORCH_CUDA_ARCH_LIST=7.0; "
    "export MAX_JOBS=${MAX_JOBS:-8}; "
    "{python} setup.py build_ext --inplace"
)


@dataclass(frozen=True)
class RuntimeSourcePreparation:
    command: tuple[str, ...]
    cwd: str
    timeout_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.command, tuple) or not self.command:
            raise ContractError("command: expected non-empty tuple")
        for index, part in enumerate(self.command):
            require_str(part, f"command[{index}]")
        selected_cwd = require_str(self.cwd, "cwd")
        pure = PurePosixPath(selected_cwd)
        if pure.is_absolute() or ".." in pure.parts or str(pure) != selected_cwd:
            raise ContractError("cwd: expected canonical relative path")
        require_int(self.timeout_ms, "timeout_ms", minimum=1)


def build_runtime_source_preparation(
    profile: RuntimeProfile,
    python_executable: str,
    overlay_paths: tuple[str, ...],
) -> RuntimeSourcePreparation | None:
    if not isinstance(profile, RuntimeProfile):
        raise ContractError("profile: expected RuntimeProfile")
    selected_python = require_str(python_executable, "python_executable")
    if not isinstance(overlay_paths, tuple):
        raise ContractError("overlay_paths: expected tuple")
    normalized: list[str] = []
    for index, value in enumerate(overlay_paths):
        selected = require_str(value, f"overlay_paths[{index}]")
        pure = PurePosixPath(selected)
        if (
            pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or str(pure) != selected
        ):
            raise ContractError(
                f"overlay_paths[{index}]: expected canonical relative path"
            )
        normalized.append(selected)
    if len(set(normalized)) != len(normalized):
        raise ContractError("overlay_paths: duplicate path")

    mode = profile.source_loading_mode
    if mode == "none":
        return None
    if mode == "python_overlay":
        top_levels = {PurePosixPath(path).parts[0] for path in normalized}
        if len(top_levels) != 1:
            raise ContractError("source_overlay_package_ambiguous")
        command = (
            selected_python,
            "-I",
            "-c",
            _PYTHON_OVERLAY_PROGRAM,
            json.dumps(
                {
                    "package": next(iter(top_levels)),
                    "paths": normalized,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    elif mode == "inplace_build":
        command_template = (
            _CPU_INPLACE_BUILD_COMMAND
            if profile.runtime_tier == "cpu_source_snapshot_fuller"
            else _CUDA_INPLACE_BUILD_COMMAND
        )
        command = (
            "bash",
            "-lc",
            command_template.replace(
                "{python}",
                shlex.quote(selected_python),
            ),
        )
    else:
        raise ContractError("source_loading_mode_unsupported")
    return RuntimeSourcePreparation(
        command=command,
        cwd=".",
        timeout_ms=profile.timeout_ms,
    )


__all__ = [
    "RuntimeSourcePreparation",
    "build_runtime_source_preparation",
]
