from __future__ import annotations

import json
import shlex

from op_bench.task import TaskManifest


PYTHON_OVERLAY_SYNC_CODE = r"""
import hashlib
import importlib
import json
import os
import pathlib
import shutil
import site
import sys

cfg = json.loads(sys.argv[1])
workspace = pathlib.Path(cfg["workspace_dir"])
runtime_site_packages = pathlib.Path(cfg["runtime_site_packages"])
package = cfg["installed_package"]
overlay_paths = cfg["overlay_paths"]

runtime_site_packages.mkdir(parents=True, exist_ok=True)
os.chdir("/tmp")

module = importlib.import_module(package)
package_source = pathlib.Path(module.__file__).resolve().parent
package_destination = runtime_site_packages / package.split(".")[0]
if not package_destination.exists():
    shutil.copytree(package_source, package_destination, symlinks=True)

package_libs_source = package_source.parent / f"{package.split('.')[0]}.libs"
package_libs_destination = runtime_site_packages / package_libs_source.name
if package_libs_source.exists() and not package_libs_destination.exists():
    shutil.copytree(package_libs_source, package_libs_destination, symlinks=True)

pth_line = "import sys; sys.path.insert(0, {!r})\n".format(str(runtime_site_packages))
for site_packages in site.getsitepackages():
    site_packages_path = pathlib.Path(site_packages)
    if site_packages_path.exists():
        (site_packages_path / "op_bench_runtime_overlay.pth").write_text(pth_line, encoding="utf-8")

synced = []
for relative in overlay_paths:
    relative_path = pathlib.PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"invalid overlay path: {relative}")
    source = workspace / pathlib.Path(*relative_path.parts)
    destination = runtime_site_packages / pathlib.Path(*relative_path.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    synced.append({"workspace": str(source), "runtime": str(destination), "sha256": f"sha256:{digest}"})

print(json.dumps({"mode": "python_overlay", "package": package, "overlay_files": synced}, sort_keys=True))
""".strip()


# Default build command for inplace_build mode (PyTorch-style develop install).
# Tasks can override via source_loading.build_command in task.json.
#
# The build is streamed to /workspace/.op_bench_build.log (visible on the host
# via the mounted workspace) *and* to a limited tail on stdout. This way:
#   - operator can `tail -f /path/to/workspace/.op_bench_build.log` in real time
#   - if the outer timeout kills the command, the log file is still present
#     with everything up to the moment of kill (unlike the previous `| tail -100`
#     which buffered indefinitely and produced empty stdout on timeout)
#   - the last 200 lines are still printed to stdout at the end for evidence
DEFAULT_INPLACE_BUILD_COMMAND = (
    "set -o pipefail; "
    "cd {workspace_dir} && "
    "test -f setup.py || { echo 'ERROR: setup.py missing in workspace' >&2; exit 2; } && "
    # If MAX_JOBS isn't already set (e.g. old container image), default to nproc
    # so we saturate the host CPU. `command -v nproc` guards against BusyBox etc.
    "export MAX_JOBS=${MAX_JOBS:-$(command -v nproc >/dev/null && nproc || echo 8)}; "
    "echo \"MAX_JOBS=$MAX_JOBS\"; "
    # Tee to a log file inside the workspace so progress is visible on the host
    # (workspace is bind-mounted) and survives a timeout kill.
    # stdbuf -oL/-eL forces line-buffered output so the outer subprocess.PIPE
    # captures progress as it happens (previously `| tail -100` waited for
    # end-of-stream and produced empty output on timeout).
    "stdbuf -oL -eL python setup.py develop --no-deps 2>&1 | "
    "stdbuf -oL -eL tee .op_bench_build.log"
)


def build_source_loading_command(task: TaskManifest) -> list[str] | None:
    source_loading = task.source_loading
    if not source_loading:
        return None
    mode = source_loading.get("mode")
    if mode == "python_overlay":
        return _build_python_overlay_command(task, source_loading)
    if mode == "inplace_build":
        return _build_inplace_build_command(task, source_loading)
    return None


def _build_python_overlay_command(task: TaskManifest, source_loading: dict) -> list[str]:
    config = {
        "workspace_dir": task.environment_workspace_dir,
        "installed_package": str(source_loading["installed_package"]),
        "overlay_paths": task.source_loading_overlay_paths,
        "runtime_site_packages": str(source_loading["runtime_site_packages"]),
    }
    return [
        task.environment_python_executable,
        "-c",
        PYTHON_OVERLAY_SYNC_CODE,
        json.dumps(config, sort_keys=True),
    ]


def _build_inplace_build_command(task: TaskManifest, source_loading: dict) -> list[str]:
    """Build command for inplace_build mode.

    The agent's modifications to the source tree (including .cpp/.cu kernels) are
    re-compiled in place via setup.py develop. The build command runs from the
    workspace directory so the resulting binaries replace the installed ones.

    Tasks can override via source_loading.build_command (string with {workspace_dir}
    and {python} placeholders). Default does a no-deps incremental rebuild.
    """
    template = source_loading.get("build_command", DEFAULT_INPLACE_BUILD_COMMAND)
    # Use str.replace instead of .format() to avoid colliding with literal `{...}` in shell
    rendered = (
        template
        .replace("{workspace_dir}", task.environment_workspace_dir)
        .replace("{python}", shlex.quote(task.environment_python_executable))
    )
    return ["bash", "-lc", rendered]
