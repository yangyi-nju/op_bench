from __future__ import annotations

import json

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


def build_source_loading_command(task: TaskManifest) -> list[str] | None:
    source_loading = task.source_loading
    if not source_loading:
        return None
    mode = source_loading.get("mode")
    if mode != "python_overlay":
        return None
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
