"""Workspace environment and build-cache rules shared by solving and grading.

This module owns execution artifacts only; it does not evaluate patches or
interpret model attempts and result records.
"""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import time
import uuid

from op_bench.data.task import TaskSpec
from op_bench.runtime.execution import ExecutionError


def relocate_environment(task: TaskSpec, source: Path, destination: Path) -> TaskSpec:
    if task.environment.backend == "docker":
        return task
    # Local development has absolute workspace variables; clone sessions
    # must reference their own artifact copies, never the frozen build copy.
    return replace(task, environment=replace(task.environment, environment={
        key: value.replace(str(source), str(destination))
        for key, value in task.environment.environment.items()}))


def age_cached_workspace(workspace: Path) -> None:
    """Keep untouched dependencies ordered, with room for patch mtimes.

    Some build systems compare only whole seconds. A freshly cached output
    must precede a same-second patch or it can be incorrectly reused. Cap
    only recent/future timestamps before applying the patch; do not touch
    every source to a time newer than its compiled outputs.
    """
    cutoff = (int(time.time()) - 2) * 1_000_000_000
    for path in workspace.rglob("*"):
        info = path.lstat()
        if info.st_mtime_ns > cutoff:
            os.utime(path, ns=(info.st_atime_ns, cutoff), follow_symlinks=False)


def workspace_environment(task: TaskSpec, workspace: Path, artifact_contract: dict) -> TaskSpec:
    """Make ordinary user installs/caches/tmp files portable workspace data.

    The supplied artifact contract receives the reserved runtime paths.
    This is an explicit environment contract, not detection of every write.
    Absolute-path installations elsewhere must be moved into the image or
    rewritten as workspace outputs when a task is admitted.
    """
    runtime = workspace / (".opbench-build-" + uuid.uuid4().hex)
    runtime.mkdir()
    base = Path("/workspace") if task.environment.backend == "docker" else workspace
    paths = {"HOME": "home", "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp",
             "PYTHONUSERBASE": "userbase", "XDG_CACHE_HOME": "cache"}
    variables = {key: value.replace("{workspace}", str(base)).replace("{runtime}", str(base / runtime.name))
                 for key, value in task.environment.environment.items()}
    reserved = {runtime.name}
    planned = []
    for key, directory in paths.items():
        default = str(base / runtime.name / directory)
        value = variables.setdefault(key, default)
        candidate = Path(value)
        if not candidate.is_absolute() or ".." in candidate.parts or not candidate.is_relative_to(base):
            raise ExecutionError(f"workspace-only evaluation requires {key} to stay inside the workspace; preinstall external dependencies in the image")
        relative = candidate.relative_to(base)
        if ".git" in relative.parts or (relative.parts and relative.parts[0] == ".opbench"):
            raise ExecutionError(f"Runtime directory for {key} overlaps controller metadata: {relative}; use {{runtime}}")
        local = workspace / relative
        if not local.resolve().is_relative_to(workspace.resolve()):
            raise ExecutionError(f"Runtime directory for {key} escapes the workspace through a symlink")
        if not local.is_relative_to(runtime):
            if local.exists() or local.is_symlink():
                raise ExecutionError(f"Runtime directory for {key} overlaps existing source: {relative}; use a new directory or {{runtime}}")
            reserved.add(relative.as_posix())
        planned.append(local)
    for directory in planned:
        directory.mkdir(parents=True, exist_ok=True)
    # Keep only the outermost reserved directories, including when several
    # runtime variables deliberately share one new subtree.
    reserved = sorted(name for name in reserved if not any(
        name != other and Path(name).is_relative_to(other) for other in reserved))
    artifact_contract["runtime_directory"] = runtime.name
    artifact_contract["runtime_paths"] = reserved
    return replace(task, environment=replace(task.environment, environment=variables))
