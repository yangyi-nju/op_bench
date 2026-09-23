"""Explicit source preparation, separate from offline evaluation and agent runs.

Only a caller that requested source fetching should call ``provision_source``.
Existing source directories are inspected read-only. Newly fetched Git history
stays in the private source cache; materialize_source exports the selected tree
without that history for both solvers and graders.
"""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit

from op_bench.runtime.execution import ExecutionError, run_process
from op_bench.data.task import TaskSpec


class SourceProvisionError(ExecutionError):
    """Source preparation failed; ``result`` preserves bounded stage evidence."""

    def __init__(self, message: str, result: dict | None = None):
        super().__init__(message)
        self.result = result


class _GitCommands:
    def __init__(self, logs: Path, deadline: float, max_output_bytes: int, result: dict):
        self.logs = logs
        self.deadline = deadline
        self.max_output_bytes = max_output_bytes
        self.output_bytes = 0
        self.result = result
        self.environment = os.environ.copy()
        self.environment.update({"GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
                                 "GIT_NO_LAZY_FETCH": "1"})

    def run(self, cwd: Path, *args: str, stage: str, timeout_sec: float = 60) -> bytes:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise SourceProvisionError("Source preparation exceeded its total time limit")
        available = self.max_output_bytes - self.output_bytes
        if available <= 0:
            raise SourceProvisionError("Source preparation exceeded its total Git output limit")
        path = self.logs / f"{len(self.result['stages']):03d}.log"
        # Hook/config execution is unnecessary for materializing a source tree.
        # Only the HTTPS and local transports in this source contract are used.
        command = ["git", "-c", "core.hooksPath=" + os.devnull,
                   "-c", "protocol.allow=never", "-c", "protocol.https.allow=always",
                   "-c", "protocol.file.allow=always", *args]
        evidence = run_process(command, cwd=cwd, timeout_sec=min(timeout_sec, remaining),
                               log_path=path, environment=self.environment,
                               max_output_bytes=available)
        output = path.read_bytes()
        self.output_bytes += len(output)
        record = {"stage": stage, **evidence.to_dict()}
        self.result["stages"].append(record)
        if evidence.exit_code != 0 or evidence.timed_out or evidence.output_limited:
            record["output_tail"] = output[-4096:].decode("utf-8", errors="replace")
            raise SourceProvisionError(f"Source preparation failed during {stage}: "
                f"exit_code={evidence.exit_code}, timed_out={evidence.timed_out}, "
                f"output_limited={evidence.output_limited}. {record['output_tail'].strip()}")
        return output


def _repository_location(task: TaskSpec) -> str:
    location = task.source.repo_url
    if location is None:
        raise SourceProvisionError("Missing source directory requires source.repo_url")
    parsed = urlsplit(location)
    if parsed.scheme == "https":
        if not parsed.netloc:
            raise SourceProvisionError("source.repo_url must have a valid HTTPS host")
        return location
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise SourceProvisionError("file source URLs must refer to this host")
        location = unquote(parsed.path)
    elif parsed.scheme:
        raise SourceProvisionError("source.repo_url supports HTTPS or a local Git repository path")
    local = Path(location).expanduser()
    if not local.is_absolute():
        local = task.task_dir / local
    return str(local.resolve())


def _validate_checkout(source: Path, revision: str, git: _GitCommands) -> tuple[str, list[dict]]:
    root = Path(os.fsdecode(git.run(source, "rev-parse", "--show-toplevel", stage="inspect_repository")).strip()).resolve()
    if root != source.resolve():
        raise SourceProvisionError("source.path must be a complete Git checkout root")
    commit = git.run(source, "rev-parse", "--verify", "--end-of-options", revision + "^{commit}",
                     stage="resolve_revision").decode("ascii").strip()
    tree = git.run(source, "ls-tree", "-rz", commit, stage="inspect_submodules")
    submodules: list[dict] = []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        details, raw_path = entry.split(b"\t", 1)
        mode, _, object_id = details.split(b" ")
        if mode != b"160000":
            continue
        relative = Path(os.fsdecode(raw_path))
        checkout = source / relative
        expected = object_id.decode("ascii")
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise SourceProvisionError(f"Required submodule {relative} at {expected} is not prepared")
        current = git.run(checkout, "rev-parse", "HEAD", stage="inspect_submodule_revision").decode("ascii").strip()
        if current != expected:
            raise SourceProvisionError(f"Submodule {relative} is not checked out at required commit {expected}")
        _, children = _validate_checkout(checkout, expected, git)
        submodules.append({"path": relative.as_posix(), "revision": expected})
        submodules.extend({"path": (relative / child["path"]).as_posix(), "revision": child["revision"]}
                          for child in children)
    return commit, submodules


def publish_directory(checkout: Path, destination: Path) -> None:
    """Atomically publish without replacing even a concurrently created directory.

    POSIX rename alone can replace an existing empty directory. Use the native
    no-replace operation on supported hosts instead of deleting/reserving a user
    path and hoping it remains unused until publication.
    """
    if os.path.lexists(destination):
        raise SourceProvisionError("Source destination appeared during preparation; it was left unchanged")
    if os.name == "nt":
        os.rename(checkout, destination)  # Windows rename rejects an existing target.
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        status = rename(os.fsencode(checkout), os.fsencode(destination), 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        status = rename(-100, os.fsencode(checkout), -100, os.fsencode(destination), 1)  # RENAME_NOREPLACE
    else:
        raise SourceProvisionError("This host lacks atomic directory publication without replacement")
    if status != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise SourceProvisionError("Source destination appeared during preparation; it was left unchanged")
        raise OSError(error, os.strerror(error), str(destination))


def provision_source(task: TaskSpec, *, timeout_sec: float = 900,
                     max_output_bytes: int = 16 * 1024 * 1024) -> dict:
    """Prepare the declared source only when explicitly invoked by the caller.

    No evaluation/admission success is implied. Existing checkouts are never
    fetched, checked out, reset, cleaned, or modified. If their requested objects
    or submodules are unavailable, choose a new managed source.path and fetch it.
    """
    import math

    if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (float, int)) or not math.isfinite(timeout_sec) or timeout_sec <= 0:
        raise SourceProvisionError("timeout_sec must be a positive finite number")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes <= 0:
        raise SourceProvisionError("max_output_bytes must be a positive integer")
    started = time.monotonic()
    source = task.source.path.resolve()
    result = {"schema_version": 1, "source_path": str(source), "status": "failed",
              "revision": None, "submodules": [], "stages": [], "duration_sec": 0.0}
    try:
        if source.exists():
            if not source.is_dir():
                raise SourceProvisionError("Existing source.path is not a directory; it was left unchanged")
            if not (source / ".git").exists():
                if task.source.revision is not None:
                    raise SourceProvisionError("Existing source has no Git history for the requested revision. "
                                               "Use a new managed source.path; the existing directory was left unchanged")
                result.update(status="existing", kind="directory_snapshot")
            else:
                with tempfile.TemporaryDirectory(prefix="opbench-source-check-") as temporary:
                    git = _GitCommands(Path(temporary), started + timeout_sec, max_output_bytes, result)
                    try:
                        commit, submodules = _validate_checkout(source, task.source.revision or "HEAD", git)
                    except SourceProvisionError as exc:
                        raise SourceProvisionError(str(exc) + ". Use a new managed source.path to fetch a complete checkout; "
                                                   "the existing source was left unchanged") from exc
                result.update(status="existing", kind="git", revision=commit, submodules=submodules)
        else:
            if not task.source.repo_url or not task.source.revision:
                raise SourceProvisionError("Missing source requires both source.repo_url and source.revision")
            location = _repository_location(task)
            source.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".opbench-provision-", dir=source.parent) as temporary:
                staging = Path(temporary)
                checkout = staging / "checkout"
                git = _GitCommands(staging / "logs", started + timeout_sec, max_output_bytes, result)
                if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", task.source.revision):
                    # A pinned upstream commit needs its tree, not years of
                    # unrelated framework history. Submodules are still fetched
                    # and validated recursively at their declared gitlinks.
                    git.run(staging, "init", str(checkout), stage="initialize")
                    git.run(checkout, "remote", "add", "origin", location, stage="set_origin")
                    git.run(checkout, "fetch", "--depth=1", "--no-tags", "--", "origin",
                            task.source.revision, stage="fetch_revision", timeout_sec=timeout_sec)
                else:
                    # Existing recipes may use expressions such as HEAD~1.
                    # These need history and retain the general clone behavior.
                    git.run(staging, "clone", "--no-checkout", "--no-hardlinks", "--no-local", "--",
                            location, str(checkout), stage="clone", timeout_sec=timeout_sec)
                commit = git.run(checkout, "rev-parse", "--verify", "--end-of-options",
                                 task.source.revision + "^{commit}", stage="resolve_revision").decode("ascii").strip()
                git.run(checkout, "checkout", "--detach", commit, stage="checkout", timeout_sec=timeout_sec)
                git.run(checkout, "submodule", "update", "--init", "--recursive", "--checkout", "--depth=1",
                        stage="prepare_submodules", timeout_sec=timeout_sec)
                commit, submodules = _validate_checkout(checkout, commit, git)
                publish_directory(checkout, source)
                result.update(status="prepared", kind="git", revision=commit, submodules=submodules)
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        result["duration_sec"] = time.monotonic() - started
        if isinstance(exc, SourceProvisionError):
            exc.result = result
            raise
        raise SourceProvisionError(str(exc), result=result) from exc
    result["duration_sec"] = time.monotonic() - started
    return result
