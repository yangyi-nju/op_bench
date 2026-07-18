from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import signal
import stat
import subprocess
import threading
import time
from typing import Callable, Protocol

from op_bench.runtime.actions import CommandExecution
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import RuntimeProfile
from op_bench.runtime.resources import (
    AttemptResourceLedger,
    RuntimeCleanupEntry,
    RuntimeCleanupReport,
    RuntimeLease,
    RuntimeLeaseStore,
    RuntimeResourceHandle,
)
from op_bench.runtime.source_materialization import materialize_frozen_git_revision
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_int,
    require_str,
)


_BACKENDS = ("local", "docker", "remote_docker", "scripted")


class RuntimeBackendUnavailable(Exception):
    """An exact requested runtime target cannot serve the attempt."""

    def __init__(self, reason_code: str, private_message: str = "") -> None:
        self.reason_code = require_str(reason_code, "reason_code")
        self.private_message = require_str(
            private_message,
            "private_message",
            min_length=0,
        )
        super().__init__(self.private_message or self.reason_code)


@dataclass(frozen=True, repr=False)
class RuntimeTargetBinding:
    backend: str
    local_workspace_parent: Path
    host_alias: str | None = None
    remote_user: str | None = None
    ssh_port: int = 22
    identity_file: Path | None = None
    docker_binary: str = "docker"
    ssh_binary: str = "ssh"
    rsync_binary: str = "rsync"
    remote_workspace_root: str = "/tmp/opbench-runtime"
    remote_ccache_seed: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in _BACKENDS:
            raise ContractError(f"backend: unsupported value {self.backend!r}")
        if not isinstance(self.local_workspace_parent, Path):
            raise ContractError("local_workspace_parent: expected Path")
        if (
            self.local_workspace_parent.is_symlink()
            or not self.local_workspace_parent.is_dir()
        ):
            raise ContractError("local_workspace_parent: expected real directory")
        for value, path in (
            (self.host_alias, "host_alias"),
            (self.remote_user, "remote_user"),
        ):
            if value is not None:
                require_str(value, path)
        require_int(self.ssh_port, "ssh_port", minimum=1)
        if self.ssh_port > 65535:
            raise ContractError("ssh_port: must be <= 65535")
        if self.identity_file is not None:
            if not isinstance(self.identity_file, Path):
                raise ContractError("identity_file: expected Path")
            if self.identity_file.is_symlink() or not self.identity_file.is_file():
                raise ContractError("identity_file: expected real file")
        for value, path in (
            (self.docker_binary, "docker_binary"),
            (self.ssh_binary, "ssh_binary"),
            (self.rsync_binary, "rsync_binary"),
        ):
            require_str(value, path)
        remote_root = PurePosixPath(
            require_str(self.remote_workspace_root, "remote_workspace_root")
        )
        if not remote_root.is_absolute() or ".." in remote_root.parts:
            raise ContractError("remote_workspace_root: expected safe absolute path")
        if self.remote_ccache_seed is not None:
            seed = PurePosixPath(
                require_str(self.remote_ccache_seed, "remote_ccache_seed")
            )
            if not seed.is_absolute() or ".." in seed.parts:
                raise ContractError("remote_ccache_seed: expected safe absolute path")
            if self.backend != "remote_docker":
                raise ContractError("remote_ccache_seed: requires remote_docker")

    @property
    def public_binding_hash(self) -> str:
        return canonical_sha256(
            {
                "binding_type": "runtime_target_binding",
                "backend": self.backend,
                "local_workspace_parent": str(self.local_workspace_parent),
                "host_alias": self.host_alias,
                "remote_user": self.remote_user,
                "ssh_port": self.ssh_port,
                "identity_file": (
                    None if self.identity_file is None else str(self.identity_file)
                ),
                "docker_binary": self.docker_binary,
                "ssh_binary": self.ssh_binary,
                "rsync_binary": self.rsync_binary,
                "remote_workspace_root": self.remote_workspace_root,
                "remote_ccache_seed": self.remote_ccache_seed,
            }
        )


def load_runtime_target_binding(
    path: Path | str,
    *,
    local_workspace_parent: Path | None = None,
) -> RuntimeTargetBinding:
    """Load one exact private target without probing or discovering alternatives."""

    config_path = Path(path)
    if config_path.is_symlink():
        raise ContractError("target_config: symlink is denied")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(config_path, flags)
    except OSError as exc:
        raise ContractError("target_config: cannot open regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("target_config: expected regular file")
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as handle:
            encoded = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("target_config: invalid JSON") from exc
    finally:
        os.close(descriptor)
    if not isinstance(encoded, dict):
        raise ContractError("target_config: expected object")
    allowed = {
        "backend",
        "local_workspace_parent",
        "host_alias",
        "remote_user",
        "ssh_port",
        "identity_file",
        "docker_binary",
        "ssh_binary",
        "rsync_binary",
        "remote_workspace_root",
        "remote_ccache_seed",
    }
    if set(encoded) == {"hosts"}:
        return _load_single_legacy_target(
            encoded,
            config_path=config_path,
            local_workspace_parent=local_workspace_parent,
        )
    if set(encoded) - allowed:
        raise ContractError("target_config: unknown field")
    if "backend" not in encoded or "local_workspace_parent" not in encoded:
        raise ContractError("target_config: missing required field")

    def private_path(value: object, name: str) -> Path:
        raw = require_str(value, f"target_config.{name}")
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (config_path.resolve().parent / candidate).resolve()
        return candidate

    backend = require_str(encoded["backend"], "target_config.backend")
    binding = RuntimeTargetBinding(
        backend=backend,
        local_workspace_parent=private_path(
            encoded["local_workspace_parent"],
            "local_workspace_parent",
        ),
        host_alias=(
            None
            if encoded.get("host_alias") is None
            else require_str(encoded["host_alias"], "target_config.host_alias")
        ),
        remote_user=(
            None
            if encoded.get("remote_user") is None
            else require_str(encoded["remote_user"], "target_config.remote_user")
        ),
        ssh_port=require_int(
            encoded.get("ssh_port", 22),
            "target_config.ssh_port",
            minimum=1,
        ),
        identity_file=(
            None
            if encoded.get("identity_file") is None
            else private_path(encoded["identity_file"], "identity_file")
        ),
        docker_binary=require_str(
            encoded.get("docker_binary", "docker"),
            "target_config.docker_binary",
        ),
        ssh_binary=require_str(
            encoded.get("ssh_binary", "ssh"),
            "target_config.ssh_binary",
        ),
        rsync_binary=require_str(
            encoded.get("rsync_binary", "rsync"),
            "target_config.rsync_binary",
        ),
        remote_workspace_root=require_str(
            encoded.get("remote_workspace_root", "/tmp/opbench-runtime"),
            "target_config.remote_workspace_root",
        ),
        remote_ccache_seed=(
            None
            if encoded.get("remote_ccache_seed") is None
            else require_str(
                encoded["remote_ccache_seed"],
                "target_config.remote_ccache_seed",
            )
        ),
    )
    if backend == "remote_docker" and (
        binding.host_alias is None
        or binding.remote_user is None
        or binding.identity_file is None
    ):
        raise ContractError("target_config: incomplete remote target")
    return binding


def _load_single_legacy_target(
    encoded: dict[str, object],
    *,
    config_path: Path,
    local_workspace_parent: Path | None,
) -> RuntimeTargetBinding:
    hosts = encoded.get("hosts")
    if not isinstance(hosts, dict) or len(hosts) != 1:
        raise ContractError("target_config: expected one exact host binding")
    raw = next(iter(hosts.values()))
    if not isinstance(raw, dict):
        raise ContractError("target_config: invalid exact host binding")
    required = {
        "hostname",
        "user",
        "port",
        "identity_file",
        "remote_workspace_root",
    }
    allowed = required | {"remote_ccache_seed"}
    if set(raw) - allowed or not required.issubset(raw):
        raise ContractError("target_config: invalid exact host fields")
    if local_workspace_parent is None:
        raise ContractError("target_config: local workspace parent is required")
    identity_raw = require_str(raw["identity_file"], "target_config.identity_file")
    identity = Path(identity_raw).expanduser()
    if not identity.is_absolute():
        identity = (config_path.resolve().parent / identity).resolve()
    return RuntimeTargetBinding(
        backend="remote_docker",
        local_workspace_parent=local_workspace_parent,
        host_alias=require_str(raw["hostname"], "target_config.hostname"),
        remote_user=require_str(raw["user"], "target_config.user"),
        ssh_port=require_int(raw["port"], "target_config.port", minimum=1),
        identity_file=identity,
        remote_workspace_root=require_str(
            raw["remote_workspace_root"],
            "target_config.remote_workspace_root",
        ),
        remote_ccache_seed=(
            None
            if raw.get("remote_ccache_seed") is None
            else require_str(
                raw["remote_ccache_seed"],
                "target_config.remote_ccache_seed",
            )
        ),
    )


@dataclass(frozen=True)
class RuntimeAttemptContext:
    attempt_id: str
    retry_index: int
    runtime_profile_hash: str
    frozen_source_directory: Path
    frozen_source_revision: str
    resource_ledger: AttemptResourceLedger
    lease_store: RuntimeLeaseStore
    target_binding: RuntimeTargetBinding

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=r"attempt:v1:[0-9a-f]{64}")
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(
            self.runtime_profile_hash,
            "runtime_profile_hash",
            pattern=r"sha256:[0-9a-f]{64}",
        )
        if not isinstance(self.frozen_source_directory, Path):
            raise ContractError("frozen_source_directory: expected Path")
        if (
            self.frozen_source_directory.is_symlink()
            or not self.frozen_source_directory.is_dir()
        ):
            raise ContractError("frozen_source_directory: expected real directory")
        require_str(self.frozen_source_revision, "frozen_source_revision")
        if not isinstance(self.resource_ledger, AttemptResourceLedger):
            raise ContractError("resource_ledger: expected AttemptResourceLedger")
        if not isinstance(self.lease_store, RuntimeLeaseStore):
            raise ContractError("lease_store: expected RuntimeLeaseStore")
        if not isinstance(self.target_binding, RuntimeTargetBinding):
            raise ContractError("target_binding: expected RuntimeTargetBinding")
        for owner, path in (
            (self.resource_ledger, "resource_ledger"),
            (self.lease_store, "lease_store"),
        ):
            if (
                owner.attempt_id != self.attempt_id
                or owner.retry_index != self.retry_index
                or owner.runtime_profile_hash != self.runtime_profile_hash
            ):
                raise ContractError(f"{path}: identity mismatch")


@dataclass(frozen=True)
class RuntimeCommandResult:
    command: tuple[str, ...]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool

    def __post_init__(self) -> None:
        if not isinstance(self.command, tuple) or not self.command:
            raise ContractError("command: expected non-empty tuple")
        for index, item in enumerate(self.command):
            require_str(item, f"command[{index}]")
        require_str(self.cwd, "cwd")
        require_int(self.exit_code, "exit_code")
        require_str(self.stdout, "stdout", min_length=0)
        require_str(self.stderr, "stderr", min_length=0)
        require_int(self.duration_ms, "duration_ms", minimum=0)
        require_bool(self.timed_out, "timed_out")


@dataclass(frozen=True)
class RuntimeEvidence:
    attempt_id: str
    retry_index: int
    runtime_profile_hash: str
    resource_ledger_hash: str
    resource_states: tuple[tuple[str, str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "retry_index": self.retry_index,
            "runtime_profile_hash": self.runtime_profile_hash,
            "resource_ledger_hash": self.resource_ledger_hash,
            "resource_states": [
                {
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "status": status,
                }
                for resource_id, resource_type, status in self.resource_states
            ],
        }


@dataclass(frozen=True)
class RuntimeCleanupResult:
    report: RuntimeCleanupReport


class RuntimeBackend(Protocol):
    def prepare(
        self,
        profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
    ) -> RuntimeLease: ...

    def run(
        self,
        lease: RuntimeLease,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> RuntimeCommandResult: ...

    def collect(self, lease: RuntimeLease) -> RuntimeEvidence: ...

    def cleanup(self, lease: RuntimeLease) -> RuntimeCleanupResult: ...


@dataclass
class _BackendState:
    profile: RuntimeProfile
    context: RuntimeAttemptContext
    lease: RuntimeLease
    workspace: Path | None
    cleanup_result: RuntimeCleanupResult | None = None
    local_sync_hash: str | None = None


class LocalProcessBackend:
    def __init__(self) -> None:
        self._states: dict[tuple[str, int, str], _BackendState] = {}
        self._lock = threading.RLock()

    def prepare(
        self,
        profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
    ) -> RuntimeLease:
        _validate_prepare(profile, attempt_context, expected_backend="local")
        key = _context_key(attempt_context)
        with self._lock:
            if key in self._states:
                raise ContractError("attempt context is already prepared")
            ordinal = _next_ordinal(
                attempt_context.resource_ledger,
                "workspace",
            )
            declared = attempt_context.resource_ledger.declare(
                "workspace",
                ordinal,
            )
            workspace = _local_workspace_path(attempt_context, ordinal)
            try:
                if workspace.exists() or workspace.is_symlink():
                    raise ContractError("workspace path already exists")
                workspace.parent.mkdir(parents=True, exist_ok=True)
                snapshot = materialize_frozen_git_revision(
                    attempt_context.frozen_source_directory,
                    attempt_context.frozen_source_revision,
                    workspace,
                    include_submodules=(
                        profile.source_loading_mode == "inplace_build"
                    ),
                )
                handle = attempt_context.lease_store.put_exact(
                    declared.resource_id,
                    "workspace",
                    ordinal,
                    str(snapshot.workspace),
                )
                attempt_context.resource_ledger.created(
                    declared.resource_id,
                    handle.raw_handle_hash,
                )
            except Exception as exc:  # noqa: BLE001 - stable preparation boundary.
                if _last_transition(attempt_context, declared.resource_id) == "declared":
                    attempt_context.resource_ledger.create_failed(declared.resource_id)
                raise RuntimeBackendUnavailable(
                    "workspace_prepare_failed",
                    str(exc),
                ) from exc
            lease = RuntimeLease(
                attempt_id=attempt_context.attempt_id,
                retry_index=attempt_context.retry_index,
                runtime_profile_hash=attempt_context.runtime_profile_hash,
                handles=(handle,),
            )
            self._states[key] = _BackendState(
                profile=profile,
                context=attempt_context,
                lease=lease,
                workspace=workspace,
            )
            return lease

    def run(
        self,
        lease: RuntimeLease,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> RuntimeCommandResult:
        state = self._state_for(lease)
        argv = _validate_command(command)
        relative_cwd = _validate_cwd(cwd)
        timeout = require_int(timeout_ms, "timeout_ms", minimum=1)
        assert state.workspace is not None
        actual_cwd = state.workspace.joinpath(*relative_cwd.parts)
        _require_within_workspace(state.workspace, actual_cwd, "cwd")
        if not actual_cwd.is_dir():
            raise ContractError("cwd: expected existing workspace directory")
        context = state.context
        ordinal = _next_ordinal(context.resource_ledger, "process")
        declared = context.resource_ledger.declare("process", ordinal)
        started = time.monotonic_ns()
        try:
            process = subprocess.Popen(
                argv,
                cwd=actual_cwd,
                env=_runtime_subprocess_environment(actual_cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            context.resource_ledger.create_failed(declared.resource_id)
            raise RuntimeBackendUnavailable("process_launch_failed", str(exc)) from exc
        handle = context.lease_store.put_exact(
            declared.resource_id,
            "process",
            ordinal,
            f"pid:{process.pid}",
        )
        context.resource_ledger.created(declared.resource_id, handle.raw_handle_hash)
        timed_out = False
        try:
            stdout, stderr = process.communicate(timeout=timeout / 1000.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_exact_process_group(
                process,
                grace_ms=state.profile.cleanup_policy.grace_ms,
            )
            stdout, stderr = process.communicate()
        finally:
            if process.poll() is None:
                _terminate_exact_process_group(
                    process,
                    grace_ms=state.profile.cleanup_policy.grace_ms,
                )
            context.resource_ledger.released(declared.resource_id)
        duration_ms = max(0, (time.monotonic_ns() - started) // 1_000_000)
        return RuntimeCommandResult(
            command=argv,
            cwd=cwd,
            exit_code=(process.returncode if process.returncode is not None else -1),
            stdout=stdout,
            stderr=stderr,
            duration_ms=duration_ms,
            timed_out=timed_out,
        )

    def collect(self, lease: RuntimeLease) -> RuntimeEvidence:
        return _collect(self._state_for(lease))

    def cleanup(self, lease: RuntimeLease) -> RuntimeCleanupResult:
        state = self._state_for(lease)
        with self._lock:
            if state.cleanup_result is not None:
                return state.cleanup_result
            _verify_lease_handles(state)
            workspace_handle = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "workspace"
            )
            assert state.workspace is not None
            if workspace_handle.raw_handle != str(state.workspace):
                raise ContractError("lease workspace handle mismatch")
            try:
                if state.profile.cleanup_policy.remove_workspace:
                    shutil.rmtree(state.workspace)
                    _prune_empty_workspace_parents(
                        state.workspace,
                        state.context.target_binding.local_workspace_parent,
                    )
                state.context.resource_ledger.released(workspace_handle.resource_id)
            except OSError:
                state.context.resource_ledger.cleanup_failed(
                    workspace_handle.resource_id
                )
            result = RuntimeCleanupResult(_cleanup_report(state.context))
            state.cleanup_result = result
            return result

    def _state_for(self, lease: RuntimeLease) -> _BackendState:
        if not isinstance(lease, RuntimeLease):
            raise ContractError("lease: expected RuntimeLease")
        key = (lease.attempt_id, lease.retry_index, lease.runtime_profile_hash)
        with self._lock:
            state = self._states.get(key)
        if state is None or state.lease != lease:
            raise ContractError("lease: unknown or mismatched lease")
        return state


class ScriptedRuntimeBackend(LocalProcessBackend):
    def __init__(
        self,
        outcomes: tuple[RuntimeCommandResult, ...],
        *,
        cleanup_failure_code: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(outcomes, tuple) or not outcomes:
            raise ContractError("outcomes: expected non-empty tuple")
        if any(not isinstance(item, RuntimeCommandResult) for item in outcomes):
            raise ContractError("outcomes: expected RuntimeCommandResult values")
        self._outcomes = outcomes
        self._outcome_index = 0
        self._cleanup_failure_code = cleanup_failure_code

    @classmethod
    def success(
        cls,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = 0,
    ) -> "ScriptedRuntimeBackend":
        return cls(
            (
                RuntimeCommandResult(
                    command=("scripted",),
                    cwd=".",
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_ms=0,
                    timed_out=False,
                ),
            )
        )

    def prepare(
        self,
        profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
    ) -> RuntimeLease:
        if attempt_context.target_binding.backend != "local":
            raise ContractError("target binding backend mismatch")
        if profile.content_hash != attempt_context.runtime_profile_hash:
            raise ContractError("runtime profile hash mismatch")
        key = _context_key(attempt_context)
        with self._lock:
            declared = attempt_context.resource_ledger.declare("workspace", 1)
            handle = attempt_context.lease_store.put_exact(
                declared.resource_id,
                "workspace",
                1,
                f"scripted:{declared.resource_id}",
            )
            attempt_context.resource_ledger.created(
                declared.resource_id,
                handle.raw_handle_hash,
            )
            lease = RuntimeLease(
                attempt_id=attempt_context.attempt_id,
                retry_index=attempt_context.retry_index,
                runtime_profile_hash=attempt_context.runtime_profile_hash,
                handles=(handle,),
            )
            self._states[key] = _BackendState(
                profile=profile,
                context=attempt_context,
                lease=lease,
                workspace=None,
            )
            return lease

    def run(
        self,
        lease: RuntimeLease,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> RuntimeCommandResult:
        state = self._state_for(lease)
        argv = _validate_command(command)
        _validate_cwd(cwd)
        require_int(timeout_ms, "timeout_ms", minimum=1)
        ordinal = _next_ordinal(state.context.resource_ledger, "process")
        declared = state.context.resource_ledger.declare("process", ordinal)
        handle = state.context.lease_store.put_exact(
            declared.resource_id,
            "process",
            ordinal,
            f"scripted:{declared.resource_id}",
        )
        state.context.resource_ledger.created(
            declared.resource_id,
            handle.raw_handle_hash,
        )
        try:
            result = self._outcomes[
                min(self._outcome_index, len(self._outcomes) - 1)
            ]
            self._outcome_index += 1
            return RuntimeCommandResult(
                command=argv,
                cwd=cwd,
                exit_code=result.exit_code,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=result.duration_ms,
                timed_out=result.timed_out,
            )
        finally:
            state.context.resource_ledger.released(declared.resource_id)

    def cleanup(self, lease: RuntimeLease) -> RuntimeCleanupResult:
        state = self._state_for(lease)
        if state.cleanup_result is not None:
            return state.cleanup_result
        _verify_lease_handles(state)
        workspace = lease.handles[0]
        if self._cleanup_failure_code is None:
            state.context.resource_ledger.released(workspace.resource_id)
        else:
            state.context.resource_ledger.cleanup_failed(workspace.resource_id)
        result = RuntimeCleanupResult(_cleanup_report(state.context))
        state.cleanup_result = result
        return result


ArgvRunner = Callable[..., RuntimeCommandResult]


class DockerRuntimeBackend(LocalProcessBackend):
    def __init__(self, *, argv_runner: ArgvRunner | None = None) -> None:
        super().__init__()
        self._argv_runner = argv_runner or _default_argv_runner

    def prepare(
        self,
        profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
    ) -> RuntimeLease:
        _validate_prepare(profile, attempt_context, expected_backend="docker")
        key = _context_key(attempt_context)
        with self._lock:
            if key in self._states:
                raise ContractError("attempt context is already prepared")
            workspace_record, workspace_handle, workspace = _materialize_local_workspace(
                profile,
                attempt_context,
            )
            container_ordinal = _next_ordinal(
                attempt_context.resource_ledger,
                "container",
            )
            container_record = attempt_context.resource_ledger.declare(
                "container",
                container_ordinal,
            )
            container_name = _container_name(attempt_context)
            create = _container_create_command(
                profile,
                docker_binary=attempt_context.target_binding.docker_binary,
                container_name=container_name,
                container_resource_id=container_record.resource_id,
                workspace_source=str(workspace),
            )
            try:
                created = self._argv_runner(create, None, profile.timeout_ms)
            except RuntimeBackendUnavailable:
                attempt_context.resource_ledger.create_failed(
                    container_record.resource_id
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if created.exit_code != 0 or created.timed_out:
                attempt_context.resource_ledger.create_failed(
                    container_record.resource_id
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("container_create_failed")
            container_handle = attempt_context.lease_store.put_exact(
                container_record.resource_id,
                "container",
                container_ordinal,
                container_name,
            )
            attempt_context.resource_ledger.created(
                container_record.resource_id,
                container_handle.raw_handle_hash,
            )
            try:
                started = self._argv_runner(
                    (
                        attempt_context.target_binding.docker_binary,
                        "start",
                        container_name,
                    ),
                    None,
                    profile.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                _release_created_resource_after_prepare_failure(
                    attempt_context,
                    container_record.resource_id,
                    (
                        attempt_context.target_binding.docker_binary,
                        "rm",
                        "--force",
                        container_name,
                    ),
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if started.exit_code != 0 or started.timed_out:
                _release_created_resource_after_prepare_failure(
                    attempt_context,
                    container_record.resource_id,
                    (
                        attempt_context.target_binding.docker_binary,
                        "rm",
                        "--force",
                        container_name,
                    ),
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("container_start_failed")
            lease = RuntimeLease(
                attempt_id=attempt_context.attempt_id,
                retry_index=attempt_context.retry_index,
                runtime_profile_hash=attempt_context.runtime_profile_hash,
                handles=(workspace_handle, container_handle),
            )
            self._states[key] = _BackendState(
                profile=profile,
                context=attempt_context,
                lease=lease,
                workspace=workspace,
            )
            return lease

    def run(
        self,
        lease: RuntimeLease,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> RuntimeCommandResult:
        state = self._state_for(lease)
        argv = _validate_command(command)
        relative = _validate_cwd(cwd)
        timeout = require_int(timeout_ms, "timeout_ms", minimum=1)
        container = next(
            handle for handle in lease.handles if handle.resource_type == "container"
        )
        ordinal = _next_ordinal(state.context.resource_ledger, "process")
        declared = state.context.resource_ledger.declare("process", ordinal)
        process_handle = state.context.lease_store.put_exact(
            declared.resource_id,
            "process",
            ordinal,
            f"container-exec:{container.raw_handle}:{ordinal}",
        )
        state.context.resource_ledger.created(
            declared.resource_id,
            process_handle.raw_handle_hash,
        )
        target_cwd = _logical_container_cwd(
            state.profile.mount_policy.workspace_target,
            relative,
        )
        executed = _container_exec_command(
            docker_binary=state.context.target_binding.docker_binary,
            workspace_target=state.profile.mount_policy.workspace_target,
            workdir=target_cwd,
            container_name=container.raw_handle,
            command=argv,
            python_path=(
                state.profile.mount_policy.workspace_target
                if state.profile.source_loading_mode == "inplace_build"
                else None
            ),
        )
        try:
            raw = self._argv_runner(executed, None, timeout)
            return RuntimeCommandResult(
                command=argv,
                cwd=cwd,
                exit_code=raw.exit_code,
                stdout=raw.stdout,
                stderr=raw.stderr,
                duration_ms=raw.duration_ms,
                timed_out=raw.timed_out,
            )
        finally:
            state.context.resource_ledger.released(declared.resource_id)

    def cleanup(self, lease: RuntimeLease) -> RuntimeCleanupResult:
        state = self._state_for(lease)
        with self._lock:
            if state.cleanup_result is not None:
                return state.cleanup_result
            _verify_lease_handles(state)
            container = next(
                handle for handle in lease.handles if handle.resource_type == "container"
            )
            try:
                removed = self._argv_runner(
                    (
                        state.context.target_binding.docker_binary,
                        "rm",
                        "--force",
                        container.raw_handle,
                    ),
                    None,
                    state.profile.cleanup_policy.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                removed = None
            if removed is not None and removed.exit_code == 0 and not removed.timed_out:
                state.context.resource_ledger.released(container.resource_id)
            else:
                state.context.resource_ledger.cleanup_failed(container.resource_id)
            workspace = next(
                handle for handle in lease.handles if handle.resource_type == "workspace"
            )
            assert state.workspace is not None
            try:
                shutil.rmtree(state.workspace)
                _prune_empty_workspace_parents(
                    state.workspace,
                    state.context.target_binding.local_workspace_parent,
                )
                state.context.resource_ledger.released(workspace.resource_id)
            except OSError:
                state.context.resource_ledger.cleanup_failed(workspace.resource_id)
            result = RuntimeCleanupResult(_cleanup_report(state.context))
            state.cleanup_result = result
            return result


class RemoteDockerRuntimeBackend(LocalProcessBackend):
    def __init__(self, *, argv_runner: ArgvRunner | None = None) -> None:
        super().__init__()
        self._argv_runner = argv_runner or _default_argv_runner

    def prepare(
        self,
        profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
    ) -> RuntimeLease:
        _validate_prepare(profile, attempt_context, expected_backend="remote_docker")
        binding = attempt_context.target_binding
        if (
            binding.host_alias is None
            or binding.remote_user is None
            or binding.identity_file is None
        ):
            raise ContractError("explicit remote target binding is required")
        key = _context_key(attempt_context)
        with self._lock:
            if key in self._states:
                raise ContractError("attempt context is already prepared")
            workspace_record, workspace_handle, workspace = _materialize_local_workspace(
                profile,
                attempt_context,
            )
            try:
                local_sync_hash = _workspace_sync_fingerprint(workspace)
            except RuntimeBackendUnavailable:
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            remote_ordinal = _next_ordinal(
                attempt_context.resource_ledger,
                "remote_workspace",
            )
            remote_record = attempt_context.resource_ledger.declare(
                "remote_workspace", remote_ordinal
            )
            remote_path = _remote_workspace_path(attempt_context)
            target = _remote_target(binding)
            try:
                mkdir = self._argv_runner(
                    _ssh_command(binding, ("mkdir", "-p", "--", remote_path)),
                    None,
                    profile.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                attempt_context.resource_ledger.create_failed(
                    remote_record.resource_id
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if mkdir.exit_code != 0 or mkdir.timed_out:
                attempt_context.resource_ledger.create_failed(remote_record.resource_id)
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("remote_workspace_create_failed")
            remote_handle = attempt_context.lease_store.put_exact(
                remote_record.resource_id,
                "remote_workspace",
                remote_ordinal,
                f"{target}:{remote_path}",
            )
            attempt_context.resource_ledger.created(
                remote_record.resource_id,
                remote_handle.raw_handle_hash,
            )
            try:
                synced = self._argv_runner(
                    _rsync_command(binding, workspace, remote_path),
                    None,
                    profile.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if synced.exit_code != 0 or synced.timed_out:
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("remote_source_sync_failed")

            if (
                profile.source_loading_mode == "inplace_build"
                and binding.remote_ccache_seed is not None
            ):
                try:
                    seeded = self._argv_runner(
                        _ssh_command(
                            binding,
                            (
                                "cp",
                                "-a",
                                "--reflink=auto",
                                "--",
                                binding.remote_ccache_seed,
                                f"{remote_path}/.ccache",
                            ),
                        ),
                        None,
                        profile.timeout_ms,
                    )
                except RuntimeBackendUnavailable:
                    _release_remote_after_prepare_failure(
                        attempt_context,
                        binding,
                        remote_record.resource_id,
                        remote_path,
                        self._argv_runner,
                        profile.cleanup_policy.timeout_ms,
                    )
                    _release_workspace_after_prepare_failure(
                        attempt_context,
                        workspace_record.resource_id,
                        workspace,
                    )
                    raise
                if seeded.exit_code != 0 or seeded.timed_out:
                    _release_remote_after_prepare_failure(
                        attempt_context,
                        binding,
                        remote_record.resource_id,
                        remote_path,
                        self._argv_runner,
                        profile.cleanup_policy.timeout_ms,
                    )
                    _release_workspace_after_prepare_failure(
                        attempt_context,
                        workspace_record.resource_id,
                        workspace,
                    )
                    raise RuntimeBackendUnavailable(
                        "remote_ccache_seed_copy_failed"
                    )

            container_ordinal = _next_ordinal(
                attempt_context.resource_ledger,
                "container",
            )
            container_record = attempt_context.resource_ledger.declare(
                "container",
                container_ordinal,
            )
            container_name = _container_name(attempt_context)
            create_inner = _container_create_command(
                profile,
                docker_binary=binding.docker_binary,
                container_name=container_name,
                container_resource_id=container_record.resource_id,
                workspace_source=remote_path,
            )
            try:
                created = self._argv_runner(
                    _ssh_command(binding, create_inner),
                    None,
                    profile.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                attempt_context.resource_ledger.create_failed(
                    container_record.resource_id
                )
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if created.exit_code != 0 or created.timed_out:
                attempt_context.resource_ledger.create_failed(
                    container_record.resource_id
                )
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("container_create_failed")
            container_handle = attempt_context.lease_store.put_exact(
                container_record.resource_id,
                "container",
                container_ordinal,
                container_name,
            )
            attempt_context.resource_ledger.created(
                container_record.resource_id,
                container_handle.raw_handle_hash,
            )
            remove_container = _ssh_command(
                binding,
                (binding.docker_binary, "rm", "--force", container_name),
            )
            try:
                started = self._argv_runner(
                    _ssh_command(
                        binding,
                        (binding.docker_binary, "start", container_name),
                    ),
                    None,
                    profile.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                _release_created_resource_after_prepare_failure(
                    attempt_context,
                    container_record.resource_id,
                    remove_container,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise
            if started.exit_code != 0 or started.timed_out:
                _release_created_resource_after_prepare_failure(
                    attempt_context,
                    container_record.resource_id,
                    remove_container,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_remote_after_prepare_failure(
                    attempt_context,
                    binding,
                    remote_record.resource_id,
                    remote_path,
                    self._argv_runner,
                    profile.cleanup_policy.timeout_ms,
                )
                _release_workspace_after_prepare_failure(
                    attempt_context,
                    workspace_record.resource_id,
                    workspace,
                )
                raise RuntimeBackendUnavailable("container_start_failed")
            lease = RuntimeLease(
                attempt_id=attempt_context.attempt_id,
                retry_index=attempt_context.retry_index,
                runtime_profile_hash=attempt_context.runtime_profile_hash,
                handles=(workspace_handle, remote_handle, container_handle),
            )
            self._states[key] = _BackendState(
                profile=profile,
                context=attempt_context,
                lease=lease,
                workspace=workspace,
                local_sync_hash=local_sync_hash,
            )
            return lease

    def run(
        self,
        lease: RuntimeLease,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> RuntimeCommandResult:
        state = self._state_for(lease)
        argv = _validate_command(command)
        relative = _validate_cwd(cwd)
        timeout = require_int(timeout_ms, "timeout_ms", minimum=1)
        _sync_remote_workspace_if_changed(state, self._argv_runner, timeout)
        container = next(
            handle for handle in lease.handles if handle.resource_type == "container"
        )
        ordinal = _next_ordinal(state.context.resource_ledger, "process")
        declared = state.context.resource_ledger.declare("process", ordinal)
        process_handle = state.context.lease_store.put_exact(
            declared.resource_id,
            "process",
            ordinal,
            f"remote-container-exec:{container.raw_handle}:{ordinal}",
        )
        state.context.resource_ledger.created(
            declared.resource_id,
            process_handle.raw_handle_hash,
        )
        inner = _container_exec_command(
            docker_binary=state.context.target_binding.docker_binary,
            workspace_target=state.profile.mount_policy.workspace_target,
            workdir=_logical_container_cwd(
                state.profile.mount_policy.workspace_target,
                relative,
            ),
            container_name=container.raw_handle,
            command=argv,
            ccache_directory=(
                f"{state.profile.mount_policy.workspace_target}/.ccache"
                if state.profile.source_loading_mode == "inplace_build"
                and state.context.target_binding.remote_ccache_seed is not None
                else "/tmp/op_bench_runtime/ccache"
            ),
            python_path=(
                state.profile.mount_policy.workspace_target
                if state.profile.source_loading_mode == "inplace_build"
                else None
            ),
        )
        try:
            raw = self._argv_runner(
                _ssh_command(state.context.target_binding, inner),
                None,
                timeout,
            )
            return RuntimeCommandResult(
                command=argv,
                cwd=cwd,
                exit_code=raw.exit_code,
                stdout=raw.stdout,
                stderr=raw.stderr,
                duration_ms=raw.duration_ms,
                timed_out=raw.timed_out,
            )
        finally:
            state.context.resource_ledger.released(declared.resource_id)

    def cleanup(self, lease: RuntimeLease) -> RuntimeCleanupResult:
        state = self._state_for(lease)
        with self._lock:
            if state.cleanup_result is not None:
                return state.cleanup_result
            _verify_lease_handles(state)
            binding = state.context.target_binding
            container = next(
                handle for handle in lease.handles if handle.resource_type == "container"
            )
            try:
                removed = self._argv_runner(
                    _ssh_command(
                        binding,
                        (
                            binding.docker_binary,
                            "rm",
                            "--force",
                            container.raw_handle,
                        ),
                    ),
                    None,
                    state.profile.cleanup_policy.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                removed = None
            if removed is not None and removed.exit_code == 0 and not removed.timed_out:
                state.context.resource_ledger.released(container.resource_id)
            else:
                state.context.resource_ledger.cleanup_failed(container.resource_id)
            remote = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "remote_workspace"
            )
            remote_path = remote.raw_handle.split(":", 1)[1]
            try:
                workspace_removed = self._argv_runner(
                    _ssh_command(binding, ("rm", "-rf", "--", remote_path)),
                    None,
                    state.profile.cleanup_policy.timeout_ms,
                )
            except RuntimeBackendUnavailable:
                workspace_removed = None
            if (
                workspace_removed is not None
                and workspace_removed.exit_code == 0
                and not workspace_removed.timed_out
            ):
                state.context.resource_ledger.released(remote.resource_id)
            else:
                state.context.resource_ledger.cleanup_failed(remote.resource_id)
            workspace = next(
                handle for handle in lease.handles if handle.resource_type == "workspace"
            )
            assert state.workspace is not None
            try:
                shutil.rmtree(state.workspace)
                _prune_empty_workspace_parents(
                    state.workspace,
                    state.context.target_binding.local_workspace_parent,
                )
                state.context.resource_ledger.released(workspace.resource_id)
            except OSError:
                state.context.resource_ledger.cleanup_failed(workspace.resource_id)
            result = RuntimeCleanupResult(_cleanup_report(state.context))
            state.cleanup_result = result
            return result


class RuntimeCommandBackend:
    def __init__(self, backend: RuntimeBackend, lease: RuntimeLease) -> None:
        if not callable(getattr(backend, "run", None)):
            raise ContractError("backend: expected RuntimeBackend")
        if not isinstance(lease, RuntimeLease):
            raise ContractError("lease: expected RuntimeLease")
        self._backend = backend
        self._lease = lease

    def run(
        self,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> CommandExecution:
        result = self._backend.run(self._lease, command, cwd, timeout_ms)
        return CommandExecution(
            command=result.command,
            cwd=result.cwd,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=result.duration_ms,
            timed_out=result.timed_out,
        )


def _validate_prepare(
    profile: RuntimeProfile,
    context: RuntimeAttemptContext,
    *,
    expected_backend: str,
) -> None:
    if not isinstance(profile, RuntimeProfile):
        raise ContractError("profile: expected RuntimeProfile")
    if not isinstance(context, RuntimeAttemptContext):
        raise ContractError("attempt_context: expected RuntimeAttemptContext")
    if profile.content_hash != context.runtime_profile_hash:
        raise ContractError("runtime profile hash mismatch")
    if profile.backend != expected_backend:
        raise ContractError("runtime profile backend mismatch")
    if context.target_binding.backend != expected_backend:
        raise ContractError("target binding backend mismatch")


def _runtime_subprocess_environment(workspace: Path) -> dict[str, str]:
    allowed = (
        "PATH",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "TMPDIR",
        "TMP",
        "TEMP",
        "SYSTEMROOT",
        "WINDIR",
    )
    environment = {
        name: value
        for name in allowed
        if isinstance((value := os.environ.get(name)), str) and value
    }
    environment.update(
        {
            "HOME": str(workspace),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _context_key(context: RuntimeAttemptContext) -> tuple[str, int, str]:
    return (context.attempt_id, context.retry_index, context.runtime_profile_hash)


def _local_workspace_path(
    context: RuntimeAttemptContext,
    ordinal: int = 1,
) -> Path:
    digest = context.attempt_id.removeprefix("attempt:v1:")
    return (
        context.target_binding.local_workspace_parent
        / digest
        / f"retry-{context.retry_index:04d}"
        / ("workspace" if ordinal == 1 else f"workspace-{ordinal:04d}")
    )


def _materialize_local_workspace(
    profile: RuntimeProfile,
    context: RuntimeAttemptContext,
) -> tuple[object, RuntimeResourceHandle, Path]:
    ordinal = _next_ordinal(context.resource_ledger, "workspace")
    declared = context.resource_ledger.declare("workspace", ordinal)
    workspace = _local_workspace_path(context, ordinal)
    try:
        if workspace.exists() or workspace.is_symlink():
            raise ContractError("workspace path already exists")
        workspace.parent.mkdir(parents=True, exist_ok=True)
        snapshot = materialize_frozen_git_revision(
            context.frozen_source_directory,
            context.frozen_source_revision,
            workspace,
            include_submodules=(profile.source_loading_mode == "inplace_build"),
        )
        handle = context.lease_store.put_exact(
            declared.resource_id,
            "workspace",
            ordinal,
            str(snapshot.workspace),
        )
        context.resource_ledger.created(
            declared.resource_id,
            handle.raw_handle_hash,
        )
        return declared, handle, snapshot.workspace
    except Exception as exc:  # noqa: BLE001 - stable materialization boundary.
        try:
            _prune_empty_workspace_parents(
                workspace,
                context.target_binding.local_workspace_parent,
            )
        except OSError:
            pass
        if _last_transition(context, declared.resource_id) == "declared":
            context.resource_ledger.create_failed(declared.resource_id)
        raise RuntimeBackendUnavailable("workspace_prepare_failed", str(exc)) from exc


def _release_workspace_after_prepare_failure(
    context: RuntimeAttemptContext,
    resource_id: str,
    workspace: Path,
) -> None:
    try:
        shutil.rmtree(workspace)
        _prune_empty_workspace_parents(
            workspace,
            context.target_binding.local_workspace_parent,
        )
        context.resource_ledger.released(resource_id)
    except OSError:
        context.resource_ledger.cleanup_failed(resource_id)


def _release_remote_after_prepare_failure(
    context: RuntimeAttemptContext,
    binding: RuntimeTargetBinding,
    resource_id: str,
    remote_path: str,
    argv_runner: ArgvRunner,
    timeout_ms: int,
) -> None:
    try:
        removed = argv_runner(
            _ssh_command(binding, ("rm", "-rf", "--", remote_path)),
            None,
            timeout_ms,
        )
    except RuntimeBackendUnavailable:
        removed = None
    if removed is not None and removed.exit_code == 0 and not removed.timed_out:
        context.resource_ledger.released(resource_id)
    else:
        context.resource_ledger.cleanup_failed(resource_id)


def _release_created_resource_after_prepare_failure(
    context: RuntimeAttemptContext,
    resource_id: str,
    remove_command: tuple[str, ...],
    argv_runner: ArgvRunner,
    timeout_ms: int,
) -> None:
    try:
        removed = argv_runner(remove_command, None, timeout_ms)
    except RuntimeBackendUnavailable:
        removed = None
    if removed is not None and removed.exit_code == 0 and not removed.timed_out:
        context.resource_ledger.released(resource_id)
    else:
        context.resource_ledger.cleanup_failed(resource_id)


def _workspace_sync_fingerprint(workspace: Path) -> str:
    diff = subprocess.run(
        (
            "git",
            "-C",
            str(workspace),
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeBackendUnavailable("workspace_sync_state_failed")
    untracked = subprocess.run(
        (
            "git",
            "-C",
            str(workspace),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if untracked.returncode != 0:
        raise RuntimeBackendUnavailable("workspace_sync_state_failed")
    digest = hashlib.sha256()
    digest.update(diff.stdout)
    for raw_path in sorted(item for item in untracked.stdout.split(b"\0") if item):
        try:
            relative = raw_path.decode("utf-8")
            candidate = workspace.joinpath(*PurePosixPath(relative).parts)
            _require_within_workspace(workspace, candidate, "untracked path")
            digest.update(raw_path)
            digest.update(candidate.read_bytes())
        except (OSError, UnicodeDecodeError, ContractError) as exc:
            raise RuntimeBackendUnavailable("workspace_sync_state_failed") from exc
    return digest.hexdigest()


def _sync_remote_workspace_if_changed(
    state: _BackendState,
    argv_runner: ArgvRunner,
    timeout_ms: int,
) -> None:
    if state.workspace is None:
        raise ContractError("remote Runtime is missing local authoritative workspace")
    observed = _workspace_sync_fingerprint(state.workspace)
    if observed == state.local_sync_hash:
        return
    remote = next(
        handle
        for handle in state.lease.handles
        if handle.resource_type == "remote_workspace"
    )
    remote_path = remote.raw_handle.split(":", 1)[1]
    synced = argv_runner(
        _rsync_command(
            state.context.target_binding,
            state.workspace,
            remote_path,
        ),
        None,
        timeout_ms,
    )
    if synced.exit_code != 0 or synced.timed_out:
        raise RuntimeBackendUnavailable("remote_source_sync_failed")
    state.local_sync_hash = observed


def _container_name(context: RuntimeAttemptContext) -> str:
    digest = context.attempt_id.removeprefix("attempt:v1:")
    return f"opbench-{digest[:20]}-r{context.retry_index:04d}"


def _container_create_command(
    profile: RuntimeProfile,
    *,
    docker_binary: str,
    container_name: str,
    container_resource_id: str,
    workspace_source: str,
) -> tuple[str, ...]:
    policy = profile.resource_policy
    command: list[str] = [
        docker_binary,
        "create",
        "--name",
        container_name,
        "--label",
        f"opbench.resource-id={container_resource_id}",
        "--network",
        "none" if profile.network_policy == "denied" else "bridge",
    ]
    if policy.cpu_millis is not None:
        cpu = policy.cpu_millis / 1000
        command.extend(("--cpus", f"{cpu:g}"))
    if policy.memory_bytes is not None:
        command.extend(("--memory", str(policy.memory_bytes)))
    if policy.pids_limit is not None:
        command.extend(("--pids-limit", str(policy.pids_limit)))
    if policy.gpu_count:
        command.extend(("--gpus", str(policy.gpu_count)))
    if profile.mount_policy.root_filesystem == "read_only_container":
        command.extend(
            (
                "--read-only",
                "--tmpfs",
                "/tmp:rw,exec,nosuid,size=4294967296",
            )
        )
    command.extend(
        (
            "--volume",
            f"{workspace_source}:{profile.mount_policy.workspace_target}:rw",
            profile.image.identifier,
            "sleep",
            "infinity",
        )
    )
    return tuple(command)


def _container_exec_command(
    *,
    docker_binary: str,
    workspace_target: str,
    workdir: str,
    container_name: str,
    command: tuple[str, ...],
    ccache_directory: str = "/tmp/op_bench_runtime/ccache",
    python_path: str | None = None,
) -> tuple[str, ...]:
    return (
        docker_binary,
        "exec",
        "--env",
        "GIT_CONFIG_COUNT=1",
        "--env",
        "GIT_CONFIG_KEY_0=safe.directory",
        "--env",
        f"GIT_CONFIG_VALUE_0={workspace_target}",
        "--env",
        "XDG_CACHE_HOME=/tmp/op_bench_runtime/xdg-cache",
        "--env",
        "TRITON_CACHE_DIR=/tmp/op_bench_runtime/triton-cache",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/op_bench_runtime/torchinductor-cache",
        "--env",
        f"CCACHE_DIR={ccache_directory}",
        "--env",
        "CCACHE_MAXSIZE=2G",
        "--env",
        "CUDA_CACHE_PATH=/tmp/op_bench_runtime/cuda-cache",
        "--env",
        "TORCH_EXTENSIONS_DIR=/tmp/op_bench_runtime/torch-extensions",
        *(
            ()
            if python_path is None
            else ("--env", f"PYTHONPATH={python_path}")
        ),
        "--workdir",
        workdir,
        container_name,
        *command,
    )


def _logical_container_cwd(target: str, relative: PurePosixPath) -> str:
    base = PurePosixPath(target)
    if str(relative) in {"", "."}:
        return str(base)
    return str(base.joinpath(*relative.parts))


def _remote_workspace_path(context: RuntimeAttemptContext) -> str:
    digest = context.attempt_id.removeprefix("attempt:v1:")
    root = context.target_binding.remote_workspace_root.rstrip("/")
    return f"{root}/{digest}/retry-{context.retry_index:04d}/workspace"


def _remote_target(binding: RuntimeTargetBinding) -> str:
    if binding.remote_user is None or binding.host_alias is None:
        raise ContractError("explicit remote target binding is required")
    return f"{binding.remote_user}@{binding.host_alias}"


def _ssh_command(
    binding: RuntimeTargetBinding,
    remote_command: tuple[str, ...],
) -> tuple[str, ...]:
    if binding.identity_file is None:
        raise ContractError("explicit remote target binding is required")
    return (
        binding.ssh_binary,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=30",
        "-p",
        str(binding.ssh_port),
        "-i",
        str(binding.identity_file),
        _remote_target(binding),
        shlex.join(remote_command),
    )


def _rsync_command(
    binding: RuntimeTargetBinding,
    source: Path,
    remote_path: str,
) -> tuple[str, ...]:
    if binding.identity_file is None:
        raise ContractError("explicit remote target binding is required")
    transport = shlex.join(
        (
            binding.ssh_binary,
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            "-p",
            str(binding.ssh_port),
            "-i",
            str(binding.identity_file),
        )
    )
    return (
        binding.rsync_binary,
        "-a",
        "--delete",
        "--exclude=.ccache/",
        "-e",
        transport,
        str(source) + "/",
        f"{_remote_target(binding)}:{remote_path}/",
    )


def _default_argv_runner(
    command: tuple[str, ...],
    cwd: Path | None,
    timeout_ms: int,
) -> RuntimeCommandResult:
    started = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_ms / 1000.0,
            check=False,
        )
        timed_out = False
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = -1
        stdout = "" if exc.stdout is None else str(exc.stdout)
        stderr = "" if exc.stderr is None else str(exc.stderr)
    except OSError as exc:
        raise RuntimeBackendUnavailable("runtime_command_unavailable", str(exc)) from exc
    return RuntimeCommandResult(
        command=command,
        cwd="." if cwd is None else str(cwd),
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_ms=max(0, (time.monotonic_ns() - started) // 1_000_000),
        timed_out=timed_out,
    )


def _validate_command(command: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(command, tuple) or not command:
        raise ContractError("command: expected non-empty tuple")
    for index, item in enumerate(command):
        require_str(item, f"command[{index}]")
    return command


def _validate_cwd(cwd: str) -> PurePosixPath:
    value = require_str(cwd, "cwd")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ContractError("cwd: expected workspace-relative path")
    return path


def _require_within_workspace(workspace: Path, candidate: Path, path: str) -> None:
    root = workspace.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError(f"{path}: escapes workspace") from exc


def _next_ordinal(ledger: AttemptResourceLedger, resource_type: str) -> int:
    return 1 + sum(
        record.resource_type == resource_type and record.transition == "declared"
        for record in ledger.records
    )


def _last_transition(context: RuntimeAttemptContext, resource_id: str) -> str | None:
    history = [
        record.transition
        for record in context.resource_ledger.records
        if record.resource_id == resource_id
    ]
    return history[-1] if history else None


def _terminate_exact_process_group(
    process: subprocess.Popen[str],
    *,
    grace_ms: int,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=max(0.001, grace_ms / 1000.0))
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _verify_lease_handles(state: _BackendState) -> None:
    for handle in state.lease.handles:
        try:
            owned = state.context.lease_store.get_exact(handle.resource_id)
        except ContractError as exc:
            raise ContractError(
                "lease handle is not owned by exact retry store"
            ) from exc
        if owned != handle:
            raise ContractError("lease handle is not owned by exact retry store")


def _prune_empty_workspace_parents(workspace: Path, boundary: Path) -> None:
    current = workspace.parent
    root = boundary.resolve(strict=True)
    while True:
        resolved = current.resolve(strict=True)
        if resolved == root:
            return
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ContractError("workspace cleanup escapes configured parent") from exc
        try:
            current.rmdir()
        except OSError as exc:
            if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                return
            raise
        current = current.parent


def _collect(state: _BackendState) -> RuntimeEvidence:
    records = state.context.resource_ledger.verify()
    final: dict[str, tuple[str, str]] = {}
    for record in records:
        final[record.resource_id] = (record.resource_type, record.transition)
    states = tuple(
        (resource_id, resource_type, transition)
        for resource_id, (resource_type, transition) in sorted(final.items())
    )
    return RuntimeEvidence(
        attempt_id=state.context.attempt_id,
        retry_index=state.context.retry_index,
        runtime_profile_hash=state.context.runtime_profile_hash,
        resource_ledger_hash=canonical_sha256(
            [record.to_dict() for record in records]
        ),
        resource_states=states,
    )


def _cleanup_report(context: RuntimeAttemptContext) -> RuntimeCleanupReport:
    final = {}
    for record in context.resource_ledger.records:
        final[record.resource_id] = record
    entries = tuple(
        RuntimeCleanupEntry(
            resource_id=resource_id,
            resource_type=record.resource_type,
            status=record.transition,
            error_code=(
                "workspace_remove_failed"
                if record.transition == "cleanup_failed"
                and record.resource_type in {"workspace", "remote_workspace"}
                else (
                    "container_remove_failed"
                    if record.transition == "cleanup_failed"
                    and record.resource_type == "container"
                    else (
                        "process_remove_failed"
                        if record.transition == "cleanup_failed"
                        else None
                    )
                )
            ),
        )
        for resource_id, record in sorted(final.items())
    )
    return RuntimeCleanupReport(
        attempt_id=context.attempt_id,
        retry_index=context.retry_index,
        runtime_profile_hash=context.runtime_profile_hash,
        entries=entries,
        all_released=all(
            entry.status in {"released", "create_failed"} for entry in entries
        ),
    )


__all__ = [
    "DockerRuntimeBackend",
    "LocalProcessBackend",
    "RuntimeAttemptContext",
    "RuntimeBackend",
    "RuntimeBackendUnavailable",
    "RuntimeCleanupResult",
    "RuntimeCommandBackend",
    "RuntimeCommandResult",
    "RuntimeEvidence",
    "RuntimeLease",
    "RuntimeTargetBinding",
    "RemoteDockerRuntimeBackend",
    "ScriptedRuntimeBackend",
]
