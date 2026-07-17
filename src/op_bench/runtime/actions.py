from __future__ import annotations

import base64
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import re
import shlex
import threading
from typing import Protocol

from op_bench.runtime.contracts import (
    ActionObservation,
    ActionRequest,
    BudgetDelta,
    BudgetPolicy,
    CapabilityPolicy,
    ContentIdentity,
)
from op_bench.runtime.events import EventJournal
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_bool, require_int, require_str
from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    WorkspaceError,
    WorkspacePolicyError,
    WorkspaceStateError,
    _normalize_relative_path,
    _patch_paths_from_bytes,
    _path_in_scopes,
)


@dataclass(frozen=True)
class CommandExecution:
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
        for index, part in enumerate(self.command):
            require_str(part, f"command[{index}]")
        require_str(self.cwd, "cwd")
        require_int(self.exit_code, "exit_code")
        require_str(self.stdout, "stdout", min_length=0)
        require_str(self.stderr, "stderr", min_length=0)
        require_int(self.duration_ms, "duration_ms", minimum=0)
        require_bool(self.timed_out, "timed_out")


class CommandBackend(Protocol):
    def run(
        self,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> CommandExecution:
        ...


@dataclass(frozen=True)
class RegisteredTest:
    selector_id: str
    command: tuple[str, ...]
    cwd: str
    timeout_ms: int

    def __post_init__(self) -> None:
        require_str(self.selector_id, "selector_id")
        if not isinstance(self.command, tuple) or not self.command:
            raise ContractError("command: expected non-empty tuple")
        for index, part in enumerate(self.command):
            require_str(part, f"command[{index}]")
        _normalize_cwd(self.cwd)
        require_int(self.timeout_ms, "timeout_ms", minimum=1)


@dataclass(frozen=True)
class ActionUsage:
    actions: int
    tests: int
    commands: int
    output_bytes: int
    wall_clock_ms: int


@dataclass(frozen=True)
class ActionExchange:
    request: ActionRequest
    observation: ActionObservation

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "observation": self.observation.to_dict(),
        }


@dataclass(frozen=True)
class _Outcome:
    data: dict[str, object]
    mutation_state: str = "none"
    message: str = "ok"
    output_bytes: int = 0
    ok: bool = True
    error_code: str = "ok"
    tests: int = 0
    commands: int = 0


class _ActionFailure(Exception):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class CanonicalActionService:
    """Single policy and execution authority for Agent-facing v0.6 actions."""

    def __init__(
        self,
        *,
        session_id: str,
        workspace: AuthoritativeWorkspace,
        capability_policy: CapabilityPolicy,
        budget_policy: BudgetPolicy,
        command_backend: CommandBackend,
        test_registry: Mapping[str, RegisteredTest],
        clock_ms: Callable[[], int],
        event_journal: EventJournal | None = None,
    ) -> None:
        require_str(session_id, "session_id")
        if not isinstance(workspace, AuthoritativeWorkspace):
            raise ContractError("workspace: expected AuthoritativeWorkspace")
        if not isinstance(capability_policy, CapabilityPolicy):
            raise ContractError("capability_policy: expected CapabilityPolicy")
        if not isinstance(budget_policy, BudgetPolicy):
            raise ContractError("budget_policy: expected BudgetPolicy")
        if not callable(getattr(command_backend, "run", None)):
            raise ContractError("command_backend: expected run method")
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        if event_journal is not None and not isinstance(event_journal, EventJournal):
            raise ContractError("event_journal: expected EventJournal")
        if event_journal is not None and event_journal.session_id != session_id:
            raise ContractError("event_journal: session does not match service")
        registry: dict[str, RegisteredTest] = {}
        for selector_id, registered in test_registry.items():
            if not isinstance(selector_id, str) or not isinstance(registered, RegisteredTest):
                raise ContractError("test_registry: expected string to RegisteredTest mapping")
            if selector_id != registered.selector_id:
                raise ContractError("test_registry: selector key does not match entry")
            registry[selector_id] = registered
        command_prefixes: list[tuple[str, ...]] = []
        for index, encoded in enumerate(capability_policy.allowed_command_prefixes):
            try:
                parsed = tuple(shlex.split(encoded, posix=True))
            except ValueError as exc:
                raise ContractError(
                    f"allowed_command_prefixes[{index}]: invalid argv prefix"
                ) from exc
            if not parsed or any(not part for part in parsed):
                raise ContractError(
                    f"allowed_command_prefixes[{index}]: expected non-empty argv prefix"
                )
            command_prefixes.append(parsed)

        self.session_id = session_id
        self._workspace = workspace
        self.capability_policy = capability_policy
        self.budget_policy = budget_policy
        self._command_backend = command_backend
        self._test_registry = registry
        self._command_prefixes = tuple(
            sorted(command_prefixes, key=lambda prefix: (-len(prefix), prefix))
        )
        for index, scope in enumerate(capability_policy.writable_paths):
            directory = scope.endswith("/")
            candidate = scope[:-1] if directory else scope
            try:
                normalized = _normalize_relative_path(candidate)
            except WorkspacePolicyError as exc:
                raise ContractError(
                    f"writable_paths[{index}]: invalid workspace scope"
                ) from exc
            if scope != normalized + ("/" if directory else ""):
                raise ContractError(
                    f"writable_paths[{index}]: expected canonical workspace scope"
                )
        self._clock_ms = clock_ms
        self._event_journal = event_journal
        self._started_at_ms = self._now()
        self._lock = threading.RLock()
        self._running = True
        self._actions = 0
        self._tests = 0
        self._commands = 0
        self._output_bytes = 0
        self._last_sequence = 0
        self._sequence_owner: dict[int, str] = {}
        self._cache: dict[str, tuple[str, ActionObservation]] = {}
        self._audit: list[ActionExchange] = []
        self._finish_data: dict[str, object] | None = None
        self._finish_exemption_consumed = False

    @property
    def workspace_identity(self) -> ContentIdentity:
        return self._workspace.identity

    @property
    def event_journal(self) -> EventJournal | None:
        return self._event_journal

    @property
    def usage(self) -> ActionUsage:
        with self._lock:
            return ActionUsage(
                actions=self._actions,
                tests=self._tests,
                commands=self._commands,
                output_bytes=self._output_bytes,
                wall_clock_ms=max(0, self._now() - self._started_at_ms),
            )

    @property
    def audit_exchanges(self) -> tuple[ActionExchange, ...]:
        with self._lock:
            return tuple(self._audit)

    def execute(self, request: ActionRequest) -> ActionObservation:
        if not isinstance(request, ActionRequest):
            raise ContractError("request: expected ActionRequest")
        with self._lock:
            cached = self._cache.get(request.action_id)
            if cached is not None:
                request_hash, observation = cached
                if request_hash == request.content_hash:
                    return observation
                return self._uncached_failure(request, "conflict", "action_id was reused")

            started_at = self._now()
            if request.session_id != self.session_id:
                return self._uncached_failure(
                    request,
                    "session_not_running",
                    "request session does not match active session",
                )
            if self._event_journal is not None:
                try:
                    self._event_journal.record_action_requested(request)
                except Exception:  # noqa: BLE001 - fixed public persistence boundary.
                    self._running = False
                    return self._uncached_failure(
                        request,
                        "platform_error",
                        "action trajectory persistence failed",
                    )
            if request.deadline_ms <= started_at:
                return self._record_failure(
                    request,
                    started_at,
                    "timeout",
                    "request deadline has expired",
                    count_action=False,
                )
            if started_at - self._started_at_ms >= self.budget_policy.wall_clock_ms:
                return self._record_failure(
                    request,
                    started_at,
                    "budget_exhausted",
                    "wall-clock budget is exhausted",
                    count_action=False,
                )
            if request.action_name not in self.capability_policy.allowed_actions:
                return self._record_failure(
                    request,
                    started_at,
                    "capability_denied",
                    f"action {request.action_name!r} is not allowed",
                    count_action=False,
                )
            sequence_owner = self._sequence_owner.get(request.client_sequence)
            if sequence_owner is not None and sequence_owner != request.action_id:
                return self._record_failure(
                    request,
                    started_at,
                    "conflict",
                    "client_sequence is already owned by another action",
                    count_action=False,
                )
            if request.client_sequence <= self._last_sequence:
                return self._record_failure(
                    request,
                    started_at,
                    "conflict",
                    "client_sequence must increase for unique actions",
                    count_action=False,
                )
            if request.action_name == "session_finish":
                try:
                    _arguments(request, required=(), optional=())
                except _ActionFailure as exc:
                    if self._actions < self.budget_policy.max_actions:
                        self._last_sequence = request.client_sequence
                        self._sequence_owner[request.client_sequence] = request.action_id
                        return self._record_failure(
                            request,
                            started_at,
                            exc.error_code,
                            exc.message,
                            count_action=True,
                        )
                    return self._complete_uncached_failure(
                        request, exc.error_code, exc.message
                    )
            if not self._running and request.action_name != "session_finish":
                return self._record_failure(
                    request,
                    started_at,
                    "session_not_running",
                    "session no longer accepts actions",
                    count_action=False,
                )
            finish_exempt = (
                request.action_name == "session_finish"
                and not self._finish_exemption_consumed
            )
            if not finish_exempt and self._actions >= self.budget_policy.max_actions:
                return self._record_failure(
                    request,
                    started_at,
                    "budget_exhausted",
                    "action budget is exhausted",
                    count_action=False,
                )

            self._last_sequence = request.client_sequence
            self._sequence_owner[request.client_sequence] = request.action_id
            self._actions += 1
            if finish_exempt:
                self._finish_exemption_consumed = True
            before_tests = self._tests
            before_commands = self._commands
            before_output = self._output_bytes
            try:
                outcome = self._dispatch(request)
            except _ActionFailure as exc:
                outcome = _Outcome(
                    data={},
                    message=exc.message,
                    ok=False,
                    error_code=exc.error_code,
                )
            except WorkspaceStateError as exc:
                outcome = _Outcome(
                    data={},
                    message=str(exc),
                    ok=False,
                    error_code="session_not_running",
                )
            except WorkspacePolicyError as exc:
                outcome = _Outcome(
                    data={},
                    message=str(exc),
                    ok=False,
                    error_code="workspace_error",
                )
            except WorkspaceError as exc:
                outcome = _Outcome(
                    data={},
                    message=str(exc),
                    ok=False,
                    error_code="workspace_error",
                )
            except ContractError as exc:
                outcome = _Outcome(
                    data={},
                    message=str(exc),
                    ok=False,
                    error_code="invalid_request",
                )
            except Exception as exc:  # noqa: BLE001 - stable runtime observation boundary.
                outcome = _Outcome(
                    data={},
                    message="command backend failed",
                    ok=False,
                    error_code="runtime_error",
                )

            ended_at = self._now()
            outcome = self._apply_post_execution_limits(
                request,
                outcome,
                ended_at=ended_at,
            )
            self._tests += outcome.tests
            self._commands += outcome.commands
            self._output_bytes += outcome.output_bytes
            observation = ActionObservation(
                session_id=self.session_id,
                action_id=request.action_id,
                ok=outcome.ok,
                error_code=outcome.error_code,
                message=outcome.message,
                data=outcome.data,
                started_at_ms=started_at,
                ended_at_ms=ended_at,
                budget_delta=BudgetDelta(
                    wall_clock_ms=max(0, ended_at - started_at),
                    actions=1,
                    tests=self._tests - before_tests,
                    commands=self._commands - before_commands,
                    output_bytes=self._output_bytes - before_output,
                    provider_tokens=0,
                ),
                mutation_state=outcome.mutation_state,
            )
            return self._complete_observation(request, observation)

    def invalid_request_observation(
        self,
        *,
        action_id: str,
        message: str,
    ) -> ActionObservation:
        safe_action_id = action_id if isinstance(action_id, str) and action_id else "invalid"
        now = self._now()
        return ActionObservation(
            session_id=self.session_id,
            action_id=safe_action_id,
            ok=False,
            error_code="invalid_request",
            message=message,
            data={},
            started_at_ms=now,
            ended_at_ms=now,
            budget_delta=_zero_delta(),
            mutation_state="none",
        )

    def _dispatch(self, request: ActionRequest) -> _Outcome:
        handlers = {
            "workspace_list": self._workspace_list,
            "workspace_search": self._workspace_search,
            "workspace_read": self._workspace_read,
            "workspace_write": self._workspace_write,
            "workspace_apply_patch": self._workspace_apply_patch,
            "command_run": self._command_run,
            "test_run": self._test_run,
            "vcs_diff": self._vcs_diff,
            "session_finish": self._session_finish,
        }
        handler = handlers.get(request.action_name)
        if handler is None:
            raise _ActionFailure("unsupported_action", "action is not implemented")
        return handler(request)

    def _workspace_list(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(request, required=(), optional=("path", "recursive", "max_entries", "max_depth"))
        path = _string_argument(arguments, "path", default=".")
        recursive = _bool_argument(arguments, "recursive", default=False)
        max_entries = _bounded_int(arguments, "max_entries", default=200, minimum=1, maximum=1_000)
        max_depth = _bounded_int(arguments, "max_depth", default=4, minimum=1, maximum=32)
        if self._output_limit() < 2:
            raise _ActionFailure("budget_exhausted", "output budget is exhausted")
        try:
            entries = self._workspace.list_entries(
                path,
                recursive=recursive,
                max_entries=max_entries,
                max_depth=max_depth,
            )
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc
        encoded = [
            {
                "path": entry.path,
                "entry_type": entry.entry_type,
                "mode": entry.mode,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
        ]
        limited, output_bytes, truncated = _truncate_json_items(encoded, self._output_limit())
        return _Outcome(
            data={"entries": limited, "truncated": truncated},
            output_bytes=output_bytes,
        )

    def _workspace_search(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(
            request,
            required=("query",),
            optional=("path", "max_matches", "max_files"),
        )
        query = _string_argument(arguments, "query")
        path = _string_argument(arguments, "path", default=".")
        max_matches = _bounded_int(
            arguments, "max_matches", default=50, minimum=1, maximum=1_000
        )
        max_files = _bounded_int(
            arguments, "max_files", default=500, minimum=1, maximum=5_000
        )
        if self._output_limit() < 2:
            raise _ActionFailure("budget_exhausted", "output budget is exhausted")
        try:
            entries = self._workspace.list_entries(
                path,
                recursive=True,
                max_entries=max_files,
                max_depth=32,
            )
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc
        matches: list[dict[str, object]] = []
        match_limit_hit = False
        for entry in entries:
            if entry.entry_type != "file":
                continue
            try:
                content = self._workspace.read(
                    entry.path,
                    max_bytes=self.capability_policy.max_read_bytes,
                ).content.decode("utf-8", errors="strict")
            except UnicodeDecodeError as exc:
                raise _ActionFailure(
                    "workspace_error", f"path {entry.path!r}: search requires UTF-8 text"
                ) from exc
            except WorkspacePolicyError as exc:
                raise _ActionFailure("workspace_error", str(exc)) from exc
            for line_number, line in enumerate(content.splitlines(), start=1):
                column = line.find(query)
                if column < 0:
                    continue
                matches.append(
                    {
                        "path": entry.path,
                        "line": line_number,
                        "column": column + 1,
                        "text": line,
                    }
                )
                if len(matches) >= max_matches:
                    match_limit_hit = True
                    break
            if match_limit_hit:
                break
        limited, output_bytes, output_truncated = _truncate_json_items(
            matches, self._output_limit()
        )
        return _Outcome(
            data={
                "matches": limited,
                "truncated": match_limit_hit or output_truncated,
            },
            output_bytes=output_bytes,
        )

    def _workspace_read(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(request, required=("path",), optional=("max_bytes",))
        path = _string_argument(arguments, "path")
        requested = _bounded_int(
            arguments,
            "max_bytes",
            default=self.capability_policy.max_read_bytes,
            minimum=1,
            maximum=self.capability_policy.max_read_bytes,
        )
        output_limit = self._output_limit()
        if output_limit <= 0:
            raise _ActionFailure("budget_exhausted", "output budget is exhausted")
        try:
            read = self._workspace.read(path, max_bytes=requested)
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc
        returned = read.content[:output_limit]
        return _Outcome(
            data={
                "path": read.path,
                "mode": read.mode,
                "content_base64": base64.b64encode(returned).decode("ascii"),
                "truncated": len(returned) < len(read.content),
            },
            output_bytes=len(returned),
        )

    def _workspace_write(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(request, required=("path", "content"), optional=("mode",))
        path = _string_argument(arguments, "path")
        self._require_capability_write_path(path)
        content = _string_argument(arguments, "content", min_length=0)
        if len(content.encode("utf-8")) > self.capability_policy.max_write_bytes:
            raise _ActionFailure("path_denied", "content exceeds capability max_write_bytes")
        mode = None
        if "mode" in arguments:
            mode = _bounded_int(arguments, "mode", minimum=0, maximum=0o777)
        try:
            mutation = self._workspace.write(path, content, mode=mode)
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc
        return _Outcome(
            data={"path": mutation.path, "changed": mutation.changed},
            mutation_state="mutated" if mutation.changed else "unchanged",
        )

    def _workspace_apply_patch(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(request, required=("patch",), optional=())
        patch = _string_argument(arguments, "patch", min_length=0)
        if len(patch.encode("utf-8")) > self.capability_policy.max_write_bytes:
            raise _ActionFailure("path_denied", "patch exceeds capability max_write_bytes")
        try:
            paths = _patch_paths_from_bytes(patch.encode("utf-8"))
        except WorkspacePolicyError as exc:
            raise _ActionFailure("invalid_request", str(exc)) from exc
        for path in paths:
            self._require_capability_write_path(path)
        try:
            mutation = self._workspace.apply_patch(patch)
        except WorkspacePolicyError as exc:
            message = str(exc)
            error_code = "path_denied" if "scope" in message else "invalid_request"
            raise _ActionFailure(error_code, message) from exc
        return _Outcome(
            data={"paths": list(mutation.paths), "changed": mutation.changed},
            mutation_state="mutated" if mutation.changed else "unchanged",
        )

    def _command_run(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(
            request,
            required=("command",),
            optional=("cwd", "timeout_ms"),
        )
        command = _command_argument(arguments)
        matched_prefix = next(
            (
                prefix
                for prefix in self._command_prefixes
                if command[: len(prefix)] == prefix
            ),
            None,
        )
        if matched_prefix is None:
            raise _ActionFailure("capability_denied", "command argv prefix is not allowed")
        self._validate_command_shape(command, matched_prefix)
        if self._commands >= self.budget_policy.max_commands:
            raise _ActionFailure("budget_exhausted", "command budget is exhausted")
        cwd = _normalize_cwd(_string_argument(arguments, "cwd", default="."))
        self._validate_cwd(cwd)
        remaining_ms = self._execution_time_limit(request)
        requested_timeout = _unbounded_positive_int(
            arguments,
            "timeout_ms",
            default=remaining_ms,
        )
        timeout_ms = min(requested_timeout, remaining_ms)
        self._commands += 1
        execution = self._command_backend.run(command, cwd, timeout_ms)
        if not isinstance(execution, CommandExecution):
            raise _ActionFailure("runtime_error", "command backend returned invalid execution")
        if execution.command != command or execution.cwd != cwd:
            raise _ActionFailure(
                "runtime_error",
                "command backend returned mismatched execution metadata",
            )
        return self._execution_outcome(execution)

    def _test_run(self, request: ActionRequest) -> _Outcome:
        arguments = _arguments(request, required=("selector_id",), optional=())
        selector_id = _string_argument(arguments, "selector_id")
        if (
            selector_id not in self.capability_policy.registered_tests
            or selector_id not in self._test_registry
        ):
            raise _ActionFailure("selector_denied", "test selector is not registered")
        if self._tests >= self.budget_policy.max_tests:
            raise _ActionFailure("budget_exhausted", "test budget is exhausted")
        registered = self._test_registry[selector_id]
        cwd = _normalize_cwd(registered.cwd)
        self._validate_cwd(cwd)
        remaining_ms = self._execution_time_limit(request)
        timeout_ms = min(registered.timeout_ms, remaining_ms)
        self._tests += 1
        execution = self._command_backend.run(registered.command, cwd, timeout_ms)
        if not isinstance(execution, CommandExecution):
            raise _ActionFailure("runtime_error", "command backend returned invalid execution")
        if execution.command != registered.command or execution.cwd != cwd:
            raise _ActionFailure(
                "runtime_error",
                "command backend returned mismatched execution metadata",
            )
        outcome = self._execution_outcome(execution)
        return _Outcome(
            data={"selector_id": selector_id, **outcome.data},
            mutation_state=outcome.mutation_state,
            message=outcome.message,
            output_bytes=outcome.output_bytes,
            ok=outcome.ok,
            error_code=outcome.error_code,
            tests=outcome.tests,
            commands=outcome.commands,
        )

    def _vcs_diff(self, request: ActionRequest) -> _Outcome:
        _arguments(request, required=(), optional=())
        diff = self._workspace.diff()
        if len(diff.patch_bytes) > self._output_limit():
            raise _ActionFailure(
                "budget_exhausted", "canonical patch exceeds remaining output budget"
            )
        return self._patch_outcome(diff.patch_bytes, mutation_state="none")

    def _session_finish(self, request: ActionRequest) -> _Outcome:
        _arguments(request, required=(), optional=())
        if self._finish_data is not None:
            return _Outcome(data=dict(self._finish_data), mutation_state="frozen")
        self._running = False
        frozen = self._workspace.freeze()
        self._finish_data = _patch_metadata(
            patch=frozen.patch,
            size_bytes=len(frozen.patch_bytes),
            changed_paths=frozen.changed_paths,
            empty=frozen.empty,
        )
        return _Outcome(data=dict(self._finish_data), mutation_state="frozen")

    def _patch_outcome(self, patch_bytes: bytes, *, mutation_state: str) -> _Outcome:
        from op_bench.runtime.workspace import raw_patch_identity

        identity = raw_patch_identity(
            patch_bytes,
            identifier=f"{self.workspace_identity.identifier}:final.patch",
        )
        return _Outcome(
            data=_patch_data(
                patch_bytes,
                patch=identity,
                changed_paths=_patch_paths_from_bytes(patch_bytes),
                empty=not patch_bytes,
            ),
            mutation_state=mutation_state,
            output_bytes=len(patch_bytes),
        )

    def _execution_outcome(
        self,
        execution: CommandExecution,
        *,
        tests: int = 0,
        commands: int = 0,
    ) -> _Outcome:
        limit = self._output_limit()
        safe_stdout, stdout_redacted = _safe_backend_output(execution.stdout)
        safe_stderr, stderr_redacted = _safe_backend_output(execution.stderr)
        stdout, stdout_bytes, stdout_truncated = _truncate_text(safe_stdout, limit)
        stderr, stderr_bytes, stderr_truncated = _truncate_text(
            safe_stderr,
            max(0, limit - stdout_bytes),
        )
        timed_out = execution.timed_out
        return _Outcome(
            data={
                "exit_code": execution.exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "duration_ms": execution.duration_ms,
                "timed_out": timed_out,
                "truncated": (
                    stdout_redacted
                    or stderr_redacted
                    or stdout_truncated
                    or stderr_truncated
                ),
            },
            message="command timed out" if timed_out else "ok",
            output_bytes=stdout_bytes + stderr_bytes,
            ok=not timed_out,
            error_code="timeout" if timed_out else "ok",
            tests=tests,
            commands=commands,
        )

    def _require_capability_write_path(self, path: str) -> None:
        try:
            normalized = _normalize_relative_path(path)
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc
        if not _path_in_scopes(normalized, self.capability_policy.writable_paths):
            raise _ActionFailure("path_denied", f"path {normalized!r} is not writable")

    def _validate_cwd(self, cwd: str) -> None:
        try:
            self._workspace.list_entries(
                cwd,
                recursive=False,
                max_entries=10_000,
                max_depth=1,
            )
        except WorkspacePolicyError as exc:
            raise _ActionFailure("path_denied", str(exc)) from exc

    def _validate_command_shape(
        self,
        command: tuple[str, ...],
        prefix: tuple[str, ...],
    ) -> None:
        suffix = command[len(prefix) :]
        if prefix == ("python", "-m", "unittest"):
            simple_flags = {
                "-b",
                "--buffer",
                "-c",
                "--catch",
                "-f",
                "--failfast",
                "-q",
                "--quiet",
                "-v",
                "--verbose",
            }
            selector_pattern = re.compile(
                r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
            )
            for argument in suffix:
                if argument in simple_flags or selector_pattern.fullmatch(argument):
                    continue
                raise _ActionFailure(
                    "capability_denied",
                    "unittest command argument shape is not allowed",
                )
            return
        if prefix == ("git", "diff"):
            safe_flags = {
                "--binary",
                "--check",
                "--color=never",
                "--name-only",
                "--name-status",
                "--no-color",
                "--numstat",
                "--shortstat",
                "--stat",
                "-w",
                "--ignore-all-space",
            }
            paths_started = False
            for argument in suffix:
                if argument == "--" and not paths_started:
                    paths_started = True
                    continue
                if not paths_started and (
                    argument in safe_flags
                    or re.fullmatch(r"-U\d+", argument)
                    or re.fullmatch(r"--unified=\d+", argument)
                ):
                    continue
                if paths_started:
                    try:
                        _normalize_relative_path(argument)
                    except WorkspacePolicyError as exc:
                        raise _ActionFailure(
                            "capability_denied",
                            "git diff pathspec is not a canonical workspace path",
                        ) from exc
                    continue
                raise _ActionFailure(
                    "capability_denied",
                    "git diff command argument shape is not allowed",
                )
            return
        if suffix:
            raise _ActionFailure(
                "capability_denied",
                "configured command prefix only permits an exact argv match",
            )

    def _output_limit(self) -> int:
        remaining = max(0, self.budget_policy.max_output_bytes - self._output_bytes)
        return min(self.capability_policy.max_output_bytes, remaining)

    def _wall_remaining_ms(self, *, now: int | None = None) -> int:
        observed = self._now() if now is None else now
        return self.budget_policy.wall_clock_ms - (
            observed - self._started_at_ms
        )

    def _execution_time_limit(self, request: ActionRequest) -> int:
        now = self._now()
        request_remaining = request.deadline_ms - now
        if request_remaining <= 0:
            raise _ActionFailure("timeout", "request deadline has expired")
        wall_remaining = self._wall_remaining_ms(now=now)
        if wall_remaining <= 0:
            raise _ActionFailure("budget_exhausted", "wall-clock budget is exhausted")
        return min(request_remaining, wall_remaining)

    def _apply_post_execution_limits(
        self,
        request: ActionRequest,
        outcome: _Outcome,
        *,
        ended_at: int,
    ) -> _Outcome:
        if ended_at >= request.deadline_ms:
            return _Outcome(
                data=outcome.data,
                mutation_state=outcome.mutation_state,
                message="request deadline expired during execution",
                output_bytes=outcome.output_bytes,
                ok=False,
                error_code="timeout",
                tests=outcome.tests,
                commands=outcome.commands,
            )
        if self._wall_remaining_ms(now=ended_at) <= 0:
            return _Outcome(
                data=outcome.data,
                mutation_state=outcome.mutation_state,
                message="wall-clock budget exhausted during execution",
                output_bytes=outcome.output_bytes,
                ok=False,
                error_code="budget_exhausted",
                tests=outcome.tests,
                commands=outcome.commands,
            )
        return outcome

    def _record_failure(
        self,
        request: ActionRequest,
        started_at: int,
        error_code: str,
        message: str,
        *,
        count_action: bool,
    ) -> ActionObservation:
        if count_action:
            self._actions += 1
        observation = ActionObservation(
            session_id=self.session_id,
            action_id=request.action_id,
            ok=False,
            error_code=error_code,
            message=message,
            data={},
            started_at_ms=started_at,
            ended_at_ms=self._now(),
            budget_delta=BudgetDelta(
                wall_clock_ms=max(0, self._now() - started_at),
                actions=1 if count_action else 0,
                tests=0,
                commands=0,
                output_bytes=0,
                provider_tokens=0,
            ),
            mutation_state="none",
        )
        return self._complete_observation(request, observation)

    def _complete_observation(
        self,
        request: ActionRequest,
        observation: ActionObservation,
    ) -> ActionObservation:
        final_observation = observation
        if self._event_journal is not None:
            try:
                self._event_journal.record_action_observed(request, observation)
            except Exception:  # noqa: BLE001 - fixed public persistence boundary.
                self._running = False
                final_observation = ActionObservation(
                    session_id=observation.session_id,
                    action_id=observation.action_id,
                    ok=False,
                    error_code="platform_error",
                    message="action trajectory persistence failed",
                    data={},
                    started_at_ms=observation.started_at_ms,
                    ended_at_ms=observation.ended_at_ms,
                    budget_delta=observation.budget_delta,
                    mutation_state=observation.mutation_state,
                )
                try:
                    self._event_journal.record_action_observed(
                        request,
                        final_observation,
                    )
                except Exception:  # noqa: BLE001 - persistence remains unavailable.
                    pass
        self._cache[request.action_id] = (request.content_hash, final_observation)
        self._audit.append(
            ActionExchange(request=request, observation=final_observation)
        )
        return final_observation

    def _complete_uncached_failure(
        self,
        request: ActionRequest,
        error_code: str,
        message: str,
    ) -> ActionObservation:
        observation = self._uncached_failure(request, error_code, message)
        if self._event_journal is None:
            return observation
        try:
            self._event_journal.record_action_observed(request, observation)
        except Exception:  # noqa: BLE001 - fixed public persistence boundary.
            self._running = False
            return ActionObservation(
                session_id=observation.session_id,
                action_id=observation.action_id,
                ok=False,
                error_code="platform_error",
                message="action trajectory persistence failed",
                data={},
                started_at_ms=observation.started_at_ms,
                ended_at_ms=observation.ended_at_ms,
                budget_delta=observation.budget_delta,
                mutation_state=observation.mutation_state,
            )
        self._cache[request.action_id] = (request.content_hash, observation)
        self._audit.append(ActionExchange(request=request, observation=observation))
        return observation

    def _uncached_failure(
        self,
        request: ActionRequest,
        error_code: str,
        message: str,
    ) -> ActionObservation:
        now = self._now()
        return ActionObservation(
            session_id=self.session_id,
            action_id=request.action_id,
            ok=False,
            error_code=error_code,
            message=message,
            data={},
            started_at_ms=now,
            ended_at_ms=now,
            budget_delta=_zero_delta(),
            mutation_state="none",
        )

    def _now(self) -> int:
        value = self._clock_ms()
        return require_int(value, "clock_ms", minimum=0)


def _arguments(
    request: ActionRequest,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...],
) -> Mapping[str, object]:
    arguments = request.arguments
    keys = set(arguments)
    missing = set(required) - keys
    unknown = keys - set(required) - set(optional)
    if missing:
        raise _ActionFailure(
            "invalid_request", f"missing arguments: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise _ActionFailure(
            "invalid_request", f"unknown arguments: {', '.join(sorted(unknown))}"
        )
    return arguments


def _string_argument(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: str | None = None,
    min_length: int = 1,
) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or len(value) < min_length:
        raise _ActionFailure("invalid_request", f"{name}: expected string")
    return value


def _bool_argument(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise _ActionFailure("invalid_request", f"{name}: expected boolean")
    return value


def _bounded_int(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int | None = None,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _ActionFailure("invalid_request", f"{name}: expected integer")
    if value < minimum or value > maximum:
        raise _ActionFailure(
            "invalid_request", f"{name}: expected {minimum}..{maximum}"
        )
    return value


def _command_argument(arguments: Mapping[str, object]) -> tuple[str, ...]:
    value = arguments.get("command")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise _ActionFailure("invalid_request", "command: expected non-empty argv list")
    command: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, str) or not part or "\x00" in part:
            raise _ActionFailure(
                "invalid_request", f"command[{index}]: expected non-empty string"
            )
        command.append(part)
    return tuple(command)


def _unbounded_positive_int(
    arguments: Mapping[str, object],
    name: str,
    *,
    default: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _ActionFailure("invalid_request", f"{name}: expected positive integer")
    return value


def _normalize_cwd(value: str) -> str:
    if value == ".":
        return value
    try:
        return _normalize_relative_path(value)
    except WorkspacePolicyError as exc:
        raise _ActionFailure("path_denied", str(exc)) from exc


def _truncate_text(value: str, limit: int) -> tuple[str, int, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, len(encoded), False
    clipped = encoded[:limit]
    while clipped:
        try:
            decoded = clipped.decode("utf-8", errors="strict")
            return decoded, len(clipped), True
        except UnicodeDecodeError:
            clipped = clipped[:-1]
    return "", 0, bool(encoded)


def _safe_backend_output(value: str) -> tuple[str, bool]:
    try:
        assert_public_artifact_safe(value)
    except ContractError:
        return "[redacted sensitive runtime output]", True
    return value, False


def _truncate_json_items(
    items: list[dict[str, object]],
    limit: int,
) -> tuple[list[dict[str, object]], int, bool]:
    selected: list[dict[str, object]] = []
    for item in items:
        candidate = [*selected, item]
        encoded = json.dumps(
            candidate,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > limit:
            current = json.dumps(
                selected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            return selected, len(current), True
        selected = candidate
    encoded = json.dumps(
        selected,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return selected, len(encoded), False


def _patch_data(
    patch_bytes: bytes,
    *,
    patch: ContentIdentity,
    changed_paths: tuple[str, ...],
    empty: bool,
) -> dict[str, object]:
    return {
        **_patch_metadata(
            patch=patch,
            size_bytes=len(patch_bytes),
            changed_paths=changed_paths,
            empty=empty,
        ),
        "patch_base64": base64.b64encode(patch_bytes).decode("ascii"),
    }


def _patch_metadata(
    *,
    patch: ContentIdentity,
    size_bytes: int,
    changed_paths: tuple[str, ...],
    empty: bool,
) -> dict[str, object]:
    return {
        "patch": patch.to_dict(),
        "size_bytes": size_bytes,
        "changed_paths": list(changed_paths),
        "empty": empty,
    }


def _zero_delta() -> BudgetDelta:
    return BudgetDelta(
        wall_clock_ms=0,
        actions=0,
        tests=0,
        commands=0,
        output_bytes=0,
        provider_tokens=0,
    )


__all__ = [
    "ActionExchange",
    "ActionUsage",
    "CanonicalActionService",
    "CommandBackend",
    "CommandExecution",
    "RegisteredTest",
]
