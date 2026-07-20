from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import subprocess
import sys
import tempfile
from typing import Mapping, Sequence

from op_bench.runtime.adapters import AdapterContext
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.codex_adapter import (
    CodexAdapterResult,
    _TERMINAL_BY_STATUS,
    _classify_result,
    _minimal_environment,
)
from op_bench.runtime.mcp import (
    MCP_MAX_MESSAGE_BYTES,
    McpAdapterTrace,
    canonical_mcp_tools,
)
from op_bench.runtime.mcp_stdio import render_mcp_stdio_launcher
from op_bench.runtime.process_actions import ProcessActionExchange
from op_bench.runtime.process_group import ProcessGroupCleanupError, ProcessGroupResult
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_str


_ADAPTER_ID = "codex_mcp_canonical"
_LAUNCHER_FILENAME = "opbench_mcp_server.py"
_TRACE_FILENAME = "mcp_trace.json"
_MAX_CONTROLLER_EXECUTABLE_BYTES = 4 * 1024 * 1024


def _toml_literal(value: object) -> str:
    if not isinstance(value, (str, list, dict, bool, int)):
        raise ContractError("MCP config value is not a supported TOML literal")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _config_override(key: str, value: object) -> tuple[str, str]:
    return "-c", f"{key}={_toml_literal(value)}"


def _write_exclusive(path: Path, content: str, mode: int) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        encoded = content.encode("utf-8")
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_binding(path: Path) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractError("MCP controller directory is invalid")
        return metadata.st_dev, metadata.st_ino
    finally:
        os.close(descriptor)


def _regular_file_binding(path: Path) -> tuple[int, int, int, str]:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("MCP controller executable is invalid")
        if metadata.st_size > _MAX_CONTROLLER_EXECUTABLE_BYTES:
            raise ContractError("MCP controller executable exceeds size limit")
        encoded = bytearray()
        while len(encoded) <= _MAX_CONTROLLER_EXECUTABLE_BYTES:
            chunk = os.read(
                descriptor,
                min(
                    65_536,
                    _MAX_CONTROLLER_EXECUTABLE_BYTES + 1 - len(encoded),
                ),
            )
            if not chunk:
                break
            encoded.extend(chunk)
        if len(encoded) > _MAX_CONTROLLER_EXECUTABLE_BYTES:
            raise ContractError("MCP controller executable exceeds size limit")
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            hashlib.sha256(encoded).hexdigest(),
        )
    finally:
        os.close(descriptor)


def _bindings_are_unchanged(
    controller_root: Path,
    controller_binding: tuple[int, int],
    executable_bindings: Mapping[Path, tuple[int, int, int, str]],
) -> bool:
    try:
        if _directory_binding(controller_root) != controller_binding:
            return False
        return all(
            _regular_file_binding(path) == binding
            for path, binding in executable_bindings.items()
        )
    except (ContractError, OSError):
        return False


def _read_trace(path: Path) -> McpAdapterTrace:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ContractError("MCP trace is not a regular file")
    if metadata.st_size > MCP_MAX_MESSAGE_BYTES:
        raise ContractError("MCP trace exceeds size limit")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        encoded = os.read(descriptor, MCP_MAX_MESSAGE_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(encoded) > MCP_MAX_MESSAGE_BYTES:
        raise ContractError("MCP trace exceeds size limit")
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ContractError("MCP trace is invalid JSON") from None
    trace = McpAdapterTrace.from_dict(value)
    if encoded != (canonical_json(trace.to_dict()) + "\n").encode("utf-8"):
        raise ContractError("MCP trace is not canonical")
    return trace


def _synthetic_trace(
    *,
    model_id: str,
    codex_cli_version: str,
    server_terminal_status: str,
) -> McpAdapterTrace:
    return McpAdapterTrace(
        adapter_id=_ADAPTER_ID,
        model_id=model_id,
        codex_cli_version=codex_cli_version,
        negotiated_protocol_version=None,
        initialize_count=0,
        tools_list_count=0,
        tools_call_count=0,
        protocol_error_count=0,
        server_terminal_status=server_terminal_status,
    )


def _trace_matches_attempt(
    trace: McpAdapterTrace,
    *,
    model_id: str,
    codex_cli_version: str,
    observation_count: int,
) -> bool:
    return (
        trace.adapter_id == _ADAPTER_ID
        and trace.model_id == model_id
        and trace.codex_cli_version == codex_cli_version
        and trace.negotiated_protocol_version is not None
        and trace.initialize_count == 1
        and trace.tools_list_count >= 1
        and trace.tools_call_count == observation_count
        and trace.server_terminal_status in {"completed", "client_closed"}
    )


class CodexMcpCanonicalAdapter:
    """Run Codex with one invocation-local attempt-scoped stdio MCP server."""

    def __init__(
        self,
        command_runner: object,
        *,
        codex_binary: str = "codex",
        model_id: str,
        codex_cli_version: str,
        python_executable: str = sys.executable,
    ) -> None:
        if not callable(command_runner):
            raise ContractError("command_runner: expected callable")
        self.command_runner = command_runner
        self.codex_binary = require_str(codex_binary, "codex_binary")
        self.model_id = require_str(model_id, "model_id")
        self.codex_cli_version = require_str(
            codex_cli_version,
            "codex_cli_version",
        )
        self.python_executable = require_str(
            python_executable,
            "python_executable",
        )
        assert_public_artifact_safe(
            {
                "adapter_id": _ADAPTER_ID,
                "model_id": self.model_id,
                "codex_cli_version": self.codex_cli_version,
            }
        )

    def run(self, context: AdapterContext) -> CodexAdapterResult:
        if not isinstance(context, AdapterContext):
            raise ContractError("context: expected AdapterContext")
        public_task = context.launch_input.task_view.to_dict()
        assert_public_artifact_safe(public_task)
        timeout_ms = context.launch_input.task_view.budget_policy.wall_clock_ms
        prompt = _build_mcp_prompt(public_task)

        completed: subprocess.CompletedProcess[str] | None = None
        immediate_status: str | None = None
        process_terminal: str | None = None
        trace: McpAdapterTrace | None = None
        trace_invalid = False

        with tempfile.TemporaryDirectory(prefix="opbench-codex-mcp-adapter-") as temporary:
            private_root = Path(temporary) / "controller"
            agent_root = Path(temporary) / "agent"
            private_root.mkdir(mode=0o700)
            agent_root.mkdir(mode=0o500)
            transport_token = secrets.token_hex(32)
            exchange = ProcessActionExchange(
                action_client=context.action_client,
                session_id=context.session_id,
                exchange_root=private_root / "exchange",
                timeout_ms=timeout_ms,
                transport_token=transport_token,
            )
            exchange.start()
            bridge_token_fd: int | None = None
            executable_bindings: dict[Path, tuple[int, int, int, str]] = {}
            controller_binding: tuple[int, int] | None = None
            try:
                bridge_token_fd, bridge_token_writer = os.pipe()
                try:
                    token_bytes = transport_token.encode("ascii")
                    view = memoryview(token_bytes)
                    while view:
                        view = view[os.write(bridge_token_writer, view):]
                finally:
                    os.close(bridge_token_writer)
                controller_binding = _directory_binding(private_root)
                launcher_path = private_root / _LAUNCHER_FILENAME
                trace_path = private_root / _TRACE_FILENAME
                _write_exclusive(
                    launcher_path,
                    render_mcp_stdio_launcher(canonical_mcp_tools()),
                    0o700,
                )
                executable_bindings = {
                    launcher_path: _regular_file_binding(launcher_path),
                    exchange.client_path: _regular_file_binding(exchange.client_path),
                }
                server_arguments = [
                    str(launcher_path),
                    "--action-client",
                    str(exchange.client_path),
                    "--python-executable",
                    self.python_executable,
                    "--trace-path",
                    str(trace_path),
                    "--model-id",
                    self.model_id,
                    "--codex-cli-version",
                    self.codex_cli_version,
                    "--bridge-token-fd",
                    str(bridge_token_fd),
                ]
                argv = (
                    self.codex_binary,
                    "exec",
                    "--ephemeral",
                    "--ignore-user-config",
                    "--model",
                    self.model_id,
                    "--cd",
                    str(agent_root),
                    "--skip-git-repo-check",
                    "--sandbox",
                    "read-only",
                    *_config_override(
                        "mcp_servers.opbench.command",
                        self.python_executable,
                    ),
                    *_config_override("mcp_servers.opbench.args", server_arguments),
                    *_config_override("mcp_servers.opbench.env", {}),
                    prompt,
                )
                try:
                    candidate = self.command_runner(
                        argv,
                        cwd=agent_root,
                        env=_minimal_environment(os.environ),
                        timeout_ms=timeout_ms,
                        pass_fds=(bridge_token_fd,),
                    )
                except FileNotFoundError:
                    immediate_status = "executable_missing"
                except ProcessGroupCleanupError:
                    raise
                except subprocess.TimeoutExpired:
                    immediate_status = "timeout"
                    process_terminal = "terminated"
                except OSError:
                    immediate_status = "provider_failure"
                except Exception:  # noqa: BLE001 - stable Adapter boundary.
                    immediate_status = "provider_failure"
                else:
                    if isinstance(candidate, ProcessGroupResult):
                        process_terminal = candidate.terminal_status
                        if candidate.terminal_status == "completed":
                            completed = subprocess.CompletedProcess(
                                argv,
                                candidate.returncode,
                                stdout=candidate.stdout,
                                stderr=candidate.stderr,
                            )
                        else:
                            immediate_status = "timeout"
                    elif isinstance(candidate, subprocess.CompletedProcess):
                        completed = candidate
                    else:
                        immediate_status = "provider_failure"
            finally:
                if bridge_token_fd is not None:
                    os.close(bridge_token_fd)
                if controller_binding is None or not _bindings_are_unchanged(
                    private_root,
                    controller_binding,
                    executable_bindings,
                ):
                    trace_invalid = True
                exchange.close(cleanup=False)
                trace_path = private_root / _TRACE_FILENAME
                if trace_path.exists() or trace_path.is_symlink():
                    try:
                        trace = _read_trace(trace_path)
                    except (ContractError, OSError):
                        trace_invalid = True

        preliminary = _classify_result(
            immediate_status=immediate_status,
            completed=completed,
            server_failure=exchange.server_failure,
            finish_count=exchange.finish_count,
        )

        if trace is None:
            if immediate_status == "executable_missing":
                server_status = "not_started"
            elif process_terminal in {"terminated", "killed"}:
                server_status = process_terminal
            elif preliminary == "provider_failure":
                server_status = "start_failed"
            else:
                server_status = "protocol_failed"
                trace_invalid = True
            trace = _synthetic_trace(
                model_id=self.model_id,
                codex_cli_version=self.codex_cli_version,
                server_terminal_status=server_status,
            )
        elif process_terminal in {"terminated", "killed"}:
            trace = replace(trace, server_terminal_status=process_terminal)

        trace_matches = _trace_matches_attempt(
            trace,
            model_id=self.model_id,
            codex_cli_version=self.codex_cli_version,
            observation_count=exchange.observation_count,
        )
        if (
            preliminary not in {"executable_missing", "provider_failure", "timeout"}
            and (trace_invalid or not trace_matches)
        ):
            status = "malformed_action_exchange"
            trace = replace(trace, server_terminal_status="protocol_failed")
        else:
            status = preliminary

        return CodexAdapterResult(
            status=status,
            terminal_reason=_TERMINAL_BY_STATUS[status],
            exit_code=(None if completed is None else completed.returncode),
            observation_count=exchange.observation_count,
            finish_count=exchange.finish_count,
            adapter_trace=trace,
        )


def _build_mcp_prompt(public_task: Mapping[str, object]) -> str:
    tools = canonical_mcp_tools()
    tool_lines = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    return (
        "You are solving an OpBench task using only the attempt-scoped OpBench MCP "
        "tools listed below. The target repository is inaccessible through ordinary "
        "shell/filesystem access. Do not invoke Docker or SSH and do not search for "
        "another repository.\n"
        f"{tool_lines}\n"
        "Read before editing, make the smallest justified change, run a registered "
        "public test where possible, inspect vcs_diff, and call session_finish exactly "
        "once. Treat a tool result with ok=false as feedback you may correct.\n\n"
        "Public task view (canonical JSON):\n"
        f"{canonical_json(dict(public_task))}\n"
    )


__all__ = ["CodexMcpCanonicalAdapter"]
