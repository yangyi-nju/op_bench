from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import TextIO

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ACTION_NAMES, ActionObservation
from op_bench.runtime.mcp import (
    MCP_MAX_MESSAGE_BYTES,
    MCP_PROTOCOL_VERSIONS,
    MCP_SERVER_NAME,
    MCP_SERVER_VERSION,
    McpToolDefinition,
)
from op_bench.runtime.validation import ContractError, require_int


ToolInvoker = Callable[[str, Mapping[str, object]], Mapping[str, object]]
TraceWriter = Callable[[Mapping[str, object]], None]


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _response(request_id: object, *, result: object) -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: object, code: int, message: str) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _request_id(value: object) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise _RpcError(-32600, "invalid JSON-RPC request id")


def _request(value: object) -> tuple[object, str, object, bool]:
    if not isinstance(value, Mapping):
        raise _RpcError(-32600, "JSON-RPC request must be an object")
    allowed = {"jsonrpc", "id", "method", "params"}
    if set(value) - allowed:
        raise _RpcError(-32600, "JSON-RPC request has unknown fields")
    if value.get("jsonrpc") != "2.0":
        raise _RpcError(-32600, "JSON-RPC version must be 2.0")
    method = value.get("method")
    if not isinstance(method, str) or not method:
        raise _RpcError(-32600, "JSON-RPC method must be a non-empty string")
    notification = "id" not in value
    request_id = None if notification else _request_id(value.get("id"))
    return request_id, method, value.get("params", {}), notification


def _exact_object(value: object, fields: set[str], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise _RpcError(-32602, f"{label} must be an object")
    if set(value) != fields:
        raise _RpcError(-32602, f"{label} has invalid fields")
    return dict(value)


def _params_without_metadata(value: object, label: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise _RpcError(-32602, f"{label} must be an object")
    selected = dict(value)
    if "_meta" in selected and not isinstance(selected.pop("_meta"), Mapping):
        raise _RpcError(-32602, f"{label} metadata must be an object")
    return selected


def _initialize_result(params: object) -> tuple[str, dict[str, object]]:
    if not isinstance(params, Mapping):
        raise _RpcError(-32602, "initialize params must be an object")
    requested = params.get("protocolVersion")
    if not isinstance(requested, str) or not requested:
        raise _RpcError(-32602, "initialize protocolVersion is required")
    negotiated = (
        requested
        if requested in MCP_PROTOCOL_VERSIONS
        else MCP_PROTOCOL_VERSIONS[-1]
    )
    return negotiated, {
        "protocolVersion": negotiated,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
    }


def _tool_result(observation: Mapping[str, object]) -> dict[str, object]:
    payload = dict(observation)
    try:
        canonical_observation = ActionObservation.from_dict(payload).to_dict()
    except ContractError:
        raise _RpcError(-32603, "action bridge returned an invalid observation")
    return {
        "content": [
            {"type": "text", "text": canonical_json(canonical_observation)}
        ],
        "structuredContent": canonical_observation,
        "isError": not canonical_observation["ok"],
    }


def _write_response(output_stream: TextIO, payload: Mapping[str, object]) -> None:
    output_stream.write(canonical_json(dict(payload)) + "\n")
    output_stream.flush()


def serve_mcp_stdio(
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
    *,
    invoke_tool: ToolInvoker,
    write_trace: TraceWriter,
    max_message_bytes: int = MCP_MAX_MESSAGE_BYTES,
) -> str:
    """Serve the approved MCP subset without owning any Action semantics."""

    if not callable(invoke_tool):
        raise ContractError("invoke_tool: expected callable")
    if not callable(write_trace):
        raise ContractError("write_trace: expected callable")
    require_int(max_message_bytes, "max_message_bytes", minimum=1)

    negotiated: str | None = None
    initialize_count = 0
    tools_list_count = 0
    tools_call_count = 0
    protocol_error_count = 0
    terminal = "client_closed"

    for line in input_stream:
        request_id: object = None
        if len(line.encode("utf-8")) > max_message_bytes:
            protocol_error_count += 1
            terminal = "protocol_failed"
            _write_response(
                output_stream,
                _error(None, -32600, "JSON-RPC message exceeds size limit"),
            )
            break
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            protocol_error_count += 1
            terminal = "protocol_failed"
            _write_response(output_stream, _error(None, -32700, "JSON parse error"))
            break

        try:
            request_id, method, params, notification = _request(value)
            if method == "notifications/initialized":
                if not notification:
                    raise _RpcError(-32600, "initialized must be a notification")
                continue
            if notification:
                raise _RpcError(-32601, "notification method not found")
            if method == "initialize":
                negotiated, result = _initialize_result(params)
                initialize_count += 1
            elif method == "ping":
                if _params_without_metadata(params, "ping params"):
                    raise _RpcError(-32602, "ping params must be empty")
                result = {}
            elif method == "tools/list":
                if negotiated is None:
                    raise _RpcError(-32602, "MCP server is not initialized")
                list_params = _params_without_metadata(params, "tools/list params")
                if set(list_params) - {"cursor"}:
                    raise _RpcError(-32602, "tools/list params has invalid fields")
                if "cursor" in list_params and (
                    not isinstance(list_params["cursor"], str)
                    or not list_params["cursor"]
                ):
                    raise _RpcError(-32602, "tools/list cursor must be a string")
                from op_bench.runtime.mcp import canonical_mcp_tools

                result = {"tools": [tool.to_dict() for tool in canonical_mcp_tools()]}
                tools_list_count += 1
            elif method == "tools/call":
                if negotiated is None:
                    raise _RpcError(-32602, "MCP server is not initialized")
                call = _exact_object(
                    _params_without_metadata(params, "tools/call params"),
                    {"name", "arguments"},
                    "tools/call params",
                )
                name = call["name"]
                arguments = call["arguments"]
                if not isinstance(name, str) or name not in ACTION_NAMES:
                    raise _RpcError(-32602, "unknown MCP tool")
                if not isinstance(arguments, Mapping):
                    raise _RpcError(-32602, "tool arguments must be an object")
                try:
                    observed = invoke_tool(name, dict(arguments))
                except _RpcError:
                    raise
                except Exception as exc:  # noqa: BLE001 - fixed protocol boundary.
                    del exc
                    error_stream.write("MCP action bridge failed\n")
                    error_stream.flush()
                    raise _RpcError(-32603, "MCP action bridge failed") from None
                if not isinstance(observed, Mapping):
                    raise _RpcError(-32603, "action bridge returned an invalid observation")
                result = _tool_result(observed)
                tools_call_count += 1
            else:
                raise _RpcError(-32601, "method not found")
        except _RpcError as exc:
            protocol_error_count += 1
            if request_id is not None or not (
                isinstance(value, Mapping) and "id" not in value
            ):
                _write_response(output_stream, _error(request_id, exc.code, exc.message))
            if exc.code in {-32600, -32603}:
                terminal = "protocol_failed"
                break
            continue

        _write_response(output_stream, _response(request_id, result=result))

    trace = {
        "negotiated_protocol_version": negotiated,
        "initialize_count": initialize_count,
        "tools_list_count": tools_list_count,
        "tools_call_count": tools_call_count,
        "protocol_error_count": protocol_error_count,
        "server_terminal_status": terminal,
    }
    write_trace(trace)
    return terminal


_LAUNCHER_TEMPLATE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import selectors
import subprocess
import sys
import uuid


MAX_MESSAGE_BYTES = 1048576
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18")
TOOLS = json.loads(__TOOLS_JSON__)
TOOL_NAMES = tuple(tool["name"] for tool in TOOLS)
OBSERVATION_FIELDS = {
    "contract_type", "schema_version", "session_id", "action_id", "ok",
    "error_code", "message", "data", "started_at_ms", "ended_at_ms",
    "budget_delta", "mutation_state",
}
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | NOFOLLOW


class RpcError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code
        self.message = message


def canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def response(request_id, result):
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def error(request_id, code, message):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def write_response(payload):
    sys.stdout.buffer.write(canonical(payload).encode("utf-8") + b"\n")
    sys.stdout.buffer.flush()


def request_id(value):
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise RpcError(-32600, "invalid JSON-RPC request id")


def parse_request(value):
    if not isinstance(value, dict):
        raise RpcError(-32600, "JSON-RPC request must be an object")
    if set(value) - {"jsonrpc", "id", "method", "params"}:
        raise RpcError(-32600, "JSON-RPC request has unknown fields")
    if value.get("jsonrpc") != "2.0":
        raise RpcError(-32600, "JSON-RPC version must be 2.0")
    method = value.get("method")
    if not isinstance(method, str) or not method:
        raise RpcError(-32600, "JSON-RPC method must be a non-empty string")
    notification = "id" not in value
    selected_id = None if notification else request_id(value.get("id"))
    return selected_id, method, value.get("params", {}), notification


def params_without_metadata(value, label):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RpcError(-32602, label + " must be an object")
    selected = dict(value)
    if "_meta" in selected and not isinstance(selected.pop("_meta"), dict):
        raise RpcError(-32602, label + " metadata must be an object")
    return selected


def initialize_result(params):
    if not isinstance(params, dict):
        raise RpcError(-32602, "initialize params must be an object")
    requested = params.get("protocolVersion")
    if not isinstance(requested, str) or not requested:
        raise RpcError(-32602, "initialize protocolVersion is required")
    negotiated = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1]
    return negotiated, {
        "protocolVersion": negotiated,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "opbench", "version": "v0.6"},
    }


def read_message():
    encoded = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
    if not encoded:
        return None, None
    if len(encoded) > MAX_MESSAGE_BYTES:
        while encoded and not encoded.endswith(b"\n"):
            encoded = sys.stdin.buffer.readline(MAX_MESSAGE_BYTES + 2)
        return None, error(None, -32600, "JSON-RPC message exceeds size limit")
    try:
        return json.loads(encoded.decode("utf-8")), None
    except (UnicodeDecodeError, ValueError):
        return None, error(None, -32700, "JSON parse error")


def bounded_process_output(command, environment):
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    selector = selectors.DefaultSelector()
    streams = {"stdout": process.stdout, "stderr": process.stderr}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        for name, stream in streams.items():
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            for key, _ in selector.select():
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target = buffers[key.data]
                remaining = MAX_MESSAGE_BYTES - len(target)
                if remaining > 0:
                    target.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    process.kill()
                    process.wait()
                    raise RpcError(-32603, "MCP action bridge output exceeds size limit")
        return process.wait(), bytes(buffers["stdout"]), bytes(buffers["stderr"])
    finally:
        selector.close()
        for stream in streams.values():
            stream.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def invoke_action(args, name, arguments):
    action_environment = dict(os.environ)
    action_environment["OPBENCH_ACTION_TRANSPORT_TOKEN"] = args.bridge_token
    returncode, stdout, stderr = bounded_process_output(
        [
            args.python_executable,
            args.action_client,
            name,
            "--arguments",
            canonical(arguments),
        ],
        action_environment,
    )
    if returncode != 0:
        raise RpcError(-32603, "MCP action bridge failed")
    try:
        decoded = stdout.decode("utf-8")
        observed = json.loads(decoded)
    except (UnicodeDecodeError, ValueError):
        raise RpcError(-32603, "MCP action bridge returned invalid JSON") from None
    if not isinstance(observed, dict) or set(observed) != OBSERVATION_FIELDS:
        raise RpcError(-32603, "MCP action bridge returned invalid observation")
    if observed.get("contract_type") != "action_observation" or observed.get("schema_version") != "v1":
        raise RpcError(-32603, "MCP action bridge returned invalid observation")
    if not isinstance(observed.get("ok"), bool):
        raise RpcError(-32603, "MCP action bridge returned invalid observation")
    if decoded != canonical(observed) + "\n":
        raise RpcError(-32603, "MCP action bridge returned noncanonical observation")
    return observed


def atomic_write_trace(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    filename = os.path.basename(path)
    parent_descriptor = os.open(parent, DIRECTORY_FLAGS)
    temporary = ".mcp-trace-" + uuid.uuid4().hex
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW,
            0o600,
            dir_fd=parent_descriptor,
        )
        encoded = (canonical(payload) + "\n").encode("utf-8")
        view = memoryview(encoded)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary,
            filename,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.fsync(parent_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def read_bridge_token(descriptor):
    try:
        encoded = os.read(descriptor, 65)
    finally:
        os.close(descriptor)
    if len(encoded) != 64 or any(
        byte not in b"0123456789abcdef" for byte in encoded
    ):
        raise RuntimeError("MCP bridge authentication channel is invalid")
    return encoded.decode("ascii")


def main():
    parser = argparse.ArgumentParser(prog="opbench_mcp_server.py")
    parser.add_argument("--action-client", required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--trace-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--codex-cli-version", required=True)
    parser.add_argument("--bridge-token-fd", required=True, type=int)
    args = parser.parse_args()
    args.bridge_token = read_bridge_token(args.bridge_token_fd)

    negotiated = None
    initialize_count = 0
    tools_list_count = 0
    tools_call_count = 0
    protocol_error_count = 0
    terminal = "client_closed"
    try:
        while True:
            value, read_error = read_message()
            if value is None and read_error is None:
                break
            if read_error is not None:
                protocol_error_count += 1
                write_response(read_error)
                terminal = "protocol_failed"
                break
            selected_id = None
            try:
                selected_id, method, params, notification = parse_request(value)
                if method == "notifications/initialized":
                    if not notification:
                        raise RpcError(-32600, "initialized must be a notification")
                    continue
                if notification:
                    raise RpcError(-32601, "notification method not found")
                if method == "initialize":
                    negotiated, result = initialize_result(params)
                    initialize_count += 1
                elif method == "ping":
                    if params_without_metadata(params, "ping params"):
                        raise RpcError(-32602, "ping params must be empty")
                    result = {}
                elif method == "tools/list":
                    if negotiated is None:
                        raise RpcError(-32602, "MCP server is not initialized")
                    list_params = params_without_metadata(params, "tools/list params")
                    if set(list_params) - {"cursor"}:
                        raise RpcError(-32602, "tools/list params has invalid fields")
                    if "cursor" in list_params and (
                        not isinstance(list_params["cursor"], str)
                        or not list_params["cursor"]
                    ):
                        raise RpcError(-32602, "tools/list cursor must be a string")
                    result = {"tools": TOOLS}
                    tools_list_count += 1
                elif method == "tools/call":
                    if negotiated is None:
                        raise RpcError(-32602, "MCP server is not initialized")
                    call = params_without_metadata(params, "tools/call params")
                    if set(call) != {"name", "arguments"}:
                        raise RpcError(-32602, "tools/call params has invalid fields")
                    name = call["name"]
                    arguments = call["arguments"]
                    if not isinstance(name, str) or name not in TOOL_NAMES:
                        raise RpcError(-32602, "unknown MCP tool")
                    if not isinstance(arguments, dict):
                        raise RpcError(-32602, "tool arguments must be an object")
                    observed = invoke_action(args, name, arguments)
                    result = {
                        "content": [{"type": "text", "text": canonical(observed)}],
                        "structuredContent": observed,
                        "isError": not observed["ok"],
                    }
                    tools_call_count += 1
                else:
                    raise RpcError(-32601, "method not found")
            except RpcError as exc:
                protocol_error_count += 1
                if selected_id is not None or not (isinstance(value, dict) and "id" not in value):
                    write_response(error(selected_id, exc.code, exc.message))
                if exc.code in {-32600, -32603}:
                    terminal = "protocol_failed"
                    break
                continue
            write_response(response(selected_id, result))
    except BaseException:
        terminal = "protocol_failed"
        sys.stderr.write("MCP server failed\n")
        sys.stderr.flush()
        return_code = 1
    else:
        return_code = 1 if terminal == "protocol_failed" else 0
    finally:
        trace = {
            "adapter_id": "codex_mcp_canonical",
            "model_id": args.model_id,
            "codex_cli_version": args.codex_cli_version,
            "negotiated_protocol_version": negotiated,
            "initialize_count": initialize_count,
            "tools_list_count": tools_list_count,
            "tools_call_count": tools_call_count,
            "protocol_error_count": protocol_error_count,
            "server_terminal_status": terminal,
        }
        try:
            atomic_write_trace(args.trace_path, trace)
        except BaseException:
            sys.stderr.write("MCP trace write failed\n")
            sys.stderr.flush()
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
'''


def render_mcp_stdio_launcher(
    tools: tuple[McpToolDefinition, ...],
) -> str:
    if not isinstance(tools, tuple) or tuple(tool.name for tool in tools) != ACTION_NAMES:
        raise ContractError("tools: expected canonical MCP tool registry")
    encoded = canonical_json([tool.to_dict() for tool in tools])
    return _LAUNCHER_TEMPLATE.replace("__TOOLS_JSON__", repr(encoded))


__all__ = ["ToolInvoker", "render_mcp_stdio_launcher", "serve_mcp_stdio"]
