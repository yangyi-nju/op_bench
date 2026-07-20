from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ACTION_NAMES, ActionRequest
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_int,
    require_str,
)


MCP_PROTOCOL_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)
MCP_MAX_MESSAGE_BYTES = 1_048_576
MCP_SERVER_NAME = "opbench"
MCP_SERVER_VERSION = "v0.6"
MCP_SERVER_TERMINAL_STATUSES = (
    "completed",
    "not_started",
    "start_failed",
    "protocol_failed",
    "client_closed",
    "terminated",
    "killed",
)


def _plain_json(value: object, path: str) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: object keys must be strings")
            result[key] = _plain_json(item, f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _plain_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ContractError(
        f"{path}: unsupported JSON value type {type(value).__name__}"
    )


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


@dataclass(frozen=True)
class McpToolDefinition:
    name: str
    description: str
    input_schema: Mapping[str, object]

    def __post_init__(self) -> None:
        require_enum(self.name, "name", ACTION_NAMES)
        require_str(self.description, "description")
        plain = _plain_json(self.input_schema, "input_schema")
        if not isinstance(plain, dict):
            raise ContractError("input_schema: expected object")
        canonical_json(plain)
        object.__setattr__(self, "input_schema", _freeze_json(plain))

    def to_dict(self) -> dict[str, object]:
        schema = _plain_json(self.input_schema, "input_schema")
        assert isinstance(schema, dict)
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": schema,
        }


@dataclass(frozen=True)
class McpAdapterTrace:
    adapter_id: str
    model_id: str
    codex_cli_version: str
    negotiated_protocol_version: str | None
    initialize_count: int
    tools_list_count: int
    tools_call_count: int
    protocol_error_count: int
    server_terminal_status: str

    def __post_init__(self) -> None:
        require_str(self.adapter_id, "adapter_id")
        require_str(self.model_id, "model_id")
        require_str(self.codex_cli_version, "codex_cli_version")
        if self.negotiated_protocol_version is not None:
            require_enum(
                self.negotiated_protocol_version,
                "negotiated_protocol_version",
                MCP_PROTOCOL_VERSIONS,
            )
        require_int(self.initialize_count, "initialize_count", minimum=0)
        require_int(self.tools_list_count, "tools_list_count", minimum=0)
        require_int(self.tools_call_count, "tools_call_count", minimum=0)
        require_int(self.protocol_error_count, "protocol_error_count", minimum=0)
        require_enum(
            self.server_terminal_status,
            "server_terminal_status",
            MCP_SERVER_TERMINAL_STATUSES,
        )
        assert_public_artifact_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "adapter_id": self.adapter_id,
            "model_id": self.model_id,
            "codex_cli_version": self.codex_cli_version,
            "negotiated_protocol_version": self.negotiated_protocol_version,
            "initialize_count": self.initialize_count,
            "tools_list_count": self.tools_list_count,
            "tools_call_count": self.tools_call_count,
            "protocol_error_count": self.protocol_error_count,
            "server_terminal_status": self.server_terminal_status,
        }

    @classmethod
    def from_dict(cls, value: object) -> "McpAdapterTrace":
        fields = (
            "adapter_id",
            "model_id",
            "codex_cli_version",
            "negotiated_protocol_version",
            "initialize_count",
            "tools_list_count",
            "tools_call_count",
            "protocol_error_count",
            "server_terminal_status",
        )
        data = require_exact_fields(value, "mcp_adapter_trace", fields)
        negotiated = data["negotiated_protocol_version"]
        if negotiated is not None:
            negotiated = require_enum(
                negotiated,
                "negotiated_protocol_version",
                MCP_PROTOCOL_VERSIONS,
            )
        return cls(
            adapter_id=require_str(data["adapter_id"], "adapter_id"),
            model_id=require_str(data["model_id"], "model_id"),
            codex_cli_version=require_str(
                data["codex_cli_version"], "codex_cli_version"
            ),
            negotiated_protocol_version=negotiated,
            initialize_count=require_int(
                data["initialize_count"], "initialize_count", minimum=0
            ),
            tools_list_count=require_int(
                data["tools_list_count"], "tools_list_count", minimum=0
            ),
            tools_call_count=require_int(
                data["tools_call_count"], "tools_call_count", minimum=0
            ),
            protocol_error_count=require_int(
                data["protocol_error_count"], "protocol_error_count", minimum=0
            ),
            server_terminal_status=require_enum(
                data["server_terminal_status"],
                "server_terminal_status",
                MCP_SERVER_TERMINAL_STATUSES,
            ),
        )


def _string_schema() -> dict[str, object]:
    return {"type": "string", "minLength": 1}


def _positive_integer_schema(*, maximum: int | None = None) -> dict[str, object]:
    schema: dict[str, object] = {"type": "integer", "minimum": 1}
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def _object_schema(
    properties: Mapping[str, object],
    *,
    required: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_MCP_TOOLS = (
    McpToolDefinition(
        name="workspace_list",
        description="List bounded entries in the authoritative workspace.",
        input_schema=_object_schema(
            {
                "path": _string_schema(),
                "recursive": {"type": "boolean"},
                "max_entries": _positive_integer_schema(maximum=1_000),
                "max_depth": _positive_integer_schema(maximum=32),
            }
        ),
    ),
    McpToolDefinition(
        name="workspace_search",
        description="Search UTF-8 workspace files for an exact text query.",
        input_schema=_object_schema(
            {
                "query": _string_schema(),
                "path": _string_schema(),
                "max_matches": _positive_integer_schema(maximum=1_000),
                "max_files": _positive_integer_schema(maximum=5_000),
            },
            required=("query",),
        ),
    ),
    McpToolDefinition(
        name="workspace_read",
        description="Read a bounded file from the authoritative workspace.",
        input_schema=_object_schema(
            {
                "path": _string_schema(),
                "max_bytes": _positive_integer_schema(),
            },
            required=("path",),
        ),
    ),
    McpToolDefinition(
        name="workspace_write",
        description="Write UTF-8 content to an allowed workspace path.",
        input_schema=_object_schema(
            {
                "path": _string_schema(),
                "content": {"type": "string"},
                "mode": {"type": "integer", "minimum": 0, "maximum": 0o777},
            },
            required=("path", "content"),
        ),
    ),
    McpToolDefinition(
        name="workspace_apply_patch",
        description="Apply one canonical unified patch within allowed paths.",
        input_schema=_object_schema(
            {"patch": {"type": "string"}},
            required=("patch",),
        ),
    ),
    McpToolDefinition(
        name="command_run",
        description="Run one allowed argv-form command in the workspace runtime.",
        input_schema=_object_schema(
            {
                "command": {
                    "type": "array",
                    "items": _string_schema(),
                    "minItems": 1,
                },
                "cwd": _string_schema(),
                "timeout_ms": _positive_integer_schema(),
            },
            required=("command",),
        ),
    ),
    McpToolDefinition(
        name="test_run",
        description="Run one public registered test selector.",
        input_schema=_object_schema(
            {"selector_id": _string_schema()},
            required=("selector_id",),
        ),
    ),
    McpToolDefinition(
        name="vcs_diff",
        description="Return the canonical workspace patch without freezing it.",
        input_schema=_object_schema({}),
    ),
    McpToolDefinition(
        name="session_finish",
        description="Freeze the canonical patch and finish the Agent session.",
        input_schema=_object_schema({}),
    ),
)


def canonical_mcp_tools() -> tuple[McpToolDefinition, ...]:
    return _MCP_TOOLS


class CanonicalMcpTransport:
    """Minimal MCP tool dispatcher that delegates every call to one action service."""

    def __init__(self, service: CanonicalActionService) -> None:
        if not isinstance(service, CanonicalActionService):
            raise ContractError("service: expected CanonicalActionService")
        self.service = service

    def call_tool(self, action_name: str, envelope: object) -> dict[str, object]:
        action_id = "invalid"
        try:
            if not isinstance(envelope, Mapping):
                raise ContractError("MCP envelope: expected object")
            candidate_action_id = envelope.get("action_id")
            if isinstance(candidate_action_id, str) and candidate_action_id:
                action_id = candidate_action_id
            payload = {
                "contract_type": "action_request",
                "schema_version": "v1",
                "session_id": envelope.get("session_id"),
                "action_id": envelope.get("action_id"),
                "action_name": action_name,
                "arguments": envelope.get("arguments"),
                "client_sequence": envelope.get("client_sequence"),
                "deadline_ms": envelope.get("deadline_ms"),
            }
            request = ActionRequest.from_dict(payload)
        except ContractError as exc:
            return self.service.invalid_request_observation(
                action_id=action_id,
                message=str(exc),
            ).to_dict()
        return self.service.execute(request).to_dict()


__all__ = [
    "CanonicalMcpTransport",
    "MCP_MAX_MESSAGE_BYTES",
    "MCP_PROTOCOL_VERSIONS",
    "MCP_SERVER_NAME",
    "MCP_SERVER_TERMINAL_STATUSES",
    "MCP_SERVER_VERSION",
    "McpAdapterTrace",
    "McpToolDefinition",
    "canonical_mcp_tools",
]
