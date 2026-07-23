from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ACTION_NAMES
from op_bench.runtime import mcp
from op_bench.runtime.validation import ContractError


EXPECTED_PROTOCOL_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
)

EXPECTED_REQUIRED = {
    "workspace_list": (),
    "workspace_search": ("query",),
    "workspace_read": ("path",),
    "workspace_write": ("path", "content"),
    "workspace_apply_patch": ("patch",),
    "command_run": ("command",),
    "test_run": ("selector_id",),
    "vcs_diff": (),
    "session_finish": (),
}

EXPECTED_PROPERTIES = {
    "workspace_list": ("path", "recursive", "max_entries", "max_depth"),
    "workspace_search": ("query", "path", "max_matches", "max_files"),
    "workspace_read": ("path", "max_bytes"),
    "workspace_write": ("path", "content", "mode"),
    "workspace_apply_patch": ("patch",),
    "command_run": ("command", "cwd", "timeout_ms"),
    "test_run": ("selector_id",),
    "vcs_diff": (),
    "session_finish": (),
}


class McpToolRegistryTests(unittest.TestCase):
    def test_protocol_constants_are_frozen_to_the_approved_boundary(self) -> None:
        self.assertEqual(
            getattr(mcp, "MCP_PROTOCOL_VERSIONS", None),
            EXPECTED_PROTOCOL_VERSIONS,
        )
        self.assertEqual(getattr(mcp, "MCP_MAX_MESSAGE_BYTES", None), 1_048_576)
        self.assertEqual(getattr(mcp, "MCP_SERVER_NAME", None), "opbench")
        self.assertEqual(getattr(mcp, "MCP_SERVER_VERSION", None), "v0.6")

    def test_registry_has_the_nine_actions_in_canonical_order(self) -> None:
        builder = getattr(mcp, "canonical_mcp_tools", None)
        self.assertIsNotNone(builder)
        tools = builder()

        self.assertIsInstance(tools, tuple)
        self.assertEqual(tuple(tool.name for tool in tools), ACTION_NAMES)
        self.assertEqual(len({tool.name for tool in tools}), 9)
        self.assertTrue(all(tool.description.strip() for tool in tools))
        self.assertEqual(
            canonical_json([tool.to_dict() for tool in tools]),
            canonical_json([tool.to_dict() for tool in builder()]),
        )

    def test_every_input_schema_is_exact_and_rejects_extra_properties(self) -> None:
        builder = getattr(mcp, "canonical_mcp_tools", None)
        self.assertIsNotNone(builder)

        for tool in builder():
            with self.subTest(tool=tool.name):
                schema = tool.to_dict()["inputSchema"]
                self.assertEqual(schema["type"], "object")
                self.assertEqual(schema["additionalProperties"], False)
                self.assertEqual(
                    tuple(schema["required"]),
                    EXPECTED_REQUIRED[tool.name],
                )
                self.assertEqual(
                    tuple(schema["properties"]),
                    EXPECTED_PROPERTIES[tool.name],
                )

    def test_schema_constraints_match_the_canonical_action_boundary(self) -> None:
        builder = getattr(mcp, "canonical_mcp_tools", None)
        self.assertIsNotNone(builder)
        schemas = {tool.name: tool.to_dict()["inputSchema"] for tool in builder()}

        self.assertEqual(
            schemas["workspace_list"]["properties"]["max_entries"],
            {"type": "integer", "minimum": 1, "maximum": 1_000},
        )
        self.assertEqual(
            schemas["workspace_list"]["properties"]["max_depth"],
            {"type": "integer", "minimum": 1, "maximum": 32},
        )
        self.assertEqual(
            schemas["workspace_search"]["properties"]["max_matches"],
            {"type": "integer", "minimum": 1, "maximum": 1_000},
        )
        self.assertEqual(
            schemas["workspace_search"]["properties"]["max_files"],
            {"type": "integer", "minimum": 1, "maximum": 5_000},
        )
        self.assertEqual(
            schemas["workspace_write"]["properties"]["mode"],
            {"type": "integer", "minimum": 0, "maximum": 0o777},
        )
        self.assertEqual(
            schemas["command_run"]["properties"]["command"],
            {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
            },
        )
        for tool_name, property_name in (
            ("workspace_read", "max_bytes"),
            ("command_run", "timeout_ms"),
        ):
            self.assertEqual(
                schemas[tool_name]["properties"][property_name],
                {"type": "integer", "minimum": 1},
            )

    def test_definitions_and_nested_schemas_are_immutable(self) -> None:
        builder = getattr(mcp, "canonical_mcp_tools", None)
        self.assertIsNotNone(builder)
        tool = builder()[0]

        with self.assertRaises(FrozenInstanceError):
            tool.name = "changed"
        with self.assertRaises(TypeError):
            tool.input_schema["type"] = "array"
        with self.assertRaises(TypeError):
            tool.input_schema["properties"]["path"]["type"] = "integer"


class McpAdapterTraceTests(unittest.TestCase):
    def valid_trace(self):
        trace_type = getattr(mcp, "McpAdapterTrace", None)
        self.assertIsNotNone(trace_type)
        return trace_type(
            adapter_id="codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
            negotiated_protocol_version="2025-06-18",
            initialize_count=1,
            tools_list_count=1,
            tools_call_count=6,
            protocol_error_count=0,
            server_terminal_status="completed",
        )

    def test_trace_round_trips_with_exact_public_fields(self) -> None:
        trace = self.valid_trace()
        payload = trace.to_dict()

        self.assertEqual(
            tuple(payload),
            (
                "adapter_id",
                "model_id",
                "codex_cli_version",
                "negotiated_protocol_version",
                "initialize_count",
                "tools_list_count",
                "tools_call_count",
                "protocol_error_count",
                "server_terminal_status",
            ),
        )
        self.assertEqual(type(trace).from_dict(payload), trace)
        self.assertEqual(canonical_json(payload), canonical_json(trace.to_dict()))

    def test_trace_rejects_unknown_fields_versions_counts_and_private_paths(self) -> None:
        trace = self.valid_trace()
        payload = trace.to_dict()
        payload["unknown"] = True
        with self.assertRaises(ContractError):
            type(trace).from_dict(payload)

        invalid_factories = (
            lambda: replace(trace, negotiated_protocol_version="2099-01-01"),
            lambda: replace(trace, initialize_count=-1),
            lambda: replace(trace, tools_list_count=True),
            lambda: replace(trace, server_terminal_status="running"),
            lambda: replace(trace, model_id="/Users/operator/private-model"),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ContractError):
                    factory()

    def test_trace_allows_no_negotiated_version_before_server_start(self) -> None:
        trace = replace(
            self.valid_trace(),
            negotiated_protocol_version=None,
            initialize_count=0,
            tools_list_count=0,
            tools_call_count=0,
            server_terminal_status="start_failed",
        )

        self.assertIsNone(type(trace).from_dict(trace.to_dict()).negotiated_protocol_version)


if __name__ == "__main__":
    unittest.main()
