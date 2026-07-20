from __future__ import annotations

import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ACTION_NAMES
from op_bench.runtime.mcp import (
    MCP_PROTOCOL_VERSIONS,
    McpAdapterTrace,
    canonical_mcp_tools,
)

try:
    from op_bench.runtime import mcp_stdio
except ImportError:
    mcp_stdio = None


def observation(*, ok: bool = True) -> dict[str, object]:
    return {
        "contract_type": "action_observation",
        "schema_version": "v1",
        "session_id": "session-mcp",
        "action_id": "action-mcp",
        "ok": ok,
        "error_code": "ok" if ok else "invalid_request",
        "message": "completed" if ok else "arguments were rejected",
        "data": {"value": 1},
        "started_at_ms": 10,
        "ended_at_ms": 11,
        "budget_delta": {
            "contract_type": "budget_delta",
            "schema_version": "v1",
            "wall_clock_ms": 1,
            "actions": 1,
            "tests": 0,
            "commands": 0,
            "output_bytes": 1,
            "provider_tokens": 0,
        },
        "mutation_state": "none",
    }


def request(
    request_id: int,
    method: str,
    params: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
    }
    if params is not None:
        payload["params"] = params
    return payload


class McpStdioProtocolTests(unittest.TestCase):
    def run_server(
        self,
        messages: list[str],
        *,
        invoke_tool=None,
        max_message_bytes: int = 1_048_576,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], str, str]:
        self.assertIsNotNone(mcp_stdio)
        stdout = io.StringIO()
        stderr = io.StringIO()
        traces: list[dict[str, object]] = []
        terminal = mcp_stdio.serve_mcp_stdio(
            io.StringIO("".join(f"{message}\n" for message in messages)),
            stdout,
            stderr,
            invoke_tool=invoke_tool or (lambda name, arguments: observation()),
            write_trace=lambda payload: traces.append(dict(payload)),
            max_message_bytes=max_message_bytes,
        )
        responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
        return responses, traces, terminal, stderr.getvalue()

    def test_initialize_negotiates_every_supported_version(self) -> None:
        for version in MCP_PROTOCOL_VERSIONS:
            with self.subTest(version=version):
                responses, traces, terminal, stderr = self.run_server(
                    [
                        canonical_json(
                            request(
                                1,
                                "initialize",
                                {
                                    "protocolVersion": version,
                                    "capabilities": {},
                                    "clientInfo": {"name": "fixture", "version": "1"},
                                },
                            )
                        )
                    ]
                )

                self.assertEqual(responses[0]["result"]["protocolVersion"], version)
                self.assertEqual(
                    responses[0]["result"]["capabilities"],
                    {"tools": {"listChanged": False}},
                )
                self.assertEqual(
                    responses[0]["result"]["serverInfo"],
                    {"name": "opbench", "version": "v0.6"},
                )
                self.assertEqual(traces[0]["negotiated_protocol_version"], version)
                self.assertEqual(terminal, "client_closed")
                self.assertEqual(stderr, "")

    def test_unsupported_version_offers_the_highest_server_version(self) -> None:
        responses, _, _, _ = self.run_server(
            [
                canonical_json(
                    request(
                        1,
                        "initialize",
                        {
                            "protocolVersion": "2099-01-01",
                            "capabilities": {},
                            "clientInfo": {"name": "fixture", "version": "1"},
                        },
                    )
                )
            ]
        )

        self.assertEqual(responses[0]["result"]["protocolVersion"], "2025-06-18")

    def test_notification_ping_list_and_call_complete_one_session(self) -> None:
        calls: list[tuple[str, dict[str, object]]] = []

        def invoke(name: str, arguments: dict[str, object]):
            calls.append((name, arguments))
            return observation()

        initialized = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        responses, traces, terminal, _ = self.run_server(
            [
                canonical_json(
                    request(
                        1,
                        "initialize",
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "fixture", "version": "1"},
                        },
                    )
                ),
                canonical_json(initialized),
                canonical_json(request(2, "ping", {})),
                canonical_json(request(3, "tools/list", {})),
                canonical_json(
                    request(
                        4,
                        "tools/call",
                        {"name": "workspace_read", "arguments": {"path": "src/a.py"}},
                    )
                ),
            ],
            invoke_tool=invoke,
        )

        self.assertEqual([response["id"] for response in responses], [1, 2, 3, 4])
        self.assertEqual(responses[1]["result"], {})
        self.assertEqual(
            tuple(tool["name"] for tool in responses[2]["result"]["tools"]),
            ACTION_NAMES,
        )
        self.assertEqual(
            responses[2]["result"]["tools"],
            [tool.to_dict() for tool in canonical_mcp_tools()],
        )
        call_result = responses[3]["result"]
        self.assertEqual(json.loads(call_result["content"][0]["text"]), observation())
        self.assertEqual(call_result["structuredContent"], observation())
        self.assertEqual(call_result["isError"], False)
        self.assertEqual(calls, [("workspace_read", {"path": "src/a.py"})])
        self.assertEqual(
            traces,
            [
                {
                    "negotiated_protocol_version": "2025-06-18",
                    "initialize_count": 1,
                    "tools_list_count": 1,
                    "tools_call_count": 1,
                    "protocol_error_count": 0,
                    "server_terminal_status": "client_closed",
                }
            ],
        )
        self.assertEqual(terminal, "client_closed")

    def test_action_error_is_a_successful_json_rpc_tool_result(self) -> None:
        responses, traces, _, _ = self.run_server(
            [
                canonical_json(
                    request(
                        1,
                        "initialize",
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "fixture", "version": "1"},
                        },
                    )
                ),
                canonical_json(
                    request(
                        2,
                        "tools/call",
                        {"name": "test_run", "arguments": {"selector_id": "bad"}},
                    )
                )
            ],
            invoke_tool=lambda name, arguments: observation(ok=False),
        )

        self.assertNotIn("error", responses[1])
        self.assertEqual(responses[1]["result"]["structuredContent"]["ok"], False)
        self.assertEqual(responses[1]["result"]["isError"], True)
        self.assertEqual(traces[0]["tools_call_count"], 1)
        self.assertEqual(traces[0]["protocol_error_count"], 0)

    def test_malformed_action_observation_is_an_internal_bridge_error(self) -> None:
        responses, traces, terminal, stderr = self.run_server(
            [
                canonical_json(
                    request(
                        1,
                        "initialize",
                        {
                            "protocolVersion": "2025-06-18",
                            "capabilities": {},
                            "clientInfo": {"name": "fixture", "version": "1"},
                        },
                    )
                ),
                canonical_json(
                    request(
                        2,
                        "tools/call",
                        {"name": "workspace_read", "arguments": {"path": "src/a.py"}},
                    )
                ),
            ],
            invoke_tool=lambda name, arguments: {"ok": True},
        )

        self.assertIn("error", responses[1])
        self.assertEqual(responses[1]["error"]["code"], -32603)
        self.assertEqual(traces[0]["tools_call_count"], 0)
        self.assertEqual(traces[0]["protocol_error_count"], 1)
        self.assertEqual(traces[0]["server_terminal_status"], "protocol_failed")
        self.assertEqual(terminal, "protocol_failed")
        self.assertEqual(stderr, "")

    def test_parse_method_and_parameter_failures_use_stable_error_codes(self) -> None:
        malformed = "{not-json"
        unknown = canonical_json(request(2, "resources/list", {}))
        invalid_params = canonical_json(
            request(3, "tools/call", {"name": "workspace_read", "extra": True})
        )
        unknown_tool = canonical_json(
            request(4, "tools/call", {"name": "not-a-tool", "arguments": {}})
        )
        responses, traces, terminal, stderr = self.run_server(
            [unknown, invalid_params, unknown_tool]
        )

        self.assertEqual(
            [response["error"]["code"] for response in responses],
            [-32601, -32602, -32602],
        )
        self.assertEqual([response["id"] for response in responses], [2, 3, 4])
        self.assertEqual(traces[0]["protocol_error_count"], 3)
        self.assertEqual(terminal, "client_closed")
        self.assertNotIn("Traceback", stderr)
        self.assertTrue(all(response["jsonrpc"] == "2.0" for response in responses))

        fatal_responses, fatal_traces, fatal_terminal, _ = self.run_server(
            [malformed, unknown]
        )
        self.assertEqual(len(fatal_responses), 1)
        self.assertEqual(fatal_responses[0]["error"]["code"], -32700)
        self.assertEqual(fatal_traces[0]["protocol_error_count"], 1)
        self.assertEqual(fatal_terminal, "protocol_failed")

    def test_oversized_message_is_rejected_without_invoking_a_tool(self) -> None:
        calls: list[object] = []
        oversized = canonical_json(request(1, "ping", {"padding": "x" * 200}))
        responses, traces, terminal, _ = self.run_server(
            [oversized],
            invoke_tool=lambda name, arguments: calls.append((name, arguments)),
            max_message_bytes=100,
        )

        self.assertEqual(responses[0]["error"]["code"], -32600)
        self.assertEqual(traces[0]["protocol_error_count"], 1)
        self.assertEqual(terminal, "protocol_failed")
        self.assertEqual(calls, [])


class RenderedMcpLauncherTests(unittest.TestCase):
    def test_launcher_stops_a_continuously_overflowing_action_bridge(self) -> None:
        self.assertIsNotNone(mcp_stdio)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "opbench_mcp_server.py"
            action_client = root / "overflow_action_client.py"
            trace_path = root / "mcp_trace.json"
            launcher.write_text(
                mcp_stdio.render_mcp_stdio_launcher(canonical_mcp_tools()),
                encoding="utf-8",
            )
            action_client.write_text(
                "import os, time\n"
                "chunk=b'x'*65536\n"
                "for _ in range(256): os.write(1, chunk)\n"
                "time.sleep(30)\n",
                encoding="utf-8",
            )
            messages = (
                request(
                    1,
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                ),
                request(
                    2,
                    "tools/call",
                    {"name": "workspace_read", "arguments": {"path": "src/a.py"}},
                ),
            )
            token_descriptor, token_writer = os.pipe()
            try:
                os.write(token_writer, b"a" * 64)
            finally:
                os.close(token_writer)
            try:
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-I",
                        str(launcher),
                        "--action-client",
                        str(action_client),
                        "--python-executable",
                        sys.executable,
                        "--trace-path",
                        str(trace_path),
                        "--model-id",
                        "gpt-5.6-sol",
                        "--codex-cli-version",
                        "codex-cli 0.145.0-alpha.18",
                        "--bridge-token-fd",
                        str(token_descriptor),
                    ),
                    input="".join(
                        f"{canonical_json(message)}\n" for message in messages
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                    env={"PATH": os.environ.get("PATH", "")},
                    pass_fds=(token_descriptor,),
                )
            finally:
                os.close(token_descriptor)

            self.assertEqual(completed.returncode, 1)
            self.assertLess(len(completed.stdout.encode("utf-8")), 10_000)
            trace = McpAdapterTrace.from_dict(json.loads(trace_path.read_text()))
            self.assertEqual(trace.protocol_error_count, 1)
            self.assertEqual(trace.server_terminal_status, "protocol_failed")

    def test_launcher_is_stdlib_only_and_round_trips_a_real_action_process(self) -> None:
        self.assertIsNotNone(mcp_stdio)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "opbench_mcp_server.py"
            action_client = root / "fake_action_client.py"
            trace_path = root / "mcp_trace.json"
            launcher.write_text(
                mcp_stdio.render_mcp_stdio_launcher(canonical_mcp_tools()),
                encoding="utf-8",
            )
            self.assertNotIn("subprocess.run(", launcher.read_text(encoding="utf-8"))
            self.assertIn("selectors.DefaultSelector", launcher.read_text(encoding="utf-8"))
            launcher.chmod(0o700)
            action_client.write_text(
                textwrap.dedent(
                    f"""\
                    import json
                    import sys

                    value = {observation()!r}
                    value["data"] = {{"argv": sys.argv[1:]}}
                    sys.stdout.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\\n")
                    """
                ),
                encoding="utf-8",
            )
            action_client.chmod(0o700)
            messages = (
                request(
                    1,
                    "initialize",
                    {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "fixture", "version": "1"},
                    },
                ),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                request(2, "tools/list", {}),
                request(
                    3,
                    "tools/call",
                    {"name": "workspace_read", "arguments": {"path": "src/a.py"}},
                ),
            )
            token_descriptor, token_writer = os.pipe()
            try:
                os.write(token_writer, b"b" * 64)
            finally:
                os.close(token_writer)
            try:
                completed = subprocess.run(
                    (
                        sys.executable,
                        "-I",
                        str(launcher),
                        "--action-client",
                        str(action_client),
                        "--python-executable",
                        sys.executable,
                        "--trace-path",
                        str(trace_path),
                        "--model-id",
                        "gpt-5.6-sol",
                        "--codex-cli-version",
                        "codex-cli 0.145.0-alpha.18",
                        "--bridge-token-fd",
                        str(token_descriptor),
                    ),
                    input="".join(
                        f"{canonical_json(message)}\n" for message in messages
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    check=False,
                    env={"PATH": os.environ.get("PATH", "")},
                    pass_fds=(token_descriptor,),
                )
            finally:
                os.close(token_descriptor)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            responses = [json.loads(line) for line in completed.stdout.splitlines()]
            self.assertEqual([response["id"] for response in responses], [1, 2, 3])
            self.assertEqual(
                responses[-1]["result"]["structuredContent"]["data"]["argv"],
                ["workspace_read", "--arguments", '{"path":"src/a.py"}'],
            )
            self.assertEqual(completed.stderr, "")
            self.assertEqual(stat.S_IMODE(launcher.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(trace_path.stat().st_mode), 0o600)
            trace = McpAdapterTrace.from_dict(json.loads(trace_path.read_text()))
            self.assertEqual(trace.adapter_id, "codex_mcp_canonical")
            self.assertEqual(trace.negotiated_protocol_version, "2025-06-18")
            self.assertEqual(trace.tools_list_count, 1)
            self.assertEqual(trace.tools_call_count, 1)
            self.assertEqual(trace.server_terminal_status, "client_closed")


if __name__ == "__main__":
    unittest.main()
