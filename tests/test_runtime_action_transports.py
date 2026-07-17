from __future__ import annotations

import base64
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.action_cli import ActionCliTransport
from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import ACTION_NAMES
from op_bench.runtime.mcp import CanonicalMcpTransport
from op_bench.runtime.workspace import AuthoritativeWorkspace
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_actions_service import FakeCommandBackend
from tests.test_runtime_contracts import SHA_A, budget_policy, capability_policy, identity
from tests.test_runtime_workspace import policy as workspace_policy


class ActionTransportConformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def service(self, name: str) -> CanonicalActionService:
        root = Path(self.temporary.name) / name
        initialize_git_repo(root)
        return CanonicalActionService(
            session_id="session-transport",
            workspace=AuthoritativeWorkspace.open(
                root,
                source=identity("source", "fixture@transport", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=ACTION_NAMES,
                writable_paths=("src/",),
                allowed_command_prefixes=("git diff",),
                registered_tests=(),
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=100_000,
                max_actions=20,
                max_tests=0,
                max_commands=5,
                max_output_bytes=10_000,
            ),
            command_backend=FakeCommandBackend(),
            test_registry={},
            clock_ms=lambda: 1_000,
        )

    def test_cli_and_mcp_call_the_same_service_api(self) -> None:
        service = self.service("shared")
        cli = ActionCliTransport(service)
        mcp = CanonicalMcpTransport(service)

        self.assertIs(cli.service, service)
        self.assertIs(mcp.service, service)

    def test_scripted_cli_and_mcp_sequences_are_canonical_equivalent(self) -> None:
        cli_service = self.service("cli")
        mcp_service = self.service("mcp")
        cli = ActionCliTransport(cli_service)
        mcp = CanonicalMcpTransport(mcp_service)
        steps = (
            (
                "workspace_write",
                {"path": "src/operator.py", "content": "VALUE = 2\n"},
            ),
            ("test_run", {"selector_id": "hidden::not-registered"}),
            ("vcs_diff", {}),
            ("session_finish", {}),
        )
        cli_results = []
        mcp_results = []
        for sequence, (name, arguments) in enumerate(steps, start=1):
            envelope = {
                "session_id": "session-transport",
                "action_id": f"action-{sequence}",
                "client_sequence": sequence,
                "deadline_ms": 2_000,
                "arguments": arguments,
            }
            cli_payload = {
                "contract_type": "action_request",
                "schema_version": "v1",
                "action_name": name,
                **envelope,
            }
            cli_results.append(cli.execute(cli_payload))
            mcp_results.append(mcp.call_tool(name, envelope))

        self.assertEqual(cli_results, mcp_results)
        self.assertEqual(
            [item["error_code"] for item in cli_results],
            ["ok", "selector_denied", "ok", "ok"],
        )
        self.assertEqual(cli_service.usage, mcp_service.usage)
        self.assertEqual(
            [exchange.to_dict() for exchange in cli_service.audit_exchanges],
            [exchange.to_dict() for exchange in mcp_service.audit_exchanges],
        )
        cli_patch = base64.b64decode(cli_results[-2]["data"]["patch_base64"])
        mcp_patch = base64.b64decode(mcp_results[-2]["data"]["patch_base64"])
        self.assertEqual(cli_patch, mcp_patch)

    def test_transport_parse_failures_are_stable_invalid_request_observations(self) -> None:
        cli = ActionCliTransport(self.service("invalid-cli"))
        mcp = CanonicalMcpTransport(self.service("invalid-mcp"))

        cli_result = cli.execute({"not": "an action request"})
        mcp_result = mcp.call_tool("workspace_read", {"path": "src/operator.py"})

        self.assertEqual(cli_result["error_code"], "invalid_request")
        self.assertEqual(mcp_result["error_code"], "invalid_request")


if __name__ == "__main__":
    unittest.main()
