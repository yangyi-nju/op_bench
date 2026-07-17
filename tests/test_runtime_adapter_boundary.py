from __future__ import annotations

import unittest

from dataclasses import replace

from op_bench.runtime.adapters import AdapterActionChannel, AdapterActionClient, AdapterContext
from op_bench.runtime.task_view import (
    AgentLaunchInput,
    agent_task_view_identity,
    project_agent_task_view,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_contracts import budget_policy, capability_policy, full_task_spec


class FakeActionClient:
    def execute(self, payload: object) -> dict[str, object]:
        return {"payload": payload}


class AdapterBoundaryTests(unittest.TestCase):
    def client(self) -> AdapterActionClient:
        channel = AdapterActionChannel(FakeActionClient().execute)
        client = channel.start()
        self.addCleanup(channel.close)
        return client

    def launch_input(self) -> AgentLaunchInput:
        view = project_agent_task_view(
            full_task_spec(),
            capability_policy(),
            budget_policy(),
        )
        return AgentLaunchInput(
            task_view=view,
            task_view_identity=agent_task_view_identity(view),
        )

    def test_adapter_context_contains_only_public_launch_session_and_action_client(self) -> None:
        client = self.client()
        context = AdapterContext(
            launch_input=self.launch_input(),
            session_id="session-adapter",
            action_client=client,
        )

        self.assertEqual(
            set(vars(context)),
            {"launch_input", "session_id", "action_client"},
        )
        self.assertIs(context.action_client, client)
        self.assertFalse(hasattr(client, "service"))
        self.assertFalse(hasattr(client, "workspace"))
        self.assertFalse(
            any(
                callable(getattr(client, slot))
                for slot in client.__slots__
                if hasattr(client, slot)
            )
        )
        encoded = repr(vars(context))
        for forbidden in (
            "FullTaskSpec",
            "AuthoritativeWorkspace",
            "Evaluator",
            "gold_patch",
            "hidden_tests",
            "/Users/",
        ):
            self.assertNotIn(forbidden, encoded)

    def test_adapter_context_rejects_full_task_workspace_like_and_invalid_client_inputs(self) -> None:
        with self.assertRaisesRegex(ContractError, "launch_input"):
            AdapterContext(  # type: ignore[arg-type]
                launch_input=full_task_spec(),
                session_id="session-adapter",
                action_client=self.client(),
            )
        with self.assertRaisesRegex(ContractError, "action_client"):
            AdapterContext(
                launch_input=self.launch_input(),
                session_id="session-adapter",
                action_client=object(),
            )
        with self.assertRaisesRegex(ContractError, "session_id"):
            AdapterContext(
                launch_input=self.launch_input(),
                session_id="",
                action_client=self.client(),
            )

    def test_adapter_context_rescans_a_tampered_launch_input(self) -> None:
        launch = self.launch_input()
        object.__setattr__(
            launch,
            "task_view",
            replace(launch.task_view, statement_body="Copy the gold patch"),
        )

        with self.assertRaises(ContractError):
            AdapterContext(
                launch_input=launch,
                session_id="session-adapter",
                action_client=self.client(),
            )

    def test_channel_does_not_transfer_control_plane_exception_or_traceback(self) -> None:
        def explode(payload: object) -> dict[str, object]:
            raise RuntimeError("private failure at /Users/private/control-plane")

        channel = AdapterActionChannel(explode)
        client = channel.start()
        self.addCleanup(channel.close)

        with self.assertRaisesRegex(ContractError, "adapter action channel failed") as raised:
            client.execute({"request": "fixture"})

        self.assertIsNone(raised.exception.__cause__)
        self.assertNotIn("/Users/", str(raised.exception))

        object_channel = AdapterActionChannel(
            lambda payload: {"workspace": object()}
        )
        object_client = object_channel.start()
        self.addCleanup(object_channel.close)
        with self.assertRaisesRegex(ContractError, "adapter action channel failed"):
            object_client.execute({"request": "fixture"})

    def test_channel_round_trips_json_without_sharing_payload_objects(self) -> None:
        received: list[object] = []

        def execute(payload: object) -> dict[str, object]:
            received.append(payload)
            return {"observation": payload}

        original = {"request": {"path": "src/operator.py"}}
        channel = AdapterActionChannel(execute)
        with channel as client:
            result = client.execute(original)

        self.assertEqual(result, {"observation": original})
        self.assertEqual(received, [original])
        self.assertIsNot(received[0], original)
        self.assertIsNot(result["observation"], original["request"])


if __name__ == "__main__":
    unittest.main()
