from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from op_bench.runtime.adapters import AdapterActionChannel
from op_bench.runtime.process_actions import ProcessActionExchange
from tests.test_runtime_wire_contracts import action_observation


COMMANDS = (
    "workspace_list",
    "workspace_search",
    "workspace_read",
    "workspace_write",
    "workspace_apply_patch",
    "command_run",
    "test_run",
    "vcs_diff",
    "session_finish",
)


class ProcessActionExchangeTests(unittest.TestCase):
    def test_generated_client_uses_the_absolute_session_deadline(self) -> None:
        received: list[dict[str, object]] = []

        def execute(payload):
            received.append(payload)
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        self.assertIn(
            "deadline_ms",
            inspect.signature(ProcessActionExchange).parameters,
        )
        with tempfile.TemporaryDirectory() as temporary:
            channel = AdapterActionChannel(execute)
            with channel as action_client:
                exchange = ProcessActionExchange(
                    action_client=action_client,
                    session_id="session-absolute-deadline",
                    exchange_root=Path(temporary) / "exchange",
                    timeout_ms=2_000,
                    deadline_ms=1_900_000,
                ).start()
                completed = subprocess.run(
                    (
                        sys.executable,
                        str(exchange.client_path),
                        "workspace_list",
                        "--arguments",
                        "{}",
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                exchange.close()

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(received[0]["deadline_ms"], 1_900_000)

    def test_transport_token_denies_direct_client_bypass(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        with tempfile.TemporaryDirectory() as temporary:
            channel = AdapterActionChannel(execute)
            with channel as action_client:
                exchange = ProcessActionExchange(
                    action_client=action_client,
                    session_id="session-token-bound",
                    exchange_root=Path(temporary) / "exchange",
                    timeout_ms=2_000,
                    deadline_ms=2_000,
                    transport_token="fixture-transport-token",
                ).start()
                command = (
                    sys.executable,
                    str(exchange.client_path),
                    "workspace_list",
                    "--arguments",
                    "{}",
                )
                denied = subprocess.run(
                    command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                allowed = subprocess.run(
                    command,
                    env={
                        **os.environ,
                        "OPBENCH_ACTION_TRANSPORT_TOKEN": "fixture-transport-token",
                    },
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                exchange.close()

        self.assertNotEqual(denied.returncode, 0)
        self.assertIn("transport authentication failed", denied.stderr)
        self.assertEqual(allowed.returncode, 0, allowed.stderr)

    def test_generated_client_round_trips_every_command_with_locked_sequence(self) -> None:
        received = []

        def execute(payload):
            received.append(payload)
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
                data={"action_name": payload["action_name"]},
            ).to_dict()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "exchange"
            channel = AdapterActionChannel(execute)
            with channel as action_client:
                exchange = ProcessActionExchange(
                    action_client=action_client,
                    session_id="session-process-actions",
                    exchange_root=root,
                    timeout_ms=2_000,
                    deadline_ms=2_000,
                )
                exchange.start()
                for command in COMMANDS:
                    completed = subprocess.run(
                        (
                            sys.executable,
                            str(exchange.client_path),
                            command,
                            "--arguments",
                            json.dumps({}),
                        ),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    response = json.loads(completed.stdout)
                    self.assertEqual(response["data"]["action_name"], command)

                fixed = {
                    "contract_type": "action_request",
                    "schema_version": "v1",
                    "session_id": "session-process-actions",
                    "action_id": "fixed-json-action",
                    "action_name": "workspace_read",
                    "arguments": {"path": "src/operator.py"},
                    "client_sequence": 10,
                    "deadline_ms": 2_000,
                }
                for _ in range(2):
                    completed = subprocess.run(
                        (
                            sys.executable,
                            str(exchange.client_path),
                            "json",
                            "--request",
                            json.dumps(fixed),
                        ),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False,
                    )
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                exchange.close()

            self.assertEqual(
                [item["client_sequence"] for item in received[:9]],
                list(range(1, 10)),
            )
            self.assertEqual(
                [item["action_id"] for item in received[-2:]],
                ["fixed-json-action", "fixed-json-action"],
            )
            self.assertEqual(exchange.observation_count, 11)

    def test_malformed_json_timeout_and_symlink_replacement_fail_closed(self) -> None:
        def slow(payload):
            time.sleep(0.3)
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            channel = AdapterActionChannel(slow, timeout_sec=1)
            with channel as action_client:
                exchange = ProcessActionExchange(
                    action_client=action_client,
                    session_id="session-timeout",
                    exchange_root=root / "exchange",
                    timeout_ms=100,
                    deadline_ms=100,
                )
                exchange.start()
                malformed = subprocess.run(
                    (
                        sys.executable,
                        str(exchange.client_path),
                        "json",
                        "--request",
                        "not-json",
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                timed_out = subprocess.run(
                    (
                        sys.executable,
                        str(exchange.client_path),
                        "workspace_read",
                        "--arguments",
                        '{"path":"src/operator.py"}',
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(malformed.returncode, 0)
                self.assertNotEqual(timed_out.returncode, 0)
                self.assertIn("timed out", timed_out.stderr.lower())

                outside = root / "outside"
                outside.mkdir()
                moved = root / "moved-requests"
                exchange.request_directory.rename(moved)
                exchange.request_directory.symlink_to(outside, target_is_directory=True)
                escaped = subprocess.run(
                    (
                        sys.executable,
                        str(exchange.client_path),
                        "workspace_list",
                        "--arguments",
                        "{}",
                    ),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertNotEqual(escaped.returncode, 0)
                self.assertEqual(list(outside.iterdir()), [])
                exchange.close(cleanup=False)


if __name__ == "__main__":
    unittest.main()
