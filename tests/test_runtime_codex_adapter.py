from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from op_bench.runtime.adapters import AdapterActionChannel, AdapterContext
from op_bench.runtime import codex_adapter
from op_bench.runtime.codex_adapter import (
    CodexCanonicalAdapter,
    subprocess_command_runner,
)
from op_bench.runtime.process_group import ProcessGroupResult
from op_bench.runtime.task_view import (
    AgentLaunchInput,
    agent_task_view_identity,
    project_agent_task_view,
)
from tests.test_runtime_contracts import budget_policy, capability_policy, full_task_spec
from tests.test_runtime_wire_contracts import action_observation


class CodexCanonicalAdapterTests(unittest.TestCase):
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

    def context(self, execute):
        channel = AdapterActionChannel(execute)
        client = channel.start()
        self.addCleanup(channel.close)
        return AdapterContext(
            launch_input=self.launch_input(),
            session_id="session-codex-adapter",
            deadline_ms=1_900_000,
            action_client=client,
        )

    def test_real_subprocess_client_is_the_only_target_interface(self) -> None:
        observations: list[dict[str, object]] = []
        captured: dict[str, object] = {}
        target_sentinel = "/private/target/repository"
        private_sentinel = "PRIVATE_RUNTIME_HANDLE_SENTINEL"
        calls = (
            ("workspace_read", {"path": "src/operator.py"}),
            ("workspace_write", {"path": "src/operator.py", "content": "fixed\n"}),
            ("test_run", {"selector_id": "public-smoke"}),
            ("vcs_diff", {}),
            ("session_finish", {}),
        )

        def execute(payload):
            observations.append(payload)
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
                data={"action_name": payload["action_name"]},
            ).to_dict()

        def runner(argv, *, cwd, env, timeout_ms):
            captured["argv"] = tuple(argv)
            captured["cwd"] = cwd
            captured["env"] = dict(env)
            captured["timeout_ms"] = timeout_ms
            scratch = Path(cwd)
            client = scratch / "opbench_action.py"
            captured["prompt"] = (scratch / "prompt.txt").read_text(encoding="utf-8")
            captured["scratch_files"] = {
                str(path.relative_to(scratch)): (
                    path.read_text(encoding="utf-8", errors="replace")
                    if path.is_file()
                    else "<directory>"
                )
                for path in scratch.rglob("*")
            }
            for command, arguments in calls:
                completed = subprocess.run(
                    (
                        sys.executable,
                        str(client),
                        command,
                        "--arguments",
                        json.dumps(arguments),
                    ),
                    cwd=cwd,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                    env=env,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            return subprocess.CompletedProcess(argv, 0, stdout="agent complete", stderr="")

        previous = os.environ.get("OPBENCH_PRIVATE_SENTINEL")
        os.environ["OPBENCH_PRIVATE_SENTINEL"] = private_sentinel
        try:
            result = CodexCanonicalAdapter(
                command_runner=runner,
                codex_binary="codex-fixture",
            ).run(self.context(execute))
        finally:
            if previous is None:
                os.environ.pop("OPBENCH_PRIVATE_SENTINEL", None)
            else:
                os.environ["OPBENCH_PRIVATE_SENTINEL"] = previous

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.terminal_reason, "agent_finished")
        self.assertEqual(result.observation_count, 5)
        self.assertEqual(result.finish_count, 1)
        self.assertEqual([item["action_name"] for item in observations], [item[0] for item in calls])

        argv_text = json.dumps(captured["argv"])
        env_text = json.dumps(captured["env"], sort_keys=True)
        scratch_text = json.dumps(captured["scratch_files"], sort_keys=True)
        prompt = str(captured["prompt"])
        self.assertIn("codex-fixture", captured["argv"])
        self.assertIn("workspace_read", prompt)
        self.assertIn("session_finish", prompt)
        self.assertIn("target repository is inaccessible", prompt.lower())
        for forbidden in (
            target_sentinel,
            private_sentinel,
            "FullTaskSpec",
            "Evaluator",
            "RuntimeTargetBinding",
            "raw_handle",
            "gold_patch",
            "hidden_tests",
            "credential",
        ):
            self.assertNotIn(forbidden, argv_text)
            self.assertNotIn(forbidden, env_text)
            self.assertNotIn(forbidden, scratch_text)
        self.assertEqual(
            set(captured["env"]),
            set(captured["env"]) & {"PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR", "CODEX_HOME"},
        )

    def test_failure_modes_map_to_stable_adapter_statuses(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        def invoke(command, *, cwd, env):
            return subprocess.run(
                command,
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        def run_case(runner):
            return CodexCanonicalAdapter(
                command_runner=runner,
                codex_binary="codex-fixture",
            ).run(self.context(execute))

        def executable_missing(argv, **kwargs):
            raise FileNotFoundError("private executable path")

        def timeout(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 1)

        def provider_failure(argv, **kwargs):
            return subprocess.CompletedProcess(
                argv,
                1,
                stdout="",
                stderr="provider_error: quota unavailable",
            )

        def nonzero(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 7, stdout="", stderr="agent failed")

        def missing_finish(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 0, stdout="done", stderr="")

        def duplicate_finish(argv, *, cwd, env, **kwargs):
            client = Path(cwd) / "opbench_action.py"
            for _ in range(2):
                completed = invoke(
                    (sys.executable, str(client), "session_finish", "--arguments", "{}"),
                    cwd=cwd,
                    env=env,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            return subprocess.CompletedProcess(argv, 0, stdout="done", stderr="")

        def malformed_exchange(argv, *, cwd, **kwargs):
            request_directory = Path(cwd) / "requests"
            (request_directory / "request-malformed.json").write_text(
                "not-json",
                encoding="utf-8",
            )
            time.sleep(0.05)
            return subprocess.CompletedProcess(argv, 0, stdout="done", stderr="")

        cases = (
            ("executable_missing", executable_missing, "provider_error"),
            ("timeout", timeout, "timeout"),
            ("provider_failure", provider_failure, "provider_error"),
            ("nonzero_exit", nonzero, "provider_error"),
            ("missing_finish", missing_finish, "agent_exited"),
            ("duplicate_finish", duplicate_finish, "agent_exited"),
            ("malformed_action_exchange", malformed_exchange, "runtime_error"),
        )
        for expected_status, runner, terminal_reason in cases:
            with self.subTest(expected_status=expected_status):
                result = run_case(runner)
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.terminal_reason, terminal_reason)
                self.assertNotIn("private executable path", repr(result))
                self.assertNotIn("quota unavailable", repr(result))

    def test_default_runner_adapts_the_exact_process_group_result(self) -> None:
        self.assertTrue(hasattr(codex_adapter, "run_process_group"))
        completed_result = ProcessGroupResult(
            argv0="codex-fixture",
            returncode=0,
            stdout="completed",
            stderr="",
            terminal_status="completed",
        )
        timeout_result = replace(
            completed_result,
            returncode=-15,
            terminal_status="terminated",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch.object(
                codex_adapter,
                "run_process_group",
                return_value=completed_result,
            ) as exact_runner:
                result = subprocess_command_runner(
                    ("codex-fixture", "exec"),
                    cwd=root,
                    env={"PATH": "/bin"},
                    timeout_ms=1_000,
                )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "completed")
            exact_runner.assert_called_once_with(
                ("codex-fixture", "exec"),
                cwd=root,
                env={"PATH": "/bin"},
                timeout_ms=1_000,
            )

            with mock.patch.object(
                codex_adapter,
                "run_process_group",
                return_value=timeout_result,
            ):
                with self.assertRaises(subprocess.TimeoutExpired):
                    subprocess_command_runner(
                        ("codex-fixture", "exec"),
                        cwd=root,
                        env={"PATH": "/bin"},
                        timeout_ms=1_000,
                    )


if __name__ == "__main__":
    unittest.main()
