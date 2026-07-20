from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

from op_bench.runtime.adapters import AdapterActionChannel, AdapterContext
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.process_group import ProcessGroupCleanupError, ProcessGroupResult
from op_bench.runtime.task_view import (
    AgentLaunchInput,
    agent_task_view_identity,
    project_agent_task_view,
)
from tests.test_runtime_contracts import budget_policy, capability_policy, full_task_spec
from tests.test_runtime_wire_contracts import action_observation

try:
    from op_bench.runtime.codex_mcp_adapter import CodexMcpCanonicalAdapter
except ImportError:
    CodexMcpCanonicalAdapter = None


MODEL_ID = "gpt-5.6-sol"
CLI_VERSION = "codex-cli 0.145.0-alpha.18"


def decode_mcp_config(argv: tuple[str, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    for index, item in enumerate(argv):
        if item != "-c":
            continue
        key, encoded = argv[index + 1].split("=", 1)
        values[key] = json.loads(encoded)
    return values


def initialize_message(request_id: int = 1) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "fake-codex", "version": "1"},
        },
    }


def call_message(
    request_id: int,
    name: str,
    arguments: dict[str, object],
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


class CodexMcpCanonicalAdapterTests(unittest.TestCase):
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

    def context(self, execute) -> AdapterContext:
        channel = AdapterActionChannel(execute)
        client = channel.start()
        self.addCleanup(channel.close)
        return AdapterContext(
            launch_input=self.launch_input(),
            session_id="session-codex-mcp-adapter",
            action_client=client,
        )

    def adapter(self, runner):
        self.assertIsNotNone(CodexMcpCanonicalAdapter)
        return CodexMcpCanonicalAdapter(
            runner,
            codex_binary="codex-fixture",
            model_id=MODEL_ID,
            codex_cli_version=CLI_VERSION,
        )

    def run_server_from_codex_argv(
        self,
        argv: tuple[str, ...],
        *,
        cwd: Path,
        env: dict[str, str],
        messages: tuple[dict[str, object], ...],
    ) -> subprocess.CompletedProcess[str]:
        config = decode_mcp_config(argv)
        command = config["mcp_servers.opbench.command"]
        arguments = config["mcp_servers.opbench.args"]
        self.assertIsInstance(command, str)
        self.assertIsInstance(arguments, list)
        token_descriptor = int(
            arguments[arguments.index("--bridge-token-fd") + 1]
        )
        return subprocess.run(
            (command, *arguments),
            cwd=cwd,
            env=env,
            input="".join(f"{canonical_json(message)}\n" for message in messages),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
            pass_fds=(token_descriptor,),
        )

    def successful_messages(self) -> tuple[dict[str, object], ...]:
        calls = (
            ("workspace_list", {}),
            ("workspace_read", {"path": "src/operator.py"}),
            ("workspace_apply_patch", {"patch": "diff --git a/a b/a\n"}),
            ("test_run", {"selector_id": "public-smoke"}),
            ("vcs_diff", {}),
            ("session_finish", {}),
        )
        return (
            initialize_message(),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            *tuple(
                call_message(index, name, arguments)
                for index, (name, arguments) in enumerate(calls, start=3)
            ),
        )

    def test_fake_codex_completes_the_real_mcp_action_path(self) -> None:
        observations: list[dict[str, object]] = []
        captured: dict[str, object] = {}

        def execute(payload):
            observations.append(payload)
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
                data={"action_name": payload["action_name"]},
            ).to_dict()

        def runner(argv, *, cwd, env, timeout_ms, pass_fds):
            selected = tuple(argv)
            captured["argv"] = selected
            captured["cwd"] = cwd
            captured["env"] = dict(env)
            captured["timeout_ms"] = timeout_ms
            captured["pass_fds"] = tuple(pass_fds)
            captured["prompt"] = selected[-1]
            config = decode_mcp_config(selected)
            server_arguments = config["mcp_servers.opbench.args"]
            launcher_path = Path(server_arguments[0])
            action_client = Path(
                server_arguments[server_arguments.index("--action-client") + 1]
            )
            captured["launcher_path"] = launcher_path
            captured["action_client"] = action_client
            captured["launcher"] = launcher_path.read_text(encoding="utf-8")
            direct = subprocess.run(
                (
                    sys.executable,
                    str(action_client),
                    "workspace_list",
                    "--arguments",
                    "{}",
                ),
                cwd=cwd,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            captured["direct_returncode"] = direct.returncode
            captured["direct_stderr"] = direct.stderr
            server = self.run_server_from_codex_argv(
                selected,
                cwd=Path(cwd),
                env=dict(env),
                messages=self.successful_messages(),
            )
            self.assertEqual(server.returncode, 0, server.stderr)
            self.assertEqual(server.stderr, "")
            captured["token_after_server"] = os.read(pass_fds[0], 1)
            captured["rpc"] = [json.loads(line) for line in server.stdout.splitlines()]
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=0,
                stdout="agent complete",
                stderr="",
                terminal_status="completed",
            )

        result = self.adapter(runner).run(self.context(execute))

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.terminal_reason, "agent_finished")
        self.assertEqual(result.observation_count, 6)
        self.assertEqual(result.finish_count, 1)
        self.assertIsNotNone(result.adapter_trace)
        self.assertEqual(result.adapter_trace.model_id, MODEL_ID)
        self.assertEqual(result.adapter_trace.codex_cli_version, CLI_VERSION)
        self.assertEqual(result.adapter_trace.negotiated_protocol_version, "2025-06-18")
        self.assertEqual(result.adapter_trace.initialize_count, 1)
        self.assertEqual(result.adapter_trace.tools_list_count, 1)
        self.assertEqual(result.adapter_trace.tools_call_count, 6)
        self.assertEqual(result.adapter_trace.protocol_error_count, 0)
        self.assertEqual(result.adapter_trace.server_terminal_status, "client_closed")
        self.assertEqual(
            [item["action_name"] for item in observations],
            [
                "workspace_list",
                "workspace_read",
                "workspace_apply_patch",
                "test_run",
                "vcs_diff",
                "session_finish",
            ],
        )

        argv = captured["argv"]
        self.assertEqual(argv[0:2], ("codex-fixture", "exec"))
        for required in (
            "--ephemeral",
            "--ignore-user-config",
            "--model",
            MODEL_ID,
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
        ):
            self.assertIn(required, argv)
        config = decode_mcp_config(argv)
        self.assertEqual(config["mcp_servers.opbench.env"], {})
        self.assertEqual(len(config), 3)
        server_arguments = config["mcp_servers.opbench.args"]
        self.assertNotIn("--bridge-token", server_arguments)
        self.assertIn("--bridge-token-fd", server_arguments)
        token_descriptor = int(
            server_arguments[server_arguments.index("--bridge-token-fd") + 1]
        )
        self.assertEqual(captured["pass_fds"], (token_descriptor,))
        self.assertEqual(captured["token_after_server"], b"")
        self.assertNotRegex(" ".join(str(item) for item in server_arguments), r"[0-9a-f]{64}")
        self.assertNotEqual(Path(captured["cwd"]).parent, Path(captured["launcher_path"]).parent)
        self.assertEqual(Path(captured["cwd"]).name, "agent")
        self.assertEqual(Path(captured["launcher_path"]).parent.name, "controller")
        self.assertTrue(
            Path(captured["action_client"]).is_relative_to(
                Path(captured["launcher_path"]).parent
            )
        )
        self.assertNotEqual(captured["direct_returncode"], 0)
        self.assertIn("transport authentication failed", captured["direct_stderr"])
        prompt = str(captured["prompt"])
        self.assertIn("workspace_read", prompt)
        self.assertIn("session_finish", prompt)
        self.assertIn("MCP", prompt)
        self.assertNotIn("opbench_action.py", prompt)
        self.assertNotIn("from op_bench", str(captured["launcher"]))
        self.assertNotIn(str(captured["cwd"]), repr(result))

    def test_failure_and_terminal_modes_have_stable_public_attribution(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        def executable_missing(argv, **kwargs):
            raise FileNotFoundError("private executable")

        def provider_failure(argv, **kwargs):
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=1,
                stdout="",
                stderr="provider_error: quota unavailable",
                terminal_status="completed",
            )

        def timeout(argv, **kwargs):
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=-15,
                stdout="",
                stderr="",
                terminal_status="terminated",
            )

        cases = (
            ("executable_missing", executable_missing, "provider_error", "not_started"),
            ("provider_failure", provider_failure, "provider_error", "start_failed"),
            ("timeout", timeout, "timeout", "terminated"),
        )
        for expected_status, runner, terminal_reason, server_status in cases:
            with self.subTest(expected_status=expected_status):
                result = self.adapter(runner).run(self.context(execute))
                self.assertEqual(result.status, expected_status)
                self.assertEqual(result.terminal_reason, terminal_reason)
                self.assertEqual(result.adapter_trace.server_terminal_status, server_status)
                self.assertNotIn("private executable", repr(result))
                self.assertNotIn("quota unavailable", repr(result))

    def test_cleanup_uncertainty_is_never_synthesized_as_killed(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        def uncertain_cleanup(argv, **kwargs):
            raise ProcessGroupCleanupError(
                "exact process group cleanup did not converge",
                process_group_id=6767,
            )

        with self.assertRaises(ProcessGroupCleanupError) as raised:
            self.adapter(uncertain_cleanup).run(self.context(execute))

        self.assertEqual(raised.exception.process_group_id, 6767)

    def test_missing_duplicate_and_recoverable_action_error_are_not_mcp_crashes(self) -> None:
        def execute(payload):
            failed = payload["action_name"] == "test_run"
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
                ok=not failed,
                error_code="selector_denied" if failed else "ok",
                message="selector denied" if failed else "completed",
            ).to_dict()

        def runner_for(messages):
            def runner(argv, *, cwd, env, **kwargs):
                server = self.run_server_from_codex_argv(
                    tuple(argv),
                    cwd=Path(cwd),
                    env=dict(env),
                    messages=messages,
                )
                self.assertEqual(server.returncode, 0, server.stderr)
                return ProcessGroupResult(
                    argv0="codex-fixture",
                    returncode=0,
                    stdout="done",
                    stderr="",
                    terminal_status="completed",
                )

            return runner

        prefix = (
            initialize_message(),
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        missing = self.adapter(runner_for(prefix)).run(self.context(execute))
        duplicate = self.adapter(
            runner_for(
                (
                    *prefix,
                    call_message(3, "session_finish", {}),
                    call_message(4, "session_finish", {}),
                )
            )
        ).run(self.context(execute))
        recovered = self.adapter(
            runner_for(
                (
                    *prefix,
                    call_message(3, "test_run", {"selector_id": "bad"}),
                    call_message(4, "session_finish", {}),
                )
            )
        ).run(self.context(execute))

        self.assertEqual(missing.status, "missing_finish")
        self.assertEqual(duplicate.status, "duplicate_finish")
        self.assertEqual(recovered.status, "completed")
        self.assertEqual(recovered.adapter_trace.tools_call_count, 2)
        self.assertEqual(recovered.adapter_trace.protocol_error_count, 0)

    def test_fatal_protocol_error_cannot_be_recovered_by_later_finish(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        def runner(argv, *, cwd, env, pass_fds, **kwargs):
            selected = tuple(argv)
            config = decode_mcp_config(selected)
            command = config["mcp_servers.opbench.command"]
            arguments = config["mcp_servers.opbench.args"]
            messages = "{not-json\n" + "".join(
                f"{canonical_json(message)}\n"
                for message in self.successful_messages()
            )
            server = subprocess.run(
                (command, *arguments),
                cwd=cwd,
                env=env,
                input=messages,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                pass_fds=tuple(pass_fds),
            )
            self.assertEqual(server.returncode, 1)
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=server.returncode,
                stdout=server.stdout,
                stderr=server.stderr,
                terminal_status="completed",
            )

        result = self.adapter(runner).run(self.context(execute))

        self.assertEqual(result.status, "malformed_action_exchange")
        self.assertEqual(result.terminal_reason, "runtime_error")
        self.assertEqual(result.observation_count, 0)
        self.assertEqual(result.adapter_trace.protocol_error_count, 1)
        self.assertEqual(result.adapter_trace.server_terminal_status, "protocol_failed")

    def test_missing_initialize_or_tampered_trace_is_malformed_exchange(self) -> None:
        def execute(payload):
            return replace(
                action_observation(),
                session_id=payload["session_id"],
                action_id=payload["action_id"],
            ).to_dict()

        def no_initialize(argv, *, cwd, env, **kwargs):
            server = self.run_server_from_codex_argv(
                tuple(argv),
                cwd=Path(cwd),
                env=dict(env),
                messages=({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {}},),
            )
            self.assertEqual(server.returncode, 0, server.stderr)
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=0,
                stdout="done",
                stderr="",
                terminal_status="completed",
            )

        def tampered(argv, *, cwd, **kwargs):
            config = decode_mcp_config(tuple(argv))
            arguments = config["mcp_servers.opbench.args"]
            trace_path = Path(arguments[arguments.index("--trace-path") + 1])
            trace_path.write_text("not-json\n", encoding="utf-8")
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=0,
                stdout="done",
                stderr="",
                terminal_status="completed",
            )

        def replaced_launcher(argv, *, cwd, env, **kwargs):
            selected = tuple(argv)
            server = self.run_server_from_codex_argv(
                selected,
                cwd=Path(cwd),
                env=dict(env),
                messages=self.successful_messages(),
            )
            self.assertEqual(server.returncode, 0, server.stderr)
            config = decode_mcp_config(selected)
            launcher = Path(config["mcp_servers.opbench.args"][0])
            launcher.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
            return ProcessGroupResult(
                argv0="codex-fixture",
                returncode=0,
                stdout="done",
                stderr="",
                terminal_status="completed",
            )

        for runner in (no_initialize, tampered, replaced_launcher):
            with self.subTest(runner=runner):
                result = self.adapter(runner).run(self.context(execute))
                self.assertEqual(result.status, "malformed_action_exchange")
                self.assertEqual(result.terminal_reason, "runtime_error")
                self.assertEqual(result.adapter_trace.server_terminal_status, "protocol_failed")


if __name__ == "__main__":
    unittest.main()
