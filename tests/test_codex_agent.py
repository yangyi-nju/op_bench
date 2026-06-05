from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from op_bench.actions import WorkspaceActions
from op_bench.agents import CodexActionBridgeAgent, agent_by_name
from op_bench.executor import LocalExecutor
from op_bench.task import TaskManifest


class CodexActionBridgeAgentTests(unittest.TestCase):
    def test_codex_action_bridge_runs_codex_in_scratch_through_action_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            self._run_git(["init"], workspace)
            self._run_git(["config", "user.name", "Test"], workspace)
            self._run_git(["config", "user.email", "test@example.com"], workspace)
            (workspace / "bug.py").write_text("value = 'bug'\n", encoding="utf-8")
            self._run_git(["add", "bug.py"], workspace)
            self._run_git(["commit", "-m", "base"], workspace)

            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "issue.md").write_text("Fix bug.py", encoding="utf-8")
            (task_dir / "task.json").write_text(json.dumps(self._manifest()), encoding="utf-8")
            task = TaskManifest.load(task_dir / "task.json")
            actions = WorkspaceActions(task=task, workspace=workspace, command_executor=LocalExecutor())
            captured: dict[str, object] = {}
            real_run = subprocess.run

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                if command[0] != "codex":
                    return real_run(command, **kwargs)
                captured["command"] = command
                captured["cwd"] = kwargs.get("cwd")
                cd_index = command.index("--cd")
                scratch = Path(command[cd_index + 1])
                action_cli = scratch / "opbench_action.py"
                read = real_run(
                    [str(action_cli), "read_file", "bug.py"],
                    cwd=str(scratch),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(read.returncode, 0, read.stderr)
                command_result = real_run(
                    [str(action_cli), "run_command", f"{sys.executable} -c \"print('shell-ok')\""],
                    cwd=str(scratch),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(command_result.returncode, 0, command_result.stderr)
                self.assertIn("shell-ok", command_result.stdout)
                write = real_run(
                    [str(action_cli), "write_file", "bug.py"],
                    input="value = 'fixed'\n",
                    cwd=str(scratch),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(write.returncode, 0, write.stderr)
                return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

            with mock.patch("op_bench.agents.subprocess.run", side_effect=fake_run):
                output = CodexActionBridgeAgent().produce_patch(task, root / "agent-output", workspace=workspace, actions=actions)

            command = captured["command"]
            self.assertIsInstance(command, list)
            command_list = command
            self.assertIn("--skip-git-repo-check", command_list)
            self.assertIn("--sandbox", command_list)
            self.assertIn("workspace-write", command_list)
            cd_index = command_list.index("--cd")
            self.assertNotEqual(Path(str(command_list[cd_index + 1])).resolve(), workspace.resolve())
            self.assertEqual(output.agent_name, "codex_action_bridge")
            self.assertEqual(output.metadata["runtime_boundary"], "op_bench_action_interface_file_cli")
            self.assertEqual(output.metadata["integrity_status"], "clean")
            self.assertEqual(output.metadata["action_count"], 3)
            self.assertIn("timeout_sec", output.metadata)
            patch = output.patch_path.read_text(encoding="utf-8")
            self.assertIn("-value = 'bug'", patch)
            self.assertIn("+value = 'fixed'", patch)

    def test_agent_registry_loads_codex_action_bridge(self) -> None:
        self.assertIsInstance(agent_by_name("codex_action_bridge"), CodexActionBridgeAgent)

    def _manifest(self) -> dict[str, object]:
        return {
            "task_id": "codex_agent_fixture",
            "version": "v1",
            "source": {
                "pr_url": "https://github.com/local/repo/pull/1",
                "issue_url": "https://github.com/local/repo/issues/1",
                "repo": "local/repo",
                "issue_number": 1,
                "pr_number": 1,
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
                "local_path": "../workspace",
            },
            "statement": {"title": "Fix bug", "body": "Fix bug.py", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "bug",
                "problem_type": "operator-behavior",
                "tags": [],
            },
            "environment": {
                "tier": "cpu-deterministic",
                "image": "local",
                "python_version": "3",
                "os": "local",
                "build_mode": "editable-python",
                "hardware": {"device": "cpu", "min_memory_gb": 1},
                "dependencies": [],
            },
            "agent_visible": {
                "repo_setup_commands": [],
                "known_constraints": [],
                "allowed_test_commands": [],
            },
            "evaluation": {
                "setup_commands": [],
                "fail_to_pass": ["unused"],
                "pass_to_pass": ["unused"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {
                "difficulty": "easy",
                "curation_status": "verified",
                "deterministic": True,
                "estimated_runtime_min": 1,
            },
        }

    def _run_git(self, args: list[str], cwd: Path) -> None:
        import subprocess

        subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
