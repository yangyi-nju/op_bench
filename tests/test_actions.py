from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from op_bench.actions import WorkspaceActions
from op_bench.executor import LocalExecutor
from op_bench.task import TaskManifest


class WorkspaceActionsTests(unittest.TestCase):
    def test_file_actions_are_scoped_to_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            actions = WorkspaceActions(
                task=self._task(Path(tmp), workspace),
                workspace=workspace,
                command_executor=LocalExecutor(),
            )

            actions.write_file("pkg/value.py", "VALUE = 1\n")

            self.assertEqual(actions.read_file("pkg/value.py"), "VALUE = 1\n")
            with self.assertRaises(ValueError):
                actions.read_file("../outside.py")

    def test_command_and_diff_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            self._run_git(["init"], workspace)
            self._run_git(["config", "user.name", "Test"], workspace)
            self._run_git(["config", "user.email", "test@example.com"], workspace)
            (workspace / "value.py").write_text("VALUE = 0\n", encoding="utf-8")
            self._run_git(["add", "value.py"], workspace)
            self._run_git(["commit", "-m", "base"], workspace)
            actions = WorkspaceActions(
                task=self._task(Path(tmp), workspace),
                workspace=workspace,
                command_executor=LocalExecutor(),
            )

            result = actions.run_command([sys.executable, "-c", "print('ok')"], timeout_sec=5)
            actions.write_file("value.py", "VALUE = 1\n")
            diff = actions.git_diff()

            self.assertEqual(result.exit_code, 0)
            self.assertIn("ok", result.stdout)
            self.assertIn("-VALUE = 0", diff)
            self.assertIn("+VALUE = 1", diff)

    def test_git_diff_is_limited_to_source_loading_overlay_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir()
            self._run_git(["init"], workspace)
            self._run_git(["config", "user.name", "Test"], workspace)
            self._run_git(["config", "user.email", "test@example.com"], workspace)
            (workspace / "pkg").mkdir()
            (workspace / "tests").mkdir()
            (workspace / "pkg" / "value.py").write_text("VALUE = 0\n", encoding="utf-8")
            (workspace / "tests" / "test_value.py").write_text("def test_value(): pass\n", encoding="utf-8")
            self._run_git(["add", "pkg/value.py", "tests/test_value.py"], workspace)
            self._run_git(["commit", "-m", "base"], workspace)
            actions = WorkspaceActions(
                task=self._task(Path(tmp), workspace, overlay_paths=["pkg/value.py"]),
                workspace=workspace,
                command_executor=LocalExecutor(),
            )

            actions.write_file("pkg/value.py", "VALUE = 1\n")
            actions.write_file("tests/test_value.py", "def test_value(): assert False\n")
            diff = actions.git_diff()

            self.assertIn("diff --git a/pkg/value.py b/pkg/value.py", diff)
            self.assertNotIn("diff --git a/tests/test_value.py b/tests/test_value.py", diff)

    def _task(self, root: Path, workspace: Path, overlay_paths: list[str] | None = None) -> TaskManifest:
        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "gold.patch").write_text("", encoding="utf-8")
        (artifacts / "test.patch").write_text("", encoding="utf-8")
        manifest = {
            "task_id": "actions__fixture",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(workspace),
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "actions", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "op",
                "problem_type": "tooling",
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
                "allowed_test_commands": ["{python} -m unittest {test}"],
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
                "curation_status": "draft",
                "deterministic": True,
                "estimated_runtime_min": 1,
            },
        }
        if overlay_paths is not None:
            manifest["environment"]["source_loading"] = {
                "mode": "python_overlay",
                "installed_package": "pkg",
                "overlay_paths": overlay_paths,
            }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return TaskManifest.load(task_dir / "task.json")

    def _run_git(self, args: list[str], cwd: Path) -> None:
        subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
