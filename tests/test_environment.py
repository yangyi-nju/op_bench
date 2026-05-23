from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from op_bench.environment import EnvironmentManager
from op_bench.evaluator import Evaluator
from op_bench.executor import DockerExecutor
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]


class DockerEnvironmentTests(unittest.TestCase):
    def test_docker_executor_builds_mount_command(self) -> None:
        executor = DockerExecutor("example/image:tag", "/repo")
        command = executor.command_for_run(["python", "-m", "pytest"], Path("/tmp/workspace"))

        self.assertEqual(command[:2], ["docker", "run"])
        self.assertIn("--volume", command)
        volume = command[command.index("--volume") + 1]
        self.assertTrue(volume.endswith("/tmp/workspace:/repo"), volume)
        self.assertIn("--workdir", command)
        self.assertIn("/repo", command)
        self.assertEqual(command[-3:], ["python", "-m", "pytest"])

    def test_docker_executor_builds_persistent_container_commands(self) -> None:
        executor = DockerExecutor("example/image:tag", "/repo", container_name="op-bench-test")

        start_command = executor.command_for_start(Path("/tmp/workspace"))
        run_command = executor.command_for_run(["python", "-m", "pytest"], Path("/tmp/workspace"))

        self.assertEqual(start_command[:5], ["docker", "run", "--detach", "--name", "op-bench-test"])
        self.assertIn("--volume", start_command)
        self.assertTrue(start_command[start_command.index("--volume") + 1].endswith("/tmp/workspace:/repo"))
        self.assertEqual(start_command[-3:], ["tail", "-f", "/dev/null"])
        self.assertEqual(
            run_command,
            ["docker", "exec", "--workdir", "/repo", "op-bench-test", "python", "-m", "pytest"],
        )

    def test_docker_executor_can_run_commands_outside_workspace(self) -> None:
        executor = DockerExecutor(
            "example/image:tag",
            "/repo",
            container_name="op-bench-test",
            command_workdir="/tmp",
        )

        command = executor.command_for_run(["python", "-c", "import torch"], Path("/tmp/workspace"))

        self.assertEqual(
            command,
            ["docker", "exec", "--workdir", "/tmp", "op-bench-test", "python", "-c", "import torch"],
        )

    def test_environment_manager_reports_docker_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self._docker_task(Path(tmp))
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = ""
            try:
                preparation = EnvironmentManager().prepare(task, Path(tmp))
            finally:
                os.environ["PATH"] = old_path

            self.assertFalse(preparation.available)
            self.assertEqual(preparation.status, "environment_unavailable")
            self.assertEqual(preparation.error, "docker command not found")
            self.assertFalse(preparation.evidence["docker_available"])

    def test_evaluator_returns_environment_unavailable_before_tests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "bug.py").write_text("def value():\n    return 0\n", encoding="utf-8")
            task = self._docker_task(root, source=source)
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = ""
            try:
                result = Evaluator().evaluate_baseline(task)
            finally:
                os.environ["PATH"] = old_path

            self.assertEqual(result.status, "environment_unavailable")
            self.assertEqual(result.fail_to_pass_passed, 0)

    def test_evaluator_uses_repo_cache_workspace_for_docker_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self._docker_task(Path(tmp))

            workspace_parent = Evaluator()._workspace_parent(task)

            self.assertEqual(workspace_parent, str(ROOT / ".op_bench_cache" / "workspaces"))

    def _docker_task(self, root: Path, source: Path | None = None) -> TaskManifest:
        source = source or root
        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "test.patch").write_text("", encoding="utf-8")
        (artifacts / "gold.patch").write_text("", encoding="utf-8")
        manifest = {
            "task_id": "docker__missing",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "docker", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "op",
                "problem_type": "environment",
                "tags": [],
            },
            "environment": {
                "backend": "docker",
                "tier": "cpu-deterministic",
                "image": "op-bench/test:latest",
                "workspace_dir": "/workspace",
                "python_executable": "python",
                "python_version": "3",
                "os": "linux",
                "build_mode": "editable-python",
                "hardware": {"device": "cpu", "min_memory_gb": 1},
                "dependencies": [],
                "preflight_commands": ["{python} --version"],
            },
            "agent_visible": {
                "repo_setup_commands": [],
                "known_constraints": [],
                "allowed_test_commands": ["{python} -m unittest {test}"],
            },
            "evaluation": {
                "setup_commands": [],
                "fail_to_pass": ["test_missing"],
                "pass_to_pass": ["test_missing"],
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
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return TaskManifest.load(task_dir / "task.json")


if __name__ == "__main__":
    unittest.main()
