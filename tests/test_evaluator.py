from __future__ import annotations

import unittest
import json
import tempfile
import textwrap
from pathlib import Path

from op_bench.environment import EnvironmentPreparation
from op_bench.executor import CommandResult
from op_bench.evaluator import Evaluator
from op_bench.task import TaskManifest


class _FakeRemoteExecutor:
    name = "remote_docker"

    def __init__(self) -> None:
        self.sync_count = 0

    def sync_to_remote(self, workspace: Path, timeout_sec: int = 600) -> CommandResult:
        self.sync_count += 1
        return CommandResult(
            command=["sync_to_remote", str(workspace)],
            cwd=str(workspace),
            exit_code=0,
            stdout="",
            stderr="",
            duration_sec=0,
        )

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        command_text = " ".join(command)
        if "test_failing" in command_text:
            exit_code = 1 if self.sync_count >= 2 else 0
        else:
            exit_code = 0
        return CommandResult(
            command=command,
            cwd=str(cwd),
            exit_code=exit_code,
            stdout="",
            stderr="",
            duration_sec=0,
        )

    def collect_environment(self) -> dict[str, object]:
        return {"executor": "remote_docker"}

    def close(self, timeout_sec: int = 30) -> CommandResult | None:
        return None


class _FakeRemoteEnvironmentManager:
    def __init__(self, executor: _FakeRemoteExecutor) -> None:
        self.executor = executor

    def precheck(self, task: TaskManifest) -> None:
        return None

    def prepare(self, task: TaskManifest, workspace: Path) -> EnvironmentPreparation:
        initial_sync = self.executor.sync_to_remote(workspace, timeout_sec=task.timeout_sec)
        return EnvironmentPreparation(
            status="ready",
            executor=self.executor,
            evidence={"executor": "remote_docker"},
            commands=[initial_sync],
        )

    def cleanup(self, preparation: EnvironmentPreparation) -> CommandResult | None:
        return None


class EvaluatorTests(unittest.TestCase):
    def test_baseline_reproduces_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Evaluator().evaluate_baseline(TaskManifest.load(self._fixable_task(Path(tmp)) / "task.json"))
            self.assertEqual(result.status, "baseline_reproduced")
            self.assertEqual(result.fail_to_pass_passed, 0)
            self.assertEqual(result.pass_to_pass_passed, 1)

    def test_gold_resolves_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = Evaluator().evaluate_gold(TaskManifest.load(self._fixable_task(Path(tmp)) / "task.json"))
            self.assertEqual(result.status, "resolved")
            self.assertEqual(result.fail_to_pass_passed, 1)
            self.assertEqual(result.pass_to_pass_passed, 1)

    def test_missing_dependency_is_environment_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")

            task_dir = root / "task"
            artifacts = task_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "test.patch").write_text(
                textwrap.dedent(
                    """\
                    diff --git a/test_env.py b/test_env.py
                    new file mode 100644
                    index 0000000..73a0f22
                    --- /dev/null
                    +++ b/test_env.py
                    @@ -0,0 +1,8 @@
                    +import unittest
                    +
                    +import package_that_is_not_installed
                    +
                    +
                    +class TestEnv(unittest.TestCase):
                    +    def test_import(self):
                    +        self.assertTrue(package_that_is_not_installed)
                    """
                ),
                encoding="utf-8",
            )
            (artifacts / "gold.patch").write_text("", encoding="utf-8")
            manifest = {
                "task_id": "local__missing_dependency",
                "version": "v1",
                "source": {
                    "repo": "local/repo",
                    "local_path": str(source),
                    "base_commit": "local",
                    "checkout_mode": "local-copy",
                },
                "statement": {"title": "missing dep", "body": "body", "labels": []},
                "operator": {
                    "framework": "pytorch",
                    "component": "test",
                    "operator_name": "env",
                    "problem_type": "environment",
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
                "evaluation": {
                    "setup_commands": [],
                    "fail_to_pass": ["test_env.TestEnv.test_import"],
                    "pass_to_pass": ["test_env.TestEnv.test_import"],
                    "test_command": "{python} -m unittest {test}",
                    "timeout_sec": 30,
                },
                "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
                "metadata": {"curation_status": "draft"},
            }
            (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")

            result = Evaluator().evaluate_baseline(TaskManifest.load(task_dir / "task.json"))

            self.assertEqual(result.status, "environment_error")

    def test_remote_workspace_is_resynced_after_local_patches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._remote_patch_sync_task(root)
            executor = _FakeRemoteExecutor()
            environment_manager = _FakeRemoteEnvironmentManager(executor)

            result = Evaluator(environment_manager=environment_manager).evaluate_baseline(
                TaskManifest.load(task_dir / "task.json")
            )

            self.assertEqual(executor.sync_count, 2)
            self.assertEqual(result.status, "baseline_reproduced")

    def _fixable_task(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "op_lib").mkdir()
        (source / "op_lib" / "__init__.py").write_text("from .special import expit\n", encoding="utf-8")
        (source / "op_lib" / "special.py").write_text(
            "from __future__ import annotations\n\n"
            "import math\n\n\n"
            "def expit(value: float) -> float:\n"
            "    if math.isnan(value):\n"
            "        return 0.0\n"
            "    return 1.0 / (1.0 + math.exp(-value))\n",
            encoding="utf-8",
        )

        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "test.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/test_special.py b/test_special.py
                new file mode 100644
                index 0000000..4136fae
                --- /dev/null
                +++ b/test_special.py
                @@ -0,0 +1,12 @@
                +import math
                +import unittest
                +
                +from op_lib import expit
                +
                +
                +class TestSpecialExpit(unittest.TestCase):
                +    def test_nan_is_preserved(self):
                +        self.assertTrue(math.isnan(expit(float("nan"))))
                +
                +    def test_regular_value(self):
                +        self.assertAlmostEqual(expit(0.0), 0.5)
                """
            ),
            encoding="utf-8",
        )
        (artifacts / "gold.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/op_lib/special.py b/op_lib/special.py
                index 3d0f57e..30a970d 100644
                --- a/op_lib/special.py
                +++ b/op_lib/special.py
                @@ -6,4 +6,4 @@
                 def expit(value: float) -> float:
                     if math.isnan(value):
                -        return 0.0
                +        return float("nan")
                     return 1.0 / (1.0 + math.exp(-value))
                """
            ),
            encoding="utf-8",
        )
        manifest = {
            "task_id": "local__fixable",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "local",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "preserve NaN", "body": "expit should preserve NaN", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "expit",
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
            "evaluation": {
                "setup_commands": [],
                "fail_to_pass": ["test_special.TestSpecialExpit.test_nan_is_preserved"],
                "pass_to_pass": ["test_special.TestSpecialExpit.test_regular_value"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {"curation_status": "draft"},
        }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return task_dir

    def _remote_patch_sync_task(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        (source / "test_remote.py").write_text("VALUE = 'old'\n", encoding="utf-8")

        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "test.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/test_remote.py b/test_remote.py
                index 7234b7e..8f0df4c 100644
                --- a/test_remote.py
                +++ b/test_remote.py
                @@ -1 +1 @@
                -VALUE = 'old'
                +VALUE = 'patched'
                """
            ),
            encoding="utf-8",
        )
        (artifacts / "gold.patch").write_text("", encoding="utf-8")
        manifest = {
            "task_id": "remote__patch_sync",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "local",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "remote sync", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "sync",
                "problem_type": "runner",
                "tags": [],
            },
            "environment": {
                "backend": "remote_docker",
                "host": "gpu-a10",
                "tier": "cuda_python_overlay",
                "image": "op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
                "python_executable": "python",
                "hardware": {"requires_gpu": True},
            },
            "evaluation": {
                "setup_commands": [],
                "fail_to_pass": ["test_failing"],
                "pass_to_pass": ["test_passing"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {"curation_status": "draft"},
        }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return task_dir


if __name__ == "__main__":
    unittest.main()
