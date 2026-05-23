from __future__ import annotations

import json
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from op_bench.executor import CommandResult


ROOT = Path(__file__).resolve().parents[1]


class RunExperimentTests(unittest.TestCase):
    def test_cli_runs_noop_and_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            task_dir = self._fixable_git_task(root)

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--task",
                    str(task_dir),
                    "--agent",
                    "noop",
                    "--agent",
                    "gold",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["agents"]["gold"]["resolved_rate"], 1.0)
            self.assertEqual(summary["agents"]["noop"]["resolved_rate"], 0.0)

    def test_cli_can_repeat_agent_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            task_dir = self._fixable_git_task(root)

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--task",
                    str(task_dir),
                    "--agent",
                    "gold",
                    "--agent-repeat",
                    "2",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["agents"]["gold"]["total"], 2)
            self.assertEqual(summary["agents"]["gold"]["resolved"], 2)
            records = [
                json.loads(line)
                for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            agent_records = [record for record in records if record["agent"] == "gold"]
            self.assertEqual([record["attempt"] for record in agent_records], [1, 2])
            self.assertNotEqual(agent_records[0]["patch_path"], agent_records[1]["patch_path"])

    def test_cli_skips_agents_when_baseline_is_not_reproduced(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._baseline_not_reproduced_task(root)
            output_dir = root / "out"

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--task",
                    str(task_dir),
                    "--agent",
                    "noop",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["status"], "baseline_not_reproduced")
            self.assertEqual(records[1]["status"], "task_not_reproduced")
            self.assertEqual(records[1]["baseline_status"], "baseline_not_reproduced")

    def test_cli_accepts_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = self._baseline_not_reproduced_task(root)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            dataset_path = dataset_dir / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "local_dataset",
                        "version": "v0",
                        "status": "draft",
                        "tasks": [
                            {
                                "task_id": "local__not_reproduced",
                                "task_path": str(task_dir),
                                "admission_status": "draft",
                                "environment_status": "pending",
                                "source_status": "pending",
                                "replay_status": "pending",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "out"

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--dataset",
                    str(dataset_path),
                    "--agent",
                    "noop",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[0]["task_id"], "local__not_reproduced")
            self.assertEqual(records[1]["agent"], "noop")

    def test_cli_can_filter_dataset_to_verified_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            draft_task_dir = self._baseline_not_reproduced_task(root / "draft", task_id="local__draft")
            verified_task_dir = self._baseline_not_reproduced_task(root / "verified", task_id="local__verified")
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            dataset_path = dataset_dir / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "local_dataset",
                        "version": "v0",
                        "status": "draft",
                        "tasks": [
                            {
                                "task_id": "local__draft",
                                "task_path": str(draft_task_dir),
                                "admission_status": "draft",
                                "environment_status": "pending",
                                "source_status": "pending",
                                "replay_status": "pending",
                            },
                            {
                                "task_id": "local__verified",
                                "task_path": str(verified_task_dir),
                                "admission_status": "verified",
                                "environment_status": "ready",
                                "source_status": "ready",
                                "replay_status": "verified",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output_dir = root / "out"

            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--dataset",
                    str(dataset_path),
                    "--verified-only",
                    "--agent",
                    "noop",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            records = [
                json.loads(line)
                for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({record["task_id"] for record in records}, {"local__verified"})

    def test_agent_runtime_failure_record_includes_environment_cleanup(self) -> None:
        import scripts.run_experiment as run_experiment

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "out"
            cleanup_result = CommandResult(
                command=["docker", "rm", "-f", "op-bench-test"],
                cwd="",
                exit_code=0,
                stdout="op-bench-test\n",
                stderr="",
                duration_sec=0.1,
            )
            preparation_command = CommandResult(
                command=["docker", "run", "--detach", "--name", "op-bench-test"],
                cwd=str(root),
                exit_code=0,
                stdout="container-id\n",
                stderr="",
                duration_sec=0.1,
            )

            class FakeEnvironmentManager:
                def prepare(self, task, workspace):
                    return SimpleNamespace(
                        available=True,
                        evidence={"executor": "docker"},
                        executor=SimpleNamespace(name="docker"),
                        commands=[preparation_command],
                        commands_as_dicts=lambda: [preparation_command.to_dict()],
                    )

                def cleanup(self, preparation):
                    return cleanup_result

            class FakeBaseline:
                status = "baseline_reproduced"

                def to_dict(self):
                    return {
                        "task_id": "local__fixable",
                        "mode": "baseline",
                        "status": "baseline_reproduced",
                        "fail_to_pass_total": 1,
                        "fail_to_pass_passed": 0,
                        "pass_to_pass_total": 1,
                        "pass_to_pass_passed": 1,
                        "duration_sec": 0.0,
                        "environment": {},
                        "commands": [],
                    }

            class FakeEvaluator:
                def __init__(self, environment_manager):
                    pass

                def evaluate_baseline(self, task):
                    return FakeBaseline()

                def prepare_workspace(self, task, workspace):
                    workspace.mkdir(parents=True, exist_ok=True)
                    return None

            task_dir = self._fixable_git_task(root)
            with mock.patch.object(run_experiment, "EnvironmentManager", FakeEnvironmentManager), mock.patch.object(
                run_experiment, "Evaluator", FakeEvaluator
            ):
                exit_code = run_experiment.main(
                    [
                        "--task",
                        str(task_dir),
                        "--agent",
                        "codex",
                        "--output-dir",
                        str(output_dir),
                    ]
                )

            self.assertEqual(exit_code, 0)
            records = [
                json.loads(line)
                for line in (output_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            agent_record = next(record for record in records if record["agent"] == "codex")
            self.assertEqual(agent_record["status"], "agent_runtime_unsupported")
            self.assertEqual(agent_record["commands"][-1]["command"], ["docker", "rm", "-f", "op-bench-test"])

    def _baseline_not_reproduced_task(self, root: Path, task_id: str = "local__not_reproduced") -> Path:
        source = root / "source"
        source.mkdir(parents=True)
        (source / "test_ok.py").write_text(
            textwrap.dedent(
                """\
                import unittest


                class TestOk(unittest.TestCase):
                    def test_ok(self):
                        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "gold.patch").write_text("", encoding="utf-8")
        (artifacts / "test.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/test_extra.py b/test_extra.py
                new file mode 100644
                index 0000000..1d0b6fc
                --- /dev/null
                +++ b/test_extra.py
                @@ -0,0 +1,6 @@
                +import unittest
                +
                +
                +class TestExtra(unittest.TestCase):
                +    def test_extra(self):
                +        self.assertTrue(True)
                """
            ),
            encoding="utf-8",
        )
        manifest = {
            "task_id": task_id,
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "not reproduced", "body": "body", "labels": []},
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
                "fail_to_pass": ["test_ok.TestOk.test_ok"],
                "pass_to_pass": ["test_ok.TestOk.test_ok"],
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
        return task_dir

    def _fixable_git_task(self, root: Path) -> Path:
        source = root / "fixable_source"
        source.mkdir()
        self._run_git(["init"], source)
        self._run_git(["config", "user.name", "Test"], source)
        self._run_git(["config", "user.email", "test@example.com"], source)
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
        self._run_git(["add", "op_lib"], source)
        self._run_git(["commit", "-m", "base"], source)

        task_dir = root / "fixable_task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
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
        (artifacts / "test.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/tests/test_special.py b/tests/test_special.py
                new file mode 100644
                index 0000000..4af1891
                --- /dev/null
                +++ b/tests/test_special.py
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
        manifest = {
            "task_id": "local__fixable",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "fix nan", "body": "Fix expit NaN handling", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "expit",
                "problem_type": "numerical",
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
                "fail_to_pass": ["tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
                "pass_to_pass": ["tests.test_special.TestSpecialExpit.test_regular_value"],
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
        return task_dir

    def _run_git(self, args: list[str], cwd: Path) -> None:
        subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
