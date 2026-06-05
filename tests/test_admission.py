from __future__ import annotations

import json
import tempfile
import textwrap
import unittest
from datetime import datetime, timezone
from pathlib import Path

from op_bench.admission import AdmissionRunner
from op_bench.evaluator import EvaluationResult
from op_bench.task import TaskManifest
from scripts.run_admission import main as run_admission_main


class FakeEvaluator:
    def __init__(self, baseline_status: str, gold_status: str = "resolved") -> None:
        self.baseline_status = baseline_status
        self.gold_status = gold_status
        self.gold_calls = 0

    def evaluate_baseline(self, task: TaskManifest) -> EvaluationResult:
        return self._result(task, "baseline", self.baseline_status)

    def evaluate_gold(self, task: TaskManifest) -> EvaluationResult:
        self.gold_calls += 1
        return self._result(task, "gold", self.gold_status)

    def _result(self, task: TaskManifest, mode: str, status: str) -> EvaluationResult:
        baseline = mode == "baseline"
        reproduced = status == "baseline_reproduced"
        resolved = status == "resolved"
        return EvaluationResult(
            task_id=task.task_id,
            mode=mode,
            status=status,
            fail_to_pass_total=1,
            fail_to_pass_passed=0 if baseline and reproduced else int(resolved),
            pass_to_pass_total=1,
            pass_to_pass_passed=1 if reproduced or resolved else 0,
            duration_sec=1.5,
            environment={"image_id": "sha256:runtime"},
            commands=[],
        )


class AdmissionRunnerTests(unittest.TestCase):
    def test_verified_decision_records_replay_and_asset_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(Path(tmp))
            evaluator = FakeEvaluator("baseline_reproduced", "resolved")

            evidence = AdmissionRunner(evaluator=evaluator, now=self._now).run(task)

            self.assertEqual(evidence.decision, "verified")
            self.assertTrue(evidence.verified)
            self.assertEqual(evidence.failure_classification, None)
            self.assertEqual(evidence.baseline["status"], "baseline_reproduced")
            self.assertEqual(evidence.gold["status"], "resolved")
            self.assertEqual(evidence.environment["id"], "pytorch-cpu")
            self.assertEqual(evidence.environment["runtime_tier"], "cpu_python_overlay")
            self.assertEqual(evidence.source["id"], "pytorch-base")
            self.assertTrue(evidence.task_manifest_hash.startswith("sha256:"))
            self.assertEqual(evaluator.gold_calls, 1)

    def test_environment_failure_blocks_without_running_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = FakeEvaluator("environment_unavailable")

            evidence = AdmissionRunner(evaluator=evaluator, now=self._now).run(self._task(Path(tmp)))

            self.assertEqual(evidence.decision, "blocked_environment")
            self.assertEqual(evidence.failure_classification, "environment_unavailable")
            self.assertIsNone(evidence.gold)
            self.assertEqual(evaluator.gold_calls, 0)

    def test_non_reproduced_baseline_is_not_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = FakeEvaluator("baseline_not_reproduced")

            evidence = AdmissionRunner(evaluator=evaluator, now=self._now).run(self._task(Path(tmp)))

            self.assertEqual(evidence.decision, "not_reproduced")
            self.assertEqual(evidence.failure_classification, "baseline_not_reproduced")
            self.assertIsNone(evidence.gold)

    def test_failed_gold_is_classified_and_bundle_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evaluator = FakeEvaluator("baseline_reproduced", "fail_to_pass_failed")
            runner = AdmissionRunner(evaluator=evaluator, now=self._now)
            evidence = runner.run(self._task(root))

            runner.write_bundle(evidence, root / "run")

            self.assertEqual(evidence.decision, "gold_failed")
            self.assertEqual(evidence.failure_classification, "fail_to_pass_failed")
            written = json.loads((root / "run/evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(written["admission"]["decision"], "gold_failed")
            self.assertTrue((root / "run/baseline.log").exists())
            self.assertTrue((root / "run/gold.log").exists())
            self.assertTrue((root / "run/environment.json").exists())
            self.assertTrue((root / "run/source.json").exists())

    def test_cli_admits_local_task_and_writes_stable_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self._local_fixable_task(root)
            output = root / "admission-run"

            exit_code = run_admission_main(
                [
                    "--task",
                    str(task.task_dir),
                    "--output-dir",
                    str(output),
                    "--write-task-evidence",
                    "--quiet",
                ]
            )

            self.assertEqual(exit_code, 0)
            evidence = json.loads((output / "evidence.json").read_text(encoding="utf-8"))
            stable = json.loads((task.task_dir / "admission/evidence.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["admission"]["decision"], "verified")
            self.assertEqual(evidence["baseline"]["status"], "baseline_reproduced")
            self.assertEqual(evidence["gold"]["status"], "resolved")
            self.assertIn("commands", evidence["baseline"])
            self.assertNotIn("commands", stable["baseline"])
            self.assertNotIn("observed", stable["environment"])
            self.assertNotIn("snapshot_path", stable["source"])

    def _now(self) -> datetime:
        return datetime(2026, 6, 4, tzinfo=timezone.utc)

    def _task(self, root: Path) -> TaskManifest:
        task_dir = root / "task"
        task_dir.mkdir(parents=True)
        manifest = {
            "task_id": "pytorch__fixture",
            "version": "v1",
            "environment_ref": "pytorch-cpu",
            "runtime_tier": "cpu_python_overlay",
            "source_ref": "pytorch-base",
            "source": {
                "repo": "pytorch/pytorch",
                "base_commit": "abcdef1",
                "snapshot_path": "snapshot/source",
                "snapshot_hash": "sha256:source",
            },
            "environment": {
                "backend": "docker",
                "tier": "cpu-deterministic",
                "image": "op-bench/pytorch-cpu:test",
                "image_digest": "sha256:image",
                "platform": "linux/amd64",
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "evaluation": {
                "fail_to_pass": ["test_fail"],
                "pass_to_pass": ["test_pass"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "metadata": {"curation_status": "draft"},
        }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return TaskManifest.load(task_dir / "task.json")

    def _local_fixable_task(self, root: Path) -> TaskManifest:
        source = root / "source"
        source.mkdir()
        (source / "bug.py").write_text("def value():\n    return 0\n", encoding="utf-8")
        task_dir = root / "local-task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "test.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/test_bug.py b/test_bug.py
                new file mode 100644
                index 0000000..5070fa8
                --- /dev/null
                +++ b/test_bug.py
                @@ -0,0 +1,10 @@
                +import unittest
                +from bug import value
                +
                +
                +class TestBug(unittest.TestCase):
                +    def test_value(self):
                +        self.assertEqual(value(), 1)
                +
                +    def test_regular(self):
                +        self.assertIn(value(), {0, 1})
                """
            ),
            encoding="utf-8",
        )
        (artifacts / "gold.patch").write_text(
            textwrap.dedent(
                """\
                diff --git a/bug.py b/bug.py
                index 7c3c178..76ac8a0 100644
                --- a/bug.py
                +++ b/bug.py
                @@ -1,2 +1,2 @@
                 def value():
                -    return 0
                +    return 1
                """
            ),
            encoding="utf-8",
        )
        manifest = {
            "task_id": "local__admission",
            "version": "v1",
            "runtime_tier": "cpu_python_overlay",
            "source": {
                "repo": "local/repo",
                "pr_url": "https://github.com/local/repo/pull/1",
                "issue_url": "https://github.com/local/repo/issues/1",
                "issue_number": 1,
                "pr_number": 1,
                "local_path": str(source),
                "base_commit": "localbase",
                "merge_commit": "localmerge",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "bug", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "value",
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
                "allowed_test_commands": ["{python} -m unittest {test}"],
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "evaluation": {
                "fail_to_pass": ["test_bug.TestBug.test_value"],
                "pass_to_pass": ["test_bug.TestBug.test_regular"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "metadata": {
                "difficulty": "easy",
                "curation_status": "draft",
                "deterministic": True,
            },
        }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return TaskManifest.load(task_dir / "task.json")


if __name__ == "__main__":
    unittest.main()
