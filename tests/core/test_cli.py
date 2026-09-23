"""Public CLI workflows: validate data, score a patch, and read model results."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from _support import make_task, CORRECT_PATCH, FakeModel, decide
from op_bench.evaluation.controls import check_task
from op_bench.data.task import TaskSpec
from op_bench.runtime.snapshot import snapshot_task
from op_bench.evaluation.verification import verify_evaluation
from op_bench.runner.experiment import run_experiment


class CliTests(unittest.TestCase):
    def cli(self, cwd, *arguments):
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
        return subprocess.run([sys.executable, "-m", "op_bench", *map(str, arguments)], cwd=cwd,
                              env=env, text=True, capture_output=True, timeout=30)

    def test_portable_task_and_patch_evaluation_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "bundle")
            elsewhere = root / "elsewhere"
            elsewhere.mkdir()
            validated = self.cli(elsewhere, "datasets", "validate", "--dataset", root / "bundle/dataset.json")
            self.assertEqual(validated.returncode, 0, validated.stderr)
            patch_file = root / "fix.patch"
            patch_file.write_text(CORRECT_PATCH)
            evaluated = self.cli(elsewhere, "evaluate", "--task", task_file, "--patch", patch_file,
                                 "--output", root / "result")
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertEqual(json.loads(evaluated.stdout)["status"], "resolved")
            verified = self.cli(elsewhere, "verify", "--evaluation", root / "result")
            self.assertEqual(verified.returncode, 0, verified.stderr)
            baseline = self.cli(elsewhere, "evaluate", "--task", task_file, "--baseline", "--output", root / "baseline")
            self.assertEqual(baseline.returncode, 0, baseline.stderr)
            self.assertEqual(json.loads(baseline.stdout)["status"], "test_failed")

    def test_report_is_rebuilt_from_independent_results_without_new_model_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_task(root / "bundle")
            spec = root / "experiment.json"
            spec.write_text(json.dumps({"schema_version": 2, "dataset": "bundle/dataset.json",
                "information_profile": "development", "models": [{"model_id": "fixture", "backend": "chat_completions",
                "model": "fixture", "base_url": "http://127.0.0.1:1/v1"}]}))
            model = FakeModel(decide("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)"),
                              decide("finish", summary="repaired"))
            with patch("op_bench.runner.attempt.create_client", return_value=model):
                initial = run_experiment(spec, root / "run")
            attempt = root / "run/attempts/attempt-000001/attempt.json"
            saved = json.loads(attempt.read_text())
            self.assertNotIn("evaluation", saved)
            # A stale derived report must not override the independent score.
            (root / "run/report.json").write_text('{"obsolete":true}')
            result = self.cli(root, "report", "--run", root / "run")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["models"], initial["models"])
            self.assertTrue(report["complete"])
            self.assertEqual(json.loads(attempt.read_text()), saved)
            self.assertFalse((root / "run/report-history").exists())

    def test_snapshot_pins_docker_image_and_preserves_declared_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "bundle")
            declaration = json.loads(task_file.read_text())
            declaration["environment"].update(backend="docker", image="fixture:tag")
            declaration["metadata"] = {"admission_status": "unreviewed", "provenance": "synthetic fixture"}
            task_file.write_text(json.dumps(declaration))
            image_id = "sha256:" + "a" * 64
            inspected = subprocess.CompletedProcess([], 0, stdout=image_id + "\n", stderr="")
            with patch("op_bench.runtime.execution.subprocess.run", return_value=inspected) as inspect:
                saved = snapshot_task(TaskSpec.load(task_file), root / "snapshot")
            self.assertEqual(saved.environment.image, image_id)
            self.assertEqual(saved.metadata, declaration["metadata"])
            self.assertEqual(inspect.call_args.args[0][:3], ["docker", "image", "inspect"])
            shutil.rmtree(root / "bundle")
            reloaded = TaskSpec.load(root / "snapshot/task.json")
            self.assertTrue((reloaded.source.path / "operator_impl.py").is_file())
            self.assertTrue((reloaded.grader_dir / "check.py").is_file())
            with self.assertRaises(FileExistsError):
                snapshot_task(reloaded, root / "snapshot")

    def test_quality_controls_save_the_executed_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = TaskSpec.load(make_task(root / "bundle"))
            alternative = CORRECT_PATCH.replace("[sum(row) for row in rows]", "list(map(sum, rows))")
            incorrect = CORRECT_PATCH.replace("[sum(row) for row in rows]", "[sum(row) + 1 for row in rows]")
            result = check_task(task, CORRECT_PATCH, root / "controls", alternatives=[alternative], mutations=[incorrect])
            self.assertTrue(result["execution_checks_passed"])
            self.assertEqual(result, json.loads((root / "controls/controls.json").read_text()))
            self.assertEqual(result["reference"]["status"], "resolved")
            self.assertEqual(result["mutations"][0]["status"], "test_failed")

    def test_verify_detects_semantic_score_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = TaskSpec.load(make_task(root / "bundle"))
            from op_bench.evaluation.evaluator import PatchEvaluator
            result = PatchEvaluator().evaluate(task, "", root / "evaluation")
            result["resolved"] = True
            (root / "evaluation/result.json").write_text(json.dumps(result))
            self.assertFalse(verify_evaluation(root / "evaluation")["valid"])


if __name__ == "__main__":
    unittest.main()
