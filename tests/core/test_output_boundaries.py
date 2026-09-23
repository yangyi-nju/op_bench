"""Result-producing entry points reject destinations inside trusted task inputs."""
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from _support import CORRECT_PATCH, make_task
from op_bench.data.task import TaskInputError, TaskSpec, TaskVariant
from op_bench.evaluation.controls import check_task
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.evaluation.replay import replay_attempt
from op_bench.io import write_json
from op_bench.runner.attempt import run_attempt
from op_bench.runner.experiment import run_experiment
from op_bench.runner.model_client import ModelSpec
from op_bench.runtime.baseline import build_baseline, fork_baseline
from op_bench.runtime.snapshot import snapshot_task


class OutputBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task_file = make_task(self.root / "task")
        self.task = TaskSpec.load(self.task_file)
        self.source_alias, self.grader_alias = self.root / "source-alias", self.root / "grader-alias"
        self.source_alias.symlink_to(self.task.source.path, target_is_directory=True)
        self.grader_alias.symlink_to(self.task.grader_dir, target_is_directory=True)
        self.input_directories = (self.task.source.path, self.task.grader_dir, self.source_alias, self.grader_alias)
        self.model = ModelSpec("fixture", "chat_completions", "fixture", base_url="http://127.0.0.1:1/v1")
        self.experiment = self.root / "experiment.json"
        write_json(self.experiment, {"schema_version": 2, "dataset": "task/dataset.json",
            "information_profile": "development", "models": [self.model.to_dict()]})
        self.original_attempt = self.root / "original-attempt"
        write_json(self.original_attempt / "attempt.json", {
            "task_id": self.task.task_id, "submission": {"status": "frozen"}})
        (self.original_attempt / "patch.diff").write_text(CORRECT_PATCH)
        self.reference = self.root / "reference.patch"
        self.reference.write_text(CORRECT_PATCH)

    def test_direct_workflows_reject_outputs_before_writing_or_calling_a_model(self):
        variant_task = replace(self.task, variants=(TaskVariant("cpu", self.task.environment, self.task.tests),))
        workflows = {
            "attempt": lambda output: run_attempt(self.task, self.model, output),
            "evaluate": lambda output: PatchEvaluator().evaluate(self.task, CORRECT_PATCH, output),
            "variants": lambda output: PatchEvaluator().evaluate(variant_task, CORRECT_PATCH, output),
            "controls": lambda output: check_task(self.task, CORRECT_PATCH, output),
            "replay": lambda output: replay_attempt(self.original_attempt, self.task_file, output, reason="Recheck"),
            "snapshot": lambda output: snapshot_task(self.task, output),
            "baseline": lambda output: build_baseline(self.task, output),
            "fork": lambda output: fork_baseline(self.task, self.root / "absent-artifact", output),
            "experiment": lambda output: run_experiment(self.experiment, output),
        }
        with patch("op_bench.runner.attempt.create_client") as create_client:
            for directory in self.input_directories:
                for name, run in workflows.items():
                    output = directory / f"unsafe-{name}"
                    with self.subTest(workflow=name, directory=directory), self.assertRaisesRegex(
                        TaskInputError, "outside task source and grader"
                    ):
                        run(output)
                    self.assertFalse(output.exists(), f"{name} polluted task input before rejection")
            create_client.assert_not_called()

    def test_source_and_grader_roots_are_rejected_and_valid_paths_are_only_resolved(self):
        for directory in self.input_directories:
            with self.subTest(directory=directory), self.assertRaises(TaskInputError):
                self.task.validate_output_path(directory)
        output = self.root / "results" / "evaluation"
        self.assertEqual(output.resolve(), self.task.validate_output_path(output))
        self.assertFalse(output.parent.exists())

    def test_cli_workflows_reject_source_and_grader_outputs_without_partial_artifacts(self):
        workflows = (
            ("prepare", "--task", self.task_file),
            ("evaluate", "--task", self.task_file, "--baseline"),
            ("build-baseline", "--task", self.task_file),
            ("check-task", "--task", self.task_file, "--reference", self.reference),
            ("replay", "--attempt", self.original_attempt, "--task", self.task_file, "--reason", "Recheck"),
            ("run", "--experiment", self.experiment),
        )
        environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
        for directory in self.input_directories:
            for arguments in workflows:
                output = directory / f"unsafe-cli-{arguments[0]}"
                with self.subTest(command=arguments[0], directory=directory):
                    completed = subprocess.run(
                        [sys.executable, "-m", "op_bench", *map(str, arguments), "--output", str(output)],
                        cwd=self.root, env=environment, text=True, capture_output=True, timeout=10)
                    self.assertEqual(2, completed.returncode, completed.stderr)
                    self.assertIn("outside task source and grader", completed.stderr)
                    self.assertFalse(output.exists(), "CLI left artifacts in task input before rejection")


if __name__ == "__main__":
    unittest.main()
