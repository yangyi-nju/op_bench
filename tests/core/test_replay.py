import json
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

from op_bench.runner.attempt import HarnessSpec, run_attempt
from op_bench.runner.model_client import ModelSpec
from op_bench.runtime.baseline import build_baseline
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.evaluation.replay import replay_attempt
from op_bench.data.task import TaskSpec
from _support import CORRECT_PATCH, make_task, FakeModel, decide
from test_baseline_evaluation import BUGGY_SOURCE, FIXED_SOURCE, MAKEFILE


class ReplayTests(unittest.TestCase):
    @staticmethod
    def _compiled_grader(task_file, revision, expected):
        (task_file.parent / "grader/check.py").write_text(
            "import json, pathlib, subprocess, sys\n"
            "workspace, case = pathlib.Path(sys.argv[1]), sys.argv[2]\n"
            "arguments = ['4', '5'] if case == 'rows' else []\n"
            "value = int(subprocess.check_output([str(workspace / 'app'), *arguments], text=True, timeout=5))\n"
            "links = len((workspace / 'link-events.txt').read_text().splitlines())\n"
            f"print(json.dumps({{'grader': {revision!r}, 'value': value, 'links': links}}), flush=True)\n"
            f"assert value == ({expected!r} if case == 'rows' else 0)\n")
        raw = json.loads(task_file.read_text())
        raw["scoring_revision"] = revision
        task_file.write_text(json.dumps(raw))

    def _compiled_attempt(self, root):
        task_file = make_task(root / "task")
        source = task_file.parent / "source"
        (source / "main.cpp").write_text(BUGGY_SOURCE)
        (source / "Makefile").write_text(MAKEFILE)
        for args in (("init", "-q"), ("add", "."),
                     ("-c", "user.name=Replay Fixture", "-c", "user.email=replay@example.invalid",
                      "commit", "-qm", "Buggy operator baseline")):
            subprocess.run(["git", *args], cwd=source, check=True, capture_output=True)
        raw = json.loads(task_file.read_text())
        raw["environment"].update(build=[[shutil.which("make")]], build_timeout_sec=30)
        task_file.write_text(json.dumps(raw))
        self._compiled_grader(task_file, "score-1", 9)
        task = TaskSpec.load(task_file)
        artifact = root / "baseline"
        self.assertEqual("ready", build_baseline(task, artifact)["status"])
        model = FakeModel(decide("write_file", path="main.cpp", content=FIXED_SOURCE),
                          decide("finish", summary="Fix reduction"))
        with mock_patch("op_bench.runner.attempt.create_client", return_value=model):
            original = run_attempt(task, ModelSpec("cpp-fixture", "codex_cli", "fixture"),
                                   root / "attempt", baseline_artifact=artifact,
                                   harness=HarnessSpec(budget_sec=30))
        self.assertTrue(original["evaluation"]["resolved"], original)
        return task_file, artifact, original

    def _compiled_observation(self, evaluation, directory):
        case, = [case for case in evaluation["cases"] if case["id"] == "rows"]
        values = [json.loads(line) for line in (directory / case["log"]).read_text().splitlines()
                  if line.startswith('{"grader":')]
        self.assertEqual(1, len(values))
        return values[0]

    def test_revised_grading_replays_patch_and_preserves_original_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "task")
            model = FakeModel(decide("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)"),
                              decide("finish", summary="Fix row indexing"))
            with mock_patch("op_bench.runner.attempt.create_client", return_value=model):
                original = run_attempt(TaskSpec.load(task_file), ModelSpec("fixture", "codex_cli", "fixture"), root / "attempt",
                                       harness=HarnessSpec(budget_sec=30))
            self.assertTrue(original["evaluation"]["resolved"])
            saved_original = json.loads((root / "attempt/attempt.json").read_text())
            grader = root / "task/grader/check.py"
            grader.write_text(grader.read_text().replace("[3, 9]", "[3, 99]"))
            result = replay_attempt(root / "attempt", task_file, root / "replay", reason="Exercise a changed scoring contract")
            self.assertEqual(result["evaluation"]["status"], "test_failed")
            stored = json.loads((root / "replay/replay.json").read_text())
            self.assertNotIn("evaluation", stored)
            self.assertEqual(result["evaluation"], json.loads((root / "replay" / stored["evaluation_path"]).read_text()))
            self.assertFalse(result["generation_performed"])
            self.assertFalse(result["new_independent_repeat"])
            self.assertIsNone(result["baseline_artifact"])
            self.assertEqual(result["generation_comparability"], "requires_review")
            self.assertEqual(result["origin"]["attempt_id"], original["attempt_id"])
            self.assertEqual(result["origin"]["timing_protocol_revision"], original["timing_protocol_revision"])
            self.assertEqual(result["origin"]["budget"], original["budget"])
            self.assertEqual(json.loads((root / "attempt/attempt.json").read_text()), saved_original)
            self.assertEqual((root / "attempt/patch.diff").read_text(), (root / "replay/patch.diff").read_text())
            with self.assertRaises(FileExistsError):
                replay_attempt(root / "attempt", task_file, root / "replay", reason="Do not replace a physical execution")

    @unittest.skipUnless(shutil.which("c++") and shutil.which("make"), "C++ and make are required")
    def test_cached_replay_rebuilds_frozen_patch_with_current_grader_and_default_stays_fresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file, artifact, original = self._compiled_attempt(root)
            saved_original = json.loads((root / "attempt/attempt.json").read_text())
            patch = (root / "attempt/patch.diff").read_text()
            self._compiled_grader(task_file, "score-2", 99)
            replay = replay_attempt(root / "attempt", task_file, root / "replay",
                                    reason="Review a changed numerical oracle", baseline_artifact=artifact)
            self.assertEqual("test_failed", replay["evaluation"]["status"])
            self.assertEqual({"grader": "score-2", "value": 9, "links": 2},
                             self._compiled_observation(replay["evaluation"], root / "replay/evaluation"))
            self.assertEqual(str(artifact.resolve()), replay["baseline_artifact"]["path"])
            self.assertEqual("git", replay["baseline_artifact"]["source_provenance"]["kind"])
            self.assertEqual("score-1", replay["evaluation"]["baseline_artifact"]["producer_task_identity"]["scoring_revision"])
            self.assertEqual("score-2", replay["evaluation"]["task_identity"]["scoring_revision"])
            self.assertFalse((root / "replay/inputs/task/source/.git").exists())

            # The public CLI must select the same cache explicitly and preserve
            # the original patch even when another grader revision now passes.
            self._compiled_grader(task_file, "score-3", 9)
            env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
            invoked = subprocess.run([sys.executable, "-m", "op_bench", "replay",
                "--attempt", str(root / "attempt"), "--task", str(task_file),
                "--baseline-artifact", str(artifact), "--output", str(root / "cli-replay"),
                "--reason", "Restore the reviewed numerical oracle"],
                cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(0, invoked.returncode, invoked.stderr)
            cli_replay = json.loads(invoked.stdout)
            self.assertEqual("resolved", cli_replay["evaluation"]["status"])
            self.assertEqual({"grader": "score-3", "value": 9, "links": 2},
                             self._compiled_observation(cli_replay["evaluation"], root / "cli-replay/evaluation"))

            # Although the original attempt selected a cache, omitting the
            # option makes a fresh patched build, with only its own link event.
            fresh = replay_attempt(root / "attempt", task_file, root / "fresh-replay", reason="Fresh build control")
            self.assertEqual("resolved", fresh["evaluation"]["status"])
            self.assertIsNone(fresh["baseline_artifact"])
            self.assertEqual({"grader": "score-3", "value": 9, "links": 1},
                             self._compiled_observation(fresh["evaluation"], root / "fresh-replay/evaluation"))
            self.assertEqual(3, len({entry["physical_execution_id"] for entry in (replay, cli_replay, fresh)}))
            for name, entry in (("replay", replay), ("cli-replay", cli_replay), ("fresh-replay", fresh)):
                self.assertFalse(entry["generation_performed"])
                self.assertFalse(entry["new_independent_repeat"])
                self.assertEqual(original["attempt_id"], entry["origin"]["attempt_id"])
                self.assertEqual(patch, (root / name / "patch.diff").read_text())
                self.assertEqual(patch, (root / name / "evaluation/patch.diff").read_text())
            self.assertEqual(saved_original, json.loads((root / "attempt/attempt.json").read_text()))
            self.assertEqual(4, int(subprocess.check_output(
                [str(artifact / "workspace/app"), "4", "5"], text=True, timeout=5)))
            self.assertEqual(["linked"], (artifact / "workspace/link-events.txt").read_text().splitlines())

    def test_pending_submission_does_not_become_frozen_by_attaching_an_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "task")
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "attempt.json").write_text(json.dumps({"task_id": "row-sum", "terminal_status": "running",
                "submission": {"status": "pending", "path": "patch.diff"}}))
            (attempt / "patch.diff").write_text(CORRECT_PATCH)
            self.assertTrue(PatchEvaluator().evaluate(TaskSpec.load(task_file), CORRECT_PATCH, attempt / "evaluation")["resolved"])
            with self.assertRaisesRegex(ValueError, "did not freeze a submission"):
                replay_attempt(attempt, task_file, root / "replay", reason="Review an unfinished submission")
            self.assertFalse((root / "replay").exists())

    @unittest.skipUnless(shutil.which("c++") and shutil.which("make"), "C++ and make are required")
    def test_cached_replay_rejects_a_different_source_revision_without_fresh_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file, artifact, _ = self._compiled_attempt(root)
            source = task_file.parent / "source"
            (source / "README.md").write_text("A separately revised source baseline.\n")
            subprocess.run(["git", "add", "README.md"], cwd=source, check=True, capture_output=True)
            subprocess.run(["git", "-c", "user.name=Replay Fixture", "-c", "user.email=replay@example.invalid",
                            "commit", "-qm", "Revise source baseline"], cwd=source, check=True, capture_output=True)
            replay = replay_attempt(root / "attempt", task_file, root / "replay",
                                    reason="Check declared source compatibility", baseline_artifact=artifact)
            evaluation = replay["evaluation"]
            self.assertEqual("environment_error", evaluation["status"])
            self.assertIn("incompatible", evaluation["error"]["message"])
            self.assertTrue(all(case["status"] == "not_run" for case in evaluation["cases"]))
            self.assertFalse(any(stage["stage"] == "build" for stage in evaluation["stages"]))
            self.assertTrue((root / "replay/inputs/task/source/README.md").is_file())

    def test_empty_placeholder_from_unstarted_attempt_is_not_a_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = make_task(root / "task")
            attempt = root / "attempt"
            attempt.mkdir()
            (attempt / "attempt.json").write_text(json.dumps({"task_id": "row-sum", "terminal_status": "runner_error"}))
            (attempt / "patch.diff").write_text("")
            with self.assertRaisesRegex(ValueError, "did not freeze a submission"):
                replay_attempt(attempt, task, root / "replay", reason="Incomplete attempt")
            self.assertFalse((root / "replay").exists())

            # A historical evaluation does not replace an explicit freeze
            # declaration. Its original files remain readable as evidence.
            (attempt / "patch.diff").write_text(CORRECT_PATCH)
            PatchEvaluator().evaluate(TaskSpec.load(task), CORRECT_PATCH, attempt / "evaluation")
            with self.assertRaisesRegex(ValueError, "did not freeze a submission"):
                replay_attempt(attempt, task, root / "replay", reason="Missing original freeze declaration")
            self.assertFalse((root / "replay").exists())

    def test_replay_rejects_another_patchs_score_attached_to_a_frozen_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "task")
            task = TaskSpec.load(task_file)
            with mock_patch("op_bench.runner.attempt.create_client", return_value=FakeModel(decide("finish", summary="No changes"))):
                original = run_attempt(task, ModelSpec("noop", "codex_cli", "fixture"), root / "attempt",
                                       harness=HarnessSpec(budget_sec=30))
            self.assertEqual("frozen", original["submission"]["status"])
            self.assertEqual("test_failed", original["evaluation"]["status"])
            saved_original = json.loads((root / "attempt/attempt.json").read_text())
            (root / "attempt/evaluation").rename(root / "original-evaluation")
            other = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "attempt/evaluation")
            self.assertEqual("resolved", other["status"])
            with self.assertRaisesRegex(ValueError, "differs from the original frozen submission"):
                replay_attempt(root / "attempt", task_file, root / "replay", reason="Review a misfiled evaluation")
            self.assertFalse((root / "replay").exists())
            self.assertEqual("", (root / "attempt/patch.diff").read_text())
            self.assertEqual(saved_original, json.loads((root / "attempt/attempt.json").read_text()))

    def test_interrupted_cached_replay_preserves_executed_cases_and_frozen_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_file = make_task(root / "task")
            task = TaskSpec.load(task_file)
            artifact = root / "baseline"
            self.assertEqual("ready", build_baseline(task, artifact)["status"])
            model = FakeModel(decide("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)"),
                              decide("finish", summary="Fix row indexing"))
            with mock_patch("op_bench.runner.attempt.create_client", return_value=model):
                original = run_attempt(task, ModelSpec("fixture", "codex_cli", "fixture"), root / "attempt",
                                       baseline_artifact=artifact, harness=HarnessSpec(budget_sec=30))
            self.assertTrue(original["evaluation"]["resolved"])
            saved_original = json.loads((root / "attempt/attempt.json").read_text())
            tests = PatchEvaluator._tests

            def interrupt_after_first_case(task, session, output, result):
                tests(replace(task, tests=task.tests[:1]), session, output, result)
                raise KeyboardInterrupt("Stopped after one actual grading case")

            with mock_patch.object(PatchEvaluator, "_tests", side_effect=interrupt_after_first_case):
                with self.assertRaises(KeyboardInterrupt):
                    replay_attempt(root / "attempt", task_file, root / "replay",
                                   reason="Exercise interruption recovery", baseline_artifact=artifact)
            replay = json.loads((root / "replay/replay.json").read_text())
            evaluation = json.loads((root / "replay/evaluation/result.json").read_text())
            self.assertEqual("interrupted", replay["status"])
            self.assertEqual("KeyboardInterrupt", replay["error"]["type"])
            self.assertNotIn("evaluation", replay)
            self.assertEqual(evaluation, json.loads((root / "replay" / replay["evaluation_path"]).read_text()))
            self.assertEqual("evaluation_error", evaluation["status"])
            self.assertFalse(evaluation["resolved"])
            self.assertEqual(["passed", "not_run"], [case["status"] for case in evaluation["cases"]])
            self.assertTrue((root / "replay/evaluation" / evaluation["cases"][0]["log"]).is_file())
            self.assertFalse(replay["generation_performed"])
            self.assertFalse(replay["new_independent_repeat"])
            self.assertEqual((root / "attempt/patch.diff").read_text(), (root / "replay/patch.diff").read_text())
            self.assertEqual(saved_original, json.loads((root / "attempt/attempt.json").read_text()))


if __name__ == "__main__":
    unittest.main()
