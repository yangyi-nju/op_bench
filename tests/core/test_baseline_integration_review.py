"""Baseline reuse across grading revisions and control-preparation failures."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from _support import CORRECT_PATCH, make_task
from op_bench.evaluation.controls import check_task
from op_bench.runtime.baseline import build_baseline
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.data.task import TaskSpec
from op_bench.runtime.snapshot import snapshot_task
from op_bench.evaluation.verification import verify_evaluation
from op_bench.io import read_json


class BaselineIntegrationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-baseline-integration-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task_file = make_task(self.root / "task")
        self.raw = json.loads(self.task_file.read_text())
        self.raw.update(schema_version=2, task_revision="task-1", scoring_revision="oracle-1",
                        scope="operator", defect_group="row-sum")
        self.raw["environment"].update(environment_id="local-stdlib", revision="1")
        self.build_events = self.root / "build-events.txt"
        self.grader_visits = self.root / "grader-visits.txt"
        grader = self.root / "task/grader/check.py"
        grader.write_text(
            "from pathlib import Path\n"
            f"with Path({str(self.grader_visits)!r}).open('a') as stream: stream.write('graded\\n')\n"
            + grader.read_text())

    def task(self):
        self.task_file.write_text(json.dumps(self.raw))
        return TaskSpec.load(self.task_file)

    def variants(self, modes):
        build = [sys.executable, "-c",
                 "import os; from pathlib import Path; mode=os.environ['MODE']; "
                 f"p=Path({str(self.build_events)!r}); "
                 "p.write_text((p.read_text() if p.exists() else '')+mode+'\\n'); "
                 "assert mode != 'unavailable', 'fixture baseline dependency unavailable'"]
        self.raw["variants"] = []
        for index, mode in enumerate(modes, 1):
            env = deepcopy(self.raw["environment"])
            env.update(environment={"MODE": mode}, build=[build])
            self.raw["variants"].append({"variant_id": f"mode-{index}", "environment": env})
        return self.task()

    def test_new_grading_revision_reuses_build_through_snapshot_but_runs_new_oracle(self):
        original = self.task()
        artifact = self.root / "baseline"
        self.assertEqual("ready", build_baseline(original, artifact)["status"])
        self.assertFalse(self.grader_visits.exists())
        grader = original.grader_dir / "check.py"
        grader.write_text(grader.read_text().replace("[[1, 2], [4, 5]]) == [3, 9]",
                                                   "[[6, 7], [1, 2]]) == [13, 3]"))
        revised = replace(original, task_revision="task-2", scoring_revision="oracle-2")
        stale_values = CORRECT_PATCH.replace("+    return [sum(row) for row in rows]",
                                             "+    return [sum(row) for row in rows] if len(rows)==1 else [3,9]")
        result = check_task(revised, CORRECT_PATCH, self.root / "controls",
                            mutations=[stale_values], baseline_artifact=artifact)
        self.assertTrue(result["execution_checks_passed"], result)
        self.assertEqual("test_failed", result["mutations"][0]["status"])
        for row in [result["baseline"], result["reference"], *result["mutations"]]:
            control = read_json(self.root / "controls" / row["evaluation_path"])
            self.assertEqual(revised.identity_dict(), control["task_identity"])
            self.assertEqual(original.identity_dict(), control["baseline_artifact"]["producer_task_identity"])
        frozen = TaskSpec.load(self.root / "controls/inputs/task.json")
        self.assertNotEqual(original.source.path, frozen.source.path)
        self.assertEqual("oracle-2", frozen.scoring_revision)
        self.assertTrue(verify_evaluation(self.root / "controls/reference")["valid"])

    def test_reused_baseline_and_snapshot_keep_original_commit_when_head_moves(self):
        source = self.root / "task/source"

        def git(*args):
            return subprocess.run(["git", *args], cwd=source, check=True,
                                  capture_output=True, text=True).stdout.strip()

        git("init", "-q")
        git("add", ".")
        commit_options = ("-c", "user.name=Admission Fixture", "-c", "user.email=admission@example.invalid")
        git(*commit_options, "commit", "-qm", "Original operator baseline")
        original_commit = git("rev-parse", "HEAD")
        self.raw["source"]["revision"] = "HEAD"
        task = self.task()
        artifact = self.root / "baseline"
        self.assertEqual("ready", build_baseline(task, artifact)["status"])

        def move_head_then_snapshot(pinned_task, directory):
            (source / "operator_impl.py").write_text("def row_sum(rows):\n    return [-1 for row in rows]\n")
            git("add", "operator_impl.py")
            git(*commit_options, "commit", "-qm", "A different operator baseline")
            return snapshot_task(pinned_task, directory)

        output = self.root / "controls"
        with patch("op_bench.evaluation.controls.snapshot_task", side_effect=move_head_then_snapshot):
            result = check_task(task, CORRECT_PATCH, output, baseline_artifact=artifact)
        self.assertTrue(result["execution_checks_passed"], result)
        self.assertNotEqual(original_commit, git("rev-parse", "HEAD"))
        self.assertEqual("HEAD", task.source.revision)
        origin = json.loads((output / "inputs/origin.json").read_text())
        self.assertEqual(original_commit, origin["revision"])
        for row in (result["baseline"], result["reference"]):
            control = read_json(output / row["evaluation_path"])
            self.assertEqual(original_commit, control["baseline_artifact"]["compatibility"]["source"]["revision"])
        observed = subprocess.check_output([sys.executable, "-I", "-c",
            "import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from operator_impl import row_sum; print(json.dumps(row_sum([[1, 2], [4, 5]])))",
            str(output / "inputs/source")], text=True, timeout=5)
        self.assertEqual([3, 3], json.loads(observed))

    def test_variant_environment_mismatch_is_attributed_to_its_own_unexecuted_child(self):
        task = self.variants(["one", "two"])
        artifact = self.root / "baseline"
        self.assertEqual("ready", build_baseline(task, artifact)["status"])
        self.raw["variants"][1]["environment"]["environment"]["MODE"] = "three"
        revised = self.task()
        result = PatchEvaluator().evaluate(revised, CORRECT_PATCH, self.root / "evaluation",
                                            baseline_artifact=artifact)
        self.assertEqual("environment_error", result["status"], result)
        self.assertEqual("resolved", result["variants"]["mode-1"]["status"])
        failed = read_json(self.root / "evaluation" / result["variants"]["mode-2"]["evaluation_path"])
        self.assertEqual("environment_error", failed["status"])
        self.assertEqual("source", failed["error"]["stage"])
        self.assertTrue(all(case["status"] == "not_run" for case in failed["cases"]))
        # Both baseline variants built once; only the compatible first variant
        # ran its required candidate build. The second did not reuse wrong inputs.
        self.assertEqual(["one", "two", "one"], self.build_events.read_text().splitlines())
        self.assertTrue(verify_evaluation(self.root / "evaluation")["valid"])

    def test_failed_variant_baseline_retains_all_controls_without_rebuilding_or_grading(self):
        task = self.variants(["ready", "unavailable"])
        output = self.root / "controls"
        result = check_task(task, CORRECT_PATCH, output, alternatives=[CORRECT_PATCH],
                            mutations=[""], reuse_baseline=True)
        self.assertEqual(result, json.loads((output / "controls.json").read_text()))
        self.assertFalse(result["execution_checks_passed"])
        self.assertEqual("baseline_build", result["error"]["stage"])
        self.assertEqual("failed", result["baseline_build"]["status"])
        self.assertEqual("ready", result["baseline_build"]["variants"]["mode-1"]["status"])
        self.assertEqual("build", result["baseline_build"]["variants"]["mode-2"]["error"]["stage"])
        controls = [result["baseline"], result["reference"], *result["alternatives"], *result["mutations"]]
        self.assertEqual(4, len(controls))
        for control in controls:
            self.assertEqual("not_run", control["status"])
            self.assertEqual("baseline_build", control["error"]["stage"])
            self.assertFalse((output / control["evaluation_path"]).exists())
        self.assertEqual(["ready", "unavailable"], self.build_events.read_text().splitlines())
        self.assertFalse(self.grader_visits.exists())
        self.assertFalse(result["checks"]["provided_mutations_rejected"])
        self.assertEqual({"baseline", "reference", "alternative-1", "mutation-1"},
                         {row["control"] for row in result["control_plan"]})

    def test_existing_baseline_missing_required_variant_is_recorded_once_and_controls_stay_unrun(self):
        first = self.variants(["one"])
        artifact = self.root / "baseline"
        self.assertEqual("ready", build_baseline(first, artifact)["status"])
        task = self.variants(["one", "two"])
        output = self.root / "controls"
        result = check_task(task, CORRECT_PATCH, output, mutations=[""], baseline_artifact=artifact)
        self.assertEqual(result, json.loads((output / "controls.json").read_text()))
        self.assertFalse(result["execution_checks_passed"])
        self.assertEqual("baseline_evaluation", result["error"]["stage"])
        self.assertEqual("baseline_artifact", result["baseline"]["error"]["stage"])
        self.assertEqual("evaluation_error", result["baseline"]["status"])
        self.assertEqual("not_run", result["reference"]["status"])
        self.assertEqual("not_run", result["mutations"][0]["status"])
        self.assertEqual(["one"], self.build_events.read_text().splitlines())
        self.assertFalse(self.grader_visits.exists())
        self.assertFalse(result["checks"]["provided_mutations_rejected"])
        self.assertTrue(verify_evaluation(output / "baseline")["valid"])

    def test_interrupted_variant_preserves_executed_cases_in_the_control_matrix(self):
        task = self.variants(["one", "two"])
        output = self.root / "controls"
        original_tests = PatchEvaluator._tests

        def interrupt_after_first_test(task, session, directory, result):
            if Path(directory).parts[-3:] == ("reference", "variants", "000002"):
                original_tests(replace(task, tests=task.tests[:1]), session, directory, result)
                raise KeyboardInterrupt("fixture interruption after an executed case")
            original_tests(task, session, directory, result)

        with patch.object(PatchEvaluator, "_tests", side_effect=interrupt_after_first_test):
            with self.assertRaises(KeyboardInterrupt):
                check_task(task, CORRECT_PATCH, output, mutations=[""])
        saved = json.loads((output / "controls.json").read_text())
        reference = read_json(output / saved["reference"]["evaluation_path"])
        self.assertEqual("test_failed", saved["baseline"]["status"])
        self.assertEqual("evaluation_error", reference["status"])
        self.assertEqual("resolved", reference["variants"]["mode-1"]["status"])
        interrupted = read_json(output / "reference" / reference["variants"]["mode-2"]["evaluation_path"])
        self.assertEqual(["passed", "not_run"], [case["status"] for case in interrupted["cases"]])
        self.assertEqual("KeyboardInterrupt", saved["error"]["type"])
        self.assertEqual("not_run", saved["mutations"][0]["status"])
        self.assertFalse(saved["execution_checks_passed"])
        self.assertTrue(verify_evaluation(output / "reference")["valid"])


if __name__ == "__main__":
    unittest.main()
