import copy
from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from _support import make_task, CORRECT_PATCH
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runtime.execution import ExecutionError
from op_bench.io import read_json, write_json
from op_bench.results.records import load_attempt
from op_bench.results.report import build_report
from op_bench.data.task import TaskSpec
from op_bench.evaluation.verification import verify_evaluation


class RecordLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task = TaskSpec.load(make_task(self.root / "bundle"))
        self.attempt = self.root / "attempt"
        self.record = {"schema_version": 1, "attempt_id": "attempt-1", "task_id": self.task.task_id,
                       "model_id": "fixture", "repeat": 1, "terminal_status": "finished", "evaluation": None,
                       "submission": {"status": "frozen", "path": "patch.diff"}}
        write_json(self.attempt / "attempt.json", self.record)

    def evaluate(self, patch=CORRECT_PATCH):
        (self.attempt / "patch.diff").write_text(patch)
        return PatchEvaluator().evaluate(self.task, patch, self.attempt / "evaluation")

    def test_completed_evaluation_is_loaded_without_writing_or_execution(self):
        expected = self.evaluate()
        self.assertTrue((self.attempt / "patch.diff").exists())
        recovered = load_attempt(self.attempt / "attempt.json")
        self.assertEqual(recovered["evaluation"], expected)
        self.assertEqual(recovered["terminal_status"], "finished")
        self.assertEqual(recovered["evaluation_recovery"]["status"], "verified")
        self.assertEqual(self.record, read_json(self.attempt / "attempt.json"))
        self.assertEqual(load_attempt(self.attempt), recovered)

    def test_legacy_labels_are_normalized_only_when_reading(self):
        self.evaluate()
        legacy = {**self.record, "terminal_status": "completed", "agent": {"name": "fixture"}}
        legacy["agent_id"] = legacy.pop("model_id")
        write_json(self.attempt / "attempt.json", legacy)
        recovered = load_attempt(self.attempt)
        self.assertEqual(("fixture", "finished"), (recovered["model_id"], recovered["terminal_status"]))
        self.assertEqual({"name": "fixture"}, recovered["model"])
        self.assertNotIn("agent_id", recovered)
        self.assertNotIn("agent", recovered)
        self.assertTrue(build_report([recovered], planned=[legacy])["complete"])
        self.assertEqual(legacy, read_json(self.attempt / "attempt.json"))
        write_json(self.attempt / "attempt.json", {**legacy, "model": {"name": "other"}})
        with self.assertRaisesRegex(ValueError, "conflicting model"):
            load_attempt(self.attempt)

    def test_another_patch_evaluation_cannot_resolve_a_frozen_submission(self):
        frozen_patch = ""
        (self.attempt / "patch.diff").write_text(frozen_patch)
        other = PatchEvaluator().evaluate(self.task, CORRECT_PATCH, self.attempt / "evaluation")
        self.assertTrue(other["resolved"])
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        recovered = load_attempt(self.attempt)
        self.assertIsNone(recovered["evaluation"])
        self.assertEqual("rejected", recovered["evaluation_recovery"]["status"])
        self.assertIn("frozen submission", " ".join(recovered["evaluation_recovery"]["findings"]))
        self.assertEqual(self.record, read_json(self.attempt / "attempt.json"))
        self.assertEqual(other, read_json(self.attempt / "evaluation/result.json"))
        summary = build_report([recovered], planned=[self.record])["models"]["fixture"]
        self.assertEqual((0, 1), (summary["resolved"], summary["errors"]))
        self.assertEqual(0, summary["resolved_rate"])
        self.assertEqual(recovered, load_attempt(self.attempt))

    def test_frozen_submission_requires_its_patch_and_accepts_matching_evidence(self):
        evaluated = self.evaluate()
        frozen = {**self.record, "submission": {"status": "frozen", "path": "patch.diff"},
                  "evaluation": evaluated}
        write_json(self.attempt / "attempt.json", frozen)
        (self.attempt / "patch.diff").unlink()
        rejected = load_attempt(self.attempt)
        self.assertIsNone(rejected["evaluation"])
        self.assertEqual("rejected", rejected["evaluation_recovery"]["status"])
        self.assertIn("cannot read frozen submission", " ".join(rejected["evaluation_recovery"]["findings"]))

        (self.attempt / "patch.diff").write_text(CORRECT_PATCH)
        recovered = load_attempt(self.attempt)
        self.assertEqual(evaluated, recovered["evaluation"])
        self.assertEqual("verified", recovered["evaluation_recovery"]["status"])
        self.assertNotIn("unverified_evaluation", recovered)
        (self.attempt / "evaluation/patch.diff").unlink()
        self.assertIsNone(load_attempt(self.attempt)["evaluation"])

    def test_correct_evaluation_cannot_resolve_a_capture_failure(self):
        original = {**self.record, "capture_error": "capture storage unavailable",
                    "submission": {"status": "capture_failed"}}
        write_json(self.attempt / "attempt.json", original)
        other = self.evaluate()
        recovered = load_attempt(self.attempt)
        self.assertIsNone(recovered["evaluation"])
        self.assertEqual("rejected", recovered["evaluation_recovery"]["status"])
        self.assertIn("capture_error", " ".join(recovered["evaluation_recovery"]["findings"]))
        self.assertEqual(original, read_json(self.attempt / "attempt.json"))
        self.assertEqual(other, read_json(self.attempt / "evaluation/result.json"))
        summary = build_report([recovered], planned=[original])["models"]["fixture"]
        self.assertEqual((0, 1), (summary["resolved"], summary["errors"]))

    def test_pending_submission_or_capture_error_cannot_use_valid_evaluation(self):
        evaluated = self.evaluate()
        (self.attempt / "patch.diff").write_text(CORRECT_PATCH)
        for declaration in (
            {"submission": {"status": "pending", "path": "patch.diff"}},
            {"submission": None},
            {"capture_error": "No reliable patch was retrieved"},
            {"submission": {"status": "frozen", "path": "patch.diff"},
             "capture_error": "Recorded capture failure"},
        ):
            with self.subTest(declaration=declaration):
                original = {**self.record, **declaration, "evaluation": evaluated}
                write_json(self.attempt / "attempt.json", original)
                recovered = load_attempt(self.attempt)
                self.assertIsNone(recovered["evaluation"])
                self.assertEqual("rejected", recovered["evaluation_recovery"]["status"])
                summary = build_report([recovered], planned=[original])["models"]["fixture"]
                self.assertEqual((0, 1), (summary["resolved"], summary["errors"]))

    def test_different_declared_task_identity_is_rejected(self):
        declared = replace(self.task, task_revision="task-1", scoring_revision="score-1",
                           scope="operator", defect_group="row-sum",
                           environment=replace(self.task.environment, environment_id="local-stdlib", revision="1"))
        original = {**self.record, "task_identity": declared.identity_dict(),
                    "submission": {"status": "frozen", "path": "patch.diff"}}
        write_json(self.attempt / "attempt.json", original)
        (self.attempt / "patch.diff").write_text(CORRECT_PATCH)
        revised = replace(declared, scoring_revision="score-2")
        evaluated = PatchEvaluator().evaluate(revised, CORRECT_PATCH, self.attempt / "evaluation")
        self.assertTrue(evaluated["resolved"])
        recovered = load_attempt(self.attempt)
        self.assertIsNone(recovered["evaluation"])
        self.assertEqual("rejected", recovered["evaluation_recovery"]["status"])
        self.assertIn("task_identity", " ".join(recovered["evaluation_recovery"]["findings"]))
        self.assertEqual(original["task_identity"], recovered["task_identity"])
        self.assertEqual(evaluated, read_json(self.attempt / "evaluation/result.json"))
        summary = build_report([recovered], planned=[original])["models"]["fixture"]
        self.assertEqual((0, 1), (summary["resolved"], summary["errors"]))

    def test_wrong_task_evaluation_is_rejected_and_remains_unknown(self):
        evaluation = self.evaluate()
        evaluation["task_id"] = "different-task"
        write_json(self.attempt / "evaluation/result.json", evaluation)
        recovered = load_attempt(self.attempt)
        self.assertIsNone(recovered["evaluation"])
        self.assertEqual(recovered["evaluation_recovery"]["status"], "rejected")
        self.assertIn("task_id", " ".join(recovered["evaluation_recovery"]["findings"]))

    def test_missing_result_is_explicitly_unknown(self):
        recovered = load_attempt(self.attempt)
        self.assertIsNone(recovered["evaluation"])
        self.assertEqual(recovered["evaluation_recovery"]["status"], "unavailable")

    def test_inline_success_cannot_hide_missing_execution_logs(self):
        result = self.evaluate()
        write_json(self.attempt / "attempt.json", {**self.record, "evaluation": result})
        (self.attempt / "evaluation" / result["cases"][0]["log"]).unlink()
        record = load_attempt(self.attempt)
        self.assertIsNone(record["evaluation"])
        self.assertEqual(read_json(self.attempt / "evaluation/result.json"), result)
        report = build_report([record], planned=[self.record])
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"]["fixture"]["errors"], 1)
        self.assertEqual(report["models"]["fixture"]["resolved_rate"], 0)

    def test_inline_success_without_independent_result_stays_unknown(self):
        result = self.evaluate()
        write_json(self.attempt / "attempt.json", {**self.record, "evaluation": result})
        (self.attempt / "evaluation/result.json").unlink()
        record = load_attempt(self.attempt)
        self.assertIsNone(record["evaluation"])
        self.assertEqual(record["evaluation_recovery"]["status"], "unavailable")
        self.assertEqual(load_attempt(self.attempt), record)

    def test_valid_independent_score_replaces_an_inconsistent_inline_cache(self):
        independent = self.evaluate("")
        cached = {**independent, "status": "resolved", "resolved": True}
        original = {**self.record, "evaluation": cached, "unverified_evaluation": cached,
                    "evaluation_recovery": {"status": "recovered"}}
        write_json(self.attempt / "attempt.json", original)
        record = load_attempt(self.attempt)
        self.assertEqual(record["evaluation"], independent)
        self.assertEqual("verified", record["evaluation_recovery"]["status"])
        self.assertNotIn("unverified_evaluation", record)
        self.assertEqual(original, read_json(self.attempt / "attempt.json"))
        report = build_report([record], planned=[self.record])
        self.assertEqual(report["models"]["fixture"]["unresolved"], 1)
        self.assertEqual(report["models"]["fixture"]["resolved"], 0)

    def test_missing_log_fields_and_missing_log_files_are_rejected(self):
        original = self.evaluate()
        modified = copy.deepcopy(original)
        for case in modified["cases"]:
            case.pop("log")
        write_json(self.attempt / "evaluation/result.json", modified)
        self.assertFalse(verify_evaluation(self.attempt / "evaluation")["valid"])
        self.assertIsNone(load_attempt(self.attempt)["evaluation"])
        write_json(self.attempt / "evaluation/result.json", original)
        (self.attempt / "evaluation" / original["cases"][0]["log"]).unlink()
        self.assertFalse(verify_evaluation(self.attempt / "evaluation")["valid"])

    def test_malformed_cases_and_inconsistent_process_outcomes_are_rejected(self):
        original = self.evaluate()
        changes = [lambda d: d["cases"][0].update(id=""),
                   lambda d: d["cases"][0].update(group="unregistered"),
                   lambda d: d["cases"][0].update(exit_code=True),
                   lambda d: d["cases"][0].update(timed_out="false"),
                   lambda d: d["cases"][0].update(status="skipped"),
                   lambda d: d["cases"][0].update(log="../outside.log"),
                   lambda d: d.update(status="test_failed", resolved=False),
                   lambda d: d["groups"]["fail_to_pass"].update(total=99)]
        for change in changes:
            with self.subTest(change=change):
                modified = copy.deepcopy(original)
                change(modified)
                write_json(self.attempt / "evaluation/result.json", modified)
                self.assertFalse(verify_evaluation(self.attempt / "evaluation")["valid"])
        write_json(self.attempt / "evaluation/result.json", ["not an object"])
        self.assertFalse(verify_evaluation(self.attempt / "evaluation")["valid"])

    def test_pretest_failure_allows_not_run_cases_without_test_logs(self):
        evaluation = self.evaluate("This is not a patch")
        self.assertEqual(evaluation["status"], "invalid_patch")
        self.assertTrue(all(case["status"] == "not_run" for case in evaluation["cases"]))
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        self.assertEqual(load_attempt(self.attempt)["evaluation"]["status"], "invalid_patch")

    def test_unrepresentable_durations_reject_evidence_without_crashing_the_report(self):
        original = self.evaluate()
        for mutate in (lambda result: result.update(duration_sec=10**400),
                       lambda result: result["cases"][0].update(duration_sec=10**400)):
            changed = copy.deepcopy(original)
            mutate(changed)
            write_json(self.attempt / "evaluation/result.json", changed)
            with self.subTest(result=changed):
                record = load_attempt(self.attempt)
                self.assertIsNone(record["evaluation"])
                self.assertEqual("rejected", record["evaluation_recovery"]["status"])
                self.assertIn("duration_sec", " ".join(record["evaluation_recovery"]["findings"]))
                summary = build_report([record], planned=[self.record])["models"]["fixture"]
                self.assertEqual((0, 1), (summary["resolved"], summary["errors"]))

    def unittest_task(self, script):
        (self.task.grader_dir / "unit_case.py").write_text(script)
        case = replace(self.task.tests[0], kind="unittest", expected_tests=1,
                       argv=("{python}", "{grader}/unit_case.py", "{workspace}"))
        self.task = replace(self.task, tests=(case, *self.task.tests[1:]))

    def test_empty_unittest_suite_is_a_valid_recoverable_failure(self):
        self.unittest_task("import unittest\nunittest.TextTestRunner().run(unittest.TestSuite())\n")
        result = self.evaluate()
        self.assertEqual("test_failed", result["status"])
        self.assertEqual(0, result["cases"][0]["exit_code"])
        self.assertEqual(0, result["cases"][0]["witness"]["tests_run"])
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        recovered = load_attempt(self.attempt)
        self.assertEqual("verified", recovered["evaluation_recovery"]["status"])
        self.assertEqual("test_failed", recovered["evaluation"]["status"])

    def test_absent_unittest_execution_is_a_valid_recoverable_failure(self):
        self.unittest_task("")
        result = self.evaluate()
        self.assertIsNone(result["cases"][0]["witness"])
        self.assertEqual(0, result["cases"][0]["exit_code"])
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        self.assertEqual("test_failed", load_attempt(self.attempt)["evaluation"]["status"])

    def test_skipped_unittest_is_a_valid_recoverable_failure(self):
        self.unittest_task("import unittest\nclass OperatorTests(unittest.TestCase):\n"
                           "    @unittest.skip('fixture unavailable')\n"
                           "    def test_case(self):\n        pass\n"
                           "unittest.main(argv=['fixture'])\n")
        result = self.evaluate()
        self.assertEqual(1, result["cases"][0]["witness"]["skipped"])
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        self.assertEqual("test_failed", load_attempt(self.attempt)["evaluation"]["status"])

    def test_unittest_success_requires_matching_witness_and_declared_case_count(self):
        self.unittest_task("import sys,unittest\nsys.path.insert(0,sys.argv.pop(1))\n"
                           "from operator_impl import row_sum\n"
                           "class OperatorTests(unittest.TestCase):\n"
                           "    def test_case(self):\n        self.assertEqual([3,9],row_sum([[1,2],[4,5]]))\n"
                           "unittest.main()\n")
        original = self.evaluate()
        evaluation_dir = self.attempt / "evaluation"
        write_json(self.attempt / "attempt.json", {**self.record, "evaluation": original})
        self.assertTrue(verify_evaluation(evaluation_dir)["valid"])
        witness_path = evaluation_dir / original["cases"][0]["witness_log"]
        changed_witness = {**original["cases"][0]["witness"], "tests_run": 0}
        write_json(witness_path, changed_witness)
        self.assertFalse(verify_evaluation(evaluation_dir)["valid"])
        write_json(witness_path, original["cases"][0]["witness"])
        changes = [lambda d: d["cases"][0].pop("witness_log"),
                   lambda d: d["cases"][0].update(expected_tests=2),
                   lambda d: d["cases"][0].update(expected_tests=0),
                   lambda d: d["cases"][0].update(kind="unknown"),
                   lambda d: d["cases"][0].update(kind="command"),
                   lambda d: d["cases"][0].pop("kind"),
                   lambda d: d["cases"][0].update(witness=None)]
        for change in changes:
            with self.subTest(change=change):
                modified = copy.deepcopy(original)
                change(modified)
                write_json(evaluation_dir / "result.json", modified)
                self.assertFalse(verify_evaluation(evaluation_dir)["valid"])
        modified = copy.deepcopy(original)
        modified["cases"][0]["witness"] = changed_witness
        write_json(witness_path, changed_witness)
        write_json(evaluation_dir / "result.json", modified)
        # Agreement between two files is insufficient: zero actual tests cannot
        # support a passed case or a resolved score.
        self.assertFalse(verify_evaluation(evaluation_dir)["valid"])
        self.assertIsNone(load_attempt(self.attempt)["evaluation"])
        self.assertEqual(read_json(self.attempt / "attempt.json")["evaluation"], original)

    def test_witness_collection_failure_preserves_execution_and_recovers_as_unknown(self):
        self.unittest_task("import unittest\nclass OperatorTests(unittest.TestCase):\n"
                           "    def test_case(self):\n        self.assertEqual(3,1+2)\n"
                           "unittest.main(argv=['fixture'])\n")
        with patch("op_bench.runtime.execution.ExecutionSession._read_unittest_witness",
                   side_effect=ExecutionError("temporary evidence transport unavailable")):
            result = self.evaluate()
        self.assertEqual("evaluation_error", result["status"])
        self.assertEqual("error", result["cases"][0]["status"])
        self.assertEqual(0, result["cases"][0]["exit_code"])
        self.assertTrue((self.attempt / "evaluation" / result["cases"][0]["log"]).is_file())
        self.assertTrue(verify_evaluation(self.attempt / "evaluation")["valid"])
        recovered = load_attempt(self.attempt)
        self.assertEqual("evaluation_error", recovered["evaluation"]["status"])
        self.assertFalse(recovered["evaluation"]["resolved"])


if __name__ == "__main__":
    unittest.main()
