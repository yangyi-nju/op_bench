import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runner.experiment import run_experiment
from op_bench.data.task import TaskSpec
from op_bench.evaluation.verification import verify_evaluation
from op_bench.io import read_json, write_json
from _support import CORRECT_PATCH, make_task, FakeModel, decide


def variant_task(root):
    path = make_task(root)
    value = json.loads(path.read_text())
    value.update(schema_version=2, task_revision="1", scope="operator", defect_group="row-indexing",
                 scoring_revision="1")
    value["environment"].update(environment_id="stdlib-cpu", revision="1")
    (root / "grader/check_more.py").write_text(
        "import sys\nsys.path.insert(0,sys.argv[1])\nfrom operator_impl import row_sum\n"
        "assert row_sum([[10],[20]])==[10,20]\n")
    other_tests = [dict(value["tests"][0], argv=["{python}", "{grader}/check_more.py", "{workspace}"]), value["tests"][1]]
    value["variants"] = [{"variant_id": "ordinary"}, {"variant_id": "other-values", "tests": other_tests}]
    path.write_text(json.dumps(value))
    return TaskSpec.load(path)


class VariantTests(unittest.TestCase):
    def test_same_patch_must_pass_every_required_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            partial = CORRECT_PATCH.replace("+    return [sum(row) for row in rows]",
                                            "+    return [sum(rows[0])] if len(rows)==1 else [3,9]")
            result = PatchEvaluator().evaluate(task, partial, root / "partial")
            self.assertEqual(result["variants"]["ordinary"]["status"], "resolved")
            self.assertEqual(result["variants"]["other-values"]["status"], "test_failed")
            self.assertFalse(result["resolved"])
            self.assertEqual(result["status"], "test_failed")
            correct = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "correct")
            self.assertTrue(correct["resolved"])
            self.assertNotIn("cases", correct)
            self.assertNotIn("groups", correct)
            for child in correct["variants"].values():
                saved = read_json(root / "correct" / child["evaluation_path"])
                self.assertEqual(saved["groups"]["fail_to_pass"], {"passed": 1, "total": 1})
                self.assertEqual(set(child), {"evaluation_path", "status", "resolved"})
            self.assertTrue(verify_evaluation(root / "correct")["valid"])
            self.assertIn("sum(rows[0])", (task.source.path / "operator_impl.py").read_text())

    def test_variant_child_outcome_cannot_be_omitted_from_saved_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            result = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "result")
            child = root / "result" / result["variants"]["other-values"]["evaluation_path"]
            child.unlink()
            self.assertFalse(verify_evaluation(root / "result")["valid"])

    def test_variant_results_require_the_declared_revision_and_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            result = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "result")
            child_path = root / "result" / result["variants"]["other-values"]["evaluation_path"]
            original_child = read_json(child_path)
            for mutate in (
                lambda identity: identity.update(scoring_revision="different-scoring"),
                lambda identity: identity.update(task_revision="different-task"),
                lambda identity: identity["solver_environment"].update(environment_id="different-hardware"),
                lambda identity: identity["required_variants"][0]["environment"].update(revision="different-build"),
            ):
                changed = deepcopy(result)
                child = deepcopy(original_child)
                mutate(child["task_identity"])
                write_json(child_path, child)
                write_json(root / "result/result.json", changed)
                with self.subTest(identity=child["task_identity"]):
                    checked = verify_evaluation(root / "result")
                    self.assertFalse(checked["valid"])
                    self.assertIn("variant task identity", " ".join(checked["findings"]))

    def test_unreadable_variant_patch_is_rejected_without_crashing_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "result")
            (root / "result/patch.diff").write_bytes(b"\xff")
            checked = verify_evaluation(root / "result")
            self.assertFalse(checked["valid"])
            self.assertIn("cannot read submitted patch", " ".join(checked["findings"]))

    def test_legacy_parent_reads_authoritative_children_without_inline_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            result = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "result")
            legacy = {**result, "schema_version": 1,
                      "variant_artifacts": {name: str(Path(row["evaluation_path"]).parent) for name, row in result["variants"].items()},
                      "variants": {name: read_json(root / "result" / row["evaluation_path"])
                                   for name, row in result["variants"].items()}}
            write_json(root / "result/result.json", legacy)
            self.assertTrue(verify_evaluation(root / "result")["valid"])
            (root / "result" / legacy["variant_artifacts"]["ordinary"] / "result.json").unlink()
            self.assertFalse(verify_evaluation(root / "result")["valid"])

    def test_variant_summary_cannot_invent_outcome_or_reuse_another_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = variant_task(root / "task")
            result = PatchEvaluator().evaluate(task, CORRECT_PATCH, root / "result")
            for mutate in (
                lambda value: value["variants"]["ordinary"].update(status="test_failed", resolved=False),
                lambda value: value["variants"]["ordinary"].update(resolved=1),
                lambda value: value["variants"]["ordinary"].update(evaluation_path=value["variants"]["other-values"]["evaluation_path"]),
                lambda value: value["variants"].pop("ordinary"),
            ):
                changed = deepcopy(result)
                mutate(changed)
                write_json(root / "result/result.json", changed)
                self.assertFalse(verify_evaluation(root / "result")["valid"])

    def test_fixed_experiment_counts_all_required_variants_as_one_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            variant_task(root / "task")
            config = {"schema_version": 2, "dataset": "task/dataset.json", "harness": {"budget_sec": 30},
                      "information_profile": "development", "models": [
                          {"model_id": "fixture", "backend": "codex_cli", "model": "fixture"}]}
            (root / "experiment.json").write_text(json.dumps(config))
            model = FakeModel(decide("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)"),
                              decide("finish", summary="Fix all row values"))
            with patch("op_bench.runner.attempt.create_client", return_value=model):
                report = run_experiment(root / "experiment.json", root / "run")
            summary = report["models"]["fixture"]
            self.assertEqual(summary["planned"], 1)
            self.assertEqual(summary["resolved"], 1)
            self.assertEqual(summary["resolved_rate"], 1)
            self.assertEqual(len(summary["tasks"]), 1)
            with patch("op_bench.runner.attempt.create_client", side_effect=AssertionError("Resume must not call model")):
                resumed = run_experiment(root / "experiment.json", root / "run", resume=True)
            self.assertEqual(resumed["models"], report["models"])
            self.assertEqual(resumed["complete"], report["complete"])
            self.assertEqual(resumed["software_identity"], report["software_identity"])


if __name__ == "__main__":
    unittest.main()
