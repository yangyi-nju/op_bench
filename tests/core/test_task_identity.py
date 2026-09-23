"""Task revisions and grading variants preserve one public repair problem."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from _support import make_task
from op_bench.data.task import TaskInputError, TaskSpec


class TaskIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-identities-")
        self.addCleanup(self.temporary.cleanup)
        self.path = make_task(Path(self.temporary.name))
        self.raw = json.loads(self.path.read_text())

    def load(self):
        self.path.write_text(json.dumps(self.raw))
        return TaskSpec.load(self.path)

    def identify(self):
        self.raw.update(schema_version=2, task_revision="task-r1", scope="operator",
                        defect_group="row-reduction-defect", scoring_revision="score-r1")
        self.raw["environment"].update(environment_id="cpu-reference", revision="env-r1")

    def numeric(self):
        self.raw["tests"] = [
            {"id": "ordinary", "group": "fail_to_pass", "kind": "numeric",
             "argv": ["{python}", "candidate_worker.py"], "oracle": "cases/ordinary.json"},
            {"id": "boundary", "group": "pass_to_pass", "kind": "numeric",
             "argv": ["{python}", "candidate_worker.py"], "oracle": "cases/boundary.json"},
        ]

    def test_older_task_input_requires_explicit_migration_even_with_verified_metadata(self):
        self.raw["schema_version"] = 1
        self.raw["metadata"] = {"verified": True, "scope": "operator", "task_revision": "claimed"}
        with self.assertRaisesRegex(TaskInputError, "requires task schema 2.*Migrate older task bundles"):
            self.load()

    def test_new_task_identity_survives_serialization_without_defaults(self):
        task = self.load()
        self.assertEqual(task.identity_dict()["status"], "declared")
        self.assertEqual([v.variant_id for v in task.required_variants()], ["default"])
        self.raw = task.to_dict()
        self.assertEqual(self.load().identity_dict(), task.identity_dict())

    def test_schema_two_requires_each_identity_instead_of_defaulting_versions(self):
        self.identify()
        complete = deepcopy(self.raw)
        for field in ("task_revision", "scope", "defect_group", "scoring_revision"):
            self.raw = deepcopy(complete)
            del self.raw[field]
            with self.subTest(field=field), self.assertRaises(TaskInputError):
                self.load()
        for field in ("environment_id", "revision"):
            self.raw = deepcopy(complete)
            del self.raw["environment"][field]
            with self.subTest(environment_field=field), self.assertRaises(TaskInputError):
                self.load()

    def test_variants_share_source_statement_and_one_logical_task(self):
        self.identify()
        other_environment = deepcopy(self.raw["environment"])
        other_environment.update(environment_id="alternate-cpu", revision="env-r2")
        self.raw["variants"] = [{"variant_id": "cpu"},
                                {"variant_id": "alternate", "environment": other_environment}]
        task = self.load()
        self.assertEqual(task.identity_dict()["status"], "declared")
        variant = task.as_variant("alternate")
        self.assertEqual(variant.task_id, task.task_id)
        self.assertEqual(variant.source, task.source)
        self.assertEqual(variant.statement, task.statement)
        self.assertEqual(variant.grader_dir, task.grader_dir)
        self.assertEqual(variant.environment.environment_id, "alternate-cpu")
        self.assertEqual(variant.variants, ())
        self.assertEqual(len(task.variants), 2)
        self.raw = task.to_dict()
        restored = self.load()
        self.assertEqual(restored.identity_dict(), task.identity_dict())
        self.assertEqual(restored.to_dict(), task.to_dict())
        with self.assertRaisesRegex(TaskInputError, "unknown required variant"):
            task.as_variant("missing")

    def test_variants_cannot_change_source_or_public_problem(self):
        self.identify()
        for field in ("source", "statement", "grader_dir", "unknown_option"):
            self.raw["variants"] = [{"variant_id": "cpu", field: "different"}]
            with self.subTest(field=field), self.assertRaisesRegex(TaskInputError, "unknown variant fields"):
                self.load()

    def test_required_variants_reject_duplicates_optional_and_unsafe_ids(self):
        self.identify()
        for variants in ([{"variant_id": "cpu"}, {"variant_id": "cpu"}],
                         [{"variant_id": "cpu", "required": False}],
                         [{"variant_id": "cpu", "required": 1}],
                         [{"variant_id": "../outside"}]):
            self.raw["variants"] = variants
            with self.subTest(variants=variants), self.assertRaises(TaskInputError):
                self.load()

    def test_variant_environment_and_tests_are_validated_independently(self):
        self.identify()
        self.raw["variants"] = [{"variant_id": "cpu", "environment": {"backend": "local"}}]
        with self.assertRaisesRegex(TaskInputError, "environment_id"):
            self.load()
        self.raw["variants"] = [{"variant_id": "cpu", "tests": []}]
        with self.assertRaisesRegex(TaskInputError, "at least one"):
            self.load()

    def test_visible_variant_contract_omits_private_tests_and_oracle_paths(self):
        self.identify()
        self.numeric()
        self.raw["variants"] = [{"variant_id": "cpu"}]
        task = self.load()
        visible = json.dumps(task.visible_dict())
        self.assertIn("cpu-reference", visible)
        self.assertNotIn("ordinary.json", visible)
        self.assertNotIn("oracle", visible)
        self.assertNotIn("defect_group", visible)
        self.assertNotIn("grader_dir", visible)

    def test_numeric_contract_allows_same_worker_for_distinct_oracles(self):
        self.numeric()
        task = self.load()
        self.assertEqual(task.tests[0].argv, task.tests[1].argv)
        self.assertNotEqual(task.tests[0].oracle, task.tests[1].oracle)
        self.assertEqual(task.tests[0].to_dict()["oracle"], "cases/ordinary.json")
        self.raw["tests"][1]["oracle"] = self.raw["tests"][0]["oracle"]
        with self.assertRaisesRegex(TaskInputError, "selectors must be unique"):
            self.load()

    def test_numeric_oracle_cannot_escape_grader_or_supply_grader_to_worker(self):
        self.numeric()
        for oracle in ("/private.json", "../private.json", "cases/../private.json",
                       "C:\\private.json", "..\\private.json", "", "."):
            self.raw["tests"][0]["oracle"] = oracle
            with self.subTest(oracle=oracle), self.assertRaises(TaskInputError):
                self.load()
        self.numeric()
        self.raw["tests"][0]["argv"] = ["{python}", "{grader}/worker.py"]
        with self.assertRaisesRegex(TaskInputError, "must not reference"):
            self.load()
        self.numeric()
        self.raw.pop("grader_dir")
        with self.assertRaisesRegex(TaskInputError, "require grader_dir"):
            self.load()

    def test_numeric_is_one_protocol_case_and_other_kinds_reject_oracle(self):
        self.numeric()
        self.raw["tests"][0]["expected_tests"] = 2
        with self.assertRaisesRegex(TaskInputError, "1 protocol case"):
            self.load()
        self.numeric()
        self.raw["tests"][0]["kind"] = "command"
        with self.assertRaisesRegex(TaskInputError, "only valid for numeric"):
            self.load()


if __name__ == "__main__":
    unittest.main()
