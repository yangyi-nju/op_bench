import unittest
import tempfile
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from unittest.mock import patch

from op_bench.results.report import build_report, summarize_run, REPORT_PROTOCOL_REVISION


def row(task, repeat=1, resolved=True, model="a", status=None):
    status = status or ("resolved" if resolved else "test_failed")
    return {"attempt_id": f"{task}-{model}-{repeat}", "task_id": task, "model_id": model,
            "repeat": repeat, "terminal_status": "finished", "model": {"model": model},
            "submission": {"status": "frozen", "path": "patch.diff"},
            "evaluation_recovery": {"status": "verified"},
            "evaluation": {"task_id": task, "status": status, "resolved": resolved}}


def identified_row(task, *, model="a", resolved=True, variants=None):
    record = row(task, resolved=resolved, model=model)
    identity = {"status": "declared", "task_id": task, "task_revision": "task-1",
                "scope": "operator", "defect_group": task, "scoring_revision": "score-1",
                "solver_environment": {"environment_id": "cpu", "revision": "env-1"},
                "required_variants": [{"variant_id": name,
                                       "environment": {"environment_id": name, "revision": "env-1"}}
                                      for name in (variants or ["default"])]}
    record.update(task_identity=identity,
                  dataset_identity={"dataset_id": "fixture", "dataset_version": "1"},
                  software_identity={"version": "0.8.0.dev1"})
    record["evaluation"]["task_identity"] = deepcopy(identity)
    if variants:
        record["evaluation"]["variants"] = {
            name: {"task_id": task, "status": "resolved" if resolved else "test_failed", "resolved": resolved}
            for name in variants}
    return record


class ReportTests(unittest.TestCase):
    def test_success_failure_and_missing_use_the_fixed_task_denominator(self):
        plan = [row("fixed"), row("failed", resolved=False), row("not-started")]
        records = deepcopy(plan[:2])
        original = deepcopy(records)
        report = build_report(records, planned=plan)
        model = report["models"]["a"]
        self.assertEqual((3, 2, 1, 1, 1, 0), tuple(model[k] for k in
                         ("planned", "started", "resolved", "unresolved", "missing", "errors")))
        self.assertEqual(1 / 3, model["resolved_rate"])
        self.assertFalse(report["complete"])
        self.assertEqual("incomplete", model["status"])
        self.assertEqual(records, original)
        self.assertNotIn("agents", report)
        self.assertNotIn("task_macro", model)

    def test_environment_and_service_failures_do_not_resolve_partial_patches(self):
        for terminal, evaluation_status in (("finished", "environment_error"),
                                            ("finished", "evaluation_error"),
                                            ("runner_error", "resolved"),
                                            ("model_service_error", "resolved"),
                                            ("tool_error", "resolved"),
                                            ("agent_error", "resolved"),
                                            ("cancelled", "resolved")):
            failed = row("failed", resolved=evaluation_status == "resolved", status=evaluation_status)
            failed["terminal_status"] = terminal
            plan = [row("ok"), failed]
            with self.subTest(terminal=terminal, evaluation=evaluation_status):
                report = build_report(plan, planned=plan)
                result = report["models"]["a"]
                self.assertEqual(.5, result["resolved_rate"])
                self.assertEqual(1, result["errors"])
                self.assertEqual(0, result["unresolved"])
                self.assertFalse(report["complete"])

    def test_deadline_submission_is_scored_when_freezing_and_grading_succeeded(self):
        for resolved in (True, False):
            attempt = row("task", resolved=resolved)
            attempt["terminal_status"] = "timed_out"
            result = build_report([attempt], planned=[attempt])["models"]["a"]
            self.assertTrue(result["complete"])
            self.assertEqual(int(resolved), result["resolved"])
            self.assertEqual(int(not resolved), result["unresolved"])

    def test_invalid_model_tool_decisions_use_the_independent_patch_score(self):
        from _support import make_task, CORRECT_PATCH
        from op_bench.evaluation.evaluator import PatchEvaluator
        from op_bench.io import write_json
        from op_bench.data.task import TaskSpec

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = TaskSpec.load(make_task(root / "task"))
            output = root / "run"
            planned = []
            for model, patch_text, terminal in (("fixed", CORRECT_PATCH, "model_protocol_error"),
                                                ("unchanged", "", "model_protocol_error"),
                                                ("service-failure", CORRECT_PATCH, "model_service_error")):
                attempt = output / "attempts" / model
                attempt.mkdir(parents=True)
                record = {"attempt_id": model, "task_id": task.task_id, "model_id": model,
                          "repeat": 1, "model": {"model": model}, "terminal_status": terminal,
                          "task_identity": task.identity_dict(),
                          "submission": {"status": "frozen", "path": "patch.diff"}, "evaluation": None}
                planned.append(deepcopy(record))
                write_json(attempt / "attempt.json", record)
                (attempt / "patch.diff").write_text(patch_text)
                PatchEvaluator().evaluate(task, patch_text, attempt / "evaluation")
            write_json(output / "plan.json", {"mode": "fixed_models", "schedule": planned})
            report = summarize_run(output)
            self.assertEqual("fixed_models", report["mode"])
            self.assertFalse((output / "report.json").exists())
            self.assertEqual((1, True), (report["models"]["fixed"]["resolved_rate"], report["models"]["fixed"]["complete"]))
            self.assertEqual((0, 1, True), tuple(report["models"]["unchanged"][field] for field in ("resolved_rate", "unresolved", "complete")))
            self.assertEqual((0, 1, False), tuple(report["models"]["service-failure"][field] for field in ("resolved_rate", "errors", "complete")))

    def test_directory_report_rejects_plan_conditions_that_differ_from_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            for field, value in (("harness", {"budget_sec": 600}),
                                 ("harness_version", "different-loop"),
                                 ("information_profile", "controlled"),
                                 ("tool_version", "1"),
                                 ("dataset_identity", {"dataset_id": "other", "dataset_version": "2"}),
                                 ("software_identity", {"version": "other"}),
                                 ("timing_protocol_revision", "3")):
                plan = {"schedule": [row("missing")], field: value}
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, f"plan {field} differs"):
                    summarize_run(Path(temporary), plan)

    def test_inline_success_cannot_replace_verified_frozen_submission(self):
        original = row("task")
        mutations = [lambda r: r.pop("submission"),
                     lambda r: r["submission"].update(status="pending"),
                     lambda r: r.update(capture_error="could not stop writer"),
                     lambda r: r.pop("evaluation_recovery"),
                     lambda r: r["evaluation_recovery"].update(status="rejected"),
                     lambda r: r.update(evaluation=None)]
        for mutate in mutations:
            record = deepcopy(original)
            mutate(record)
            with self.subTest(record=record):
                result = build_report([record], planned=[original])["models"]["a"]
                self.assertEqual(0, result["resolved_rate"])
                self.assertEqual(1, result["errors"])

    def test_multiple_models_share_the_task_set_without_best_of_repeats(self):
        records = [row(task, model=model, resolved=model == "a")
                   for task in ("x", "y") for model in ("a", "b")]
        report = build_report(records, planned=records)
        self.assertTrue(report["complete"])
        self.assertEqual(1, report["models"]["a"]["resolved_rate"])
        self.assertEqual(0, report["models"]["b"]["resolved_rate"])
        with self.assertRaisesRegex(ValueError, "same planned task set"):
            build_report(records[:-1], planned=records[:-1])
        with self.assertRaisesRegex(ValueError, "one attempt"):
            build_report([row("x", repeat=2)])
        with self.assertRaisesRegex(ValueError, "duplicate model/task"):
            build_report([row("x"), dict(row("x"), attempt_id="retried")])

    def test_comparison_requires_the_same_tools_and_information_conditions(self):
        for field, first, second in (("harness", {"toolset": "fixed-v1"}, {"toolset": "fixed-v2"}),
                                     ("harness_version", "1", "2"),
                                     ("tool_version", "1", "2"),
                                     ("information_profile", "closed", "development")):
            records = [row("x", model="a"), row("x", model="b")]
            records[0][field] = first
            records[1][field] = second
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "same planned"):
                build_report(records, planned=records)

    def test_missing_plan_is_an_observed_incomplete_report(self):
        report = build_report([row("x")])
        self.assertFalse(report["complete"])
        self.assertFalse(report["planned_schedule_supplied"])
        self.assertEqual(1, report["models"]["a"]["resolved_rate"])
        empty = build_report([], planned=[])
        self.assertFalse(empty["complete"])
        self.assertEqual({}, empty["models"])

    def test_legacy_model_labels_are_read_without_dual_identity_output(self):
        record = row("x")
        record["agent_id"] = record.pop("model_id")
        record["agent"] = record.pop("model")
        report = build_report([record], planned=[record])
        self.assertTrue(report["complete"])
        self.assertIn("a", report["models"])
        self.assertNotIn("agents", report)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            build_report([dict(record, model_id="another")])

    def test_plan_generation_and_grading_identities_cannot_drift(self):
        plan = identified_row("x")
        for field in ("task_identity", "model", "harness", "harness_version", "tool_version", "information_profile"):
            record = deepcopy(plan)
            if field == "task_identity":
                record[field]["scoring_revision"] = "other"
            else:
                record[field] = {"changed": True} if field != "information_profile" else "open"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs"):
                build_report([record], planned=[plan])
        record = deepcopy(plan)
        record["evaluation"]["task_identity"]["scoring_revision"] = "other"
        result = build_report([record], planned=[plan])["models"]["a"]
        self.assertEqual(1, result["errors"])
        self.assertEqual({"evaluation_identity_mismatch": 1}, result["statuses"])

    def test_required_variants_cannot_be_omitted_or_averaged(self):
        record = identified_row("x", variants=["cpu", "cuda"])
        self.assertTrue(build_report([record], planned=[record])["complete"])
        for mutation in (lambda r: r["evaluation"]["variants"].pop("cuda"),
                         lambda r: r["evaluation"]["variants"]["cuda"].update(status="test_failed", resolved=False),
                         lambda r: r["evaluation"]["variants"]["cuda"].update(status="environment_error", resolved=False)):
            changed = deepcopy(record)
            mutation(changed)
            result = build_report([changed], planned=[record])["models"]["a"]
            self.assertEqual(1, result["errors"])
            self.assertEqual(0, result["resolved"])

    def test_invalid_record_and_mixed_provenance_fail_clearly(self):
        for evaluation in ({"status": "made_up", "resolved": False},
                           {"status": "resolved", "resolved": False},
                           {"status": [], "resolved": False}):
            with self.subTest(evaluation=evaluation), self.assertRaises(ValueError):
                build_report([dict(row("x"), evaluation=evaluation)])
        for field, key in (("software_identity", "version"), ("dataset_identity", "dataset_version")):
            first, second = identified_row("x"), identified_row("y")
            second[field][key] = "different"
            with self.assertRaisesRegex(ValueError, field):
                build_report([first, second], planned=[first, second])
        with self.assertRaisesRegex(ValueError, "planned identity"):
            build_report([row("y")], planned=[row("x")])

    def test_recomputation_records_reporter_without_relabeling_execution(self):
        record = identified_row("x")
        record["software_identity"]["version"] = "old-executor"
        before = deepcopy(record)
        current = {"version": "new-reporter"}
        with patch("op_bench.results.report.observed_software_identity", return_value=current):
            report = build_report([record], planned=[record])
        self.assertEqual(current, report["report_generation"]["software_identity"])
        self.assertEqual(REPORT_PROTOCOL_REVISION, report["report_generation"]["report_protocol_revision"])
        self.assertIsNotNone(datetime.fromisoformat(report["report_generation"]["generated_at"]).utcoffset())
        self.assertEqual("old-executor", report["software_identity"]["version"])
        self.assertEqual(before, record)


if __name__ == "__main__":
    unittest.main()
