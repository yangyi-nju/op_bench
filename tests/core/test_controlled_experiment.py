"""Real fixed-loop tool execution and scoring, with deterministic model responses."""
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from op_bench.runner.attempt import HarnessSpec, run_attempt
from op_bench.runner.experiment import run_experiment
from op_bench.runner.model_client import ModelError, ModelSpec
from op_bench.results.records import load_attempt
from op_bench.results.report import build_report
from op_bench.data.task import TaskSpec
from _support import make_task


MODEL = ModelSpec("fixture-model", "chat_completions", "fixture", base_url="http://127.0.0.1:1/v1")


def decision(name, **arguments):
    return {"content": "", "tool_calls": [{"id": "call", "name": name, "arguments": arguments}], "usage": None}


class StepClient:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.calls = 0

    def complete(self, messages, tools, timeout_sec):
        self.calls += 1
        value = next(self.steps)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            return value()
        return value


def repair_steps():
    return [decision("read_file", path="operator_impl.py"),
            decision("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)"),
            decision("finish", summary="Fixed independent row selection")]


class ControlledExperimentTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.task_file = make_task(self.root / "task")
        self.task = TaskSpec.load(self.task_file)

    def test_harness_settings_are_validated_at_python_and_config_entry_points(self):
        for field in ("budget_sec", "per_tool_timeout_sec", "max_turns", "max_context_chars"):
            values = [True, None, "10", 0, -1, float("inf"), float("nan")]
            values += [10**400] if field.endswith("sec") else [1.5]
            for value in values:
                for make in (lambda data: HarnessSpec(**data), HarnessSpec.from_dict):
                    with self.subTest(field=field, value=value, entrypoint=make), self.assertRaises(ValueError):
                        make({field: value})
        configured = HarnessSpec(budget_sec=.5, max_turns=1, max_context_chars=20, per_tool_timeout_sec=.1)
        self.assertEqual(configured, HarnessSpec.from_dict(configured.to_dict()))

    def test_actual_tool_repair_frozen_and_independently_scored(self):
        client = StepClient(repair_steps())
        with patch("op_bench.runner.attempt.create_client", return_value=client):
            record = run_attempt(self.task, MODEL, self.root / "run")
        self.assertEqual(record["terminal_status"], "finished")
        self.assertEqual(record["evaluation"]["status"], "resolved")
        self.assertEqual(record["submission"]["status"], "frozen")
        saved = json.loads((self.root / "run/attempt.json").read_text())
        self.assertNotIn("harness_result", saved)
        self.assertNotIn("evaluation", saved)
        harness = json.loads((self.root / "run" / saved["harness_path"]).read_text())
        self.assertEqual(harness["state"], "completed")
        self.assertEqual(harness["harness_version"], saved["harness_version"])
        self.assertEqual(load_attempt(self.root / "run/attempt.json")["evaluation_recovery"]["status"], "verified")
        events = [json.loads(line) for line in (self.root / "run/harness/trajectory.jsonl").read_text().splitlines()]
        public = events[0]["public_task"]
        self.assertNotIn(str(self.task.grader_dir), json.dumps(public))
        self.assertNotIn("tests", public)
        self.assertEqual([event["name"] for event in events if event["event"] == "tool_call"],
                         ["read_file", "replace_text", "finish"])
        self.assertFalse(record["environment"]["grader_access"])

    def test_full_schedule_resume_does_not_call_models_again(self):
        experiment = {"schema_version": 2, "dataset": "task/dataset.json", "information_profile": "development",
                      "reuse_baseline": True, "models": [MODEL.to_dict(), replace(MODEL, model_id="noop").to_dict()]}
        path = self.root / "experiment.json"
        path.write_text(json.dumps(experiment))
        def client(model, output):
            return StepClient(repair_steps() if model.model_id == "fixture-model" else [decision("finish", summary="No change")])
        with patch("op_bench.runner.attempt.create_client", side_effect=client) as create:
            report = run_experiment(path, self.root / "run")
            calls = create.call_count
            again = run_experiment(path, self.root / "run", resume=True)
            with patch("op_bench.runner.experiment.TOOL_VERSION", "changed-tools"), \
                 patch("op_bench.runner.experiment.HARNESS_VERSION", "changed-harness"):
                historical = run_experiment(path, self.root / "run", resume=True)
            self.assertEqual(create.call_count, calls)
        self.assertTrue(report["complete"])
        self.assertEqual(report["models"]["fixture-model"]["resolved_rate"], 1)
        self.assertEqual(report["models"]["noop"]["resolved_rate"], 0)
        self.assertEqual(report["models"], again["models"])
        self.assertEqual(report["models"], historical["models"])
        self.assertEqual(len(list((self.root / "run/attempts").glob("*/attempt.json"))), 2)

    def test_expired_model_decision_cannot_edit_workspace(self):
        def delayed():
            time.sleep(.15)
            return decision("replace_text", path="operator_impl.py", old_text="sum(rows[0])", new_text="sum(row)")
        with patch("op_bench.runner.attempt.create_client", return_value=StepClient([delayed])):
            record = run_attempt(self.task, MODEL, self.root / "run", harness=HarnessSpec(budget_sec=.1))
        self.assertEqual(record["terminal_status"], "timed_out")
        self.assertEqual(record["evaluation"]["status"], "test_failed")
        self.assertEqual((self.root / "run/patch.diff").read_text(), "")

    def test_service_failure_does_not_become_a_valid_model_score(self):
        with patch("op_bench.runner.attempt.create_client", return_value=StepClient([
            repair_steps()[1], ModelError("fixture_service_unavailable")])):
            raw = run_attempt(self.task, MODEL, self.root / "run")
        self.assertEqual(raw["evaluation"]["status"], "resolved")
        record = load_attempt(self.root / "run/attempt.json")
        plan = [{key: record[key] for key in ("attempt_id", "task_id", "model_id", "model", "harness", "harness_version", "information_profile", "timing_protocol_revision", "tool_version", "task_identity", "dataset_identity", "software_identity")}]
        report = build_report([record], planned=plan)
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"][MODEL.model_id]["errors"], 1)

    def test_invalid_model_tool_request_stops_and_scores_existing_patch(self):
        for repaired in (False, True):
            with self.subTest(repaired=repaired):
                steps = ([repair_steps()[1]] if repaired else []) + [decision("unknown_tool")]
                output = self.root / ("repaired" if repaired else "unchanged")
                with patch("op_bench.runner.attempt.create_client", return_value=StepClient(steps)):
                    run_attempt(self.task, MODEL, output)
                record = load_attempt(output / "attempt.json")
                self.assertEqual(record["terminal_status"], "model_protocol_error")
                self.assertEqual(record["evaluation"]["status"], "resolved" if repaired else "test_failed")
                plan = [{key: record[key] for key in ("attempt_id", "task_id", "model_id", "model", "harness", "harness_version", "information_profile", "timing_protocol_revision", "tool_version", "task_identity", "dataset_identity", "software_identity")}]
                report = build_report([record], planned=plan)
                self.assertTrue(report["complete"])
                self.assertEqual(report["models"][MODEL.model_id]["resolved"], int(repaired))
                self.assertEqual(report["models"][MODEL.model_id]["errors"], 0)

    def test_failed_preparation_keeps_plan_and_can_resume(self):
        experiment = {"schema_version": 2, "dataset": "task/dataset.json", "information_profile": "development",
                      "models": [MODEL.to_dict()]}
        path = self.root / "experiment.json"
        path.write_text(json.dumps(experiment))
        with patch("op_bench.runner.preparation.snapshot_task", side_effect=OSError("temporary source copy unavailable")):
            with self.assertRaises(OSError):
                run_experiment(path, self.root / "run")
        report = json.loads((self.root / "run/report.json").read_text())
        self.assertEqual(report["models"][MODEL.model_id]["missing"], 1)
        with patch("op_bench.runner.attempt.create_client", return_value=StepClient(repair_steps())):
            report = run_experiment(path, self.root / "run", resume=True)
        self.assertTrue(report["complete"])
        plan = json.loads((self.root / "run/plan.json").read_text())
        self.assertGreaterEqual(len(plan["preparation"]["tasks"][0]["attempts"]), 2)

    def test_changed_protocol_cannot_resume_unstarted_model_calls(self):
        path = self.root / "experiment.json"
        path.write_text(json.dumps({"schema_version": 2, "dataset": "task/dataset.json",
            "information_profile": "development", "models": [MODEL.to_dict()]}))
        with patch("op_bench.runner.preparation.snapshot_task", side_effect=OSError("copy interrupted")):
            with self.assertRaises(OSError):
                run_experiment(path, self.root / "run")
        for field, reason in (("TOOL_VERSION", "Tool protocol"), ("HARNESS_VERSION", "Harness protocol")):
            with self.subTest(protocol=field), \
                 patch("op_bench.runner.experiment." + field, "changed-protocol"), \
                 patch("op_bench.runner.attempt.create_client") as create:
                with self.assertRaisesRegex(ValueError, reason + " differs"):
                    run_experiment(path, self.root / "run", resume=True)
                create.assert_not_called()
        report = json.loads((self.root / "run/report.json").read_text())
        self.assertFalse(report["complete"])
        self.assertEqual(report["models"][MODEL.model_id]["missing"], 1)

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER") == "1", "Docker explicitly enabled")
    def test_docker_tool_build_and_test_have_no_grader_or_controller_credentials(self):
        env = replace(self.task.environment, backend="docker", python="python3",
                      image=os.environ.get("OPBENCH_TEST_DOCKER_IMAGE", "opbench/cpp-fixture:cpu"),
                      build=(("{python}", "-c", "import os; assert 'OPBENCH_SECRET_TEST' not in os.environ"),))
        task = replace(self.task, environment=env, public_commands=(("{python}", "-c",
            "import os,pathlib; assert not pathlib.Path('/grader').exists(); assert 'OPBENCH_SECRET_TEST' not in os.environ"),))
        steps = repair_steps()[:-1] + [decision("build"), decision("run_public_tests"), repair_steps()[-1]]
        with patch.dict(os.environ, {"OPBENCH_SECRET_TEST": "controller-only"}), \
             patch("op_bench.runner.attempt.create_client", return_value=StepClient(steps)):
            record = run_attempt(task, MODEL, self.root / "run")
        self.assertEqual(record["evaluation"]["status"], "resolved")
        self.assertEqual(record["information_boundary"]["status"], "enforced")
        self.assertEqual(record["cleanup_errors"], [])
        self.assertTrue(record["environment"]["cleanup_complete"])


if __name__ == "__main__":
    unittest.main()
