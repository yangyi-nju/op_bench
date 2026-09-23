"""Finite observations are judged without importing the candidate or its oracle."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from op_bench.data.task import TaskSpec
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runtime.execution import ExecutionSession, run_observation_process
from op_bench.evaluation.numeric import NumericFormatError, judge, validate_oracle
from op_bench.evaluation.verification import verify_evaluation


BUGGY = "import math\ndef softmax(values):\n    numerators = [math.exp(v) for v in values]\n    return [v / sum(numerators) for v in numerators]\n"
FIXED = "import math\ndef softmax(values):\n    numerators = [math.exp(v - max(values)) for v in values]\n    return [v / sum(numerators) for v in numerators]\n"
ALTERNATIVE = "import math\ndef softmax(values):\n    anchor = max(values)\n    log_total = math.log(math.fsum(math.exp(v - anchor) for v in values))\n    return [math.exp((v - anchor) - log_total) for v in values]\n"
MUTANT = "def softmax(values):\n    return [1 / len(values)] * len(values)\n"
WORKER = (
    "import json, pathlib, sys\nfrom ops import softmax\n"
    "data = json.load(sys.stdin)\n"
    "assert set(data) == {'values'}\n"
    "assert not pathlib.Path('/grader').exists()\n"
    "assert not pathlib.Path('previous-case').exists()\n"
    "pathlib.Path('previous-case').write_text('normal per-case state')\n"
    "print('worker diagnostic', file=sys.stderr)\n"
    "output = softmax(data['values'])\n"
    "print(json.dumps({'shape': [len(output)], 'values': output}))\n"
)


def oracle(values, expected):
    return {"schema_version": 1, "input": {"values": values},
            "expected": {"shape": [len(expected)], "values": expected}, "atol": 1e-12, "rtol": 1e-12}


class NumericEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-numeric-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.grader = self.root / "grader"
        self.source.mkdir()
        self.grader.mkdir()
        (self.source / "ops.py").write_text(BUGGY)
        (self.source / "worker.py").write_text(WORKER)
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline")
        # References are fixed independently of candidate imports: these are
        # logistic(1), logistic(0), rounded to binary64 from decimal values.
        (self.grader / "large.json").write_text(json.dumps(oracle([1000, 1001],
            [0.268941421369995120748840758178, 0.731058578630004879251159241822])))
        (self.grader / "zero.json").write_text(json.dumps(oracle([0, 0], [.5, .5])))
        self.raw = {"schema_version": 2, "task_id": "numeric-softmax", "statement": "Stable finite softmax.",
            "task_revision": "1", "scoring_revision": "1", "scope": "operator", "defect_group": "fixture:softmax-stability",
            "source": {"path": "source", "revision": "HEAD"}, "grader_dir": "grader",
            "environment": {"backend": "local", "python": sys.executable,
                            "environment_id": "fixture-stdlib-local", "revision": "1"},
            "tests": [{"id": name, "group": group, "kind": "numeric", "oracle": name + ".json",
                       "argv": ["{python}", "{workspace}/worker.py"], "timeout_sec": 2}
                      for name, group in (("large", "fail_to_pass"), ("zero", "pass_to_pass"))]}
        self.counter = 0

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], text=True, capture_output=True, check=True).stdout

    def candidate(self, content):
        (self.source / "ops.py").write_text(content)
        return self.git("diff", "HEAD")

    def evaluate(self, candidate=""):
        self.counter += 1
        (self.root / "task.json").write_text(json.dumps(self.raw))
        output = self.root / f"result-{self.counter}"
        result = PatchEvaluator().evaluate(TaskSpec.load(self.root / "task.json"), candidate, output)
        self.assertEqual(result, json.loads((output / "result.json").read_text()))
        self.assertTrue(verify_evaluation(output)["valid"], verify_evaluation(output))
        return result, output

    def test_baseline_reference_alternative_and_normalized_mutation(self):
        baseline, output = self.evaluate()
        self.assertEqual("test_failed", baseline["status"])
        self.assertEqual("numeric_process_failed", baseline["cases"][0]["reason"])
        self.assertIn("OverflowError", (output / "logs/test-000.log").read_text())
        self.assertEqual("passed", baseline["cases"][1]["status"])
        for content in (FIXED, ALTERNATIVE):
            fixed, output = self.evaluate(self.candidate(content))
            self.assertEqual("resolved", fixed["status"])
            self.assertTrue(all(case["numeric"]["compared_values"] == 2 for case in fixed["cases"]))
            self.assertNotIn("worker diagnostic", (output / "logs/test-000.observation.json").read_text())
            self.assertIn("worker diagnostic", (output / "logs/test-000.log").read_text())
            infos = [value for value in fixed["sessions"].values() if value]
            self.assertEqual(3, len({info["session_id"] for info in infos}))
            self.assertTrue(all(info["grader_access"] is False and info["execution_stopped"] for info in infos))
        mutant, _ = self.evaluate(self.candidate(MUTANT))
        self.assertEqual("test_failed", mutant["status"])
        self.assertEqual("numeric_value_mismatch", mutant["cases"][0]["reason"])
        self.assertEqual("passed", mutant["cases"][1]["status"])

    def test_absent_and_illegal_observations_are_failures(self):
        for code, reason in (("pass", "missing_numeric_observation"),
                             ("print('not json')", "invalid_numeric_observation"),
                             ("print('{\"shape\": [2], \"values\": [true, false]}')", "invalid_numeric_observation"),
                             ("print('{\"shape\": [2], \"values\": [NaN, 0]}')", "invalid_numeric_observation"),
                             ("print('{\"shape\": [3], \"values\": [1, 2]}')", "invalid_numeric_observation"),
                             ("print('{\"shape\": [1,2], \"values\": [.5,.5]}'.replace('.5', '0.5'))", "numeric_shape_mismatch")):
            with self.subTest(code=code):
                self.raw["tests"][0]["argv"] = ["{python}", "-c", code]
                result, _ = self.evaluate()
                self.assertEqual("test_failed", result["status"])
                self.assertEqual(reason, result["cases"][0]["reason"])
                self.assertEqual(0, result["cases"][0]["exit_code"])

    def test_timeout_nonzero_and_output_limit_keep_execution_symptoms(self):
        for code, limit in (("import time; time.sleep(2)", "timed_out"),
                            ("raise RuntimeError('normal runtime error fixture')", None),
                            ("print('a' * 512)", "output_limited")):
            self.raw["tests"][0].update(argv=["{python}", "-c", code], timeout_sec=.1)
            original = ExecutionSession.observe
            def observe_small(session, *args, **kwargs):
                return original(session, *args, **kwargs, max_output_bytes=256)
            with patch.object(ExecutionSession, "observe", observe_small):
                result, _ = self.evaluate()
            self.assertEqual("test_failed", result["status"])
            self.assertEqual("numeric_process_failed", result["cases"][0]["reason"])
            if limit:
                self.assertTrue(result["cases"][0][limit])

    def test_invalid_private_oracle_is_unknown_without_candidate_execution(self):
        (self.grader / "large.json").write_text('{"schema_version":1}')
        result, _ = self.evaluate()
        self.assertEqual("evaluation_error", result["status"])
        self.assertEqual("numeric_oracle", result["error"]["stage"])
        self.assertEqual("not_run", result["cases"][0]["status"])

    def test_integration_workspace_changes_do_not_replace_frozen_numeric_artifacts(self):
        self.raw["tests"].append({"id": "integration", "group": "pass_to_pass", "kind": "command",
            "argv": ["{python}", "-c", "from pathlib import Path; Path('ops.py').write_text('def softmax(values): return [0.0]*len(values)\\n')"]})
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("resolved", result["status"])
        self.assertTrue(result["sessions"]["grading"]["grader_access"])
        self.assertFalse(result["sessions"]["numeric-000"]["grader_access"])

    def test_verification_recomputes_saved_comparison(self):
        result, output = self.evaluate(self.candidate(FIXED))
        result["cases"][0]["numeric"]["compared_values"] = 99
        (output / "result.json").write_text(json.dumps(result))
        self.assertFalse(verify_evaluation(output)["valid"])
        result["cases"][0]["numeric"]["compared_values"] = 2
        (output / "result.json").write_text(json.dumps(result))
        (output / result["cases"][0]["observation_log"]).write_text('{"shape":[2],"values":[0.5,0.5]}')
        self.assertFalse(verify_evaluation(output)["valid"])

    def test_numeric_judgment_waits_for_stopped_execution(self):
        actual_close = ExecutionSession.close
        def incomplete_close(session):
            actual_close(session)
            if session.workspace.name.startswith("numeric-"):
                session.execution_stopped = False
        with patch.object(ExecutionSession, "close", incomplete_close):
            result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("evaluation_error", result["status"])
        self.assertEqual("numeric_execution_boundary_unavailable", result["cases"][0]["reason"])

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER") == "1", "set OPBENCH_TEST_DOCKER=1 for real Docker execution")
    def test_docker_numeric_uses_only_current_input_and_fresh_no_grader_sessions(self):
        self.raw["environment"] = {"backend": "docker", "python": "python3",
                                   "image": os.environ.get("OPBENCH_TEST_DOCKER_IMAGE", "opbench/numeric-fixture:py3.12"),
                                   "environment_id": "fixture-stdlib-container", "revision": "1"}
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("resolved", result["status"], result.get("error"))
        self.assertEqual([], result["cleanup_errors"])
        for info in result["sessions"].values():
            if info is not None:
                self.assertEqual("none", info["observed_network_mode"])
                self.assertFalse(info["grader_access"])
                self.assertTrue(info["execution_stopped"])


class NumericProtocolTests(unittest.TestCase):
    def test_empty_tensors_and_scalar_are_explicit(self):
        for shape, values in (([0], []), ([], [4])):
            reference = validate_oracle({"schema_version": 1, "input": None,
                "expected": {"shape": shape, "values": values}, "atol": 0, "rtol": 0})
            self.assertTrue(judge(reference, json.dumps(reference["expected"]).encode())["passed"])
        with self.assertRaises(NumericFormatError):
            validate_oracle({"schema_version": 1, "input": None,
                "expected": {"shape": [], "values": []}, "atol": 0, "rtol": 0})

    def test_duplicate_keys_boolean_and_nonfinite_are_rejected(self):
        reference = validate_oracle(oracle([0, 0], [.5, .5]))
        for output in (b'{"shape":[2],"shape":[2],"values":[0.5,0.5]}',
                       b'{"shape":[true],"values":[0.5]}',
                       b'{"shape":[2],"values":[1e999,0.5]}'):
            self.assertEqual("invalid_numeric_observation", judge(reference, output)["reason"])

    def test_large_finite_difference_does_not_overflow_comparison(self):
        reference = validate_oracle(oracle([0], [1e308]))
        decision = judge(reference, b'{"shape":[1],"values":[-1e308]}')
        self.assertFalse(decision["passed"])
        self.assertNotIn("Infinity", decision["max_absolute_error_decimal"])

    def test_input_output_transport_does_not_deadlock_and_stderr_has_own_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = "import sys; sys.stderr.write('z'*512); sys.stderr.flush(); sys.stdin.read()"
            result = run_observation_process([sys.executable, "-c", script], cwd=root, timeout_sec=2,
                input_bytes=b"x" * 131072, output_path=root / "stdout", log_path=root / "stderr", max_log_bytes=256)
            self.assertTrue(result.output_limited)
            self.assertEqual(256, (root / "stderr").stat().st_size)
            self.assertEqual(b"", (root / "stdout").read_bytes())
