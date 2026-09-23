"""Behavior checks for actual patch evaluation, independent of installed agents."""

import json
import os
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

from op_bench.data.task import TaskInputError, TaskSpec
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runtime.execution import materialize_source
from op_bench.runtime.execution import DockerSession, ExecutionError, create_session, run_process


BUGGY = "def row_sum(rows):\n    return [sum(row[:-1]) for row in rows]\n"
FIXED = "def row_sum(rows):\n    return [sum(row) for row in rows]\n"
ALTERNATIVE = """def row_sum(rows):
    result = []
    for row in rows:
        accumulator = 0
        for value in row:
            accumulator += value
        result.append(accumulator)
    return result
"""


class EvaluationBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-core-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "task"
        self.source = self.bundle / "source"
        self.grader = self.bundle / "grader"
        self.source.mkdir(parents=True)
        self.grader.mkdir()
        (self.source / "ops.py").write_text(BUGGY)
        (self.source / "obsolete.py").write_text("OLD = True\n")
        self.git("init", "--quiet")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "buggy operator")
        (self.grader / "check.py").write_text(
            "import pathlib, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from ops import row_sum\n"
            "if sys.argv[2] == 'sum':\n"
            "    assert row_sum([[1, 2], [-3, 4]]) == [3, 1]\n"
            "elif sys.argv[2] == 'empty':\n"
            "    assert row_sum([[], []]) == [0, 0]\n"
            "elif sys.argv[2] == 'files':\n"
            "    assert (pathlib.Path(sys.argv[1]) / 'helper.py').is_file()\n"
            "    assert not (pathlib.Path(sys.argv[1]) / 'obsolete.py').exists()\n"
            "    assert row_sum([[4, 5]]) == [9]\n"
            "else:\n"
            "    raise ValueError('Unknown selector')\n"
        )
        self.raw = {
            "schema_version": 2, "task_id": "row-sum", "statement": "Sum every element in each row, preserving empty-row behavior.",
            "task_revision": "1", "scoring_revision": "1", "scope": "operator", "defect_group": "fixture:omitted-column",
            "source": {"path": "source", "revision": "HEAD"},
            "environment": {"backend": "local", "python": sys.executable,
                            "environment_id": "fixture-stdlib-local", "revision": "1"},
            "grader_dir": "grader",
            "tests": [
                {"id": "sum", "group": "fail_to_pass", "argv": ["{python}", "{grader}/check.py", "{workspace}", "sum"], "timeout_sec": 2},
                {"id": "empty", "group": "pass_to_pass", "argv": ["{python}", "{grader}/check.py", "{workspace}", "empty"], "timeout_sec": 2},
            ],
        }
        self.counter = 0

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], text=True,
                              capture_output=True, check=True).stdout

    def task(self):
        path = self.bundle / "task.json"
        path.write_text(json.dumps(self.raw))
        return TaskSpec.load(path)

    def candidate(self, content):
        (self.source / "ops.py").write_text(content)
        return self.git("diff", "--binary", "HEAD")

    def evaluate(self, candidate, task=None):
        self.counter += 1
        output = self.root / f"result-{self.counter}"
        result = PatchEvaluator().evaluate(task or self.task(), candidate, output)
        self.assertEqual(result, json.loads((output / "result.json").read_text()))
        self.assertEqual(candidate, (output / "patch.diff").read_text())
        return result, output

    def test_baseline_fails_gold_and_independent_alternative_pass(self):
        baseline, _ = self.evaluate("")
        self.assertEqual("test_failed", baseline["status"])
        self.assertEqual({"passed": 0, "total": 1}, baseline["groups"]["fail_to_pass"])
        self.assertEqual({"passed": 1, "total": 1}, baseline["groups"]["pass_to_pass"])
        for candidate in (FIXED, ALTERNATIVE):
            result, _ = self.evaluate(self.candidate(candidate))
            self.assertEqual("resolved", result["status"])
            self.assertTrue(result["resolved"])
        self.assertTrue(baseline["environment"]["development_only"])

    def test_regression_prevents_resolution_even_when_target_case_passes(self):
        result, _ = self.evaluate(self.candidate("def row_sum(rows):\n    return [3, 1]\n"))
        self.assertEqual("test_failed", result["status"])
        self.assertEqual(["passed", "failed"], [case["status"] for case in result["cases"]])
        self.assertFalse(result["resolved"])

    def test_invalid_patch_is_not_an_environment_retry(self):
        result, output = self.evaluate("this is not a patch\n")
        self.assertEqual("invalid_patch", result["status"])
        self.assertTrue((output / "logs/patch-check.log").is_file())
        self.assertTrue(all(case["status"] == "not_run" for case in result["cases"]))

    def test_candidate_timeout_is_a_test_failure_with_symptom(self):
        self.raw["tests"][0]["timeout_sec"] = 0.1
        self.raw["tests"][1]["timeout_sec"] = 0.1
        result, _ = self.evaluate(self.candidate("def row_sum(rows):\n    while True:\n        pass\n"))
        self.assertEqual("test_failed", result["status"])
        self.assertTrue(result["cases"][0]["timed_out"])
        self.assertNotEqual(0, result["cases"][0]["exit_code"])

    def test_output_limit_stops_execution_and_is_a_test_failure(self):
        self.raw["tests"][0]["argv"] = ["{python}", "-c", "print('x' * 512)"]
        def run_with_small_limit(*args, **kwargs):
            return run_process(*args, **{**kwargs, "max_output_bytes": 128})
        with mock_patch("op_bench.runtime.execution.run_process", side_effect=run_with_small_limit):
            result, output = self.evaluate("")
        self.assertEqual("test_failed", result["status"])
        self.assertTrue(result["cases"][0]["output_limited"])
        self.assertEqual("failed", result["cases"][0]["status"])
        self.assertEqual(128, (output / "logs/test-000.log").stat().st_size)

    def unittest_case(self, script):
        (self.grader / "unit_case.py").write_text(script)
        self.raw["tests"][0].update(kind="unittest", expected_tests=1,
            argv=["{python}", "{grader}/unit_case.py", "{workspace}"])

    def test_unittest_records_actual_case_execution(self):
        self.unittest_case(
            "import sys, unittest\n"
            "sys.path.insert(0, sys.argv.pop(1))\n"
            "from ops import row_sum\n"
            "class OperatorTests(unittest.TestCase):\n"
            "    def test_sum(self):\n"
            "        self.assertEqual([3, 1], row_sum([[1, 2], [-3, 4]]))\n"
            "unittest.main()\n"
        )
        baseline, _ = self.evaluate("")
        self.assertEqual("test_failed", baseline["status"])
        self.assertEqual(1, baseline["cases"][0]["witness"]["failures"])
        fixed, output = self.evaluate(self.candidate(FIXED))
        self.assertEqual("resolved", fixed["status"])
        self.assertEqual(1, fixed["cases"][0]["witness"]["tests_run"])
        self.assertEqual(0, fixed["cases"][0]["witness"]["skipped"])
        self.assertTrue((output / "logs/test-000.witness.json").is_file())

    def test_unittest_empty_script_and_normal_early_exit_cannot_pass(self):
        for script in ("", "raise SystemExit(0)\n"):
            self.unittest_case(script)
            result, _ = self.evaluate(self.candidate(FIXED))
            self.assertEqual("test_failed", result["status"])
            self.assertEqual(0, result["cases"][0]["exit_code"])
            self.assertEqual("missing_unittest_execution_witness", result["cases"][0]["reason"])

    def test_unittest_empty_suite_and_skipped_case_do_not_count_as_passes(self):
        scripts = [
            ("import unittest\nunittest.main(argv=['test'])\n", "unexpected_unittest_test_count"),
            ("import unittest\nclass OperatorTests(unittest.TestCase):\n"
             "    @unittest.skip('fixture has no available implementation')\n"
             "    def test_sum(self):\n        pass\n"
             "unittest.main(argv=['test'])\n", "unittest_case_failed_or_was_not_completed"),
        ]
        for script, reason in scripts:
            self.unittest_case(script)
            result, _ = self.evaluate(self.candidate(FIXED))
            self.assertEqual("test_failed", result["status"])
            self.assertEqual(reason, result["cases"][0]["reason"])

    def test_unittest_requires_the_declared_number_of_cases(self):
        self.unittest_case("import unittest\nclass OperatorTests(unittest.TestCase):\n"
                           "    def test_sum(self):\n        self.assertEqual(3, 1+2)\n"
                           "unittest.main(argv=['test'])\n")
        self.raw["tests"][0]["expected_tests"] = 2
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("test_failed", result["status"])
        self.assertEqual("unexpected_unittest_test_count", result["cases"][0]["reason"])
        self.raw["tests"][0]["expected_tests"] = 0
        with self.assertRaises(TaskInputError):
            self.task()

    def test_added_and_deleted_sources_take_effect(self):
        (self.source / "helper.py").write_text("def total(values):\n    return sum(values)\n")
        (self.source / "obsolete.py").unlink()
        self.git("add", "--intent-to-add", "helper.py")
        candidate = self.candidate("from helper import total\n\ndef row_sum(rows):\n    return [total(row) for row in rows]\n")
        self.raw["tests"].append({"id": "files", "group": "fail_to_pass",
            "argv": ["{python}", "{grader}/check.py", "{workspace}", "files"]})
        result, _ = self.evaluate(candidate)
        self.assertEqual("resolved", result["status"])

    def test_prepare_precedes_patch_and_build_sees_modified_source(self):
        self.raw["environment"]["environment"] = {"OPBENCH_TEST_VALUE": "available"}
        self.raw["environment"]["prepare"] = [["{python}", "-c",
            "import os,pathlib; assert os.environ['OPBENCH_TEST_VALUE']=='available'; "
            "assert 'row[:-1]' in pathlib.Path('ops.py').read_text(); pathlib.Path('prepared').touch()"]]
        self.raw["environment"]["build"] = [["{python}", "-c",
            "import pathlib,py_compile; assert pathlib.Path('prepared').exists(); "
            "assert 'row[:-1]' not in pathlib.Path('ops.py').read_text(); py_compile.compile('ops.py', doraise=True)"]]
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("resolved", result["status"])
        self.assertEqual(["prepare", "patch-check", "patch-apply", "build"],
                         [stage["stage"] for stage in result["stages"]])

    def test_build_and_grading_use_distinct_sessions_and_workspace_dependencies(self):
        self.raw["environment"]["environment"] = {"PYTHONPATH": "{workspace}/deps"}
        self.raw["environment"]["prepare"] = [["{python}", "-c",
            "from pathlib import Path; import os,tempfile; "
            "assert Path(tempfile.gettempdir()).is_relative_to(Path.cwd()); "
            "Path('deps').mkdir(); Path('deps/helper.py').write_text('def total(row): return sum(row)\\n'); "
            "Path(os.environ['TMPDIR'],'prepared.txt').write_text('portable')"]]
        self.raw["environment"]["build"] = [["{python}", "-c",
            "from pathlib import Path; import helper,os; "
            "assert Path(os.environ['TMPDIR'],'prepared.txt').read_text()=='portable'; "
            "assert helper.total([2,3])==5"]]
        observed = []
        @contextmanager
        def record_session(task, workspace, grader_dir=None, **kwargs):
            if observed:
                self.assertTrue(observed[0].closed)
                self.assertTrue(observed[0].workspace_capture_ready)
            with create_session(task, workspace, grader_dir, **kwargs) as session:
                observed.append(session)
                yield session
        with mock_patch("op_bench.evaluation.evaluator.create_session", side_effect=record_session):
            result, _ = self.evaluate(self.candidate("from helper import total\ndef row_sum(rows):\n    return [total(row) for row in rows]\n"))
        self.assertTrue(result["resolved"])
        self.assertEqual(2, len(observed))
        self.assertIsNone(observed[0].grader_dir)
        self.assertEqual(self.grader.resolve(), observed[1].grader_dir)
        self.assertNotEqual(result["sessions"]["build"]["session_id"], result["sessions"]["grading"]["session_id"])
        self.assertFalse(result["sessions"]["build"]["grader_access"])
        self.assertTrue(result["sessions"]["grading"]["grader_access"])
        self.assertTrue(result["artifact_contract"]["build_capture_ready"])

    def test_grader_placeholder_is_unavailable_during_preparation(self):
        self.raw["environment"]["prepare"] = [["{python}", "{grader}/check.py"]]
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("environment_error", result["status"])
        self.assertIn("no grader access", result["error"]["message"])
        self.assertFalse(result["sessions"]["build"]["grader_access"])
        self.assertIsNone(result["sessions"]["grading"])

    def test_failed_final_build_capture_prevents_scoring(self):
        from op_bench.evaluation.verification import verify_evaluation

        @contextmanager
        def missing_capture(task, workspace, grader_dir=None, **kwargs):
            self.assertIsNone(grader_dir)
            with create_session(task, workspace, grader_dir, **kwargs) as session:
                yield session
                session.workspace_capture_ready = False
                session.cleanup_errors.append("Fixture final artifact download unavailable")
        with mock_patch("op_bench.evaluation.evaluator.create_session", side_effect=missing_capture) as factory:
            result, output = self.evaluate(self.candidate(FIXED))
        self.assertEqual(1, factory.call_count)
        self.assertEqual("environment_error", result["status"])
        self.assertEqual("build_capture", result["error"]["stage"])
        self.assertIsNone(result["sessions"]["grading"])
        self.assertTrue(all(case["status"] == "not_run" for case in result["cases"]))
        self.assertIn("build: Fixture final artifact download unavailable", result["cleanup_errors"])
        self.assertTrue(verify_evaluation(output)["valid"])

    def test_scoring_start_failure_preserves_build_session_evidence(self):
        @contextmanager
        def failed_grading(task, workspace, grader_dir=None, **kwargs):
            if grader_dir is not None:
                error = ExecutionError("Fixture grading environment unavailable")
                error.execution_session_info = {"backend": "local", "development_only": True,
                    "grader_access": True, "cleanup_errors": ["Fixture grading cleanup detail"]}
                raise error
            with create_session(task, workspace, grader_dir, **kwargs) as session:
                yield session
        with mock_patch("op_bench.evaluation.evaluator.create_session", side_effect=failed_grading):
            result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("environment_error", result["status"])
        self.assertEqual("grading_environment", result["error"]["stage"])
        self.assertTrue(result["sessions"]["build"]["closed"])
        self.assertTrue(result["artifact_contract"]["build_capture_ready"])
        self.assertIn("grading: Fixture grading cleanup detail", result["cleanup_errors"])

    def test_declared_external_user_install_location_is_environment_error(self):
        self.raw["environment"]["environment"] = {"PYTHONUSERBASE": "/tmp/nonportable-userbase"}
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("environment_error", result["status"])
        self.assertIn("workspace-only evaluation requires PYTHONUSERBASE", result["error"]["message"])
        self.assertIsNone(result["sessions"]["grading"])

    def test_cleanup_exception_keeps_the_original_build_failure(self):
        from op_bench.runtime.execution import ExecutionSession

        self.raw["environment"]["build"] = [["{python}", "-c", "raise RuntimeError('ordinary build failure')"]]
        with mock_patch.object(ExecutionSession, "close", side_effect=OSError("fixture cleanup unavailable")):
            result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("build_failed", result["status"])
        self.assertEqual("build", result["error"]["stage"])
        self.assertTrue(any("fixture cleanup unavailable" in item for item in result["cleanup_errors"]))
        self.assertIsNone(result["sessions"]["grading"])

    def test_prepare_failure_and_build_failure_are_distinct(self):
        self.raw["environment"]["prepare"] = [["{python}", "-c", "raise RuntimeError('environment unavailable')"]]
        result, _ = self.evaluate(self.candidate(FIXED))
        self.assertEqual("environment_error", result["status"])
        self.raw["environment"]["prepare"] = []
        self.raw["environment"]["build"] = [["{python}", "-m", "py_compile", "ops.py"]]
        result, _ = self.evaluate(self.candidate("this is invalid Python syntax !\n"))
        self.assertEqual("build_failed", result["status"])

    def test_unknown_exception_and_interrupt_leave_a_result(self):
        task = self.task()
        with mock_patch("op_bench.evaluation.evaluator.PatchEvaluator._tests", side_effect=RuntimeError("unexpected failure")):
            result, _ = self.evaluate("", task)
        self.assertEqual("evaluation_error", result["status"])
        self.assertEqual("tests", result["error"]["stage"])
        output = self.root / "interrupted"
        with mock_patch("op_bench.evaluation.evaluator.PatchEvaluator._tests", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                PatchEvaluator().evaluate(task, "", output)
        self.assertEqual("KeyboardInterrupt", json.loads((output / "result.json").read_text())["error"]["type"])

    def test_no_cases_and_duplicate_selectors_are_input_errors(self):
        original = self.raw["tests"]
        self.raw["tests"] = []
        with self.assertRaises(TaskInputError):
            self.task()
        self.raw["tests"] = [original[0], {**original[0], "id": "duplicate-command"}]
        with self.assertRaises(TaskInputError):
            self.task()

    def test_visible_task_omits_private_grader_and_metadata(self):
        self.raw["metadata"] = {"reference_patch": "private answer"}
        visible = self.task().visible_dict()
        self.assertNotIn("tests", visible)
        self.assertNotIn("grader_dir", visible)
        self.assertNotIn("metadata", visible)
        self.assertNotIn("source", visible)

    def test_source_is_the_requested_revision_without_history_or_answers(self):
        task = self.task()
        self.candidate(FIXED)
        (self.bundle / "gold.patch").write_text("private answer")
        destination = self.root / "materialized"
        info = materialize_source(task, destination)
        self.assertEqual("git", info["kind"])
        self.assertEqual(BUGGY, (destination / "ops.py").read_text())
        self.assertFalse((destination / ".git").exists())
        self.assertFalse((destination / "gold.patch").exists())
        self.raw["source"]["path"] = "."
        with self.assertRaises(TaskInputError):
            self.task()

    def test_unavailable_submodule_is_an_explicit_environment_failure(self):
        commit = self.git("rev-parse", "HEAD").strip()
        self.git("update-index", "--add", "--cacheinfo", f"160000,{commit},vendor")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "record unavailable submodule")
        result, _ = self.evaluate("")
        self.assertEqual("environment_error", result["status"])
        self.assertIn("Prepare required submodule", result["error"]["message"])

    def test_available_submodule_content_is_exported_at_required_commit(self):
        dependency = self.root / "dependency"
        dependency.mkdir()
        (dependency / "kernel.py").write_text("def identity(value):\n    return value\n")
        def dependency_git(*args):
            return subprocess.run(["git", "-C", str(dependency), *args], check=True,
                                  capture_output=True, text=True).stdout
        dependency_git("init", "--quiet")
        dependency_git("add", ".")
        dependency_git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "kernel")
        self.git("-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qam", "vendor kernel")
        destination = self.root / "with-submodule"
        materialize_source(self.task(), destination)
        self.assertEqual((dependency / "kernel.py").read_text(), (destination / "vendor/kernel.py").read_text())
        self.assertFalse((destination / "vendor/.git").exists())

    def test_directory_sources_are_supported_without_git(self):
        import shutil

        shutil.rmtree(self.source / ".git")
        self.raw["source"].pop("revision")
        result, _ = self.evaluate("")
        self.assertEqual("directory_snapshot", result["source"]["kind"])
        self.assertEqual("test_failed", result["status"])

    def test_final_workspace_capture_is_distinct_from_resource_cleanup(self):
        from dataclasses import replace

        task = self.task()
        task = replace(task, environment=replace(task.environment, backend="docker", image="fixture"))
        def allocated_session():
            session = DockerSession(task, self.source, self.grader)
            session._container_created = True
            session._workspace_uploaded = True
            return session
        successful_control = subprocess.CompletedProcess([], 0, "true\n", "")
        session = allocated_session()
        self.assertFalse(session.info["workspace_capture_ready"])
        with mock_patch.object(session, "_control", return_value=successful_control), \
             mock_patch.object(session, "sync_from_container", side_effect=ExecutionError("copy unavailable")):
            session.close()
        self.assertFalse(session.workspace_capture_ready)
        self.assertTrue(session.cleanup_complete)
        self.assertIn("Download final workspace", session.cleanup_errors[0])

        session = allocated_session()
        def fail_container_removal(argv, **kwargs):
            if argv[:2] == ["rm", "--force"]:
                raise ExecutionError("container removal unavailable")
            return successful_control
        with mock_patch.object(session, "_control", side_effect=fail_container_removal), \
             mock_patch.object(session, "sync_from_container"):
            session.close()
        self.assertTrue(session.workspace_capture_ready)
        self.assertFalse(session.cleanup_complete)
        with mock_patch.object(session, "_control", return_value=successful_control), \
             mock_patch.object(session, "sync_from_container") as download:
            session.close()
            download.assert_not_called()
        self.assertTrue(session.workspace_capture_ready)
        self.assertTrue(session.cleanup_complete)

    def test_docker_witness_reader_distinguishes_missing_file_and_collection_error(self):
        import base64

        session = DockerSession(self.task(), self.source, self.grader)
        witness = {"schema_version": 1, "runner_invocations": 1, "tests_run": 1,
                   "failures": 0, "errors": 0, "skipped": 0,
                   "expected_failures": 0, "unexpected_successes": 0}
        encoded = base64.b64encode(json.dumps(witness).encode()).decode()
        response = subprocess.CompletedProcess([], 0, json.dumps({"state": "present", "data": encoded}), "")
        host_path = self.root / "witness.json"
        with mock_patch.object(session, "_control", return_value=response):
            self.assertEqual(witness, session._read_unittest_witness("/tmp/witness.json", host_path))
        missing = subprocess.CompletedProcess([], 0, '{"state":"missing"}', "")
        with mock_patch.object(session, "_control", return_value=missing):
            self.assertIsNone(session._read_unittest_witness("/tmp/absent.json", host_path))
        with mock_patch.object(session, "_control", side_effect=ExecutionError("daemon unavailable")):
            with self.assertRaises(ExecutionError):
                session._read_unittest_witness("/tmp/witness.json", host_path)

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER_IMAGE"), "Set OPBENCH_TEST_DOCKER_IMAGE for Docker integration")
    def test_docker_build_has_no_grader_and_terminates_before_new_grading_session(self):
        self.raw["environment"].update(backend="docker", python="python3", image=os.environ["OPBENCH_TEST_DOCKER_IMAGE"])
        self.raw["environment"]["prepare"] = [["{python}", "-c",
            "from pathlib import Path; import tempfile; "
            "assert not Path('/grader').exists(); "
            "Path(tempfile.gettempdir(),'dependency.txt').write_text('prepared workspace dependency')"]]
        self.raw["environment"]["build"] = [["{python}", "-c",
            "from pathlib import Path; import subprocess,sys; "
            "assert not Path('/grader').exists(); "
            "Path('helper.py').write_text('def total(row): return sum(row)\\n'); "
            "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
            "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)"]]
        self.raw["tests"].append({"id": "workspace_dependency", "group": "pass_to_pass",
            "argv": ["{python}", "-c", "from pathlib import Path; import tempfile; "
                "assert Path('/grader/check.py').is_file(); "
                "assert Path(tempfile.gettempdir(),'dependency.txt').read_text()=='prepared workspace dependency'"]})
        observed = []
        @contextmanager
        def record_session(task, workspace, grader_dir=None, **kwargs):
            if observed:
                previous = observed[0]
                self.assertTrue(previous.closed)
                self.assertTrue(previous.workspace_capture_ready)
                self.assertTrue(previous.cleanup_complete)
                inspected = subprocess.run(["docker", "inspect", previous.container],
                    capture_output=True, timeout=15)
                self.assertNotEqual(0, inspected.returncode)
                self.assertEqual(previous.image_id, task.environment.image)
            with create_session(task, workspace, grader_dir, **kwargs) as session:
                observed.append(session)
                yield session
        with mock_patch("op_bench.evaluation.evaluator.create_session", side_effect=record_session):
            result, output = self.evaluate(self.candidate("from helper import total\ndef row_sum(rows):\n    return [total(row) for row in rows]\n"))
        self.assertEqual("resolved", result["status"], result["error"])
        self.assertEqual(2, len(observed))
        self.assertFalse(result["sessions"]["build"]["grader_access"])
        self.assertTrue(result["sessions"]["grading"]["grader_access"])
        self.assertNotEqual(result["sessions"]["build"]["session_id"], result["sessions"]["grading"]["session_id"])
        self.assertEqual(result["sessions"]["build"]["image_id"], result["sessions"]["grading"]["image_id"])
        self.assertEqual([], result["cleanup_errors"])
        from op_bench.evaluation.verification import verify_evaluation
        self.assertTrue(verify_evaluation(output)["valid"])

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER_IMAGE"), "Set OPBENCH_TEST_DOCKER_IMAGE for Docker integration")
    def test_docker_unittest_witness_reports_real_execution(self):
        from dataclasses import replace

        self.unittest_case(
            "import sys, unittest\n"
            "sys.path.insert(0, sys.argv.pop(1))\n"
            "from ops import row_sum\n"
            "class OperatorTests(unittest.TestCase):\n"
            "    def test_sum(self):\n        self.assertEqual([3], row_sum([[1, 2]]))\n"
            "unittest.main()\n"
        )
        task = self.task()
        task = replace(task, environment=replace(task.environment, backend="docker", python="python3",
                                                image=os.environ["OPBENCH_TEST_DOCKER_IMAGE"]))
        baseline, _ = self.evaluate("", task)
        self.assertEqual("test_failed", baseline["status"])
        self.assertEqual(1, baseline["cases"][0]["witness"]["tests_run"])
        self.assertEqual(1, baseline["cases"][0]["witness"]["failures"])
        fixed, _ = self.evaluate(self.candidate(FIXED), task)
        self.assertEqual("resolved", fixed["status"])
        self.assertEqual(1, fixed["cases"][0]["witness"]["tests_run"])
        self.assertEqual([], fixed["cleanup_errors"])

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER_IMAGE"), "Set OPBENCH_TEST_DOCKER_IMAGE for Docker integration")
    def test_docker_volume_roundtrip_and_timeout_preserve_the_workspace(self):
        from dataclasses import replace

        task = self.task()
        task = replace(task, environment=replace(task.environment, backend="docker", python="python3",
                                                image=os.environ["OPBENCH_TEST_DOCKER_IMAGE"]))
        workspace = self.root / "docker-workspace"
        materialize_source(task, workspace)
        with create_session(task, workspace, task.grader_dir) as session:
            first = session.run(["{python}", "-c", "from pathlib import Path; "
                "Path('ops.py').write_text('container update'); Path('obsolete.py').unlink(); "
                "Path('added.py').write_text('new'); assert Path('/grader/check.py').is_file()"],
                15, self.root / "docker-first.log")
            self.assertEqual(0, first.exit_code)
            session.sync_from_container()
            self.assertEqual("container update", (workspace / "ops.py").read_text())
            self.assertFalse((workspace / "obsolete.py").exists())
            (workspace / "added.py").unlink()
            (workspace / "host-added.py").write_text("from host")
            session.sync_to_container()
            second = session.run(["{python}", "-c", "from pathlib import Path; import time; "
                "assert not Path('added.py').exists(); assert Path('host-added.py').read_text()=='from host'; "
                "Path('before-timeout.txt').write_text('retained'); time.sleep(10)"],
                1, self.root / "docker-timeout.log")
            self.assertTrue(second.timed_out)
            self.assertTrue(session.closed)
        self.assertEqual("retained", (workspace / "before-timeout.txt").read_text())
        self.assertEqual([], session.cleanup_errors)
        self.assertTrue(session.cleanup_complete)
        self.assertTrue(session.workspace_capture_ready)
        containers = subprocess.run(["docker", "ps", "-a", "--filter", "name=" + session.container,
                                     "--format", "{{.Names}}"], check=True, capture_output=True, text=True).stdout.strip()
        volumes = subprocess.run(["docker", "volume", "ls", "--filter", "label=opbench.session=" + session.container,
                                  "--format", "{{.Name}}"], check=True, capture_output=True, text=True).stdout.strip()
        self.assertEqual("", containers)
        self.assertEqual("", volumes)


if __name__ == "__main__":
    unittest.main()
