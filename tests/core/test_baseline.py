from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from op_bench.runtime.baseline import BaselineError, build_baseline, fork_baseline
from op_bench.runtime.execution import ExecutionError, create_session
from op_bench.data.task import TaskSpec
from _support import make_task


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.task_file = make_task(self.root / "task")
        self.raw = json.loads(self.task_file.read_text())

    def task(self):
        self.task_file.write_text(json.dumps(self.raw))
        return TaskSpec.load(self.task_file)

    def test_baseline_is_built_without_grader_or_model_and_forks_are_independent(self):
        self.raw["environment"]["build"] = [[sys.executable, "-c", "from pathlib import Path; Path('built.txt').write_text('baseline')"]]
        task = self.task()
        observed = []

        def observe(task, workspace, grader_dir=None, **kwargs):
            observed.append((grader_dir, kwargs))
            return create_session(task, workspace, grader_dir, **kwargs)

        with patch("op_bench.runtime.baseline.create_session", side_effect=observe):
            record = build_baseline(task, self.root / "baseline")
        self.assertEqual(record["status"], "ready", record)
        self.assertEqual(observed, [(None, {})])
        self.assertTrue(record["session"]["execution_stopped"])
        self.assertFalse((self.root / "baseline/source/built.txt").exists())
        self.assertEqual((self.root / "baseline/workspace/built.txt").read_text(), "baseline")
        first = fork_baseline(task, self.root / "baseline", self.root / "first")
        (self.root / "first/built.txt").write_text("candidate")
        fork_baseline(task, self.root / "baseline", self.root / "second")
        self.assertEqual((self.root / "second/built.txt").read_text(), "baseline")
        self.assertNotIn("built.txt", first["source_files"])
        self.assertFalse(any("grader" in name for name in first["source_files"]))

    def test_prepare_can_change_source_but_build_cannot_rewrite_that_baseline(self):
        self.raw["environment"]["prepare"] = [[sys.executable, "-c", "from pathlib import Path; Path('prepared.py').write_text('VERSION = 1')"]]
        task = self.task()
        record = build_baseline(task, self.root / "allowed")
        self.assertEqual(record["status"], "ready", record)
        self.assertEqual((self.root / "allowed/source/prepared.py").read_text(), "VERSION = 1")
        self.assertIn("prepared.py", record["source_files"])
        self.raw["environment"]["build"] = [[sys.executable, "-c", "from pathlib import Path; Path('prepared.py').write_text('VERSION = 2')"]]
        failed = build_baseline(self.task(), self.root / "rejected")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["stage"], "source_contract")
        self.assertIn("prepared.py", failed["error"]["message"])
        with self.assertRaisesRegex(BaselineError, "completed"):
            fork_baseline(self.task(), self.root / "rejected", self.root / "invalid")

    def test_custom_runtime_dependencies_are_preserved_but_not_submitted_as_source(self):
        self.raw["environment"]["environment"] = {
            "HOME": "{workspace}/support/runtime/home", "TMPDIR": "{runtime}/temporary"}
        self.raw["environment"]["prepare"] = [[sys.executable, "-c",
            "import os; from pathlib import Path; "
            "Path(os.environ['HOME'], 'dependency.txt').write_text('prepared dependency'); "
            "Path('support/source.py').write_text('VALUE = 7')"]]
        task = self.task()
        built = build_baseline(task, self.root / "custom-runtime")
        self.assertEqual(built["status"], "ready", built)
        self.assertIn("support/runtime/home", built["artifact_contract"]["runtime_paths"])
        self.assertIn("support/source.py", built["source_files"])
        self.assertNotIn("support/runtime/home/dependency.txt", built["source_files"])
        forked = fork_baseline(task, self.root / "custom-runtime", self.root / "fork")
        dependency = Path(forked["execution_environment"]["environment"]["HOME"]) / "dependency.txt"
        self.assertEqual(dependency.read_text(), "prepared dependency")
        self.assertTrue(dependency.is_relative_to((self.root / "fork").resolve()))

    def test_runtime_cannot_reserve_existing_source_or_controller_metadata(self):
        for index, home in enumerate(("{workspace}", "{workspace}/operator_impl.py", "{workspace}/.opbench/home")):
            with self.subTest(home=home):
                self.raw["environment"]["environment"] = {"HOME": home}
                built = build_baseline(self.task(), self.root / f"overlap-{index}")
                self.assertEqual(built["status"], "failed", built)
                self.assertIn("overlaps", built["error"]["message"])
                self.assertIsNone(built["session"])

    def test_moved_local_artifact_uses_its_forked_runtime_dependencies(self):
        self.raw["environment"]["prepare"] = [[sys.executable, "-c",
            "import os; from pathlib import Path; "
            "Path(os.environ['HOME'], 'dependency.txt').write_text('prepared dependency')"]]
        task = self.task()
        staging = self.root / "staging"
        built = build_baseline(task, staging)
        self.assertEqual("ready", built["status"], built)
        published = self.root / "published"
        staging.rename(published)
        forked = fork_baseline(task, published, self.root / "candidate")
        variables = forked["execution_environment"]["environment"]
        self.assertTrue(Path(variables["HOME"]).is_relative_to((self.root / "candidate").resolve()))
        candidate = replace(task, environment=replace(task.environment, environment=variables))
        log = self.root / "candidate.log"
        with create_session(candidate, self.root / "candidate") as session:
            observed = session.run([sys.executable, "-c",
                "import os; from pathlib import Path; "
                "print(Path(os.environ['HOME'], 'dependency.txt').read_text())"], 5, log)
        self.assertEqual(0, observed.exit_code)
        self.assertEqual("prepared dependency\n", log.read_text())
        self.assertFalse(staging.exists())

    def test_build_failure_and_missing_capture_never_publish_reusable_artifact(self):
        self.raw["environment"]["build"] = [[sys.executable, "-c", "raise RuntimeError('fixture build failure')"]]
        failed = build_baseline(self.task(), self.root / "failed")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["stage"], "build")
        self.assertIn("fixture build failure", (self.root / "failed/logs/build-000.log").read_text())
        self.raw["environment"]["build"] = []
        from op_bench.runtime.execution import ExecutionSession
        original = ExecutionSession.close

        def no_capture(session):
            original(session)
            session.workspace_capture_ready = False

        with patch.object(ExecutionSession, "close", no_capture):
            failed = build_baseline(self.task(), self.root / "uncaptured")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["stage"], "capture")

    def test_start_failure_keeps_partial_session_resource_evidence(self):
        failure = ExecutionError("fixture startup failure")
        failure.execution_session_info = {"session_id": "fixture", "cleanup_errors": ["resource cleanup incomplete"]}
        with patch("op_bench.runtime.baseline.create_session", side_effect=failure):
            record = build_baseline(self.task(), self.root / "failed")
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"]["stage"], "environment")
        self.assertEqual(record["session"], failure.execution_session_info)

    def test_malformed_image_identity_cannot_publish_pinned_baseline(self):
        task = self.task()
        task = replace(task, environment=replace(task.environment, backend="docker", image="fixture:image"))
        response = subprocess.CompletedProcess([], 0, "sha256:" + "z" * 64 + "\n", "")
        with patch("op_bench.runtime.execution.subprocess.run", return_value=response):
            result = build_baseline(task, self.root / "bad-image")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["stage"], "compatibility")
        self.assertIsNone(result["compatibility"])

    def test_reuse_rejects_changed_task_build_environment_and_source(self):
        task = self.task()
        self.assertEqual(build_baseline(task, self.root / "baseline")["status"], "ready")
        for field, value in (("build", ((sys.executable, "-c", "pass"),)),
                             ("prepare", ((sys.executable, "-c", "pass"),)),
                             ("cpus", 7.0), ("environment", {"MODE": "other"})):
            changed = replace(task, environment=replace(task.environment, **{field: value}))
            with self.subTest(field=field), self.assertRaisesRegex(BaselineError, "incompatible"):
                fork_baseline(changed, self.root / "baseline", self.root / "fork")
        with self.assertRaisesRegex(BaselineError, "incompatible"):
            fork_baseline(task, self.root / "baseline", self.root / "fork", expected_source={"kind": "git", "revision": "different"})
        self.assertFalse((self.root / "fork").exists())

    def test_grading_revision_changes_reuse_same_source_build_but_keep_producer_identity(self):
        task = self.task()
        record = build_baseline(task, self.root / "baseline")
        revised = replace(task, task_revision="review-2", scoring_revision="oracle-2")
        forked = fork_baseline(revised, self.root / "baseline", self.root / "fork")
        self.assertEqual(record["status"], "ready")
        self.assertEqual(forked["producer_task_identity"], task.identity_dict())
        self.assertNotEqual(forked["producer_task_identity"], revised.identity_dict())

    def test_explicit_snapshot_origin_is_compatible_without_inventing_new_source_identity(self):
        task = self.task()
        record = build_baseline(task, self.root / "baseline")
        snapshot = self.root / "snapshot"
        shutil.copytree(self.root / "baseline/source", snapshot)
        snap_task = replace(task, source=replace(task.source, path=snapshot, revision=None))
        with self.assertRaisesRegex(BaselineError, "incompatible"):
            fork_baseline(snap_task, self.root / "baseline", self.root / "not-trusted")
        fork_baseline(snap_task, self.root / "baseline", self.root / "fork",
                      expected_source=record["compatibility"]["source"])

    def test_git_revision_is_resolved_and_future_source_revisions_are_rejected(self):
        task = self.task()
        source = task.source.path
        subprocess.run(["git", "init", "-q"], cwd=source, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=source, check=True, capture_output=True)
        commit = ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm"]
        subprocess.run([*commit, "Baseline"], cwd=source, check=True, capture_output=True)
        record = build_baseline(task, self.root / "baseline")
        self.assertEqual(record["status"], "ready", record)
        self.assertEqual(record["compatibility"]["source"]["kind"], "git")
        self.assertFalse((self.root / "baseline/workspace/.git").exists())
        (source / "new.py").write_text("new = 1")
        subprocess.run(["git", "add", "new.py"], cwd=source, check=True, capture_output=True)
        subprocess.run([*commit, "Later source"], cwd=source, check=True, capture_output=True)
        with self.assertRaisesRegex(BaselineError, "incompatible"):
            fork_baseline(task, self.root / "baseline", self.root / "fork")

    def test_every_required_variant_builds_with_its_own_environment_and_failures_remain_visible(self):
        build = [[sys.executable, "-c", "import os; from pathlib import Path; mode=os.environ['MODE']; assert mode != 'fail'; Path('built.txt').write_text(mode)"]]
        self.raw["variants"] = [
            {"variant_id": "first", "environment": {"backend": "local", "build": build, "environment": {"MODE": "one"},
                "environment_id": "fixture-mode-one", "revision": "1"},
             "tests": [{"id": "first-case", "group": "fail_to_pass", "argv": [sys.executable, "-c", "pass"]}]},
            {"variant_id": "second", "environment": {"backend": "local", "build": build, "environment": {"MODE": "two"},
                "environment_id": "fixture-mode-two", "revision": "1"},
             "tests": [{"id": "second-case", "group": "fail_to_pass", "argv": [sys.executable, "-c", "pass"]}]},
        ]
        task = self.task()
        record = build_baseline(task, self.root / "variants")
        self.assertEqual(record["kind"], "trusted_baseline_variants")
        self.assertEqual(record["status"], "ready", record)
        self.assertEqual(set(record["variants"]), {"first", "second"})
        for name, expected in (("first", "one"), ("second", "two")):
            artifact = self.root / "variants" / record["variants"][name]
            child = record["variant_results"][name]
            self.assertEqual(child["status"], "ready")
            self.assertEqual((artifact / "workspace/built.txt").read_text(), expected)
            fork_baseline(task.as_variant(name), artifact, self.root / ("fork-" + name))
        with self.assertRaisesRegex(BaselineError, "completed"):
            fork_baseline(task.as_variant("first"), self.root / "variants", self.root / "wrong-level")
        self.raw["variants"][0]["environment"]["environment"]["MODE"] = "fail"
        failed = build_baseline(self.task(), self.root / "failed-variants")
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["variant_results"]["first"]["status"], "failed")
        self.assertEqual(failed["variant_results"]["second"]["status"], "ready")

    @unittest.skipUnless(shutil.which("c++") and shutil.which("make"), "C++ toolchain required")
    def test_real_incremental_build_reuses_baseline_and_observes_source_and_build_changes(self):
        source = self.task().source.path
        (source / "main.cpp").write_text("#include <iostream>\n#ifndef VALUE\n#define VALUE 3\n#endif\nint main(){std::cout << VALUE << '\\n';}\n")
        (source / "Makefile").write_text("app: main.cpp Makefile\n\tc++ -DVALUE=3 main.cpp -o app\n")
        self.raw["environment"]["build"] = [[shutil.which("make")]]
        task = self.task()
        record = build_baseline(task, self.root / "baseline")
        self.assertEqual(record["status"], "ready", record)
        original_mtime = (self.root / "baseline/workspace/app").stat().st_mtime_ns
        fork_baseline(task, self.root / "baseline", self.root / "candidate")
        candidate = self.root / "candidate"
        subprocess.run(["make"], cwd=candidate, check=True, capture_output=True)
        self.assertEqual((candidate / "app").stat().st_mtime_ns, original_mtime)
        self.assertEqual(subprocess.check_output([str(candidate / "app")], text=True).strip(), "3")
        (candidate / "main.cpp").write_text((candidate / "main.cpp").read_text().replace("<< VALUE", "<< VALUE * 2"))
        # The consumer invalidates only changed files. This explicitly crosses
        # coarse Make timestamp precision rather than sleeping or touching every
        # baseline source. fork_baseline itself preserves all original mtimes.
        changed_time = max(time.time_ns(), original_mtime + 2_000_000_000)
        os.utime(candidate / "main.cpp", ns=(changed_time, changed_time))
        subprocess.run(["make"], cwd=candidate, check=True, capture_output=True)
        self.assertEqual(subprocess.check_output([str(candidate / "app")], text=True).strip(), "6")
        fork_baseline(task, self.root / "baseline", self.root / "option-change")
        options = self.root / "option-change"
        (options / "Makefile").write_text((options / "Makefile").read_text().replace("-DVALUE=3", "-DVALUE=7"))
        changed_time = max(time.time_ns(), original_mtime + 2_000_000_000)
        os.utime(options / "Makefile", ns=(changed_time, changed_time))
        subprocess.run(["make"], cwd=options, check=True, capture_output=True)
        self.assertEqual(subprocess.check_output([str(options / "app")], text=True).strip(), "7")
        self.assertEqual(subprocess.check_output([str(self.root / "baseline/workspace/app")], text=True).strip(), "3")


if __name__ == "__main__":
    unittest.main()
