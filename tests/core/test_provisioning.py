"""Source recipes are prepared using only local Git fixtures in this suite."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from op_bench.data.task import TaskInputError, TaskSpec
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.runtime.execution import materialize_source
from op_bench.runtime.execution import ExecutionError
from op_bench.runtime.provisioning import SourceProvisionError, publish_directory, provision_source


class SourceProvisioningTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opbench-provision-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.bundle = self.root / "bundle"
        self.bundle.mkdir()
        self.origin = self.repository("origin", {"ops.py": "def add(a, b):\n    return a - b\n"})
        leaf = self.repository("leaf", {"helper.py": "VALUE = 1\n"})
        middle = self.repository("middle", {"kernel.py": "VALUE = 2\n"})
        self.git(middle, "submodule", "add", str(leaf), "nested")
        self.git(middle, "commit", "-qam", "nested dependency")
        self.git(self.origin, "submodule", "add", str(middle), "vendor")
        self.git(self.origin, "commit", "-qam", "operator with dependency")
        self.revision = self.git(self.origin, "rev-parse", "HEAD").strip()
        (self.origin / "ops.py").write_text("def add(a, b):\n    return a + b\n")
        self.git(self.origin, "commit", "-qam", "future fix")
        self.raw = {
            "schema_version": 2, "task_id": "addition", "statement": "Implement addition correctly.",
            "task_revision": "1", "scoring_revision": "1", "scope": "operator", "defect_group": "fixture:addition",
            "source": {"path": "managed/source", "repo_url": "../origin", "revision": self.revision},
            "environment": {"backend": "local", "python": sys.executable,
                            "environment_id": "fixture-stdlib-local", "revision": "1"},
            "tests": [{"id": "add", "group": "fail_to_pass", "argv": ["{python}", "-c", "from ops import add; assert add(1,2)==3"]}],
            "public_commands": [["{python}", "-m", "py_compile", "ops.py"]],
        }

    def git(self, path, *args):
        return subprocess.run(["git", "-c", "core.hooksPath=" + os.devnull,
            "-c", "protocol.file.allow=always", "-c", "user.name=Test",
            "-c", "user.email=test@example.invalid", "-C", str(path), *args],
            capture_output=True, text=True, check=True, timeout=15).stdout

    def repository(self, name, files):
        path = self.root / name
        path.mkdir()
        self.git(path, "init", "--quiet")
        for name, content in files.items():
            (path / name).write_text(content)
        self.git(path, "add", ".")
        self.git(path, "commit", "-qm", "initial source")
        return path

    def task(self):
        path = self.bundle / "task.json"
        path.write_text(json.dumps(self.raw))
        return TaskSpec.load(path)

    def test_recipe_loads_before_source_exists_and_preserves_private_fields(self):
        task = self.task()
        self.assertFalse(task.source.path.exists())
        self.assertEqual("../origin", task.to_dict()["source"]["repo_url"])
        self.assertNotIn("source", task.visible_dict())
        self.assertEqual(self.raw["public_commands"], task.visible_dict()["public_commands"])
        with self.assertRaisesRegex(ExecutionError, "not prepared"):
            materialize_source(task, self.root / "unprepared-export")
        result = PatchEvaluator().evaluate(task, "", self.root / "unprepared-evaluation")
        self.assertEqual("environment_error", result["status"])
        self.assertFalse(result["resolved"])

    def test_missing_source_needs_both_repository_and_revision(self):
        original = dict(self.raw["source"])
        for key in ("revision", "repo_url"):
            self.raw["source"] = {name: value for name, value in original.items() if name != key}
            with self.assertRaises(TaskInputError):
                self.task()
        self.raw["source"] = original
        path = self.bundle / "file"
        path.write_text("user file")
        self.raw["source"]["path"] = "file"
        with self.assertRaises(TaskInputError):
            self.task()
        self.assertEqual("user file", path.read_text())

    def test_prepares_pinned_revision_and_recursive_submodules_without_future_history_in_export(self):
        task = self.task()
        result = provision_source(task)
        self.assertEqual("prepared", result["status"])
        self.assertEqual(self.revision, result["revision"])
        self.assertEqual({"vendor", "vendor/nested"}, {item["path"] for item in result["submodules"]})
        self.assertEqual(self.revision, self.git(task.source.path, "rev-parse", "HEAD").strip())
        self.assertEqual("1", self.git(task.source.path, "rev-list", "--count", "HEAD").strip())
        self.assertIn("a - b", (task.source.path / "ops.py").read_text())
        exported = self.root / "solver-workspace"
        materialize_source(task, exported)
        self.assertTrue((exported / "vendor/nested/helper.py").is_file())
        self.assertFalse(any(path.name == ".git" for path in exported.rglob("*")))
        self.assertNotIn("a + b", (exported / "ops.py").read_text())
        self.assertFalse(list(task.source.path.parent.glob(".opbench-provision-*")))

    def test_existing_dirty_checkout_is_not_reset_or_checked_out(self):
        task = self.task()
        provision_source(task)
        source = task.source.path
        (source / "ops.py").write_text("uncommitted user work\n")
        (source / "scratch.txt").write_text("private local scratch\n")
        before_head = self.git(source, "rev-parse", "HEAD")
        result = provision_source(task)
        self.assertEqual("existing", result["status"])
        self.assertEqual(before_head, self.git(source, "rev-parse", "HEAD"))
        self.assertEqual("uncommitted user work\n", (source / "ops.py").read_text())
        self.assertEqual("private local scratch\n", (source / "scratch.txt").read_text())

    def test_existing_missing_revision_is_rejected_without_fetch_or_mutation(self):
        self.raw["source"]["path"] = "../origin"
        self.raw["source"]["revision"] = "missing-reference"
        task = self.task()
        (self.origin / "keep.txt").write_text("keep")
        before_head = self.git(self.origin, "rev-parse", "HEAD")
        with self.assertRaisesRegex(SourceProvisionError, "new managed source.path") as caught:
            provision_source(task)
        self.assertEqual("failed", caught.exception.result["status"])
        self.assertEqual(before_head, self.git(self.origin, "rev-parse", "HEAD"))
        self.assertEqual("keep", (self.origin / "keep.txt").read_text())
        self.assertNotIn("clone", [stage["stage"] for stage in caught.exception.result["stages"]])

    def test_existing_missing_submodule_requires_a_new_path(self):
        import shutil

        task = self.task()
        provision_source(task)
        shutil.rmtree(task.source.path / "vendor/nested")
        with self.assertRaisesRegex(SourceProvisionError, "new managed source.path"):
            provision_source(task)
        self.assertFalse((task.source.path / "vendor/nested").exists())

    def test_failed_new_clone_cleans_only_owned_temporary_paths(self):
        self.raw["source"]["repo_url"] = "../does-not-exist"
        task = self.task()
        task.source.path.parent.mkdir(parents=True)
        unrelated = task.source.path.parent / "unrelated"
        unrelated.mkdir()
        (unrelated / "user.txt").write_text("preserve")
        with self.assertRaises(SourceProvisionError) as caught:
            provision_source(task)
        self.assertEqual("failed", caught.exception.result["status"])
        self.assertFalse(task.source.path.exists())
        self.assertFalse(list(task.source.path.parent.glob(".opbench-provision-*")))
        self.assertEqual("preserve", (unrelated / "user.txt").read_text())

    def test_publication_never_replaces_a_destination_created_during_fetch(self):
        task = self.task()
        original = publish_directory
        def create_competing_destination(checkout, destination):
            destination.mkdir()
            (destination / "user.txt").write_text("arrived during fetch")
            return original(checkout, destination)
        with patch("op_bench.runtime.provisioning.publish_directory", side_effect=create_competing_destination):
            with self.assertRaisesRegex(SourceProvisionError, "left unchanged"):
                provision_source(task)
        self.assertEqual("arrived during fetch", (task.source.path / "user.txt").read_text())
        self.assertFalse((task.source.path / "ops.py").exists())
        self.assertFalse(list(task.source.path.parent.glob(".opbench-provision-*")))

    def test_git_output_limit_aborts_preparation_and_leaves_no_partial_source(self):
        task = self.task()
        with self.assertRaises(SourceProvisionError) as caught:
            provision_source(task, max_output_bytes=1)
        self.assertTrue(any(stage.get("output_limited") for stage in caught.exception.result["stages"]))
        self.assertFalse(task.source.path.exists())
        self.assertFalse(list(task.source.path.parent.glob(".opbench-provision-*")))

    def test_directory_snapshot_is_recognized_without_git_mutation(self):
        source = self.bundle / "plain"
        source.mkdir()
        (source / "ops.py").write_text("source snapshot")
        self.raw["source"] = {"path": "plain"}
        result = provision_source(self.task())
        self.assertEqual("existing", result["status"])
        self.assertEqual("directory_snapshot", result["kind"])
        self.assertFalse((source / ".git").exists())


if __name__ == "__main__":
    unittest.main()
