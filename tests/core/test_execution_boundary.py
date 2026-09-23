"""Normal developer workflows and the dedicated model transport boundary."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import uuid

from op_bench.runtime.execution import DockerSession, ExecutionError, create_session, materialize_source
from op_bench.runtime.submission import freeze_patch, initialize_workspace
from op_bench.data.task import TaskSpec
from _support import make_task


class ExecutionBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opbench-boundary-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.task = TaskSpec.load(make_task(self.root / "task"))
        self.workspace = self.root / "workspace"

    def git(self, directory, *arguments, check=True):
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
        return subprocess.run(
            ["git", "-C", str(directory), "-c", "user.name=Fixture",
             "-c", "user.email=fixture@example.invalid", "-c", "commit.gpgsign=false",
             *arguments], text=True, capture_output=True, check=check, timeout=15, env=environment)

    def test_upstream_history_is_removed_but_local_diff_and_commit_work(self):
        source = self.task.source.path
        # This fixture uses ordinary upstream history and a local submodule.
        # No external service is contacted.
        dependency = self.root / "dependency"
        dependency.mkdir()
        (dependency / "helper.py").write_text("VALUE = 1\n")
        self.git(dependency, "init", "--quiet")
        self.git(dependency, "add", ".")
        self.git(dependency, "commit", "-qm", "Dependency baseline")
        self.git(source, "init", "--quiet")
        self.git(source, "-c", "protocol.file.allow=always", "submodule", "add", str(dependency), "vendor")
        self.git(source, "add", ".")
        self.git(source, "commit", "-qm", "Upstream baseline")
        baseline = self.git(source, "rev-parse", "HEAD").stdout.strip()
        original = (source / "operator_impl.py").read_text()
        fixed = original.replace("sum(rows[0])", "sum(row)")
        (source / "operator_impl.py").write_text(fixed)
        self.git(source, "commit", "-qam", "Later upstream fix")
        future = self.git(source, "rev-parse", "HEAD").stdout.strip()
        self.git(source, "remote", "add", "origin", str(self.root / "upstream-remote"))

        task = replace(self.task, source=replace(self.task.source, revision=baseline))
        materialize_source(task, self.workspace)
        self.assertEqual(original, (self.workspace / "operator_impl.py").read_text())
        self.assertEqual([], list(self.workspace.rglob(".git")))
        self.assertEqual("VALUE = 1\n", (self.workspace / "vendor/helper.py").read_text())
        database = self.root / "capture.git"
        paths = [str(path.relative_to(self.workspace)) for path in self.workspace.rglob("*") if path.is_file()]
        initialize_workspace(self.workspace, database, paths)
        self.assertEqual("1", self.git(self.workspace, "rev-list", "--all", "--count").stdout.strip())
        self.assertEqual("", self.git(self.workspace, "remote", "-v").stdout.strip())
        self.assertNotEqual(0, self.git(self.workspace, "cat-file", "-e", future + "^{commit}", check=False).returncode)
        self.assertNotEqual(0, self.git(self.workspace, "cat-file", "-e", baseline + "^{commit}", check=False).returncode)
        self.assertEqual([self.workspace / ".git"], list(self.workspace.rglob(".git")))

        (self.workspace / "operator_impl.py").write_text(fixed)
        self.assertIn("+    return [sum(row)", self.git(self.workspace, "diff").stdout)
        self.git(self.workspace, "commit", "-qam", "Local repair")
        self.assertEqual("2", self.git(self.workspace, "rev-list", "--all", "--count").stdout.strip())
        self.assertEqual("", self.git(self.workspace, "diff").stdout)
        # A normal local commit remains part of the submitted patch even though
        # the Agent's own diff is now empty.
        self.assertIn("+    return [sum(row)", freeze_patch(self.workspace, database))



if __name__ == "__main__":
    unittest.main()
