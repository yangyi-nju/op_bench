import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from op_bench.cli import main


class PreparationCliTests(unittest.TestCase):
    def test_explicit_fetch_exports_requested_tree_and_records_preparation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, bundle = root / "origin", root / "bundle"
            origin.mkdir()
            bundle.mkdir()

            def git(*args):
                return subprocess.run(["git", "-c", "user.name=Fixture", "-c",
                    "user.email=fixture@example.invalid", "-C", str(origin), *args],
                    check=True, capture_output=True, text=True).stdout.strip()

            git("init", "--quiet")
            (origin / "operator.py").write_text("VALUE = 1\n")
            git("add", ".")
            git("commit", "-qm", "baseline")
            revision = git("rev-parse", "HEAD")
            (origin / "operator.py").write_text("VALUE = 2\n")
            git("commit", "-qam", "later revision")
            declaration = {"schema_version": 2, "task_id": "prepare-fixture",
                "statement": "Repair the operator.",
                "task_revision": "1", "scoring_revision": "1", "scope": "operator", "defect_group": "fixture:value",
                "environment": {"environment_id": "fixture-stdlib-local", "revision": "1"},
                "source": {"path": "managed", "repo_url": "../origin", "revision": revision},
                "tests": [{"id": "value", "group": "fail_to_pass",
                           "argv": ["python3", "-c", "assert False"]}]}
            task = bundle / "task.json"
            task.write_text(json.dumps(declaration))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["prepare", "--task", str(task), "--output", str(root / "no-fetch")]), 2)
                self.assertFalse((bundle / "managed").exists())
                self.assertEqual(main(["prepare", "--task", str(task), "--fetch",
                                       "--output", str(root / "prepared")]), 0)
            exported = root / "prepared/workspace"
            self.assertEqual((exported / "operator.py").read_text(), "VALUE = 1\n")
            self.assertFalse((exported / ".git").exists())
            evidence = json.loads((root / "prepared/preparation.json").read_text())
            self.assertEqual((evidence["status"], evidence["revision"]), ("prepared", revision))
            self.assertNotIn("source", json.loads((root / "prepared/task_input.json").read_text()))
