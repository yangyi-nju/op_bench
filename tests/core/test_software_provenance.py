"""Identify source installs without borrowing an enclosing task's Git history."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

from op_bench import provenance

@unittest.skipUnless(shutil.which("git"), "Git is required")
class SoftwareProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-provenance-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def git(self, directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True,
                                       stderr=subprocess.PIPE).strip()

    def module_file(self, root, *, installed=False):
        parent = root / (".venv/lib/python3.12/site-packages" if installed else "src")
        path = parent / "op_bench/provenance.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(provenance.__file__, path)
        return path

    def repository(self, name, *, project="opbench", source=True):
        root = self.root / name
        root.mkdir()
        (root / "pyproject.toml").write_text(f'[project]\nname = "{project}"\nversion = "1"\n')
        (root / ".gitignore").write_text(".venv/\n__pycache__/\n")
        if source:
            self.module_file(root)
        self.git(root, "init", "-q")
        self.git(root, "add", ".")
        self.git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "-c", "commit.gpgsign=false", "commit", "-qm", "Initial source")
        return root

    def load(self, path):
        spec = importlib.util.spec_from_file_location("provenance_fixture_" + uuid.uuid4().hex, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def assert_package(self, value):
        self.assertEqual("package", value["installation"])
        self.assertIsNone(value["git_revision"])
        self.assertIsNone(value["working_tree_modified"])

    def test_declared_source_checkout_observes_its_actual_commit_and_edits(self):
        root = self.repository("source")
        path = root / "src/op_bench/provenance.py"
        loaded = self.load(path)
        identity = loaded.software_identity()
        self.assertEqual("checkout", identity["installation"])
        self.assertEqual(self.git(root, "rev-parse", "HEAD"), identity["git_revision"])
        self.assertFalse(identity["working_tree_modified"])
        path.write_text(path.read_text() + "\n# An ordinary local source edit.\n")
        self.assertTrue(loaded.software_identity()["working_tree_modified"])

    def test_installed_package_ignores_task_and_opbench_ancestor_repositories(self):
        for project in ("operator-task", "opbench"):
            with self.subTest(project=project):
                root = self.repository(project, project=project)
                installed = self.module_file(root, installed=True)
                self.assert_package(self.load(installed).software_identity())

    def test_matching_module_layout_without_opbench_project_does_not_claim_checkout(self):
        root = self.repository("unrelated-source-layout", project="operator-task")
        self.assert_package(self.load(root / "src/op_bench/provenance.py").software_identity())

    def test_source_export_inside_another_repository_does_not_borrow_its_revision(self):
        outer = self.repository("outer", project="operator-task", source=False)
        exported = outer / "vendor/opbench"
        path = self.module_file(exported)
        (exported / "pyproject.toml").write_text('[project]\nname = "opbench"\n')
        self.assert_package(self.load(path).software_identity())

    def test_git_worktree_file_is_recognized_as_its_own_checkout(self):
        root = self.repository("main-checkout")
        worktree = self.root / "worktree"
        self.git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
        self.assertTrue((worktree / ".git").is_file())
        identity = self.load(worktree / "src/op_bench/provenance.py").software_identity()
        self.assertEqual("checkout", identity["installation"])
        self.assertEqual(self.git(worktree, "rev-parse", "HEAD"), identity["git_revision"])
        self.assertFalse(identity["working_tree_modified"])


if __name__ == "__main__":
    unittest.main()
