"""Capture a framework-sized file tree without changing submission semantics."""
from pathlib import Path
import os
import shutil
import tempfile
import unittest

from op_bench.runtime.submission import freeze_patch, _git, initialize_workspace


@unittest.skipUnless(shutil.which("git"), "Git is required")
class WorkspaceGitScaleTests(unittest.TestCase):
    def test_large_baseline_and_generated_tree_capture_replay(self):
        # The production failure occurred at 86,256 source paths. Exercise a
        # comparable index plus real generated files, not mocked Git arguments.
        with tempfile.TemporaryDirectory(prefix="opbench-git-scale-") as temporary:
            root = Path(temporary).resolve()
            workspace, database = root / "workspace", root / "capture.git"
            workspace.mkdir()
            originals = []
            for bucket in range(90):
                directory = workspace / f"src/bucket-{bucket:03d}"
                directory.mkdir(parents=True)
                for index in range(1000):
                    path = directory / f"unit-{index:04d}.py"
                    path.write_text("value = 0\n")
                    originals.append(path.relative_to(workspace).as_posix())
            initial = {
                ".gitignore": "ignored/\n",
                "ignored/tracked.py": "value = 1\n",
                "retired.py": "obsolete = True\n",
                "run.sh": "#!/bin/sh\nexit 0\n",
                "src/literal[1]*?.py": "value = 2\n",
            }
            for name, content in initial.items():
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
                originals.append(name)
            generated = []
            for bucket in range(30):
                directory = workspace / f"build/bucket-{bucket:03d}"
                directory.mkdir(parents=True)
                for index in range(1000):
                    path = directory / f"unit-{index:04d}.o"
                    path.write_text("generated\n")
                    generated.append(path.relative_to(workspace).as_posix())
            runtime = workspace / "custom/home"
            runtime.mkdir(parents=True)
            (runtime / "model.log").write_text("runtime notes\n")
            reserved = (".opbench", "custom/home")
            initialize_workspace(workspace, database, originals, reserved=reserved)
            self.assertEqual(set(_git(workspace, "ls-files", "-z").split("\0")) - {""}, set(originals))
            self.assertEqual(_git(workspace, "rev-list", "--count", "HEAD").strip(), "1")
            # Generated files are deliberately not a massive .git/info/exclude
            # list. They remain ineligible for submission independently of Git.
            for git_dir in (workspace / ".git", database):
                self.assertEqual((git_dir / "info/exclude").read_text().splitlines(),
                                 ["/.opbench", "/custom/home"])

            changed = "src/bucket-045/unit-0500.py"
            (workspace / changed).write_text("value = 42\n")
            (workspace / "ignored/tracked.py").write_text("value = 3\n")
            (workspace / "src/literal[1]*?.py").write_text("value = 4\n")
            (workspace / "retired.py").unlink()
            (workspace / "run.sh").chmod(0o755)
            (workspace / ".gitignore").write_text("!ignored/\nnew*/\n")
            (workspace / "new_ops").mkdir()
            (workspace / "new_ops/helper.py").write_text("value = 5\n")
            (workspace / "ignored/explicit.py").write_text("value = 6\n")
            (workspace / "ignored/implicit.py").write_text("must not be captured\n")
            (workspace / "build/recipe.json").write_text('{"backend":"new"}\n')
            (workspace / "custom/helper.py").write_text("value = 7\n")
            (workspace / generated[0]).write_text("altered generated output\n")
            _git(workspace, "add", "--force", "--", "ignored/explicit.py", generated[0], "custom/home/model.log")
            frozen = freeze_patch(workspace, database, reserved=reserved, generated=tuple(generated))

            replay = root / "replay"
            replay.mkdir()
            for name, content in {**initial, changed: "value = 0\n"}.items():
                path = replay / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            patch = root / "patch.diff"
            patch.write_text(frozen)
            _git(replay, "apply", str(patch))
            self.assertEqual((replay / changed).read_text(), "value = 42\n")
            self.assertEqual((replay / "ignored/tracked.py").read_text(), "value = 3\n")
            self.assertEqual((replay / "src/literal[1]*?.py").read_text(), "value = 4\n")
            self.assertFalse((replay / "retired.py").exists())
            self.assertTrue(os.access(replay / "run.sh", os.X_OK))
            for name in ("new_ops/helper.py", "ignored/explicit.py", "build/recipe.json", "custom/helper.py"):
                self.assertEqual((replay / name).read_text(), (workspace / name).read_text())
            for name in (generated[0], "ignored/implicit.py", "custom/home/model.log"):
                self.assertFalse((replay / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
