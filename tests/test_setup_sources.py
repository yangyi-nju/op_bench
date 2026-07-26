from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.setup_sources import _find_uninitialized_submodules


class FindUninitializedSubmodulesTests(unittest.TestCase):
    def test_ignores_stale_nested_gitmodules_entry_that_is_not_a_gitlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "advice.addEmbeddedRepo", "false"],
                cwd=root,
                check=True,
            )
            parent = root / "third_party" / "parent"
            parent.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=parent, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=parent, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=parent, check=True)
            (root / ".gitmodules").write_text(
                "[submodule \"parent\"]\n"
                "\tpath = third_party/parent\n"
                "\turl = https://example.invalid/parent.git\n",
                encoding="utf-8",
            )
            (parent / ".gitmodules").write_text(
                "[submodule \"stale\"]\n"
                "\tpath = third-party/stale\n"
                "\turl = https://example.invalid/stale.git\n",
                encoding="utf-8",
            )
            (parent / "CMakeLists.txt").write_text("", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=parent, check=True)
            subprocess.run(["git", "commit", "-qm", "parent fixture"], cwd=parent, check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "advice.addEmbeddedRepo=false",
                    "add",
                    "third_party/parent",
                ],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(_find_uninitialized_submodules(root), [])


if __name__ == "__main__":
    unittest.main()
