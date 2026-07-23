from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    WorkspacePolicyError,
)
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_contracts import SHA_A, identity
from tests.test_runtime_workspace import policy


class ActionWorkspaceListingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)
        self.workspace = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@actions", SHA_A),
            policy=policy(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_list_is_sorted_bounded_and_never_exposes_git_metadata(self) -> None:
        top = self.workspace.list_entries(
            ".",
            recursive=False,
            max_entries=10,
            max_depth=1,
        )
        nested = self.workspace.list_entries(
            "src",
            recursive=True,
            max_entries=10,
            max_depth=3,
        )

        self.assertEqual([entry.path for entry in top], ["src", "tests"])
        self.assertEqual([entry.entry_type for entry in top], ["directory", "directory"])
        self.assertEqual(
            [entry.path for entry in nested],
            ["src/helper.py", "src/operator.py"],
        )
        self.assertTrue(all(entry.entry_type == "file" for entry in nested))
        self.assertFalse(any(".git" in entry.path for entry in (*top, *nested)))

        with self.assertRaisesRegex(WorkspacePolicyError, "max_entries"):
            self.workspace.list_entries(
                ".",
                recursive=True,
                max_entries=1,
                max_depth=3,
            )

    def test_list_rejects_traversal_git_symlink_and_special_file(self) -> None:
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        os.symlink(outside, self.root / "src" / "linked")
        candidates = ("../", ".git", "src/linked")
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.root / "src" / "pipe")
            candidates += ("src/pipe",)

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with self.assertRaises(WorkspacePolicyError):
                    self.workspace.list_entries(
                        candidate,
                        recursive=True,
                        max_entries=10,
                        max_depth=3,
                    )


if __name__ == "__main__":
    unittest.main()
