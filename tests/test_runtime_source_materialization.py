from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.source_materialization import (
    SourceMaterializationError,
    _validate_archive_entry,
    materialize_frozen_git_revision,
)
from tests.runtime_git_fixture import git, initialize_git_repo


def _source_state(source: Path) -> tuple[bytes, bytes, bytes, bytes]:
    return (
        git(source, "rev-parse", "--verify", "HEAD").stdout,
        git(source, "diff", "--binary", "--no-ext-diff", "HEAD").stdout,
        git(source, "ls-files", "--others", "--exclude-standard", "-z").stdout,
        git(
            source,
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ).stdout,
    )


class FrozenSourceMaterializationApiTests(unittest.TestCase):
    def test_materializer_module_exists(self) -> None:
        spec = importlib.util.find_spec(
            "op_bench.runtime.source_materialization"
        )

        self.assertIsNotNone(spec)


class FrozenSourceMaterializationTests(unittest.TestCase):
    def test_snapshot_tree_equals_selected_revision_and_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = initialize_git_repo(source)
            before = _source_state(source)

            first = materialize_frozen_git_revision(source, revision, root / "first")
            self.assertEqual(_source_state(source), before)
            second = materialize_frozen_git_revision(source, revision, root / "second")

            expected_tree = git(
                source,
                "rev-parse",
                "--verify",
                f"{revision}^{{tree}}",
            ).stdout.decode("ascii").strip()
            self.assertEqual(first.source_tree, expected_tree)
            self.assertEqual(second.source_tree, expected_tree)
            self.assertEqual(first.snapshot_commit, second.snapshot_commit)
            self.assertEqual(
                git(first.workspace, "status", "--porcelain=v1").stdout,
                b"",
            )
            self.assertEqual(_source_state(source), before)

    def test_unknown_revision_fails_without_leaving_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            initialize_git_repo(source)
            workspace = root / "workspace"
            before = _source_state(source)

            with self.assertRaises(SourceMaterializationError):
                materialize_frozen_git_revision(source, "f" * 40, workspace)

            self.assertFalse(workspace.exists())
            self.assertEqual(_source_state(source), before)

    def test_absolute_or_parent_archive_path_is_rejected(self) -> None:
        for name in ("/absolute", "../outside", "nested/../../outside"):
            with self.subTest(name=name):
                with self.assertRaises(SourceMaterializationError):
                    _validate_archive_entry(name)

    def test_escaping_symlink_is_rejected(self) -> None:
        with self.assertRaises(SourceMaterializationError):
            _validate_archive_entry("nested/link", "../../outside")

    def test_safe_relative_symlink_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            initialize_git_repo(source)
            (source / "links").mkdir()
            (source / "links" / "operator.py").symlink_to("../src/operator.py")
            git(source, "add", "links/operator.py")
            git(source, "commit", "--quiet", "-m", "add safe link")
            revision = git(source, "rev-parse", "HEAD").stdout.decode("ascii").strip()
            before = _source_state(source)

            snapshot = materialize_frozen_git_revision(
                source,
                revision,
                root / "workspace",
            )

            link = snapshot.workspace / "links" / "operator.py"
            self.assertTrue(link.is_symlink())
            self.assertEqual(os.readlink(link), "../src/operator.py")
            self.assertEqual(_source_state(source), before)


if __name__ == "__main__":
    unittest.main()
