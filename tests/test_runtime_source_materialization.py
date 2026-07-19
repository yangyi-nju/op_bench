from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

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
    def test_workspace_leaf_race_never_deletes_a_foreign_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = initialize_git_repo(source)
            workspace = root / "workspace"
            original_mkdir = Path.mkdir

            def racing_mkdir(path, *args, **kwargs):
                if path == workspace and not path.exists():
                    original_mkdir(path, parents=True, exist_ok=False)
                    (path / "foreign.txt").write_text(
                        "foreign\n",
                        encoding="utf-8",
                    )
                return original_mkdir(path, *args, **kwargs)

            with mock.patch.object(Path, "mkdir", new=racing_mkdir):
                with self.assertRaises(SourceMaterializationError):
                    materialize_frozen_git_revision(
                        source,
                        revision,
                        workspace,
                    )

            self.assertEqual(
                (workspace / "foreign.txt").read_text(encoding="utf-8"),
                "foreign\n",
            )

    def test_ambient_git_authority_variables_cannot_redirect_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = initialize_git_repo(source)
            decoy = root / "decoy"
            initialize_git_repo(decoy)
            workspace = root / "workspace"
            pollution = {
                "GIT_DIR": str(decoy / ".git"),
                "GIT_WORK_TREE": str(decoy),
                "GIT_INDEX_FILE": str(root / "foreign-index"),
                "GIT_OBJECT_DIRECTORY": str(decoy / ".git" / "objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(source / ".git" / "objects"),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.hooksPath",
                "GIT_CONFIG_VALUE_0": str(root / "foreign-hooks"),
            }

            with mock.patch.dict(os.environ, pollution, clear=False):
                snapshot = materialize_frozen_git_revision(
                    source,
                    revision,
                    workspace,
                )

            self.assertEqual(snapshot.revision, revision)
            self.assertEqual(
                (workspace / "src" / "operator.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            self.assertFalse((root / "foreign-index").exists())

    def test_snapshot_normalizes_git_regular_file_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            initialize_git_repo(source)
            executable = source / "src" / "helper.py"
            executable.chmod(0o755)
            git(source, "add", "src/helper.py")
            git(source, "commit", "--quiet", "-m", "make helper executable")
            revision = git(source, "rev-parse", "HEAD").stdout.decode().strip()

            snapshot = materialize_frozen_git_revision(
                source,
                revision,
                root / "workspace",
            )

            self.assertEqual(
                stat.S_IMODE((snapshot.workspace / "src" / "operator.py").stat().st_mode),
                0o644,
            )
            self.assertEqual(
                stat.S_IMODE((snapshot.workspace / "src" / "helper.py").stat().st_mode),
                0o755,
            )

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

    def test_snapshot_head_preserves_exact_source_archive_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = initialize_git_repo(source)
            (source / "src" / "operator.py").write_text(
                "VALUE = 2\n",
                encoding="utf-8",
            )
            git(source, "add", "src/operator.py")
            git(source, "commit", "--quiet", "-m", "second source revision")
            revision = git(source, "rev-parse", "HEAD").stdout.decode().strip()
            expected_archive = git(
                source,
                "archive",
                "--format=tar",
                revision,
            ).stdout

            snapshot = materialize_frozen_git_revision(
                source,
                revision,
                root / "workspace",
            )

            self.assertEqual(
                git(snapshot.workspace, "rev-parse", "HEAD").stdout.decode().strip(),
                revision,
            )
            self.assertEqual(
                git(
                    snapshot.workspace,
                    "archive",
                    "--format=tar",
                    "HEAD",
                ).stdout,
                expected_archive,
            )

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

    def test_gitlink_entries_are_preserved_without_initializing_submodules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            submodule_commit = initialize_git_repo(source)
            git(
                source,
                "update-index",
                "--add",
                "--cacheinfo",
                f"160000,{submodule_commit},vendor/dependency",
            )
            git(source, "commit", "--quiet", "-m", "add frozen gitlink")
            revision = git(source, "rev-parse", "HEAD").stdout.decode().strip()

            snapshot = materialize_frozen_git_revision(
                source,
                revision,
                root / "workspace",
            )

            entry = git(
                snapshot.workspace,
                "ls-tree",
                "HEAD",
                "vendor/dependency",
            ).stdout.decode("ascii")
            self.assertEqual(
                entry,
                f"160000 commit {submodule_commit}\tvendor/dependency\n",
            )
            dependency = snapshot.workspace / "vendor" / "dependency"
            self.assertTrue(dependency.is_dir())
            self.assertEqual(list(dependency.iterdir()), [])
            self.assertEqual(
                git(snapshot.workspace, "status", "--porcelain=v1").stdout,
                b"",
            )

    def test_populated_submodules_are_materialized_recursively_from_exact_local_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            leaf = root / "leaf"
            initialize_git_repo(leaf)

            dependency = root / "dependency"
            initialize_git_repo(dependency)
            git(
                dependency,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "--quiet",
                str(leaf),
                "nested/leaf",
            )
            git(dependency, "commit", "--quiet", "-am", "add nested dependency")

            source = root / "source"
            initialize_git_repo(source)
            git(
                source,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "add",
                "--quiet",
                str(dependency),
                "vendor/dependency",
            )
            git(source, "commit", "--quiet", "-am", "add dependency")
            git(
                source,
                "-c",
                "protocol.file.allow=always",
                "submodule",
                "update",
                "--init",
                "--recursive",
            )
            revision = git(source, "rev-parse", "HEAD").stdout.decode().strip()

            snapshot = materialize_frozen_git_revision(
                source,
                revision,
                root / "workspace",
                include_submodules=True,
            )

            dependency_copy = snapshot.workspace / "vendor" / "dependency"
            leaf_copy = dependency_copy / "nested" / "leaf"
            self.assertEqual(
                (dependency_copy / "src" / "operator.py").read_text(),
                "VALUE = 1\n",
            )
            self.assertEqual(
                (leaf_copy / "src" / "operator.py").read_text(),
                "VALUE = 1\n",
            )
            self.assertFalse((dependency_copy / ".git").exists())
            self.assertFalse((leaf_copy / ".git").exists())
            self.assertEqual(
                git(
                    snapshot.workspace,
                    "status",
                    "--porcelain=v1",
                    "--ignore-submodules=all",
                ).stdout,
                b"",
            )

    def test_submodule_archive_attributes_cannot_silently_change_the_frozen_tree(self) -> None:
        scenarios = {
            "export-ignore": (
                "src/operator.py export-ignore\n",
                "VALUE = 1\n",
            ),
            "export-subst": (
                "src/operator.py export-subst\n",
                "REVISION = '$Format:%H$'\n",
            ),
        }
        for name, (attributes, operator_source) in scenarios.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dependency = root / "dependency"
                initialize_git_repo(dependency)
                (dependency / ".gitattributes").write_text(
                    attributes,
                    encoding="utf-8",
                )
                (dependency / "src" / "operator.py").write_text(
                    operator_source,
                    encoding="utf-8",
                )
                git(dependency, "add", ".gitattributes", "src/operator.py")
                git(dependency, "commit", "--quiet", "-m", name)

                source = root / "source"
                initialize_git_repo(source)
                git(
                    source,
                    "-c",
                    "protocol.file.allow=always",
                    "submodule",
                    "add",
                    "--quiet",
                    str(dependency),
                    "vendor/dependency",
                )
                git(source, "commit", "--quiet", "-am", "add dependency")
                revision = git(source, "rev-parse", "HEAD").stdout.decode().strip()
                workspace = root / "workspace"

                with self.assertRaisesRegex(
                    SourceMaterializationError,
                    "submodule archive tree does not match frozen commit",
                ):
                    materialize_frozen_git_revision(
                        source,
                        revision,
                        workspace,
                        include_submodules=True,
                    )

                self.assertFalse(workspace.exists())


if __name__ == "__main__":
    unittest.main()
