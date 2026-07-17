from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    WorkspacePolicyError,
    WorkspaceStateError,
)
from tests.runtime_git_fixture import git, initialize_git_repo
from tests.test_runtime_contracts import SHA_A, identity
from tests.test_runtime_workspace import policy


class RootSwapWorkspace(AuthoritativeWorkspace):
    swap_root: Path
    replacement_root: Path
    held_root: Path

    def _capture_authoritative_patch(self) -> tuple[tuple[str, ...], bytes]:
        self.swap_root.rename(self.held_root)
        self.replacement_root.rename(self.swap_root)
        try:
            return super()._capture_authoritative_patch()
        finally:
            self.swap_root.rename(self.replacement_root)
            self.held_root.rename(self.swap_root)


class PatchFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def workspace(self, **policy_changes) -> AuthoritativeWorkspace:
        return AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(**policy_changes),
        )

    def test_freeze_captures_add_modify_delete_and_strictly_applies_to_clean_base(self) -> None:
        workspace = self.workspace()
        workspace.write("src/operator.py", b"VALUE = 2\n")
        workspace.write("src/new.py", b"NEW = True\n")
        workspace.delete("src/helper.py")

        frozen = workspace.freeze()

        self.assertEqual(
            frozen.changed_paths,
            ("src/helper.py", "src/new.py", "src/operator.py"),
        )
        self.assertFalse(frozen.empty)
        self.assertEqual(frozen.workspace, workspace.identity)
        self.assertEqual(frozen.source, workspace.source)
        self.assertEqual(frozen.base_commit, workspace.base_commit)
        self.assertIn(b"new file mode 100644", frozen.patch_bytes)
        self.assertIn(b"deleted file mode 100644", frozen.patch_bytes)
        self.assertIn(b"-VALUE = 1", frozen.patch_bytes)
        self.assertIn(b"+VALUE = 2", frozen.patch_bytes)

        clean = Path(self.temporary.name) / "independent-clean"
        git(Path(self.temporary.name), "clone", "--quiet", "--no-hardlinks", str(self.root), str(clean))
        patch_path = Path(self.temporary.name) / "frozen.patch"
        patch_path.write_bytes(frozen.patch_bytes)
        result = git(clean, "apply", "--check", "--index", str(patch_path), check=False)
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))

    def test_empty_freeze_has_explicit_immutable_patch_identity(self) -> None:
        workspace = self.workspace()

        frozen = workspace.freeze()

        self.assertTrue(frozen.empty)
        self.assertEqual(frozen.changed_paths, ())
        self.assertEqual(frozen.patch_bytes, b"")
        self.assertEqual(
            frozen.patch.digest,
            "sha256:e3b0c44298fc1c149afbf4c8996fb924"
            "27ae41e4649b934ca495991b7852b855",
        )
        self.assertIs(workspace.freeze(), frozen)
        self.assertEqual(workspace.diff().patch_bytes, b"")

    def test_mutations_are_closed_after_freeze(self) -> None:
        workspace = self.workspace()
        workspace.freeze()

        with self.assertRaisesRegex(WorkspaceStateError, "does not accept mutations"):
            workspace.write("src/operator.py", b"VALUE = 3\n")
        with self.assertRaisesRegex(WorkspaceStateError, "does not accept mutations"):
            workspace.delete("src/operator.py")

    def test_out_of_patch_scope_contamination_does_not_change_patch_or_hash(self) -> None:
        clean_copy = Path(self.temporary.name) / "clean-copy"
        shutil.copytree(self.root, clean_copy, symlinks=True)
        clean_workspace = AuthoritativeWorkspace.open(
            clean_copy,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )
        contaminated = self.workspace()

        clean_workspace.write("src/operator.py", b"VALUE = 2\n")
        contaminated.write("src/operator.py", b"VALUE = 2\n")
        (self.root / "tests" / "test_operator.py").write_text("raise RuntimeError\n", encoding="utf-8")
        (self.root / ".agent-cache").mkdir()
        (self.root / ".agent-cache" / "state.bin").write_bytes(b"private\x00cache")

        expected = clean_workspace.freeze()
        observed = contaminated.freeze()

        self.assertEqual(observed.patch_bytes, expected.patch_bytes)
        self.assertEqual(observed.patch, expected.patch)
        self.assertEqual(observed.changed_paths, ("src/operator.py",))

    def test_repository_local_diff_driver_cannot_change_canonical_patch_bytes(self) -> None:
        configured_root = Path(self.temporary.name) / "configured"
        plain_root = Path(self.temporary.name) / "plain"
        (self.root / ".gitattributes").write_text(
            "src/operator.py diff=custom\n",
            encoding="utf-8",
        )
        git(self.root, "add", ".gitattributes")
        git(self.root, "commit", "--quiet", "-m", "declare custom diff driver")
        shutil.copytree(self.root, configured_root, symlinks=True)
        shutil.copytree(self.root, plain_root, symlinks=True)
        git(configured_root, "config", "diff.custom.binary", "true")
        configured = AuthoritativeWorkspace.open(
            configured_root,
            source=identity("source", "fixture@diff-config", SHA_A),
            policy=policy(),
        )
        plain = AuthoritativeWorkspace.open(
            plain_root,
            source=identity("source", "fixture@diff-config", SHA_A),
            policy=policy(),
        )

        configured.write("src/operator.py", b"VALUE = 2\n")
        plain.write("src/operator.py", b"VALUE = 2\n")

        self.assertEqual(configured.identity, plain.identity)
        self.assertEqual(configured.diff().patch_bytes, plain.diff().patch_bytes)
        self.assertIn(b"-VALUE = 1", configured.diff().patch_bytes)
        self.assertEqual(configured.freeze().patch_bytes, plain.freeze().patch_bytes)

    def test_freeze_fails_closed_when_workspace_root_binding_is_replaced(self) -> None:
        replacement = Path(self.temporary.name) / "replacement"
        held = Path(self.temporary.name) / "held-authority"
        initialize_git_repo(replacement)
        (replacement / "src" / "operator.py").write_text(
            "EVIL = True\n",
            encoding="utf-8",
        )
        workspace = RootSwapWorkspace.open(
            self.root,
            source=identity("source", "fixture@root-swap", SHA_A),
            policy=policy(),
        )
        workspace.swap_root = self.root
        workspace.replacement_root = replacement
        workspace.held_root = held
        workspace.write("src/operator.py", b"VALUE = 2\n")

        with self.assertRaisesRegex(WorkspacePolicyError, "root binding changed"):
            workspace.freeze()

        self.assertEqual(workspace.state, "freeze_failed")
        self.assertEqual(
            (self.root / "src" / "operator.py").read_bytes(),
            b"VALUE = 2\n",
        )
        self.assertEqual(
            (replacement / "src" / "operator.py").read_bytes(),
            b"EVIL = True\n",
        )

    def test_patch_scope_includes_ignored_untracked_file_created_by_workspace(self) -> None:
        root = Path(self.temporary.name) / "ignored-created"
        initialize_git_repo(root)
        (root / ".gitignore").write_text("src/generated.py\n", encoding="utf-8")
        git(root, "add", ".gitignore")
        git(root, "commit", "--quiet", "-m", "ignore generated")
        workspace = AuthoritativeWorkspace.open(
            root,
            source=identity("source", "fixture@ignored", SHA_A),
            policy=policy(),
        )

        workspace.write("src/generated.py", b"GENERATED = True\n")
        frozen = workspace.freeze()

        self.assertEqual(frozen.changed_paths, ("src/generated.py",))
        self.assertIn(b"src/generated.py", frozen.patch_bytes)
        self.assertFalse(frozen.empty)

    def test_open_rejects_preexisting_ignored_materialization(self) -> None:
        root = Path(self.temporary.name) / "ignored-preexisting"
        initialize_git_repo(root)
        (root / ".gitignore").write_text("src/private-answer.txt\n", encoding="utf-8")
        git(root, "add", ".gitignore")
        git(root, "commit", "--quiet", "-m", "ignore private answer")
        (root / "src" / "private-answer.txt").write_text("GOLD ANSWER\n", encoding="utf-8")

        with self.assertRaisesRegex(WorkspacePolicyError, "workspace must be clean"):
            AuthoritativeWorkspace.open(
                root,
                source=identity("source", "fixture@ignored", SHA_A),
                policy=policy(),
            )

    def test_open_rejects_index_flags_that_hide_worktree_or_index_divergence(self) -> None:
        cases = (
            ("assume", ("--assume-unchanged",), False),
            ("skip", ("--skip-worktree",), False),
            ("staged", (), True),
        )
        for index, (name, flags, stage_change) in enumerate(cases):
            with self.subTest(name=name):
                root = Path(self.temporary.name) / f"hidden-index-{index}"
                initialize_git_repo(root)
                if flags:
                    git(root, "update-index", *flags, "src/operator.py")
                (root / "src" / "operator.py").write_text(
                    "GOLD = 42\n",
                    encoding="utf-8",
                )
                if stage_change:
                    git(root, "add", "src/operator.py")

                with self.assertRaisesRegex(
                    WorkspacePolicyError,
                    "base commit|index flags|workspace must be clean",
                ):
                    AuthoritativeWorkspace.open(
                        root,
                        source=identity("source", f"fixture@index-{index}", SHA_A),
                        policy=policy(),
                    )

    def test_freeze_rejects_symlink_mode_illegal_mode_binary_and_size_policies(self) -> None:
        cases = ("symlink", "mode", "binary", "file_size", "patch_size")
        for index, case in enumerate(cases):
            with self.subTest(case=case):
                root = Path(self.temporary.name) / f"case-{index}"
                initialize_git_repo(root)
                changes = {"max_file_bytes": 32} if case == "file_size" else {}
                if case == "patch_size":
                    changes = {"max_patch_bytes": 32}
                workspace = AuthoritativeWorkspace.open(
                    root,
                    source=identity("source", "fixture@base", SHA_A),
                    policy=policy(**changes),
                )
                target = root / "src" / "operator.py"
                if case == "symlink":
                    target.unlink()
                    os.symlink(Path(self.temporary.name) / "outside.py", target)
                    expected = "symlink"
                elif case == "mode":
                    git(root, "config", "core.filemode", "false")
                    os.chmod(target, 0o700)
                    expected = "mode"
                elif case == "binary":
                    target.write_bytes(b"binary\x00value")
                    expected = "binary"
                elif case == "file_size":
                    target.write_bytes(b"x" * 33)
                    expected = "max_file_bytes"
                else:
                    target.write_text("VALUE = 'a patch larger than thirty-two bytes'\n", encoding="utf-8")
                    expected = "max_patch_bytes"

                with self.assertRaisesRegex(WorkspacePolicyError, expected):
                    workspace.freeze()

    def test_strict_clean_apply_rejects_corrupted_context_without_fuzz(self) -> None:
        workspace = self.workspace()
        workspace.write("src/operator.py", b"VALUE = 2\n")
        valid_patch = workspace.diff().patch_bytes
        corrupted = valid_patch.replace(b"-VALUE = 1", b"-VALUE = 999")

        with patch.object(
            workspace,
            "_capture_authoritative_patch",
            return_value=(("src/operator.py",), corrupted),
        ):
            with self.assertRaisesRegex(WorkspacePolicyError, "strict clean-base apply"):
                workspace.freeze()

    def test_freeze_failure_is_terminal_for_workspace_writes(self) -> None:
        workspace = self.workspace(max_patch_bytes=32)
        workspace.write("src/operator.py", b"VALUE = 'large enough to exceed the patch cap'\n")

        with self.assertRaises(WorkspacePolicyError):
            workspace.freeze()
        self.assertEqual(workspace.state, "freeze_failed")
        with self.assertRaisesRegex(WorkspaceStateError, "freeze_failed"):
            workspace.write("src/operator.py", b"VALUE = 3\n")
        with self.assertRaisesRegex(WorkspacePolicyError, "max_patch_bytes"):
            workspace.freeze()


if __name__ == "__main__":
    unittest.main()
