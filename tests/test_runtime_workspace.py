from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    WorkspacePolicy,
    WorkspacePolicyError,
)
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_contracts import SHA_A, SHA_B, identity


def policy(**changes) -> WorkspacePolicy:
    baseline = WorkspacePolicy(
        policy_id="controlled-workspace-v1",
        writable_paths=("src/",),
        patch_paths=("src/",),
        allowed_modes=(0o644, 0o755),
        max_read_bytes=1_024,
        max_write_bytes=1_024,
        max_file_bytes=2_048,
        max_patch_bytes=8_192,
        allow_binary=False,
    )
    return replace(baseline, **changes)


class ParentSwapWorkspace(AuthoritativeWorkspace):
    swap_root: Path
    outside: Path
    swapped: bool

    def _atomic_replace(
        self,
        parent_fd: int,
        name: str,
        content: bytes,
        mode: int,
    ) -> None:
        if not self.swapped:
            self.swapped = True
            (self.swap_root / "src").rename(self.swap_root / "src-detached")
            os.symlink(self.outside, self.swap_root / "src")
        super()._atomic_replace(parent_fd, name, content, mode)


class AuthoritativeWorkspaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.base_commit = initialize_git_repo(self.root)
        self.workspace = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_identity_is_path_independent_and_contains_no_local_path(self) -> None:
        second_root = Path(self.temporary.name) / "same-repo"
        shutil.copytree(self.root, second_root, symlinks=True)
        second = AuthoritativeWorkspace.open(
            second_root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )

        self.assertEqual(self.workspace.identity, second.identity)
        self.assertEqual(self.workspace.identity.identity_type, "workspace")
        self.assertNotIn(str(self.root), canonical_json(self.workspace.identity.to_dict()))

    def test_identity_changes_with_source_base_or_policy(self) -> None:
        other_source = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_B),
            policy=policy(),
        )
        other_policy = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(max_patch_bytes=8_191),
        )

        self.assertNotEqual(self.workspace.identity, other_source.identity)
        self.assertNotEqual(self.workspace.identity, other_policy.identity)

        self.workspace.write("src/operator.py", b"VALUE = 2\n")
        with self.assertRaisesRegex(WorkspacePolicyError, "workspace must be clean"):
            AuthoritativeWorkspace.open(
                self.root,
                source=identity("source", "fixture@base", SHA_A),
                policy=policy(),
            )

    def test_regular_file_read_write_delete_and_operation_binding(self) -> None:
        read = self.workspace.read("src/operator.py")
        write = self.workspace.write("src/operator.py", "VALUE = 2\n")
        added = self.workspace.write("src/new.py", b"NEW = True\n")
        deleted = self.workspace.delete("src/helper.py")
        test_binding = self.workspace.bind_test("public::smoke")
        diff = self.workspace.diff()

        self.assertEqual(read.content, b"VALUE = 1\n")
        self.assertTrue(write.changed)
        self.assertTrue(added.changed)
        self.assertTrue(deleted.changed)
        self.assertIn(b"VALUE = 2", diff.patch_bytes)
        self.assertIn(b"src/new.py", diff.patch_bytes)
        self.assertIn(b"src/helper.py", diff.patch_bytes)
        for result in (read, write, added, deleted, test_binding, diff):
            self.assertEqual(result.workspace, self.workspace.identity)

    def test_repeated_identical_write_is_reported_unchanged(self) -> None:
        first = self.workspace.write("src/operator.py", b"VALUE = 2\n")
        second = self.workspace.write("src/operator.py", b"VALUE = 2\n")

        self.assertTrue(first.changed)
        self.assertFalse(second.changed)

    def test_apply_patch_updates_controlled_regular_files(self) -> None:
        patch_bytes = (
            b"diff --git a/src/operator.py b/src/operator.py\n"
            b"--- a/src/operator.py\n"
            b"+++ b/src/operator.py\n"
            b"@@ -1 +1 @@\n"
            b"-VALUE = 1\n"
            b"+VALUE = 2\n"
        )

        result = self.workspace.apply_patch(patch_bytes)

        self.assertTrue(result.changed)
        self.assertEqual(result.paths, ("src/operator.py",))
        self.assertEqual(result.workspace, self.workspace.identity)
        self.assertEqual((self.root / "src" / "operator.py").read_bytes(), b"VALUE = 2\n")

    def test_apply_patch_rejects_scope_symlink_and_multi_file_failure_without_mutation(self) -> None:
        outside_scope = (
            b"diff --git a/tests/test_operator.py b/tests/test_operator.py\n"
            b"--- a/tests/test_operator.py\n"
            b"+++ b/tests/test_operator.py\n"
            b"@@ -1,2 +1 @@\n"
            b"-def test_operator():\n"
            b"-    assert True\n"
            b"+raise RuntimeError\n"
        )
        with self.assertRaisesRegex(WorkspacePolicyError, "writable scope"):
            self.workspace.apply_patch(outside_scope)
        self.assertIn("assert True", (self.root / "tests" / "test_operator.py").read_text())

        invalid_second_hunk = (
            b"diff --git a/src/operator.py b/src/operator.py\n"
            b"--- a/src/operator.py\n"
            b"+++ b/src/operator.py\n"
            b"@@ -1 +1 @@\n"
            b"-VALUE = 1\n"
            b"+VALUE = 2\n"
            b"diff --git a/src/helper.py b/src/helper.py\n"
            b"--- a/src/helper.py\n"
            b"+++ b/src/helper.py\n"
            b"@@ -1,2 +1,2 @@\n"
            b"-not the current content\n"
            b"+replacement\n"
            b" def helper():\n"
        )
        with self.assertRaisesRegex(WorkspacePolicyError, "does not apply"):
            self.workspace.apply_patch(invalid_second_hunk)
        self.assertEqual((self.root / "src" / "operator.py").read_bytes(), b"VALUE = 1\n")

        limited = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(max_write_bytes=4_096, max_file_bytes=32),
        )
        oversized_result = (
            b"diff --git a/src/operator.py b/src/operator.py\n"
            b"--- a/src/operator.py\n"
            b"+++ b/src/operator.py\n"
            b"@@ -1 +1 @@\n"
            b"-VALUE = 1\n"
            b"+" + b"x" * 33 + b"\n"
        )
        with self.assertRaisesRegex(WorkspacePolicyError, "max_file_bytes"):
            limited.apply_patch(oversized_result)
        self.assertEqual((self.root / "src" / "operator.py").read_bytes(), b"VALUE = 1\n")

        outside = Path(self.temporary.name) / "outside.py"
        outside.write_text("private\n", encoding="utf-8")
        os.symlink(outside, self.root / "src" / "linked.py")
        symlink_patch = (
            b"diff --git a/src/linked.py b/src/linked.py\n"
            b"--- a/src/linked.py\n"
            b"+++ b/src/linked.py\n"
            b"@@ -1 +1 @@\n"
            b"-private\n"
            b"+changed\n"
        )
        with self.assertRaisesRegex(WorkspacePolicyError, "symlink"):
            self.workspace.apply_patch(symlink_patch)
        self.assertEqual(outside.read_text(encoding="utf-8"), "private\n")

        create_symlink = (
            b"diff --git a/src/new-link.py b/src/new-link.py\n"
            b"new file mode 120000\n"
            b"--- /dev/null\n"
            b"+++ b/src/new-link.py\n"
            b"@@ -0,0 +1 @@\n"
            b"+../outside.py\n"
        )
        with self.assertRaisesRegex(WorkspacePolicyError, "symlink mode"):
            self.workspace.apply_patch(create_symlink)

    def test_rejects_traversal_absolute_backslash_and_workspace_outside_paths(self) -> None:
        invalid = (
            "../outside.py",
            "src/../../outside.py",
            str(Path(self.temporary.name) / "outside.py"),
            r"src\\..\\outside.py",
            "./src/operator.py",
            "",
        )

        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(WorkspacePolicyError, "path"):
                    self.workspace.read(candidate)

    def test_rejects_symlink_final_and_symlink_parent_escape(self) -> None:
        outside = Path(self.temporary.name) / "outside.txt"
        outside.write_text("private\n", encoding="utf-8")
        os.symlink(outside, self.root / "src" / "outside-link.py")
        os.symlink(Path(self.temporary.name), self.root / "src" / "parent-link")

        for candidate in ("src/outside-link.py", "src/parent-link/outside.txt"):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(WorkspacePolicyError, "symlink"):
                    self.workspace.read(candidate)
                with self.assertRaisesRegex(WorkspacePolicyError, "symlink"):
                    self.workspace.write(candidate, b"overwrite\n")

    def test_parent_swap_race_never_writes_outside_workspace_authority(self) -> None:
        operations = ("write", "apply_patch")
        for index, operation in enumerate(operations):
            with self.subTest(operation=operation):
                root = Path(self.temporary.name) / f"parent-swap-{index}"
                initialize_git_repo(root)
                outside = Path(self.temporary.name) / f"outside-{index}"
                outside.mkdir()
                workspace = ParentSwapWorkspace.open(
                    root,
                    source=identity("source", f"fixture@swap-{index}", SHA_A),
                    policy=policy(),
                )
                workspace.swap_root = root
                workspace.outside = outside
                workspace.swapped = False

                with self.assertRaisesRegex(WorkspacePolicyError, "parent binding"):
                    if operation == "write":
                        workspace.write("src/new.py", b"NEW = True\n")
                    else:
                        workspace.apply_patch(
                            b"diff --git a/src/operator.py b/src/operator.py\n"
                            b"--- a/src/operator.py\n"
                            b"+++ b/src/operator.py\n"
                            b"@@ -1 +1 @@\n"
                            b"-VALUE = 1\n"
                            b"+VALUE = 2\n"
                        )

                self.assertFalse((outside / "new.py").exists())
                self.assertFalse((outside / "operator.py").exists())

    def test_rejects_directory_fifo_and_illegal_file_mode(self) -> None:
        os.chmod(self.root / "src" / "operator.py", 0o600)
        candidates = ["src", "src/operator.py"]
        if hasattr(os, "mkfifo"):
            os.mkfifo(self.root / "src" / "pipe")
            candidates.append("src/pipe")

        for candidate in candidates:
            with self.subTest(candidate=candidate):
                with self.assertRaises(WorkspacePolicyError):
                    self.workspace.read(candidate)

    def test_rejects_out_of_scope_oversized_and_binary_writes(self) -> None:
        cases = (
            ("tests/test_operator.py", b"changed\n", "writable scope"),
            ("src/large.py", b"x" * 1_025, "max_write_bytes"),
            ("src/binary.py", b"before\x00after", "binary"),
        )

        for path, content, message in cases:
            with self.subTest(path=path):
                with self.assertRaisesRegex(WorkspacePolicyError, message):
                    self.workspace.write(path, content)

    def test_rejects_invalid_requested_mode_without_partial_creation(self) -> None:
        with self.assertRaisesRegex(WorkspacePolicyError, "mode"):
            self.workspace.write("src/new.py", b"value = 1\n", mode=0o600)

        self.assertFalse((self.root / "src" / "new.py").exists())

    def test_read_limit_is_enforced_before_returning_content(self) -> None:
        (self.root / "src" / "operator.py").write_bytes(b"x" * 64)

        with self.assertRaisesRegex(WorkspacePolicyError, "read limit"):
            self.workspace.read("src/operator.py", max_bytes=32)


class WorkspacePolicyValidationTests(unittest.TestCase):
    def test_rejects_invalid_scope_modes_limits_and_boolean_integer(self) -> None:
        cases = (
            ({"writable_paths": ("../src",)}, "writable_paths"),
            ({"patch_paths": ("/src",)}, "patch_paths"),
            ({"patch_paths": (":(exclude)src/",)}, "patch_paths"),
            ({"patch_paths": ("src/*.py",)}, "patch_paths"),
            ({"allowed_modes": (0o644, 0o644)}, "allowed_modes"),
            ({"allowed_modes": (0o644, 0o120000)}, "allowed_modes"),
            ({"max_patch_bytes": True}, "max_patch_bytes"),
            ({"allow_binary": 1}, "allow_binary"),
        )

        for changes, message in cases:
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(WorkspacePolicyError, message):
                    policy(**changes)


if __name__ == "__main__":
    unittest.main()
