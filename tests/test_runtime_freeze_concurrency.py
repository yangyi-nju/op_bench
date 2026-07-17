from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
import unittest

from op_bench.runtime.workspace import AuthoritativeWorkspace, WorkspaceStateError
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_contracts import SHA_A, identity
from tests.test_runtime_workspace import policy


class BlockingWriteWorkspace(AuthoritativeWorkspace):
    entered_write: threading.Event
    release_write: threading.Event

    def _atomic_replace(
        self,
        parent_fd: int,
        name: str,
        content: bytes,
        mode: int,
    ) -> None:
        self.entered_write.set()
        if not self.release_write.wait(timeout=5):
            raise RuntimeError("test did not release write")
        super()._atomic_replace(parent_fd, name, content, mode)


class BlockingFreezeWorkspace(AuthoritativeWorkspace):
    entered_build: threading.Event
    release_build: threading.Event

    def _build_frozen_patch(self):
        self.entered_build.set()
        if not self.release_build.wait(timeout=5):
            raise RuntimeError("test did not release freeze")
        return super()._build_frozen_patch()


class FreezeConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_freeze_waits_for_in_flight_mutation_and_rejects_new_mutations(self) -> None:
        workspace = BlockingWriteWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )
        workspace.entered_write = threading.Event()
        workspace.release_write = threading.Event()
        writer_errors: list[BaseException] = []
        freeze_results = []

        def write() -> None:
            try:
                workspace.write("src/operator.py", b"VALUE = 2\n")
            except BaseException as exc:  # pragma: no cover - reported by assertion
                writer_errors.append(exc)

        writer = threading.Thread(target=write)
        writer.start()
        self.assertTrue(workspace.entered_write.wait(timeout=2))

        freezer = threading.Thread(target=lambda: freeze_results.append(workspace.freeze()))
        freezer.start()
        deadline = time.monotonic() + 2
        while workspace.state != "freezing" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(workspace.state, "freezing")
        self.assertTrue(freezer.is_alive())

        with self.assertRaisesRegex(WorkspaceStateError, "freezing"):
            workspace.write("src/new.py", b"NEW = True\n")

        workspace.release_write.set()
        writer.join(timeout=2)
        freezer.join(timeout=5)

        self.assertFalse(writer.is_alive())
        self.assertFalse(freezer.is_alive())
        self.assertEqual(writer_errors, [])
        self.assertEqual(len(freeze_results), 1)
        self.assertIn(b"+VALUE = 2", freeze_results[0].patch_bytes)

    def test_concurrent_and_repeated_freeze_returns_one_final_object(self) -> None:
        workspace = BlockingFreezeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )
        workspace.entered_build = threading.Event()
        workspace.release_build = threading.Event()
        workspace.write("src/operator.py", b"VALUE = 2\n")
        barrier = threading.Barrier(5)
        results = []
        errors: list[BaseException] = []

        def freeze() -> None:
            barrier.wait(timeout=2)
            try:
                results.append(workspace.freeze())
            except BaseException as exc:  # pragma: no cover - reported by assertion
                errors.append(exc)

        threads = [threading.Thread(target=freeze) for _ in range(4)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=2)
        self.assertTrue(workspace.entered_build.wait(timeout=2))
        workspace.release_build.set()
        for thread in threads:
            thread.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)
        self.assertEqual(len({id(result) for result in results}), 1)
        self.assertIs(workspace.freeze(), results[0])


if __name__ == "__main__":
    unittest.main()
