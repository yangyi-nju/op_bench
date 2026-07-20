from __future__ import annotations

import os
import io
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

try:
    from op_bench.runtime import process_group
    from op_bench.runtime.process_group import ProcessGroupOutputLimitError
except ImportError:
    process_group = None
    ProcessGroupOutputLimitError = RuntimeError


def minimal_env() -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "LANG": "C.UTF-8",
    }


def exact_process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def wait_for_exact_exit(pid: int, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not exact_process_exists(pid):
            return True
        time.sleep(0.01)
    return not exact_process_exists(pid)


class ProcessGroupRunnerTests(unittest.TestCase):
    def runner(self):
        self.assertIsNotNone(process_group)
        return process_group.run_process_group

    def test_normal_and_nonzero_exit_capture_bounded_private_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normal = self.runner()(
                (
                    sys.executable,
                    "-c",
                    "import sys; print('ok'); sys.stderr.write('warn\\n')",
                ),
                cwd=root,
                env=minimal_env(),
                timeout_ms=2_000,
            )
            nonzero = self.runner()(
                (sys.executable, "-c", "raise SystemExit(7)"),
                cwd=root,
                env=minimal_env(),
                timeout_ms=2_000,
            )
            bounded = self.runner()(
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.stdout.write('x'*4096); sys.stderr.write('y'*4096)",
                ),
                cwd=root,
                env=minimal_env(),
                timeout_ms=2_000,
                max_output_bytes=8_192,
            )

        self.assertEqual(normal.returncode, 0)
        self.assertEqual(normal.stdout, "ok\n")
        self.assertEqual(normal.stderr, "warn\n")
        self.assertEqual(normal.terminal_status, "completed")
        self.assertEqual(nonzero.returncode, 7)
        self.assertEqual(nonzero.terminal_status, "completed")
        self.assertEqual(len(bounded.stdout.encode("utf-8")), 4_096)
        self.assertEqual(len(bounded.stderr.encode("utf-8")), 4_096)

    def test_passes_only_explicit_controller_file_descriptors(self) -> None:
        read_descriptor, write_descriptor = os.pipe()
        try:
            os.write(write_descriptor, b"controller-secret")
        finally:
            os.close(write_descriptor)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                result = self.runner()(
                    (
                        sys.executable,
                        "-c",
                        f"import os; print(os.read({read_descriptor}, 64).decode())",
                    ),
                    cwd=Path(temporary),
                    env=minimal_env(),
                    timeout_ms=2_000,
                    pass_fds=(read_descriptor,),
                )
        finally:
            os.close(read_descriptor)

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "controller-secret\n")

    def test_output_overflow_terminates_the_exact_group_without_unbounded_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "child.pid"
            source = (
                "import os, pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
                "chunk=b'x'*65536; "
                "[(os.write(1, chunk), os.write(2, chunk)) for _ in range(256)]; "
                "time.sleep(30)"
            )

            with self.assertRaises(ProcessGroupOutputLimitError):
                self.runner()(
                    (sys.executable, "-c", source),
                    cwd=root,
                    env=minimal_env(),
                    timeout_ms=5_000,
                    terminate_grace_ms=100,
                    max_output_bytes=1_024,
                )
            child_pid = int(pid_path.read_text(encoding="utf-8"))

            self.assertTrue(wait_for_exact_exit(child_pid))

    def test_normal_parent_exit_still_terminates_its_exact_lingering_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "child.pid"
            parent_source = (
                "import pathlib, subprocess, sys; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid))"
            )

            result = self.runner()(
                (sys.executable, "-c", parent_source),
                cwd=root,
                env=minimal_env(),
                timeout_ms=2_000,
                terminate_grace_ms=100,
            )
            child_pid = int(pid_path.read_text(encoding="utf-8"))

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.terminal_status, "completed")
            self.assertTrue(wait_for_exact_exit(child_pid))

    def test_normal_parent_exit_allows_descendant_to_flush_before_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace_path = root / "trace.json"
            child_source = (
                "import pathlib, time; time.sleep(0.05); "
                f"pathlib.Path({str(trace_path)!r}).write_text('trace\\n')"
            )
            parent_source = (
                "import subprocess, sys; "
                f"subprocess.Popen([sys.executable, '-c', {child_source!r}])"
            )

            result = self.runner()(
                (sys.executable, "-c", parent_source),
                cwd=root,
                env=minimal_env(),
                timeout_ms=2_000,
                terminate_grace_ms=200,
            )

            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.terminal_status, "completed")
            self.assertEqual(trace_path.read_text(encoding="utf-8"), "trace\n")

    def test_timeout_terminates_only_the_spawned_process_group(self) -> None:
        runner = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "child.pid"
            parent_source = (
                "import pathlib, subprocess, sys, time; "
                "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(30)"
            )

            with mock.patch(
                "op_bench.runtime.process_group.os.killpg",
                wraps=os.killpg,
            ) as kill_group:
                result = runner(
                    (sys.executable, "-c", parent_source),
                    cwd=root,
                    env=minimal_env(),
                    timeout_ms=100,
                    terminate_grace_ms=100,
                )
            child_pid = int(pid_path.read_text(encoding="utf-8"))

            self.assertIn(result.terminal_status, {"terminated", "killed"})
            self.assertGreaterEqual(kill_group.call_count, 1)
            self.assertTrue(
                all(isinstance(call.args[0], int) for call in kill_group.call_args_list)
            )
            self.assertTrue(wait_for_exact_exit(child_pid))

    def test_term_resistant_group_is_killed_and_reaped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = root / "child.pid"
            child_source = (
                "import signal, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
            )
            parent_source = (
                "import pathlib, signal, subprocess, sys, time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                f"child=subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
                f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid)); "
                "time.sleep(30)"
            )

            result = self.runner()(
                (sys.executable, "-c", parent_source),
                cwd=root,
                env=minimal_env(),
                timeout_ms=200,
                terminate_grace_ms=50,
            )
            child_pid = int(pid_path.read_text(encoding="utf-8"))

            self.assertEqual(result.terminal_status, "killed")
            self.assertEqual(result.returncode, -signal.SIGKILL)
            self.assertTrue(wait_for_exact_exit(child_pid))

    def test_launch_failure_does_not_attempt_broad_cleanup(self) -> None:
        runner = self.runner()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch(
                    "op_bench.runtime.process_group.subprocess.Popen",
                    side_effect=FileNotFoundError("missing"),
                ),
                mock.patch("op_bench.runtime.process_group.os.killpg") as kill_group,
            ):
                with self.assertRaises(FileNotFoundError):
                    runner(
                        ("missing-codex", "--version"),
                        cwd=root,
                        env=minimal_env(),
                        timeout_ms=100,
                    )

            kill_group.assert_not_called()

    def test_cleanup_failure_is_fail_closed(self) -> None:
        runner = self.runner()
        fake_process = mock.Mock(pid=4343, returncode=None)
        fake_process.stdout = io.BytesIO()
        fake_process.stderr = io.BytesIO()
        fake_process.wait.side_effect = subprocess.TimeoutExpired(("fixture",), 0.05)
        fake_process.poll.return_value = None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch(
                    "op_bench.runtime.process_group.subprocess.Popen",
                    return_value=fake_process,
                ),
                mock.patch(
                    "op_bench.runtime.process_group.os.killpg",
                    side_effect=PermissionError("denied"),
                ),
            ):
                with self.assertRaisesRegex(Exception, "process group cleanup"):
                    runner(
                        (sys.executable, "-c", "pass"),
                        cwd=root,
                        env=minimal_env(),
                        timeout_ms=50,
                        terminate_grace_ms=50,
                    )

    def test_cleanup_retries_transient_permission_error_while_waiting(self) -> None:
        self.assertIsNotNone(process_group)
        with mock.patch(
            "op_bench.runtime.process_group.os.killpg",
            side_effect=[
                None,
                None,
                PermissionError("temporarily unavailable"),
                ProcessLookupError,
            ],
        ) as kill_group:
            process_group._cleanup_lingering_descendants(4545, grace_ms=50)

        self.assertEqual(
            [call.args for call in kill_group.call_args_list],
            [
                (4545, 0),
                (4545, 0),
                (4545, 0),
                (4545, 0),
            ],
        )

    def test_keyboard_interrupt_cleans_exact_group_before_propagating(self) -> None:
        self.assertIsNotNone(process_group)
        fake_process = mock.Mock(pid=4242, returncode=-signal.SIGTERM)
        fake_process.stdout = io.BytesIO()
        fake_process.stderr = io.BytesIO()
        fake_process.wait.side_effect = [KeyboardInterrupt, -signal.SIGTERM]
        fake_process.poll.return_value = None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch(
                    "op_bench.runtime.process_group.subprocess.Popen",
                    return_value=fake_process,
                ),
                mock.patch(
                    "op_bench.runtime.process_group.os.killpg",
                    side_effect=[None, ProcessLookupError],
                ) as kill_group,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    self.runner()(
                        (sys.executable, "-c", "pass"),
                        cwd=root,
                        env=minimal_env(),
                        timeout_ms=100,
                    )

        self.assertEqual(kill_group.call_args_list[0].args, (4242, signal.SIGTERM))

    def test_source_contains_no_process_enumeration_or_name_based_kill(self) -> None:
        self.assertIsNotNone(process_group)
        source = Path(process_group.__file__).read_text(encoding="utf-8")

        for forbidden in ("pkill", "pgrep", "docker ps", "ps aux", "shell=True"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
