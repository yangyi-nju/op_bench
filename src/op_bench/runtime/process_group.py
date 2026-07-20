from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time

from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_int,
    require_str,
)


PROCESS_GROUP_TERMINAL_STATUSES = ("completed", "terminated", "killed")


class ProcessGroupCleanupError(RuntimeError):
    """The exact recorded process group could not be proven converged."""

    def __init__(
        self,
        message: str,
        *,
        process_group_id: int | None = None,
    ) -> None:
        super().__init__(message)
        self.process_group_id = process_group_id


class ProcessGroupOutputLimitError(RuntimeError):
    """A spawned exact process group exceeded bounded controller output."""


class _BoundedPipeCollector:
    def __init__(
        self,
        stream,
        maximum: int,
        overflow: threading.Event,
    ) -> None:
        self.stream = stream
        self.maximum = maximum
        self.overflow = overflow
        self.encoded = bytearray()
        self.failure: OSError | None = None
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _collect(self) -> None:
        try:
            while True:
                chunk = self.stream.read(65_536)
                if not chunk:
                    return
                remaining = self.maximum - len(self.encoded)
                if remaining > 0:
                    self.encoded.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.overflow.set()
        except OSError as exc:
            self.failure = exc

    def finish(self, timeout_seconds: float) -> None:
        self.thread.join(timeout=max(0.1, timeout_seconds))
        if self.thread.is_alive():
            try:
                self.stream.close()
            except OSError:
                pass
            self.thread.join(timeout=0.1)
        else:
            try:
                self.stream.close()
            except OSError:
                pass

    def text(self) -> str:
        return bytes(self.encoded).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ProcessGroupResult:
    argv0: str
    returncode: int
    stdout: str
    stderr: str
    terminal_status: str

    def __post_init__(self) -> None:
        require_str(self.argv0, "argv0")
        require_int(self.returncode, "returncode")
        require_str(self.stdout, "stdout", min_length=0)
        require_str(self.stderr, "stderr", min_length=0)
        require_enum(
            self.terminal_status,
            "terminal_status",
            PROCESS_GROUP_TERMINAL_STATUSES,
        )


def _validated_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if isinstance(argv, (str, bytes)):
        raise ContractError("argv: expected non-empty sequence")
    selected = tuple(argv)
    if not selected:
        raise ContractError("argv: expected non-empty sequence")
    for index, item in enumerate(selected):
        require_str(item, f"argv[{index}]")
    return selected


def _validated_environment(env: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(env, Mapping):
        raise ContractError("env: expected object")
    selected: dict[str, str] = {}
    for key, value in env.items():
        selected[require_str(key, "env key")] = require_str(
            value,
            f"env[{key!r}]",
            min_length=0,
        )
    return selected


def _probe_exact_group(process_group_id: int) -> bool | None:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError as exc:
        raise ProcessGroupCleanupError(
            "exact process group cleanup status is unavailable",
            process_group_id=process_group_id,
        ) from exc
    return True


def _exact_group_exists(process_group_id: int) -> bool:
    state = _probe_exact_group(process_group_id)
    if state is None:
        raise ProcessGroupCleanupError(
            "exact process group cleanup status is unavailable",
            process_group_id=process_group_id,
        )
    return state


def _wait_for_exact_group_exit(process_group_id: int, grace_ms: int) -> bool:
    deadline = time.monotonic() + grace_ms / 1000.0
    while True:
        state = _probe_exact_group(process_group_id)
        if state is False:
            return True
        if time.monotonic() >= deadline:
            if state is None:
                raise ProcessGroupCleanupError(
                    "exact process group cleanup status is unavailable",
                    process_group_id=process_group_id,
                )
            return False
        time.sleep(min(0.01, max(0.001, deadline - time.monotonic())))


def _signal_exact_group(process_group_id: int, selected_signal: int) -> bool:
    try:
        os.killpg(process_group_id, selected_signal)
    except ProcessLookupError:
        return False
    except OSError as exc:
        raise ProcessGroupCleanupError(
            "exact process group cleanup failed",
            process_group_id=process_group_id,
        ) from exc
    return True


def exact_process_group_is_absent(process_group_id: int) -> bool:
    """Prove absence of one recorded PGID without discovery or signaling it."""

    selected = require_int(
        process_group_id,
        "process_group_id",
        minimum=1,
    )
    state = _probe_exact_group(selected)
    if state is None:
        raise ProcessGroupCleanupError(
            "exact process group cleanup status is unavailable",
            process_group_id=selected,
        )
    return state is False


def _reap_direct_process(
    process: subprocess.Popen[bytes],
    *,
    timeout_ms: int,
) -> bool:
    try:
        process.wait(timeout=timeout_ms / 1000.0)
    except subprocess.TimeoutExpired:
        return False
    return True


def _terminate_timed_out_group(
    process: subprocess.Popen[bytes],
    process_group_id: int,
    *,
    grace_ms: int,
) -> str:
    term_sent = _signal_exact_group(process_group_id, signal.SIGTERM)
    direct_reaped = _reap_direct_process(process, timeout_ms=grace_ms)
    if (
        (not term_sent or direct_reaped)
        and _wait_for_exact_group_exit(process_group_id, grace_ms)
    ):
        return "terminated"

    _signal_exact_group(process_group_id, signal.SIGKILL)
    if not direct_reaped and not _reap_direct_process(process, timeout_ms=grace_ms):
        raise ProcessGroupCleanupError(
            "exact process group cleanup did not reap child",
            process_group_id=process_group_id,
        )
    if not _wait_for_exact_group_exit(process_group_id, grace_ms):
        raise ProcessGroupCleanupError(
            "exact process group cleanup did not converge",
            process_group_id=process_group_id,
        )
    return "killed"


def _cleanup_lingering_descendants(process_group_id: int, *, grace_ms: int) -> None:
    if not _exact_group_exists(process_group_id):
        return
    _signal_exact_group(process_group_id, signal.SIGTERM)
    if _wait_for_exact_group_exit(process_group_id, grace_ms):
        return
    _signal_exact_group(process_group_id, signal.SIGKILL)
    if not _wait_for_exact_group_exit(process_group_id, grace_ms):
        raise ProcessGroupCleanupError(
            "exact process group cleanup did not converge",
            process_group_id=process_group_id,
        )


def run_process_group(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_ms: int,
    input_text: str | None = None,
    pass_fds: Sequence[int] = (),
    terminate_grace_ms: int = 2_000,
    max_output_bytes: int = 1_048_576,
) -> ProcessGroupResult:
    """Run one exact process group and converge its descendants without discovery."""

    selected_argv = _validated_argv(argv)
    if not isinstance(cwd, Path):
        raise ContractError("cwd: expected Path")
    if cwd.is_symlink() or not cwd.is_dir():
        raise ContractError("cwd: expected real directory")
    selected_env = _validated_environment(env)
    timeout = require_int(timeout_ms, "timeout_ms", minimum=1)
    grace = require_int(terminate_grace_ms, "terminate_grace_ms", minimum=1)
    maximum = require_int(max_output_bytes, "max_output_bytes", minimum=1)
    if isinstance(pass_fds, (str, bytes)) or not isinstance(pass_fds, Sequence):
        raise ContractError("pass_fds: expected sequence")
    selected_fds = tuple(
        require_int(item, f"pass_fds[{index}]", minimum=0)
        for index, item in enumerate(pass_fds)
    )
    if len(selected_fds) != len(set(selected_fds)):
        raise ContractError("pass_fds: duplicate descriptor")
    if input_text is not None:
        selected_input = require_str(input_text, "input_text", min_length=0).encode(
            "utf-8"
        )
        if len(selected_input) > maximum:
            raise ContractError("input_text: exceeds bounded process input")
    else:
        selected_input = None

    with tempfile.TemporaryFile(mode="w+b") as input_file:
        if selected_input is not None:
            input_file.write(selected_input)
            input_file.flush()
            input_file.seek(0)
        process = subprocess.Popen(
            selected_argv,
            cwd=str(cwd),
            env=selected_env,
            stdin=subprocess.DEVNULL if selected_input is None else input_file,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=selected_fds,
        )
        process_group_id = process.pid
        if process.stdout is None or process.stderr is None:
            raise ProcessGroupCleanupError(
                "private process output stream is invalid",
                process_group_id=process_group_id,
            )
        overflow = threading.Event()
        stdout_collector = _BoundedPipeCollector(process.stdout, maximum, overflow)
        stderr_collector = _BoundedPipeCollector(process.stderr, maximum, overflow)
        collectors = (stdout_collector, stderr_collector)
        for collector in collectors:
            collector.start()
        terminal_status = "completed"
        try:
            deadline = time.monotonic() + timeout / 1000.0
            while True:
                if overflow.is_set():
                    terminal_status = _terminate_timed_out_group(
                        process,
                        process_group_id,
                        grace_ms=grace,
                    )
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    terminal_status = _terminate_timed_out_group(
                        process,
                        process_group_id,
                        grace_ms=grace,
                    )
                    break
                try:
                    process.wait(timeout=min(0.01, remaining))
                except subprocess.TimeoutExpired:
                    continue
                _cleanup_lingering_descendants(process_group_id, grace_ms=grace)
                break
        except BaseException as original:
            try:
                if process.poll() is None or _exact_group_exists(process_group_id):
                    _terminate_timed_out_group(
                        process,
                        process_group_id,
                        grace_ms=grace,
                    )
            except ProcessGroupCleanupError as cleanup_error:
                raise cleanup_error from original
            raise
        finally:
            for collector in collectors:
                collector.finish(grace / 1000.0 + 0.5)

        returncode = process.returncode
        if returncode is None:
            raise ProcessGroupCleanupError("exact process group child was not reaped")
        if any(collector.failure is not None for collector in collectors):
            raise ProcessGroupOutputLimitError(
                "private process output capture failed"
            )
        if overflow.is_set():
            raise ProcessGroupOutputLimitError(
                "private process output exceeds size limit"
            )
        return ProcessGroupResult(
            argv0=selected_argv[0],
            returncode=returncode,
            stdout=stdout_collector.text(),
            stderr=stderr_collector.text(),
            terminal_status=terminal_status,
        )


__all__ = [
    "PROCESS_GROUP_TERMINAL_STATUSES",
    "ProcessGroupCleanupError",
    "ProcessGroupOutputLimitError",
    "ProcessGroupResult",
    "exact_process_group_is_absent",
    "run_process_group",
]
