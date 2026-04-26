from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentEvidence:
    executor: str
    python_executable: str
    python_version: str
    platform: str
    machine: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class LocalExecutor:
    name = "local"

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )

    def collect_environment(self) -> EnvironmentEvidence:
        return EnvironmentEvidence(
            executor=self.name,
            python_executable=sys.executable,
            python_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            machine=platform.machine(),
        )
