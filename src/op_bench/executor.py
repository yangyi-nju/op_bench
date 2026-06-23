from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


def ensure_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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


class CommandExecutor(Protocol):
    name: str

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        ...

    def collect_environment(self) -> EnvironmentEvidence | dict[str, object]:
        ...

    def close(self, timeout_sec: int = 30) -> CommandResult | None:
        ...


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
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr),
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

    def close(self, timeout_sec: int = 30) -> CommandResult | None:
        return None


class DockerExecutor:
    name = "docker"

    def __init__(
        self,
        image: str,
        workspace_dir: str = "/workspace",
        container_name: str | None = None,
        command_workdir: str | None = None,
        labels: dict[str, str] | None = None,
        gpus: str | None = None,
    ) -> None:
        self.image = image
        self.workspace_dir = workspace_dir
        self.container_name = container_name
        self.command_workdir = command_workdir or workspace_dir
        self.labels = dict(labels or {})
        self.gpus = gpus

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        docker_command = self.command_for_run(command, cwd)
        start = time.monotonic()
        try:
            completed = subprocess.run(
                docker_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=docker_command,
                cwd=str(cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=docker_command,
                cwd=str(cwd),
                exit_code=124,
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr),
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )

    def command_for_run(self, command: list[str], cwd: Path) -> list[str]:
        if self.container_name:
            return [
                "docker",
                "exec",
                "--workdir",
                self.command_workdir,
                self.container_name,
                *command,
            ]
        host_workspace = cwd.resolve()
        run_command = [
            "docker",
            "run",
            "--rm",
        ]
        if self.gpus:
            run_command.extend(["--gpus", self.gpus])
        run_command.extend([
            "--volume",
            f"{host_workspace}:{self.workspace_dir}",
            "--workdir",
            self.command_workdir,
            self.image,
            *command,
        ])
        return run_command

    def start(self, cwd: Path, timeout_sec: int = 60) -> CommandResult:
        if not self.container_name:
            raise ValueError("DockerExecutor.start requires a container_name")
        docker_command = self.command_for_start(cwd)
        start = time.monotonic()
        try:
            completed = subprocess.run(
                docker_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=docker_command,
                cwd=str(cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=docker_command,
                cwd=str(cwd),
                exit_code=124,
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr),
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )

    def command_for_start(self, cwd: Path) -> list[str]:
        if not self.container_name:
            raise ValueError("DockerExecutor.command_for_start requires a container_name")
        host_workspace = cwd.resolve()
        command = [
            "docker",
            "run",
            "--detach",
            "--name",
            self.container_name,
        ]
        if self.gpus:
            command.extend(["--gpus", self.gpus])
        for key, value in sorted(self.labels.items()):
            command.extend(["--label", f"{key}={value}"])
        command.extend([
            "--volume",
            f"{host_workspace}:{self.workspace_dir}",
            "--workdir",
            self.command_workdir,
            self.image,
            "tail",
            "-f",
            "/dev/null",
        ])
        return command

    def close(self, timeout_sec: int = 30) -> CommandResult | None:
        if not self.container_name:
            return None
        docker_command = ["docker", "rm", "-f", self.container_name]
        start = time.monotonic()
        try:
            completed = subprocess.run(
                docker_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=docker_command,
                cwd="",
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=docker_command,
                cwd="",
                exit_code=124,
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr),
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )

    def collect_environment(self) -> dict[str, object]:
        return {
            "executor": self.name,
            "image": self.image,
            "workspace_dir": self.workspace_dir,
            "command_workdir": self.command_workdir,
            "container_name": self.container_name,
            "gpus": self.gpus,
            "docker_available": shutil.which("docker") is not None,
            "host_platform": platform.platform(),
            "host_machine": platform.machine(),
        }
