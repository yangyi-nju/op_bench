"""Remote Docker executor over SSH.

This module enables running Docker containers on a remote host (typically a
GPU cloud instance) by prefixing all docker commands with `ssh user@host`.
Workspace files are synced via rsync.

Usage flow:
    1. RemoteHost describes how to connect (user, hostname, port, key, remote workspace root)
    2. RemoteDockerExecutor wraps DockerExecutor command construction with SSH
    3. EnvironmentManager calls sync_to_remote() before start, sync_from_remote() before scoring
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from op_bench.executor import CommandResult, ensure_text


@dataclass(frozen=True)
class RemoteHost:
    """Connection info for a remote host running Docker."""

    user: str
    hostname: str
    port: int = 22
    identity_file: str | None = None
    remote_workspace_root: str = "/tmp/op_bench_workspaces"
    extra_ssh_options: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: dict) -> "RemoteHost":
        return cls(
            user=str(data["user"]),
            hostname=str(data["hostname"]),
            port=int(data.get("port", 22)),
            identity_file=str(data["identity_file"]) if data.get("identity_file") else None,
            remote_workspace_root=str(data.get("remote_workspace_root", "/tmp/op_bench_workspaces")),
            extra_ssh_options=tuple(data.get("extra_ssh_options", [])),
        )

    def ssh_target(self) -> str:
        return f"{self.user}@{self.hostname}"

    def ssh_command_prefix(self) -> list[str]:
        cmd = ["ssh"]
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.identity_file:
            cmd.extend(["-i", os.path.expanduser(self.identity_file)])
        cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        cmd.extend(["-o", "ServerAliveInterval=30"])
        for opt in self.extra_ssh_options:
            cmd.extend(["-o", opt])
        cmd.append(self.ssh_target())
        return cmd

    def rsync_remote_path(self, remote_path: str) -> str:
        return f"{self.ssh_target()}:{remote_path}"


def load_hosts_config(path: Path | str | None = None) -> dict[str, RemoteHost]:
    """Load remote host configuration from a JSON file.

    The file is read from `path` if provided, else from $OP_BENCH_REMOTE_HOSTS_PATH.
    Format:
        {
          "hosts": {
            "gpu-a100": {
              "user": "ubuntu",
              "hostname": "10.0.0.42",
              "port": 22,
              "identity_file": "~/.ssh/gpu_key",
              "remote_workspace_root": "/data/op_bench"
            }
          }
        }
    """
    if path is None:
        env_path = os.environ.get("OP_BENCH_REMOTE_HOSTS_PATH")
        if env_path:
            path = Path(env_path)
        else:
            return {}
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        return {}
    data = json.loads(config_path.read_text())
    hosts = data.get("hosts", {})
    return {name: RemoteHost.from_dict(entry) for name, entry in hosts.items()}


class RemoteDockerExecutor:
    """Run docker commands on a remote host over SSH."""

    name = "remote_docker"

    def __init__(
        self,
        host: RemoteHost,
        image: str,
        workspace_dir: str = "/workspace",
        container_name: str | None = None,
        command_workdir: str | None = None,
        labels: dict[str, str] | None = None,
        gpus: str | None = "all",
        remote_workspace: str | None = None,
    ) -> None:
        self.host = host
        self.image = image
        self.workspace_dir = workspace_dir
        self.container_name = container_name
        self.command_workdir = command_workdir or workspace_dir
        self.labels = dict(labels or {})
        self.gpus = gpus
        self._remote_workspace = remote_workspace

    @property
    def remote_workspace(self) -> str:
        if self._remote_workspace:
            return self._remote_workspace
        if self.container_name:
            return f"{self.host.remote_workspace_root}/{self.container_name}"
        return f"{self.host.remote_workspace_root}/default"

    def _ssh(self, remote_command: list[str], timeout_sec: int) -> CommandResult:
        full_command = self.host.ssh_command_prefix() + remote_command
        return _run_local(full_command, timeout_sec)

    def sync_to_remote(self, local_workspace: Path, timeout_sec: int = 600) -> CommandResult:
        """rsync local workspace to remote host."""
        local_path = str(local_workspace.resolve()) + "/"
        remote_path = self.host.rsync_remote_path(self.remote_workspace + "/")
        # Pre-create remote directory
        mkdir_result = self._ssh(["mkdir", "-p", self.remote_workspace], timeout_sec=30)
        if mkdir_result.exit_code != 0:
            return mkdir_result
        cmd = self._rsync_command(local_path, remote_path)
        return _run_local(cmd, timeout_sec)

    def sync_from_remote(self, local_workspace: Path, timeout_sec: int = 600) -> CommandResult:
        """rsync remote workspace back to local."""
        local_path = str(local_workspace.resolve()) + "/"
        remote_path = self.host.rsync_remote_path(self.remote_workspace + "/")
        local_workspace.mkdir(parents=True, exist_ok=True)
        cmd = self._rsync_command(remote_path, local_path)
        return _run_local(cmd, timeout_sec)

    def _rsync_command(self, source: str, destination: str) -> list[str]:
        ssh_parts = ["ssh"]
        if self.host.port != 22:
            ssh_parts.extend(["-p", str(self.host.port)])
        if self.host.identity_file:
            ssh_parts.extend(["-i", os.path.expanduser(self.host.identity_file)])
        ssh_parts.extend(["-o", "StrictHostKeyChecking=accept-new"])
        ssh_command = " ".join(ssh_parts)
        return [
            "rsync",
            "-az",
            "--delete",
            "-e", ssh_command,
            source,
            destination,
        ]

    def command_for_start(self) -> list[str]:
        """Build the remote `docker run --detach` command."""
        if not self.container_name:
            raise ValueError("RemoteDockerExecutor.command_for_start requires a container_name")
        remote_command = [
            "docker", "run", "--detach",
            "--name", self.container_name,
        ]
        if self.gpus:
            remote_command.extend(["--gpus", self.gpus])
        for key, value in sorted(self.labels.items()):
            remote_command.extend(["--label", f"{key}={value}"])
        remote_command.extend([
            "--volume", f"{self.remote_workspace}:{self.workspace_dir}",
            "--workdir", self.command_workdir,
            self.image,
            "tail", "-f", "/dev/null",
        ])
        return self.host.ssh_command_prefix() + remote_command

    def command_for_run(self, command: list[str]) -> list[str]:
        """Build the remote `docker exec` command."""
        if not self.container_name:
            raise ValueError("RemoteDockerExecutor.command_for_run requires a container_name")
        remote_command = [
            "docker", "exec",
            "--workdir", self.command_workdir,
            self.container_name,
            *command,
        ]
        return self.host.ssh_command_prefix() + remote_command

    def start(self, cwd: Path | None = None, timeout_sec: int = 60) -> CommandResult:
        return _run_local(self.command_for_start(), timeout_sec)

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        full_command = self.command_for_run(command)
        return _run_local(full_command, timeout_sec)

    def close(self, timeout_sec: int = 30) -> CommandResult | None:
        if not self.container_name:
            return None
        rm_command = self.host.ssh_command_prefix() + ["docker", "rm", "-f", self.container_name]
        result = _run_local(rm_command, timeout_sec)
        # Best-effort remote workspace cleanup; ignore failures
        if self._remote_workspace is None:
            self._ssh(["rm", "-rf", self.remote_workspace], timeout_sec=30)
        return result

    def collect_environment(self) -> dict[str, object]:
        return {
            "executor": self.name,
            "image": self.image,
            "workspace_dir": self.workspace_dir,
            "command_workdir": self.command_workdir,
            "container_name": self.container_name,
            "gpus": self.gpus,
            "remote_host": self.host.hostname,
            "remote_user": self.host.user,
            "remote_workspace": self.remote_workspace,
            "host_platform": platform.platform(),
            "host_machine": platform.machine(),
        }


def _run_local(command: list[str], timeout_sec: int) -> CommandResult:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return CommandResult(
            command=command,
            cwd="",
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_sec=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command=command,
            cwd="",
            exit_code=124,
            stdout=ensure_text(exc.stdout),
            stderr=ensure_text(exc.stderr),
            duration_sec=time.monotonic() - start,
            timed_out=True,
        )


def check_remote_available() -> bool:
    """Check if both ssh and rsync CLIs are available on the local machine."""
    return shutil.which("ssh") is not None and shutil.which("rsync") is not None
