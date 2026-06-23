from __future__ import annotations

import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from op_bench.executor import CommandExecutor, CommandResult, DockerExecutor, LocalExecutor, ensure_text
from op_bench.progress import Progress, format_command, format_duration, noop_progress
from op_bench.remote import RemoteDockerExecutor, RemoteHost, check_remote_available, load_hosts_config
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class EnvironmentPreparation:
    status: str
    executor: CommandExecutor
    evidence: dict[str, object]
    commands: list[CommandResult]
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "ready"

    def commands_as_dicts(self) -> list[dict[str, object]]:
        return [command.to_dict() for command in self.commands]


class EnvironmentManager:
    def __init__(
        self,
        host_executor: LocalExecutor | None = None,
        progress: Progress | None = None,
        hosts_config: dict[str, RemoteHost] | None = None,
    ) -> None:
        self.host_executor = host_executor or LocalExecutor()
        self.progress = progress or noop_progress
        self._hosts_config = hosts_config

    def _resolve_host(self, host_name: str) -> RemoteHost | None:
        if self._hosts_config is None:
            self._hosts_config = load_hosts_config()
        return self._hosts_config.get(host_name)

    def prepare(self, task: TaskManifest, workspace: Path) -> EnvironmentPreparation:
        if task.environment_backend == "remote_docker":
            return self._prepare_remote_docker(task, workspace)
        if task.environment_backend == "docker":
            return self._prepare_docker(task, workspace)
        evidence = self.host_executor.collect_environment().to_dict()
        evidence.update(self._asset_evidence(task))
        return EnvironmentPreparation(
            status="ready",
            executor=self.host_executor,
            evidence=evidence,
            commands=[],
        )

    def precheck(self, task: TaskManifest) -> EnvironmentPreparation | None:
        if task.environment_backend == "remote_docker":
            if not check_remote_available():
                return EnvironmentPreparation(
                    status="environment_unavailable",
                    executor=self.host_executor,
                    evidence={
                        "executor": "remote_docker",
                        "image": task.environment_image,
                        "remote_available": False,
                        **self._asset_evidence(task),
                    },
                    commands=[],
                    error="ssh or rsync command not found on local host",
                )
            return None
        if task.environment_backend != "docker" or shutil.which("docker") is not None:
            return None
        return EnvironmentPreparation(
            status="environment_unavailable",
            executor=DockerExecutor(task.environment_image, task.environment_workspace_dir),
            evidence={
                "executor": "docker",
                "image": task.environment_image,
                "workspace_dir": task.environment_workspace_dir,
                "docker_available": False,
                **self._asset_evidence(task),
            },
            commands=[],
            error="docker command not found",
        )

    def cleanup(self, preparation: EnvironmentPreparation) -> CommandResult | None:
        return preparation.executor.close()

    def _prepare_docker(self, task: TaskManifest, workspace: Path) -> EnvironmentPreparation:
        commands: list[CommandResult] = []
        if shutil.which("docker") is None:
            return EnvironmentPreparation(
                status="environment_unavailable",
                executor=DockerExecutor(task.environment_image, task.environment_workspace_dir),
                evidence={
                    "executor": "docker",
                    "image": task.environment_image,
                    "workspace_dir": task.environment_workspace_dir,
                    "docker_available": False,
                    **self._asset_evidence(task),
                },
                commands=commands,
                error="docker command not found",
            )

        inspect_result = self._run_host(
            ["docker", "image", "inspect", task.environment_image],
            cwd=workspace,
            timeout_sec=60,
            label="inspect docker image",
        )
        commands.append(inspect_result)

        if inspect_result.exit_code != 0 and task.environment_dockerfile_path is not None:
            build_context = task.environment_build_context_path or task.environment_dockerfile_path.parent
            build_result = self._run_host(
                [
                    "docker",
                    "build",
                    "-t",
                    task.environment_image,
                    "-f",
                    str(task.environment_dockerfile_path),
                    str(build_context),
                ],
                cwd=build_context,
                timeout_sec=task.timeout_sec,
                label="build docker image",
            )
            commands.append(build_result)
            if build_result.timed_out:
                return self._unavailable(task, commands, "docker build timed out")
            if build_result.exit_code != 0:
                return self._unavailable(task, commands, "docker build failed")

            inspect_result = self._run_host(
                ["docker", "image", "inspect", task.environment_image],
                cwd=workspace,
                timeout_sec=60,
                label="inspect built docker image",
            )
            commands.append(inspect_result)

        if inspect_result.exit_code != 0:
            return self._unavailable(task, commands, "docker image unavailable")

        executor = DockerExecutor(
            task.environment_image,
            task.environment_workspace_dir,
            container_name=self._container_name(task),
            labels={
                "op-bench.managed": "true",
                "op-bench.task-id": task.task_id,
                "op-bench.environment-id": task.environment_ref or "inline",
                "op-bench.runtime-tier": task.runtime_tier or "unspecified",
            },
        )
        start_result = self._run_executor(executor.start, workspace, timeout_sec=60, label="start docker container")
        commands.append(start_result)
        if start_result.timed_out:
            cleanup_result = executor.close()
            if cleanup_result is not None:
                commands.append(cleanup_result)
            return self._unavailable(task, commands, "docker container start timed out")
        if start_result.exit_code != 0:
            cleanup_result = executor.close()
            if cleanup_result is not None:
                commands.append(cleanup_result)
            return self._unavailable(task, commands, "docker container start failed")

        preflight_executor = DockerExecutor(
            task.environment_image,
            task.environment_workspace_dir,
            container_name=executor.container_name,
            command_workdir=task.environment_preflight_workdir,
            labels=executor.labels,
        )
        for command in task.environment_preflight_commands:
            result = self._run_executor(
                preflight_executor.run,
                task.render_command(command, python_executable=task.environment_python_executable),
                workspace,
                task.timeout_sec,
                label="run environment preflight",
            )
            commands.append(result)
            if result.timed_out:
                cleanup_result = executor.close()
                if cleanup_result is not None:
                    commands.append(cleanup_result)
                return self._unavailable(task, commands, "environment preflight timed out")
            if result.exit_code != 0:
                cleanup_result = executor.close()
                if cleanup_result is not None:
                    commands.append(cleanup_result)
                return self._unavailable(task, commands, "environment preflight failed")

        evidence = executor.collect_environment()
        evidence["preflight_passed"] = True
        evidence["preflight_command_count"] = len(task.environment_preflight_commands)
        evidence["preflight_workdir"] = task.environment_preflight_workdir
        evidence.update(self._asset_evidence(task))
        return EnvironmentPreparation(
            status="ready",
            executor=executor,
            evidence=evidence,
            commands=commands,
        )

    def _prepare_remote_docker(self, task: TaskManifest, workspace: Path) -> EnvironmentPreparation:
        commands: list[CommandResult] = []
        host_name = task.environment_host
        if not host_name:
            return self._unavailable_remote(task, commands, "task does not specify environment.host")
        host = self._resolve_host(host_name)
        if host is None:
            return self._unavailable_remote(
                task, commands,
                f"remote host '{host_name}' not found in OP_BENCH_REMOTE_HOSTS_PATH config",
            )
        if not check_remote_available():
            return self._unavailable_remote(task, commands, "ssh or rsync command not found on local host")

        container_name = self._container_name(task)
        executor = RemoteDockerExecutor(
            host=host,
            image=task.environment_image,
            workspace_dir=task.environment_workspace_dir,
            container_name=container_name,
            gpus=task.environment_gpus,
            labels={
                "op-bench.managed": "true",
                "op-bench.task-id": task.task_id,
                "op-bench.environment-id": task.environment_ref or "inline",
                "op-bench.runtime-tier": task.runtime_tier or "unspecified",
            },
        )

        # Sync workspace to remote
        self.progress(f"sync workspace to remote: {host.ssh_target()}:{executor.remote_workspace}")
        sync_result = executor.sync_to_remote(workspace, timeout_sec=task.timeout_sec)
        commands.append(sync_result)
        if sync_result.exit_code != 0:
            return self._unavailable_remote(task, commands, f"rsync to remote failed: {sync_result.stderr.strip()[:200]}")

        # Optional preflight: nvidia-smi on remote
        if task.requires_gpu:
            nvidia_check = executor._ssh(["nvidia-smi"], timeout_sec=30)
            commands.append(nvidia_check)
            if nvidia_check.exit_code != 0:
                return self._unavailable_remote(task, commands, "nvidia-smi failed on remote host")

        # Inspect remote docker image; build if missing and dockerfile available
        inspect_result = executor._ssh(["docker", "image", "inspect", task.environment_image], timeout_sec=60)
        commands.append(inspect_result)
        if inspect_result.exit_code != 0:
            return self._unavailable_remote(
                task, commands,
                f"remote docker image '{task.environment_image}' not available; build it on the remote host first",
            )

        # Start the container
        self.progress(f"start remote docker container: {container_name}")
        start_result = executor.start(timeout_sec=120)
        commands.append(start_result)
        if start_result.exit_code != 0:
            cleanup_result = executor.close()
            if cleanup_result is not None:
                commands.append(cleanup_result)
            return self._unavailable_remote(
                task, commands,
                f"remote docker run failed: {start_result.stderr.strip()[:200]}",
            )

        # Run preflight commands inside container
        for command in task.environment_preflight_commands:
            rendered = task.render_command(command, python_executable=task.environment_python_executable)
            preflight_executor = RemoteDockerExecutor(
                host=host,
                image=task.environment_image,
                workspace_dir=task.environment_workspace_dir,
                container_name=container_name,
                command_workdir=task.environment_preflight_workdir,
                gpus=task.environment_gpus,
                labels=executor.labels,
                remote_workspace=executor.remote_workspace,
            )
            result = preflight_executor.run(rendered, workspace, task.timeout_sec)
            commands.append(result)
            if result.exit_code != 0:
                cleanup_result = executor.close()
                if cleanup_result is not None:
                    commands.append(cleanup_result)
                return self._unavailable_remote(task, commands, "remote environment preflight failed")

        evidence = executor.collect_environment()
        evidence["preflight_passed"] = True
        evidence["preflight_command_count"] = len(task.environment_preflight_commands)
        evidence["preflight_workdir"] = task.environment_preflight_workdir
        evidence.update(self._asset_evidence(task))
        return EnvironmentPreparation(
            status="ready",
            executor=executor,
            evidence=evidence,
            commands=commands,
        )

    def _unavailable_remote(
        self,
        task: TaskManifest,
        commands: list[CommandResult],
        error: str,
    ) -> EnvironmentPreparation:
        return EnvironmentPreparation(
            status="environment_unavailable",
            executor=self.host_executor,
            evidence={
                "executor": "remote_docker",
                "image": task.environment_image,
                "workspace_dir": task.environment_workspace_dir,
                "remote_host": task.environment_host,
                "preflight_passed": False,
                **self._asset_evidence(task),
            },
            commands=commands,
            error=error,
        )

    def _container_name(self, task: TaskManifest) -> str:
        safe_task_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in task.task_id
        )
        return f"op-bench-{safe_task_id[:48]}-{uuid.uuid4().hex[:12]}"

    def _unavailable(
        self,
        task: TaskManifest,
        commands: list[CommandResult],
        error: str,
    ) -> EnvironmentPreparation:
        return EnvironmentPreparation(
            status="environment_unavailable",
            executor=DockerExecutor(task.environment_image, task.environment_workspace_dir),
            evidence={
                "executor": "docker",
                "image": task.environment_image,
                "workspace_dir": task.environment_workspace_dir,
                "docker_available": True,
                "preflight_passed": False,
                **self._asset_evidence(task),
            },
            commands=commands,
            error=error,
        )

    def _asset_evidence(self, task: TaskManifest) -> dict[str, object]:
        return {
            "environment_id": task.environment_ref,
            "runtime_tier": task.runtime_tier,
            "image_digest": task.environment_image_digest,
            "digest_kind": task.environment_digest_kind,
            "platform": task.environment_platform,
        }

    def _run_host(self, command: list[str], cwd: Path, timeout_sec: int, label: str) -> CommandResult:
        self.progress(f"{label}: {format_command(command)}")
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
            result = CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
            self._log_result(label, result)
            return result
        except subprocess.TimeoutExpired as exc:
            result = CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=124,
                stdout=ensure_text(exc.stdout),
                stderr=ensure_text(exc.stderr),
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )
            self._log_result(label, result)
            return result

    def _run_executor(self, run: Callable[..., CommandResult], *args: object, label: str, **kwargs: object) -> CommandResult:
        command = self._command_preview(args)
        self.progress(f"{label}: {command}")
        result = run(*args, **kwargs)
        self._log_result(label, result)
        return result

    def _command_preview(self, args: tuple[object, ...]) -> str:
        if args and isinstance(args[0], list):
            return format_command([str(part) for part in args[0]])
        return "(docker start)"

    def _log_result(self, label: str, result: CommandResult) -> None:
        suffix = " timeout" if result.timed_out else ""
        self.progress(f"{label} done: exit={result.exit_code}{suffix}, duration={format_duration(result.duration_sec)}")
