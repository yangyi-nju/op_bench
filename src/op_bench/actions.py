from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from op_bench.executor import CommandExecutor, CommandResult, LocalExecutor
from op_bench.source_loading import build_source_loading_command
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class WorkspaceActions:
    task: TaskManifest
    workspace: Path
    command_executor: CommandExecutor
    file_executor: LocalExecutor = field(default_factory=LocalExecutor)

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def apply_patch(self, patch: str) -> CommandResult:
        patch_path = self.workspace / ".op_bench_action.patch"
        patch_path.write_text(patch, encoding="utf-8")
        try:
            return self.file_executor.run(["git", "apply", str(patch_path)], self.workspace, timeout_sec=30)
        finally:
            patch_path.unlink(missing_ok=True)

    def run_command(self, command: list[str], timeout_sec: int | None = None) -> CommandResult:
        return self.command_executor.run(command, self.workspace, timeout_sec or self.task.timeout_sec)

    def run_test(self, test_name: str) -> CommandResult:
        source_loading_command = build_source_loading_command(self.task)
        if source_loading_command is not None:
            source_result = self.run_command(source_loading_command, timeout_sec=self.task.timeout_sec)
            if source_result.exit_code != 0 or source_result.timed_out:
                return source_result
        return self.run_command(
            self.task.command_for_test(test_name, python_executable=self.task.environment_python_executable),
            timeout_sec=self.task.timeout_sec,
        )

    def git_diff(self) -> str:
        command = ["git", "diff", "--binary"]
        scope_paths = self.task.patch_scope_paths or self.task.source_loading_overlay_paths
        if scope_paths:
            command.extend(["--", *scope_paths])
        result = self.file_executor.run(command, self.workspace, timeout_sec=30)
        return result.stdout

    def workspace_state_digest(self) -> str:
        status = self.file_executor.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            self.workspace,
            timeout_sec=30,
        )
        payload = "\0".join(
            [
                str(status.exit_code),
                status.stdout,
                status.stderr,
                self.git_diff(),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve(self, path: str) -> Path:
        target = (self.workspace / path).resolve()
        workspace = self.workspace.resolve()
        if target != workspace and workspace not in target.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return target
