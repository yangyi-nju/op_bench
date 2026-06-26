from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from op_bench.environment import EnvironmentManager
from op_bench.executor import CommandExecutor, CommandResult, LocalExecutor
from op_bench.patch_scope import validate_patch_scope
from op_bench.progress import Progress, format_command, format_duration, noop_progress
from op_bench.source_loading import build_source_loading_command
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    mode: str
    status: str
    fail_to_pass_total: int
    fail_to_pass_passed: int
    pass_to_pass_total: int
    pass_to_pass_passed: int
    duration_sec: float
    environment: dict[str, object]
    commands: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Evaluator:
    def __init__(
        self,
        executor: LocalExecutor | None = None,
        environment_manager: EnvironmentManager | None = None,
        workspace_root: Path | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.executor = executor or LocalExecutor()
        self.progress = progress or noop_progress
        self.environment_manager = environment_manager or EnvironmentManager(self.executor, progress=self.progress)
        self.workspace_root = workspace_root

    def evaluate_baseline(self, task: TaskManifest) -> EvaluationResult:
        return self._evaluate(task=task, mode="baseline", patch_path=None)

    def evaluate_gold(self, task: TaskManifest) -> EvaluationResult:
        return self._evaluate(task=task, mode="gold", patch_path=task.gold_patch_path)

    def evaluate_patch(self, task: TaskManifest, patch_path: Path, agent_name: str) -> EvaluationResult:
        return self._evaluate(task=task, mode=f"agent:{agent_name}", patch_path=patch_path)

    def _evaluate(self, task: TaskManifest, mode: str, patch_path: Path | None) -> EvaluationResult:
        start = time.monotonic()
        self.progress(f"{mode} start: task={task.task_id}")
        command_log: list[CommandResult] = []
        environment = self.executor.collect_environment().to_dict()
        workspace_parent = self._workspace_parent(task)
        with tempfile.TemporaryDirectory(prefix=f"op-bench-{task.task_id}-", dir=workspace_parent) as tmp:
            workspace = Path(tmp) / "workspace"
            self.progress(f"{mode}: workspace={workspace}")
            environment_precheck = self.environment_manager.precheck(task)
            if environment_precheck is not None:
                environment = environment_precheck.evidence
                if environment_precheck.error:
                    environment["error"] = environment_precheck.error
                return self._result(task, mode, "environment_unavailable", start, environment, command_log)

            prepare_error = self._prepare_workspace(task, workspace, command_log)
            if prepare_error is not None:
                self.progress(f"{mode} workspace preparation failed: {prepare_error}")
                return self._result(task, mode, "runner_error", start, environment, command_log)

            self.progress(f"{mode}: prepare environment")
            environment_preparation = self.environment_manager.prepare(task, workspace)
            command_log.extend(environment_preparation.commands)
            environment = environment_preparation.evidence
            if not environment_preparation.available:
                if environment_preparation.error:
                    environment["error"] = environment_preparation.error
                    self.progress(f"{mode} environment unavailable: {environment_preparation.error}")
                return self._result(task, mode, "environment_unavailable", start, environment, command_log)
            runtime_executor = environment_preparation.executor

            def finish(status: str) -> EvaluationResult:
                cleanup_result = self.environment_manager.cleanup(environment_preparation)
                if cleanup_result is not None:
                    command_log.append(cleanup_result)
                return self._result(task, mode, status, start, environment, command_log)

            for command in task.setup_commands:
                result = self._run_logged(
                    runtime_executor,
                    task.render_command(command, python_executable=task.environment_python_executable),
                    workspace,
                    task.timeout_sec,
                    label=f"{mode} setup",
                )
                command_log.append(result)
                if result.timed_out:
                    return finish("timeout")
                if result.exit_code != 0:
                    return finish("setup_failed")

            test_patch_result = self._apply_patch(task.hidden_test_patch_path, workspace)
            command_log.append(test_patch_result)
            if test_patch_result.timed_out:
                return finish("timeout")
            if test_patch_result.exit_code != 0:
                return finish("runner_error")

            if task.public_test_patch_path is not None and task.public_test_patch_path.exists():
                public_patch_result = self._apply_patch(task.public_test_patch_path, workspace)
                command_log.append(public_patch_result)
                if public_patch_result.timed_out:
                    return finish("timeout")
                if public_patch_result.exit_code != 0:
                    return finish("runner_error")

            if patch_path is not None and patch_path.read_text(encoding="utf-8").strip():
                patch_text = patch_path.read_text(encoding="utf-8")
                if task.patch_scope_paths and mode.startswith("agent:"):
                    scope_result = validate_patch_scope(patch_text, task.patch_scope_paths, task.patch_scope_mode or "enforced")
                    if scope_result.status == "out_of_scope":
                        self.progress(f"{mode} patch out of scope: {scope_result.out_of_scope_paths}")
                        return finish("patch_out_of_scope")
                    if scope_result.status == "filtered":
                        self.progress(f"{mode} patch filtered, removed: {scope_result.out_of_scope_paths}")
                        patch_text = scope_result.filtered_patch
                        if not patch_text.strip():
                            return finish("patch_out_of_scope")
                    filtered_patch_path = workspace / ".op_bench_filtered.patch"
                    filtered_patch_path.write_text(patch_text, encoding="utf-8")
                    patch_result = self._apply_patch(filtered_patch_path, workspace)
                    filtered_patch_path.unlink(missing_ok=True)
                else:
                    patch_result = self._apply_patch(patch_path, workspace)
                command_log.append(patch_result)
                if patch_result.timed_out:
                    return finish("timeout")
                if patch_result.exit_code != 0:
                    return finish("patch_apply_failed")

            sync_to_remote = getattr(runtime_executor, "sync_to_remote", None)
            if callable(sync_to_remote):
                self.progress(f"{mode}: sync patched workspace to remote")
                sync_result = sync_to_remote(workspace, timeout_sec=task.timeout_sec)
                command_log.append(sync_result)
                if sync_result.timed_out:
                    return finish("timeout")
                if sync_result.exit_code != 0:
                    return finish("environment_unavailable")

            fail_commands, fail_results = self._run_tests(task.fail_to_pass_tests, task, workspace, runtime_executor)
            command_log.extend(fail_commands)
            if any(result.timed_out for result in fail_results):
                return finish("timeout")

            pass_commands, pass_results = self._run_tests(task.pass_to_pass_tests, task, workspace, runtime_executor)
            command_log.extend(pass_commands)
            if any(result.timed_out for result in pass_results):
                return finish("timeout")
            if self._has_environment_error(fail_results + pass_results):
                return finish("environment_error")

            fail_passed = sum(1 for result in fail_results if result.exit_code == 0)
            pass_passed = sum(1 for result in pass_results if result.exit_code == 0)
            status = self._classify(
                mode=mode,
                fail_total=len(fail_results),
                fail_passed=fail_passed,
                pass_total=len(pass_results),
                pass_passed=pass_passed,
            )
            cleanup_result = self.environment_manager.cleanup(environment_preparation)
            if cleanup_result is not None:
                command_log.append(cleanup_result)
            result = EvaluationResult(
                task_id=task.task_id,
                mode=mode,
                status=status,
                fail_to_pass_total=len(fail_results),
                fail_to_pass_passed=fail_passed,
                pass_to_pass_total=len(pass_results),
                pass_to_pass_passed=pass_passed,
                duration_sec=time.monotonic() - start,
                environment=environment,
                commands=[result.to_dict() for result in command_log],
            )
            self._log_evaluation_result(result)
            return result

    def _workspace_parent(self, task: TaskManifest) -> str | None:
        if self.workspace_root is not None:
            self.workspace_root.mkdir(parents=True, exist_ok=True)
            return str(self.workspace_root)
        if task.environment_backend != "docker":
            return None
        root = self._repo_root(task.task_dir)
        workspace_root = root / ".op_bench_cache" / "workspaces"
        workspace_root.mkdir(parents=True, exist_ok=True)
        return str(workspace_root)

    def _repo_root(self, start: Path) -> Path:
        for path in [start.resolve(), *start.resolve().parents]:
            if (path / ".git").exists():
                return path
        return Path.cwd().resolve()

    def _prepare_workspace(
        self,
        task: TaskManifest,
        workspace: Path,
        command_log: list[CommandResult],
    ) -> str | None:
        return self.prepare_workspace(task, workspace, command_log)

    def prepare_workspace(
        self,
        task: TaskManifest,
        workspace: Path,
        command_log: list[CommandResult] | None = None,
    ) -> str | None:
        snapshot_error = self._prepare_snapshot_workspace(task, workspace, command_log)
        if snapshot_error is None:
            return None
        if snapshot_error != "source snapshot not configured":
            return snapshot_error

        if task.checkout_mode == "git":
            clone_timeout = max(300, task.timeout_sec)
            workspace.mkdir(parents=True, exist_ok=True)
            for command, cwd, error_label in [
                (["git", "init"], workspace, "git init failed"),
                (["git", "remote", "add", "origin", task.repo_url], workspace, "git remote add failed"),
                (
                    [
                        "git",
                        "fetch",
                        "--depth=1",
                        "--filter=blob:none",
                        "origin",
                        task.base_commit,
                    ],
                    workspace,
                    "git fetch failed",
                ),
                (
                    ["git", "-c", "advice.detachedHead=false", "checkout", "--detach", "FETCH_HEAD"],
                    workspace,
                    "git checkout failed",
                ),
            ]:
                result = self._run_logged(self.executor, command, cwd, clone_timeout, label="prepare git workspace")
                if command_log is not None:
                    command_log.append(result)
                if result.exit_code != 0:
                    return result.stderr or result.stdout or error_label
            return None
        if task.checkout_mode != "local-copy":
            return f"unsupported checkout mode: {task.checkout_mode}"
        source = task.local_source_path
        if source is None or not source.exists():
            return f"local source path not found: {source}"
        shutil.copytree(source, workspace)
        return None

    def _prepare_snapshot_workspace(
        self,
        task: TaskManifest,
        workspace: Path,
        command_log: list[CommandResult] | None,
    ) -> str | None:
        snapshot = task.source_snapshot_path
        if snapshot is None:
            return "source snapshot not configured"
        start = time.monotonic()
        if not snapshot.exists():
            result = CommandResult(
                command=["op_bench", "copy-source-snapshot", str(snapshot), str(workspace)],
                cwd=str(workspace.parent),
                exit_code=1,
                stdout="",
                stderr=f"source snapshot not found: {snapshot}",
                duration_sec=time.monotonic() - start,
            )
            if command_log is not None:
                command_log.append(result)
            return "source snapshot not configured"
        try:
            self.progress(f"prepare source snapshot: {snapshot} -> {workspace}")
            shutil.copytree(snapshot, workspace, symlinks=True)
        except OSError as exc:
            result = CommandResult(
                command=["op_bench", "copy-source-snapshot", str(snapshot), str(workspace)],
                cwd=str(workspace.parent),
                exit_code=1,
                stdout="",
                stderr=str(exc),
                duration_sec=time.monotonic() - start,
            )
            if command_log is not None:
                command_log.append(result)
            return str(exc)
        result = CommandResult(
            command=["op_bench", "copy-source-snapshot", str(snapshot), str(workspace)],
            cwd=str(workspace.parent),
            exit_code=0,
            stdout=f"copied source snapshot from {snapshot}\n",
            stderr="",
            duration_sec=time.monotonic() - start,
        )
        if command_log is not None:
            command_log.append(result)
        self.progress(f"prepare source snapshot done: duration={format_duration(result.duration_sec)}")
        return None

    def _apply_patch(self, patch_path: Path, workspace: Path) -> CommandResult:
        result = self._run_logged(
            self.executor, ["git", "apply", str(patch_path)], workspace,
            timeout_sec=30, label="apply patch",
        )
        if result.exit_code != 0 and not result.timed_out:
            # Fallback to GNU patch with fuzz tolerance for line number drift between
            # the upstream PR base and our snapshot base commit.
            self.progress("git apply failed, retrying with patch -F 3")
            fallback = self._run_logged(
                self.executor,
                ["bash", "-lc", f"patch -p1 -F 3 -i {str(patch_path)!s}"],
                workspace, timeout_sec=30, label="apply patch (fuzz)",
            )
            if fallback.exit_code == 0:
                return fallback
            return result
        return result

    def _run_tests(
        self,
        tests: list[str],
        task: TaskManifest,
        workspace: Path,
        executor: CommandExecutor,
    ) -> tuple[list[CommandResult], list[CommandResult]]:
        command_results: list[CommandResult] = []
        test_results: list[CommandResult] = []
        source_loading_command = build_source_loading_command(task)
        for test_name in tests:
            if source_loading_command is not None:
                source_result = self._run_logged(
                    executor,
                    source_loading_command,
                    workspace,
                    task.timeout_sec,
                    label=f"sync source overlay for {test_name}",
                )
                command_results.append(source_result)
                if source_result.timed_out or source_result.exit_code != 0:
                    test_results.append(source_result)
                    continue
            test_result = self._run_logged(
                executor,
                task.command_for_test(test_name, python_executable=task.environment_python_executable),
                workspace,
                task.timeout_sec,
                label=f"run test {test_name}",
            )
            command_results.append(test_result)
            test_results.append(test_result)
        return command_results, test_results

    def _has_environment_error(self, results: list[CommandResult]) -> bool:
        if not results:
            return False
        failed_outputs = [
            f"{result.stdout}\n{result.stderr}"
            for result in results
            if result.exit_code != 0
        ]
        if not failed_outputs:
            return False
        environment_markers = (
            "ModuleNotFoundError:",
            "ImportError:",
            "OSError:",
            "No module named",
            "cannot open shared object file",
            "Library not loaded:",
            "CUDA error:",
            "CUDA is not available",
            "Found no NVIDIA driver",
            "No CUDA GPUs are available",
        )
        return any(
            any(marker in output for marker in environment_markers)
            for output in failed_outputs
        )

    def _classify(
        self,
        *,
        mode: str,
        fail_total: int,
        fail_passed: int,
        pass_total: int,
        pass_passed: int,
    ) -> str:
        if mode == "baseline":
            if fail_passed < fail_total and pass_passed == pass_total:
                return "baseline_reproduced"
            return "baseline_not_reproduced"
        if pass_passed < pass_total:
            return "pass_to_pass_regressed"
        if fail_passed < fail_total:
            return "fail_to_pass_failed"
        return "resolved"

    def _result(
        self,
        task: TaskManifest,
        mode: str,
        status: str,
        start: float,
        environment: dict[str, object],
        command_log: list[CommandResult],
    ) -> EvaluationResult:
        result = EvaluationResult(
            task_id=task.task_id,
            mode=mode,
            status=status,
            fail_to_pass_total=len(task.fail_to_pass_tests),
            fail_to_pass_passed=0,
            pass_to_pass_total=len(task.pass_to_pass_tests),
            pass_to_pass_passed=0,
            duration_sec=time.monotonic() - start,
            environment=environment,
            commands=[result.to_dict() for result in command_log],
        )
        self._log_evaluation_result(result)
        return result

    def _run_logged(
        self,
        executor: CommandExecutor,
        command: list[str],
        cwd: Path,
        timeout_sec: int,
        label: str,
    ) -> CommandResult:
        self.progress(f"{label}: {format_command(command)}")
        result = executor.run(command, cwd, timeout_sec)
        suffix = " timeout" if result.timed_out else ""
        self.progress(f"{label} done: exit={result.exit_code}{suffix}, duration={format_duration(result.duration_sec)}")
        return result

    def _log_evaluation_result(self, result: EvaluationResult) -> None:
        self.progress(
            f"{result.mode} done: status={result.status}, "
            f"fail_to_pass={result.fail_to_pass_passed}/{result.fail_to_pass_total}, "
            f"pass_to_pass={result.pass_to_pass_passed}/{result.pass_to_pass_total}, "
            f"duration={format_duration(result.duration_sec)}"
        )
