from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from op_bench.executor import CommandResult, LocalExecutor
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
    environment: dict[str, str]
    commands: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class Evaluator:
    def __init__(self, executor: LocalExecutor | None = None) -> None:
        self.executor = executor or LocalExecutor()

    def evaluate_baseline(self, task: TaskManifest) -> EvaluationResult:
        return self._evaluate(task=task, mode="baseline", patch_path=None)

    def evaluate_gold(self, task: TaskManifest) -> EvaluationResult:
        return self._evaluate(task=task, mode="gold", patch_path=task.gold_patch_path)

    def evaluate_patch(self, task: TaskManifest, patch_path: Path, agent_name: str) -> EvaluationResult:
        return self._evaluate(task=task, mode=f"agent:{agent_name}", patch_path=patch_path)

    def _evaluate(self, task: TaskManifest, mode: str, patch_path: Path | None) -> EvaluationResult:
        start = time.monotonic()
        command_log: list[CommandResult] = []
        environment = self.executor.collect_environment().to_dict()
        with tempfile.TemporaryDirectory(prefix=f"op-bench-{task.task_id}-") as tmp:
            workspace = Path(tmp) / "workspace"
            prepare_error = self._prepare_workspace(task, workspace)
            if prepare_error is not None:
                return self._result(task, mode, "runner_error", start, environment, command_log)

            for command in task.setup_commands:
                result = self.executor.run(command.split(), workspace, task.timeout_sec)
                command_log.append(result)
                if result.timed_out:
                    return self._result(task, mode, "timeout", start, environment, command_log)
                if result.exit_code != 0:
                    return self._result(task, mode, "setup_failed", start, environment, command_log)

            test_patch_result = self._apply_patch(task.test_patch_path, workspace)
            command_log.append(test_patch_result)
            if test_patch_result.timed_out:
                return self._result(task, mode, "timeout", start, environment, command_log)
            if test_patch_result.exit_code != 0:
                return self._result(task, mode, "runner_error", start, environment, command_log)

            if patch_path is not None and patch_path.read_text(encoding="utf-8").strip():
                patch_result = self._apply_patch(patch_path, workspace)
                command_log.append(patch_result)
                if patch_result.timed_out:
                    return self._result(task, mode, "timeout", start, environment, command_log)
                if patch_result.exit_code != 0:
                    return self._result(task, mode, "patch_apply_failed", start, environment, command_log)

            fail_results = self._run_tests(task.fail_to_pass_tests, task, workspace)
            command_log.extend(fail_results)
            if any(result.timed_out for result in fail_results):
                return self._result(task, mode, "timeout", start, environment, command_log)

            pass_results = self._run_tests(task.pass_to_pass_tests, task, workspace)
            command_log.extend(pass_results)
            if any(result.timed_out for result in pass_results):
                return self._result(task, mode, "timeout", start, environment, command_log)

            fail_passed = sum(1 for result in fail_results if result.exit_code == 0)
            pass_passed = sum(1 for result in pass_results if result.exit_code == 0)
            status = self._classify(
                mode=mode,
                fail_total=len(fail_results),
                fail_passed=fail_passed,
                pass_total=len(pass_results),
                pass_passed=pass_passed,
            )
            return EvaluationResult(
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

    def _prepare_workspace(self, task: TaskManifest, workspace: Path) -> str | None:
        if task.checkout_mode != "local-copy":
            return f"unsupported checkout mode: {task.checkout_mode}"
        source = task.local_source_path
        if source is None or not source.exists():
            return f"local source path not found: {source}"
        shutil.copytree(source, workspace)
        return None

    def _apply_patch(self, patch_path: Path, workspace: Path) -> CommandResult:
        return self.executor.run(["git", "apply", str(patch_path)], workspace, timeout_sec=30)

    def _run_tests(self, tests: list[str], task: TaskManifest, workspace: Path) -> list[CommandResult]:
        return [
            self.executor.run(task.command_for_test(test_name), workspace, task.timeout_sec)
            for test_name in tests
        ]

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
        environment: dict[str, str],
        command_log: list[CommandResult],
    ) -> EvaluationResult:
        return EvaluationResult(
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
