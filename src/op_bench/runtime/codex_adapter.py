from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import tempfile

from op_bench.runtime.adapters import AdapterContext
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.process_actions import ProcessActionExchange
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_int,
    require_optional_int,
    require_str,
)


CODEX_ADAPTER_STATUSES = (
    "completed",
    "executable_missing",
    "provider_failure",
    "timeout",
    "nonzero_exit",
    "malformed_action_exchange",
    "missing_finish",
    "duplicate_finish",
)

_TERMINAL_BY_STATUS = {
    "completed": "agent_finished",
    "executable_missing": "provider_error",
    "provider_failure": "provider_error",
    "timeout": "timeout",
    "nonzero_exit": "provider_error",
    "malformed_action_exchange": "runtime_error",
    "missing_finish": "agent_exited",
    "duplicate_finish": "agent_exited",
}

_PROVIDER_FAILURE_MARKERS = (
    "provider error",
    "provider_error",
    "authentication failed",
    "rate limit",
    "rate_limit",
    "quota unavailable",
    "quota exceeded",
    "status: 429",
    " 429 ",
)

_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TERM",
    "TMPDIR",
    "CODEX_HOME",
)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CodexAdapterResult:
    status: str
    terminal_reason: str
    exit_code: int | None
    observation_count: int
    finish_count: int

    def __post_init__(self) -> None:
        selected = require_enum(self.status, "status", CODEX_ADAPTER_STATUSES)
        require_str(self.terminal_reason, "terminal_reason")
        if self.terminal_reason != _TERMINAL_BY_STATUS[selected]:
            raise ContractError("terminal_reason: does not match Adapter status")
        require_optional_int(self.exit_code, "exit_code")
        require_int(self.observation_count, "observation_count", minimum=0)
        require_int(self.finish_count, "finish_count", minimum=0)
        if self.finish_count > self.observation_count:
            raise ContractError("finish_count: cannot exceed observation_count")


def subprocess_command_runner(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout_ms: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(argv),
        cwd=str(cwd),
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_ms / 1000.0,
        check=False,
    )


class CodexCanonicalAdapter:
    """Run Codex against only a public prompt and canonical JSON Action client."""

    def __init__(self, command_runner: object, codex_binary: str = "codex") -> None:
        if not callable(command_runner):
            raise ContractError("command_runner: expected callable")
        self.command_runner = command_runner
        self.codex_binary = require_str(codex_binary, "codex_binary")

    def run(self, context: AdapterContext) -> CodexAdapterResult:
        if not isinstance(context, AdapterContext):
            raise ContractError("context: expected AdapterContext")
        task_view = context.launch_input.task_view
        public_task = task_view.to_dict()
        assert_public_artifact_safe(public_task)
        timeout_ms = task_view.budget_policy.wall_clock_ms
        prompt = _build_prompt(public_task)

        completed: subprocess.CompletedProcess[str] | None = None
        immediate_status: str | None = None
        with tempfile.TemporaryDirectory(prefix="opbench-codex-adapter-") as temporary:
            scratch = Path(temporary) / "scratch"
            exchange = ProcessActionExchange(
                action_client=context.action_client,
                session_id=context.session_id,
                exchange_root=scratch,
                timeout_ms=timeout_ms,
            )
            exchange.start()
            try:
                prompt_path = scratch / "prompt.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                prompt_path.chmod(0o600)
                argv = (
                    self.codex_binary,
                    "exec",
                    "--cd",
                    str(scratch),
                    "--skip-git-repo-check",
                    "--sandbox",
                    "workspace-write",
                    prompt,
                )
                try:
                    candidate = self.command_runner(
                        argv,
                        cwd=scratch,
                        env=_minimal_environment(os.environ),
                        timeout_ms=timeout_ms,
                    )
                except FileNotFoundError:
                    immediate_status = "executable_missing"
                except subprocess.TimeoutExpired:
                    immediate_status = "timeout"
                except OSError:
                    immediate_status = "provider_failure"
                except Exception:  # noqa: BLE001 - fixed Adapter boundary classification.
                    immediate_status = "provider_failure"
                else:
                    if not isinstance(candidate, subprocess.CompletedProcess):
                        immediate_status = "provider_failure"
                    else:
                        completed = candidate
            finally:
                exchange.close()

        status = _classify_result(
            immediate_status=immediate_status,
            completed=completed,
            server_failure=exchange.server_failure,
            finish_count=exchange.finish_count,
        )
        return CodexAdapterResult(
            status=status,
            terminal_reason=_TERMINAL_BY_STATUS[status],
            exit_code=(
                None
                if completed is None
                else require_int(completed.returncode, "returncode")
            ),
            observation_count=exchange.observation_count,
            finish_count=exchange.finish_count,
        )


def _minimal_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: value
        for name in _ENVIRONMENT_ALLOWLIST
        if isinstance((value := environment.get(name)), str) and value
    }


def _build_prompt(public_task: dict[str, object]) -> str:
    task_json = canonical_json(public_task)
    return (
        "You are solving an OpBench task through a canonical action interface.\n"
        "The target repository is inaccessible except through ./opbench_action.py. "
        "Do not try to locate it elsewhere and do not invoke Docker or SSH.\n"
        "Use only these JSON commands, each with --arguments '<JSON object>':\n"
        "  ./opbench_action.py workspace_list --arguments '{}'\n"
        "  ./opbench_action.py workspace_search --arguments '{\"query\":\"symbol\"}'\n"
        "  ./opbench_action.py workspace_read --arguments '{\"path\":\"path.py\"}'\n"
        "  ./opbench_action.py workspace_write --arguments "
        "'{\"path\":\"path.py\",\"content\":\"replacement\"}'\n"
        "  ./opbench_action.py workspace_apply_patch --arguments "
        "'{\"patch\":\"diff --git ...\"}'\n"
        "  ./opbench_action.py command_run --arguments '{\"command\":\"...\"}'\n"
        "  ./opbench_action.py test_run --arguments '{\"selector_id\":\"...\"}'\n"
        "  ./opbench_action.py vcs_diff --arguments '{}'\n"
        "  ./opbench_action.py session_finish --arguments '{}'\n"
        "Read before editing, make the smallest justified change, run an allowed "
        "public test, inspect the diff, and finish exactly once.\n\n"
        "Public task view (canonical JSON):\n"
        f"{task_json}\n"
    )


def _classify_result(
    *,
    immediate_status: str | None,
    completed: subprocess.CompletedProcess[str] | None,
    server_failure: str | None,
    finish_count: int,
) -> str:
    if immediate_status is not None:
        return immediate_status
    if server_failure is not None:
        return "malformed_action_exchange"
    if completed is None:
        return "provider_failure"
    combined = f"{_output_text(completed.stdout)}\n{_output_text(completed.stderr)}".lower()
    if any(marker in combined for marker in _PROVIDER_FAILURE_MARKERS):
        return "provider_failure"
    if completed.returncode != 0:
        return "nonzero_exit"
    if finish_count == 0:
        return "missing_finish"
    if finish_count > 1:
        return "duplicate_finish"
    return "completed"


def _output_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return ""


__all__ = [
    "CODEX_ADAPTER_STATUSES",
    "CodexAdapterResult",
    "CodexCanonicalAdapter",
    "subprocess_command_runner",
]
