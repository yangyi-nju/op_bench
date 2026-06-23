from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from op_bench.action_bridge import ActionBridgeServer, action_log_count, action_log_integrity, build_action_cli
from op_bench.actions import WorkspaceActions
from op_bench.progress import Progress, noop_progress
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class AgentOutput:
    agent_name: str
    patch_path: Path
    metadata: dict[str, object]


class AgentRuntimeUnsupported(RuntimeError):
    pass


def _codex_timeout_sec() -> int:
    return int(os.environ.get("OP_BENCH_CODEX_TIMEOUT_SEC", "1200"))


def _claude_timeout_sec() -> int:
    return int(os.environ.get("OP_BENCH_CLAUDE_TIMEOUT_SEC", "1200"))


def _run_codex(command: list[str], cwd: Path) -> tuple[subprocess.CompletedProcess[str], bool]:
    timeout_sec = _codex_timeout_sec()
    try:
        return (
            subprocess.run(
                command,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            ),
            False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            subprocess.CompletedProcess(
                command,
                124,
                stdout=exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
                stderr=exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
            ),
            True,
        )


def _run_claude(command: list[str], cwd: Path) -> tuple[subprocess.CompletedProcess[str], bool]:
    timeout_sec = _claude_timeout_sec()
    try:
        return (
            subprocess.run(
                command,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            ),
            False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            subprocess.CompletedProcess(
                command,
                124,
                stdout=exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or "",
                stderr=exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or "",
            ),
            True,
        )


class GoldAgent:
    name = "gold"
    requires_workspace = False
    requires_actions = False

    def produce_patch(
        self,
        task: TaskManifest,
        output_dir: Path,
        workspace: Path | None = None,
        actions: WorkspaceActions | None = None,
    ) -> AgentOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_path = output_dir / f"{task.task_id}__gold.patch"
        shutil.copyfile(task.gold_patch_path, patch_path)
        return AgentOutput(agent_name=self.name, patch_path=patch_path, metadata={"source": "gold_patch"})


class CodexActionBridgeAgent:
    name = "codex_action_bridge"
    requires_workspace = True
    requires_actions = True
    runtime_boundary = "op_bench_action_interface_file_cli"

    def __init__(self, progress: Progress | None = None, hide_public_tests: bool = False) -> None:
        self.progress = progress or noop_progress
        self.hide_public_tests = hide_public_tests

    def produce_patch(
        self,
        task: TaskManifest,
        output_dir: Path,
        workspace: Path | None = None,
        actions: WorkspaceActions | None = None,
    ) -> AgentOutput:
        if workspace is None or actions is None:
            raise AgentRuntimeUnsupported("CodexActionBridgeAgent requires a prepared workspace and WorkspaceActions")
        output_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = output_dir / "codex_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        last_message_path = output_dir / f"{task.task_id}__{self.name}_last_message.txt"
        action_log_path = output_dir / f"{task.task_id}__{self.name}_actions.jsonl"
        initial_digest = actions.workspace_state_digest()
        start = time.monotonic()
        exchange_dir = scratch_dir / ".opbench_action_exchange"
        with ActionBridgeServer(
            actions=actions,
            log_path=action_log_path,
            exchange_dir=exchange_dir,
            progress=self.progress,
        ) as bridge:
            action_cli = scratch_dir / "opbench_action.py"
            action_cli.write_text(build_action_cli(bridge.exchange_dir, sys.executable), encoding="utf-8")
            action_cli.chmod(0o700)
            prompt = self._build_bridge_prompt(task, action_cli.name)
            completed, timed_out = _run_codex(
                [
                    "codex",
                    "exec",
                    "--cd",
                    str(scratch_dir),
                    "--skip-git-repo-check",
                    "--sandbox",
                    "workspace-write",
                    "--output-last-message",
                    str(last_message_path),
                    prompt,
                ],
                scratch_dir,
            )
        patch_path = output_dir / f"{task.task_id}__{self.name}.patch"
        patch_path.write_text(actions.git_diff(), encoding="utf-8")
        action_count = action_log_count(action_log_path)
        integrity = action_log_integrity(action_log_path, initial_digest, actions.workspace_state_digest())
        return AgentOutput(
            agent_name=self.name,
            patch_path=patch_path,
            metadata={
                "command": "codex exec",
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": timed_out,
                "timeout_sec": _codex_timeout_sec(),
                "duration_sec": time.monotonic() - start,
                "last_message_path": str(last_message_path),
                "action_log_path": str(action_log_path),
                "runtime_boundary": self.runtime_boundary,
                "shell_boundary": "workspace_write_scratch_only",
                "action_count": action_count,
                "integrity_status": integrity["status"],
                "integrity_errors": integrity["errors"],
            },
        )

    def _build_bridge_prompt(self, task: TaskManifest, action_cli_name: str) -> str:
        issue_text = ""
        if task.issue_markdown_path.exists():
            issue_text = task.issue_markdown_path.read_text(encoding="utf-8")

        scope_section = ""
        if task.patch_scope_paths:
            scope_section = (
                f"\nAllowed modification paths (patch scope): {task.patch_scope_paths}\n"
                "Only changes to these files will be evaluated. Changes outside scope will be rejected.\n"
            )

        public_tests_section = ""
        if task.public_tests and not self.hide_public_tests:
            public_tests_section = (
                f"\nPublic tests you can run during repair: {task.public_tests}\n"
                "These are visible sanity tests. Final scoring uses additional hidden tests.\n"
            )

        return (
            "You are solving an op_bench task. The target repository is not in your current directory, "
            "and you must not try to locate it on the host filesystem.\n"
            f"Use only ./{action_cli_name} as the action interface for the target repository. "
            "This CLI can read files, write files, apply patches, run commands in the task runtime, "
            "run benchmark tests, and show the final git diff. Do not call docker directly and do not "
            "commit changes.\n\n"
            "Prioritize a minimal production source fix. Do not create or modify tests unless source inspection alone is impossible; "
            "the benchmark will apply hidden tests during scoring.\n\n"
            "For PyTorch Python probes, prefer run_test. If you use run_command with python -c, run it from /tmp "
            "inside a shell command so Python does not import the unbuilt source checkout from the repository root.\n\n"
            f"{scope_section}"
            f"{public_tests_section}"
            "Action CLI examples:\n"
            f"  ./{action_cli_name} read_file torch/nn/modules/linear.py\n"
            f"  ./{action_cli_name} run_test 'TestLazyModules.test_linear_state'\n"
            f"  ./{action_cli_name} run_command 'cd /tmp && python - <<\"PY\"\\nimport torch\\nprint(torch.__version__)\\nPY'\n"
            f"  ./{action_cli_name} apply_patch <<'PATCH'\n"
            "  diff --git a/path.py b/path.py\n"
            "  ...\n"
            "  PATCH\n"
            f"  ./{action_cli_name} git_diff\n\n"
            f"Task id: {task.task_id}\n"
            "Hidden fail-to-pass tests are not visible in your repair workspace. "
            "Do not try to run hidden benchmark test names directly; use the issue text and source inspection to repair the behavior.\n"
            f"Allowed test command templates: {task.data.get('agent_visible', {}).get('allowed_test_commands', [])}\n"
            f"Known constraints: {task.data.get('agent_visible', {}).get('known_constraints', [])}\n\n"
            f"Issue:\n{issue_text or task.data['statement']['body']}\n"
        )


class ClaudeCodeActionBridgeAgent:
    name = "claude_code_action_bridge"
    requires_workspace = True
    requires_actions = True
    runtime_boundary = "op_bench_action_interface_file_cli"

    def __init__(self, progress: Progress | None = None, hide_public_tests: bool = False) -> None:
        self.progress = progress or noop_progress
        self.hide_public_tests = hide_public_tests

    def produce_patch(
        self,
        task: TaskManifest,
        output_dir: Path,
        workspace: Path | None = None,
        actions: WorkspaceActions | None = None,
    ) -> AgentOutput:
        if workspace is None or actions is None:
            raise AgentRuntimeUnsupported("ClaudeCodeActionBridgeAgent requires a prepared workspace and WorkspaceActions")
        output_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir = output_dir / "claude_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        last_message_path = output_dir / f"{task.task_id}__{self.name}_last_message.txt"
        action_log_path = output_dir / f"{task.task_id}__{self.name}_actions.jsonl"
        initial_digest = actions.workspace_state_digest()
        start = time.monotonic()
        exchange_dir = scratch_dir / ".opbench_action_exchange"
        with ActionBridgeServer(
            actions=actions,
            log_path=action_log_path,
            exchange_dir=exchange_dir,
            progress=self.progress,
        ) as bridge:
            action_cli = scratch_dir / "opbench_action.py"
            action_cli.write_text(build_action_cli(bridge.exchange_dir, sys.executable), encoding="utf-8")
            action_cli.chmod(0o700)
            prompt = self._build_bridge_prompt(task, action_cli.name)
            completed, timed_out = _run_claude(
                ["claude", "--print", "--dangerously-skip-permissions", "-p", prompt],
                scratch_dir,
            )
            if not timed_out:
                last_message_path.write_text(completed.stdout, encoding="utf-8")
        patch_path = output_dir / f"{task.task_id}__{self.name}.patch"
        patch_path.write_text(actions.git_diff(), encoding="utf-8")
        action_count = action_log_count(action_log_path)
        integrity = action_log_integrity(action_log_path, initial_digest, actions.workspace_state_digest())
        return AgentOutput(
            agent_name=self.name,
            patch_path=patch_path,
            metadata={
                "command": "claude --print",
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "timed_out": timed_out,
                "timeout_sec": _claude_timeout_sec(),
                "duration_sec": time.monotonic() - start,
                "last_message_path": str(last_message_path),
                "action_log_path": str(action_log_path),
                "runtime_boundary": self.runtime_boundary,
                "shell_boundary": "workspace_write_scratch_only",
                "action_count": action_count,
                "integrity_status": integrity["status"],
                "integrity_errors": integrity["errors"],
            },
        )

    def _build_bridge_prompt(self, task: TaskManifest, action_cli_name: str) -> str:
        issue_text = ""
        if task.issue_markdown_path.exists():
            issue_text = task.issue_markdown_path.read_text(encoding="utf-8")

        scope_section = ""
        if task.patch_scope_paths:
            scope_section = (
                f"\nAllowed modification paths (patch scope): {task.patch_scope_paths}\n"
                "Only changes to these files will be evaluated. Changes outside scope will be rejected.\n"
            )

        public_tests_section = ""
        if task.public_tests and not self.hide_public_tests:
            public_tests_section = (
                f"\nPublic tests you can run during repair: {task.public_tests}\n"
                "These are visible sanity tests. Final scoring uses additional hidden tests.\n"
            )

        return (
            "You are solving an op_bench task. The target repository is not in your current directory, "
            "and you must not try to locate it on the host filesystem.\n"
            f"Use only ./{action_cli_name} as the action interface for the target repository. "
            "This CLI can read files, write files, apply patches, run commands in the task runtime, "
            "run benchmark tests, and show the final git diff. Do not call docker directly and do not "
            "commit changes.\n\n"
            "Prioritize a minimal production source fix. Do not create or modify tests unless source inspection alone is impossible; "
            "the benchmark will apply hidden tests during scoring.\n\n"
            "For PyTorch Python probes, prefer run_test. If you use run_command with python -c, run it from /tmp "
            "inside a shell command so Python does not import the unbuilt source checkout from the repository root.\n\n"
            f"{scope_section}"
            f"{public_tests_section}"
            "Action CLI examples:\n"
            f"  ./{action_cli_name} read_file torch/nn/modules/linear.py\n"
            f"  ./{action_cli_name} run_test 'TestLazyModules.test_linear_state'\n"
            f"  ./{action_cli_name} run_command 'cd /tmp && python - <<\"PY\"\\nimport torch\\nprint(torch.__version__)\\nPY'\n"
            f"  ./{action_cli_name} apply_patch <<'PATCH'\n"
            "  diff --git a/path.py b/path.py\n"
            "  ...\n"
            "  PATCH\n"
            f"  ./{action_cli_name} git_diff\n\n"
            f"Task id: {task.task_id}\n"
            "Hidden fail-to-pass tests are not visible in your repair workspace. "
            "Do not try to run hidden benchmark test names directly; use the issue text and source inspection to repair the behavior.\n"
            f"Allowed test command templates: {task.data.get('agent_visible', {}).get('allowed_test_commands', [])}\n"
            f"Known constraints: {task.data.get('agent_visible', {}).get('known_constraints', [])}\n\n"
            f"Issue:\n{issue_text or task.data['statement']['body']}\n"
        )


def agent_by_name(
    name: str,
    progress: Progress | None = None,
    hide_public_tests: bool = False,
) -> GoldAgent | CodexActionBridgeAgent | ClaudeCodeActionBridgeAgent:
    if name == "gold":
        return GoldAgent()
    if name == "codex_action_bridge":
        return CodexActionBridgeAgent(progress=progress, hide_public_tests=hide_public_tests)
    if name == "claude_code_action_bridge":
        return ClaudeCodeActionBridgeAgent(progress=progress, hide_public_tests=hide_public_tests)
    raise ValueError(f"unknown agent: {name}")
