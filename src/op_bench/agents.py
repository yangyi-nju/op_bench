from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from op_bench.task import TaskManifest


@dataclass(frozen=True)
class AgentOutput:
    agent_name: str
    patch_path: Path
    metadata: dict[str, object]


class NoopAgent:
    name = "noop"

    def produce_patch(self, task: TaskManifest, output_dir: Path, workspace: Path | None = None) -> AgentOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_path = output_dir / f"{task.task_id}__noop.patch"
        patch_path.write_text("", encoding="utf-8")
        return AgentOutput(agent_name=self.name, patch_path=patch_path, metadata={})


class GoldAgent:
    name = "gold"

    def produce_patch(self, task: TaskManifest, output_dir: Path, workspace: Path | None = None) -> AgentOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_path = output_dir / f"{task.task_id}__gold.patch"
        shutil.copyfile(task.gold_patch_path, patch_path)
        return AgentOutput(agent_name=self.name, patch_path=patch_path, metadata={"source": "gold_patch"})


class CodexAgent:
    name = "codex"

    def produce_patch(self, task: TaskManifest, output_dir: Path, workspace: Path | None = None) -> AgentOutput:
        if workspace is None:
            raise ValueError("CodexAgent requires a prepared workspace")
        output_dir.mkdir(parents=True, exist_ok=True)
        last_message_path = output_dir / f"{task.task_id}__codex_last_message.txt"
        prompt = self._build_prompt(task)
        start = time.monotonic()
        completed = subprocess.run(
            [
                "codex",
                "exec",
                "--cd",
                str(workspace),
                "--sandbox",
                "workspace-write",
                "--ask-for-approval",
                "never",
                "--output-last-message",
                str(last_message_path),
                prompt,
            ],
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        diff = subprocess.run(
            ["git", "diff", "--binary"],
            cwd=str(workspace),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        patch_path = output_dir / f"{task.task_id}__codex.patch"
        patch_path.write_text(diff.stdout, encoding="utf-8")
        return AgentOutput(
            agent_name=self.name,
            patch_path=patch_path,
            metadata={
                "command": "codex exec",
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "duration_sec": time.monotonic() - start,
                "last_message_path": str(last_message_path),
                "diff_exit_code": diff.returncode,
                "diff_stderr": diff.stderr,
            },
        )

    def _build_prompt(self, task: TaskManifest) -> str:
        issue_text = ""
        if task.issue_markdown_path.exists():
            issue_text = task.issue_markdown_path.read_text(encoding="utf-8")
        return (
            "You are solving an op_bench task in this repository checkout.\n"
            "Edit the source code to fix the issue. Do not commit changes. "
            "Keep changes minimal and leave tests runnable by the benchmark.\n\n"
            f"Task id: {task.task_id}\n\n"
            f"Issue:\n{issue_text or task.data['statement']['body']}\n"
        )


def agent_by_name(name: str) -> NoopAgent | GoldAgent | CodexAgent:
    if name == "noop":
        return NoopAgent()
    if name == "gold":
        return GoldAgent()
    if name == "codex":
        return CodexAgent()
    raise ValueError(f"unknown agent: {name}")
