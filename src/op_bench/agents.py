from __future__ import annotations

import shutil
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

    def produce_patch(self, task: TaskManifest, output_dir: Path) -> AgentOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_path = output_dir / f"{task.task_id}__noop.patch"
        patch_path.write_text("", encoding="utf-8")
        return AgentOutput(agent_name=self.name, patch_path=patch_path, metadata={})


class GoldAgent:
    name = "gold"

    def produce_patch(self, task: TaskManifest, output_dir: Path) -> AgentOutput:
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_path = output_dir / f"{task.task_id}__gold.patch"
        shutil.copyfile(task.gold_patch_path, patch_path)
        return AgentOutput(agent_name=self.name, patch_path=patch_path, metadata={"source": "gold_patch"})


def agent_by_name(name: str) -> NoopAgent | GoldAgent:
    if name == "noop":
        return NoopAgent()
    if name == "gold":
        return GoldAgent()
    raise ValueError(f"unknown agent: {name}")
