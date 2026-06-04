from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from op_bench.task import TaskManifest


@dataclass(frozen=True)
class DatasetTaskEntry:
    dataset_dir: Path
    data: dict[str, Any]

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def task_path(self) -> Path:
        path = Path(str(self.data["task_path"]))
        if path.is_absolute():
            return path
        repo_root = self._repo_root(self.dataset_dir)
        repo_relative = (repo_root / path).resolve()
        if repo_relative.exists():
            return repo_relative
        return (self.dataset_dir / path).resolve()

    @property
    def admission_status(self) -> str:
        return str(self.data.get("admission_status", "draft"))

    @property
    def environment_status(self) -> str:
        return str(self.data.get("environment_status", "pending"))

    @property
    def source_status(self) -> str:
        return str(self.data.get("source_status", "pending"))

    @property
    def replay_status(self) -> str:
        return str(self.data.get("replay_status", "pending"))

    @property
    def admission_evidence_path(self) -> Path | None:
        value = self.data.get("admission_evidence")
        if not value:
            return None
        path = Path(str(value))
        if path.is_absolute():
            return path
        repo_root = self._repo_root(self.dataset_dir)
        repo_relative = (repo_root / path).resolve()
        if repo_relative.exists():
            return repo_relative
        return (self.dataset_dir / path).resolve()

    def load_task(self) -> TaskManifest:
        return TaskManifest.load(self.task_path / "task.json")

    def _repo_root(self, start: Path) -> Path:
        for path in [start.resolve(), *start.resolve().parents]:
            if (path / ".git").exists():
                return path
        return Path.cwd().resolve()


@dataclass(frozen=True)
class DatasetManifest:
    dataset_dir: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        manifest_path = Path(path).resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(dataset_dir=manifest_path.parent, data=data)

    @property
    def dataset_id(self) -> str:
        return str(self.data["dataset_id"])

    @property
    def version(self) -> str:
        return str(self.data["version"])

    @property
    def status(self) -> str:
        return str(self.data.get("status", "draft"))

    @property
    def tasks(self) -> list[DatasetTaskEntry]:
        return [DatasetTaskEntry(self.dataset_dir, entry) for entry in self.data.get("tasks", [])]

    @property
    def registries(self) -> dict[str, str]:
        value = self.data.get("registries", {})
        if not isinstance(value, dict):
            return {}
        return {str(key): str(path) for key, path in value.items()}

    def load_tasks(self, verified_only: bool = False) -> list[TaskManifest]:
        entries = self.tasks
        if verified_only:
            entries = [entry for entry in entries if entry.admission_status == "verified"]
        return [entry.load_task() for entry in entries]
