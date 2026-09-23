"""Dataset paths are relative to the bundle, independent of Git and cwd."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from op_bench.io import read_json
from op_bench.data.task import TaskSpec


@dataclass(frozen=True)
class Dataset:
    path: Path
    dataset_id: str
    version: str
    tasks: tuple[TaskSpec, ...]
    release_status: str | None = None
    metadata: dict | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Dataset":
        path = Path(path).resolve()
        if path.is_dir():
            path = path / "dataset.json"
        data = read_json(path)
        if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
                or data["schema_version"] != 1):
            raise ValueError("expected an OpBench dataset with schema_version=1; legacy bundles need explicit migration")
        for field in ("dataset_id", "version"):
            if not isinstance(data.get(field), str) or not data[field].strip():
                raise ValueError(f"dataset.{field}: expected a nonempty string")
        entries = data.get("tasks")
        if not isinstance(entries, list) or not entries:
            raise ValueError("dataset.tasks: expected a nonempty list of relative task.json paths")
        tasks = []
        for entry in entries:
            if not isinstance(entry, str) or Path(entry).is_absolute():
                raise ValueError("dataset task paths must be relative strings")
            selected = (path.parent / entry).resolve()
            if not selected.is_relative_to(path.parent):
                raise ValueError(f"dataset task escapes bundle: {entry}")
            tasks.append(TaskSpec.load(selected))
        identifiers = [task.task_id for task in tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("dataset contains duplicate task_id values")
        status = data.get("status")
        if status is not None and (not isinstance(status, str) or not status.strip()):
            raise ValueError("dataset.status: expected a nonempty string")
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("dataset.metadata: expected an object")
        return cls(path, data["dataset_id"], data["version"], tuple(tasks), status, metadata)

    def identity_dict(self) -> dict:
        """Record dataset provenance; data admission never changes the score."""
        identity = {"dataset_id": self.dataset_id, "dataset_version": self.version}
        if self.release_status is not None:
            identity["release_status"] = self.release_status
        return identity

    def select(self, task_ids: list[str] | None = None) -> tuple[TaskSpec, ...]:
        if task_ids is None:
            return self.tasks
        if not isinstance(task_ids, list) or any(not isinstance(value, str) or not value.strip() for value in task_ids):
            raise ValueError("task selection must be a list of nonempty task IDs")
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("duplicate task selection")
        by_id = {task.task_id: task for task in self.tasks}
        missing = sorted(set(task_ids) - by_id.keys())
        if missing:
            raise ValueError(f"unknown tasks: {', '.join(missing)}")
        return tuple(by_id[task_id] for task_id in task_ids)
