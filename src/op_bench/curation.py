from __future__ import annotations

from copy import deepcopy
from typing import Any


def curate_dataset(
    source: dict[str, Any],
    *,
    verified_only: bool,
    dataset_id: str | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    curated = deepcopy(source)
    tasks = list(curated.get("tasks", []))
    if verified_only:
        tasks = [entry for entry in tasks if entry.get("admission_status") == "verified"]
    curated["tasks"] = tasks
    if dataset_id:
        curated["dataset_id"] = dataset_id
    if version:
        curated["version"] = version
    curated["status"] = (
        "verified"
        if tasks and all(entry.get("admission_status") == "verified" for entry in tasks)
        else "draft"
    )
    return curated


def summarize_dataset(data: dict[str, Any]) -> dict[str, object]:
    tasks = [entry for entry in data.get("tasks", []) if isinstance(entry, dict)]
    return {
        "dataset_id": data.get("dataset_id"),
        "version": data.get("version"),
        "status": data.get("status"),
        "task_count": len(tasks),
        "admission_status": _counts(tasks, "admission_status"),
        "environment_status": _counts(tasks, "environment_status"),
        "source_status": _counts(tasks, "source_status"),
        "replay_status": _counts(tasks, "replay_status"),
        "runtime_tier": _counts(tasks, "runtime_tier"),
    }


def _counts(entries: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(entry.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
