from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from op_bench.runtime.validation import ContractError, require_str


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


def summarize_verified_dataset(
    data: dict[str, Any],
    *,
    dataset_hash: str,
    task_metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    if data.get("status") != "verified":
        raise ContractError("dataset.status: expected 'verified'")
    normalized_hash = require_str(
        dataset_hash,
        "dataset_hash",
        pattern=r"sha256:[0-9a-f]{64}",
    )
    if not isinstance(task_metadata, Mapping):
        raise ContractError("task_metadata: expected object")
    tasks = [
        entry
        for entry in data.get("tasks", [])
        if isinstance(entry, dict)
    ]
    if not tasks or any(
        entry.get("admission_status") != "verified"
        for entry in tasks
    ):
        raise ContractError(
            "dataset.tasks: expected non-empty verified entries"
        )
    dimensions: list[str] = []
    subclasses: list[str] = []
    failure_contracts: list[str] = []
    for entry in tasks:
        task_id = require_str(entry.get("task_id"), "task_id")
        metadata = task_metadata.get(task_id, {})
        if not isinstance(metadata, Mapping):
            raise ContractError(
                f"task_metadata.{task_id}: expected object"
            )
        dimensions.append(
            _metadata_value(
                metadata,
                "problem_dimension",
            )
        )
        subclasses.append(
            _metadata_value(
                metadata,
                "problem_subclass",
            )
        )
        failure_contracts.append(
            _metadata_value(
                metadata,
                "failure_contract",
            )
        )
    return {
        **summarize_dataset(data),
        "dataset_hash": normalized_hash,
        "problem_dimension": _value_counts(dimensions),
        "problem_subclass": _value_counts(subclasses),
        "failure_contract": _value_counts(failure_contracts),
        "verified_admission_evidence": sum(
            bool(entry.get("admission_evidence"))
            for entry in tasks
        ),
    }


def _metadata_value(
    metadata: Mapping[str, object],
    field: str,
) -> str:
    value = metadata.get(field, "unclassified")
    if not isinstance(value, str) or not value:
        raise ContractError(
            f"task_metadata.{field}: expected non-empty string"
        )
    return value


def _value_counts(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _counts(entries: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(entry.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
