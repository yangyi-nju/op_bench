#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.task import TaskManifest
from scripts.validate_task import validate_manifest


ALLOWED_DATASET_STATUSES = {"draft", "verified"}
ALLOWED_TASK_STATUSES = {
    "draft",
    "candidate",
    "environment_ready",
    "source_ready",
    "baseline_reproduced",
    "gold_verified",
    "verified",
    "blocked",
    "deprecated",
}
ALLOWED_ENVIRONMENT_STATUSES = {"pending", "ready", "unavailable"}
ALLOWED_SOURCE_STATUSES = {"pending", "ready", "unavailable"}
ALLOWED_REPLAY_STATUSES = {"pending", "verified", "failed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an op_bench dataset manifest.")
    parser.add_argument("dataset", help="Path to dataset.json")
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Fail unless the dataset and every task entry are marked verified.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print(f"dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    errors = validate_dataset(data, dataset_path.parent, require_verified=args.require_verified)
    if errors:
        print(f"{dataset_path}: invalid dataset")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"{dataset_path}: dataset looks valid ({len(data.get('tasks', []))} tasks)")
    return 0


def validate_dataset(data: dict[str, Any], dataset_dir: Path, require_verified: bool = False) -> list[str]:
    errors: list[str] = []
    for field in ("dataset_id", "version", "status", "tasks"):
        if data.get(field) in (None, "", []):
            errors.append(f"missing or empty required field: {field}")

    status = data.get("status")
    if status not in ALLOWED_DATASET_STATUSES:
        errors.append(f"invalid dataset status: {status!r}")
    if require_verified and status != "verified":
        errors.append("dataset.status must be 'verified' when --require-verified is used")

    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        return errors

    seen_task_ids: set[str] = set()
    for index, entry in enumerate(tasks):
        if not isinstance(entry, dict):
            errors.append(f"tasks[{index}] must be an object")
            continue
        task_id = str(entry.get("task_id", ""))
        if not task_id:
            errors.append(f"tasks[{index}].task_id is required")
            continue
        if task_id in seen_task_ids:
            errors.append(f"duplicate task_id: {task_id}")
        seen_task_ids.add(task_id)

        task_path_value = entry.get("task_path")
        if not task_path_value:
            errors.append(f"{task_id}: task_path is required")
            continue
        task_dir = _resolve_repo_path(dataset_dir, str(task_path_value))
        task_manifest_path = task_dir / "task.json"
        if not task_manifest_path.exists():
            errors.append(f"{task_id}: task manifest not found at {task_manifest_path}")
            continue

        try:
            task = TaskManifest.load(task_manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{task_id}: cannot load task manifest: {exc}")
            continue
        if task.task_id != task_id:
            errors.append(f"{task_id}: manifest task_id mismatch: {task.task_id}")

        for task_error in validate_manifest(task.data):
            errors.append(f"{task_id}: {task_error}")

        admission_status = entry.get("admission_status")
        if admission_status not in ALLOWED_TASK_STATUSES:
            errors.append(f"{task_id}: invalid admission_status: {admission_status!r}")
        if require_verified and admission_status != "verified":
            errors.append(f"{task_id}: admission_status must be 'verified'")

        environment_status = entry.get("environment_status")
        if environment_status not in ALLOWED_ENVIRONMENT_STATUSES:
            errors.append(f"{task_id}: invalid environment_status: {environment_status!r}")
        source_status = entry.get("source_status", "pending")
        if source_status not in ALLOWED_SOURCE_STATUSES:
            errors.append(f"{task_id}: invalid source_status: {source_status!r}")
        replay_status = entry.get("replay_status")
        if replay_status not in ALLOWED_REPLAY_STATUSES:
            errors.append(f"{task_id}: invalid replay_status: {replay_status!r}")
        if admission_status == "verified" and replay_status != "verified":
            errors.append(f"{task_id}: verified tasks must have replay_status='verified'")

    return errors


def _resolve_repo_path(dataset_dir: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    repo_relative = (ROOT / path).resolve()
    if repo_relative.exists():
        return repo_relative
    return (dataset_dir / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
