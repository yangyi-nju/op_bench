#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
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
from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.registry import EnvironmentRegistry, RegistryError, SourceRegistry
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
    "blocked_environment",
    "blocked_source",
    "blocked_test",
    "not_reproduced",
    "gold_failed",
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
    environment_registry, source_registry = _load_registries(data, dataset_dir, errors)
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
        formal_admission = task.data.get("admission")
        if isinstance(formal_admission, dict) and formal_admission.get("status") != admission_status:
            errors.append(f"{task_id}: dataset admission_status must match task admission.status")

        _validate_asset_references(task, task_id, environment_registry, source_registry, errors)
        _validate_admission_evidence(entry, task, task_id, admission_status, dataset_dir, errors)

    return errors


def _load_registries(
    data: dict[str, Any],
    dataset_dir: Path,
    errors: list[str],
) -> tuple[EnvironmentRegistry | None, SourceRegistry | None]:
    registries = data.get("registries", {})
    if registries is None:
        return None, None
    if not isinstance(registries, dict):
        errors.append("registries must be an object when provided")
        return None, None
    environment_registry = _load_registry(
        EnvironmentRegistry,
        registries.get("environments"),
        "environment",
        dataset_dir,
        errors,
    )
    source_registry = _load_registry(
        SourceRegistry,
        registries.get("sources"),
        "source",
        dataset_dir,
        errors,
    )
    return environment_registry, source_registry


def _load_registry(
    registry_type: type[EnvironmentRegistry] | type[SourceRegistry],
    path_value: object,
    label: str,
    dataset_dir: Path,
    errors: list[str],
) -> EnvironmentRegistry | SourceRegistry | None:
    if not path_value:
        return None
    path = _resolve_repo_path(dataset_dir, str(path_value))
    try:
        return registry_type.load(path)
    except RegistryError as exc:
        errors.append(f"cannot load {label} registry: {exc}")
        return None


def _validate_asset_references(
    task: TaskManifest,
    task_id: str,
    environment_registry: EnvironmentRegistry | None,
    source_registry: SourceRegistry | None,
    errors: list[str],
) -> None:
    if task.environment_ref:
        if environment_registry is None:
            errors.append(f"{task_id}: environment_ref requires a dataset environment registry")
        else:
            try:
                environment_asset = environment_registry.get(task.environment_ref)
                if task.runtime_tier != environment_asset.runtime_tier:
                    errors.append(f"{task_id}: task runtime_tier must match environment asset runtime_tier")
                if (
                    task.source_loading_mode
                    and task.source_loading_mode not in environment_asset.source_loading_modes
                ):
                    errors.append(f"{task_id}: task source_loading mode is not supported by environment asset")
            except RegistryError as exc:
                errors.append(f"{task_id}: {exc}")
    if task.source_ref:
        if source_registry is None:
            errors.append(f"{task_id}: source_ref requires a dataset source registry")
        else:
            try:
                source_asset = source_registry.get(task.source_ref)
                if task.base_commit != source_asset.commit:
                    errors.append(f"{task_id}: task base_commit must match source asset commit")
                if task.source_loading_mode and task.source_loading_mode not in source_asset.source_loading_modes:
                    errors.append(f"{task_id}: task source_loading mode is not supported by source asset")
            except RegistryError as exc:
                errors.append(f"{task_id}: {exc}")


def _validate_admission_evidence(
    entry: dict[str, Any],
    task: TaskManifest,
    task_id: str,
    admission_status: object,
    dataset_dir: Path,
    errors: list[str],
) -> None:
    evidence_value = entry.get("admission_evidence")
    if not evidence_value:
        if admission_status == "verified":
            errors.append(f"{task_id}: admission_evidence is required for verified tasks")
        return
    evidence_path = _resolve_repo_path(dataset_dir, str(evidence_value))
    if not evidence_path.exists():
        errors.append(f"{task_id}: admission evidence not found at {evidence_path}")
        return
    try:
        with evidence_path.open("r", encoding="utf-8") as handle:
            evidence = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{task_id}: cannot load admission evidence: {exc}")
        return
    if evidence.get("task_id") != task_id:
        errors.append(f"{task_id}: admission evidence task_id mismatch: {evidence.get('task_id')}")
    hash_kind = evidence.get("task_manifest_hash_kind")
    if hash_kind == REPLAY_SPEC_HASH_KIND:
        expected_hash = replay_spec_hash(task)
    elif hash_kind is None:
        expected_hash = f"sha256:{hashlib.sha256(task.task_json_path.read_bytes()).hexdigest()}"
    else:
        expected_hash = None
        errors.append(f"{task_id}: unsupported admission evidence hash kind: {hash_kind!r}")
    if expected_hash is not None and evidence.get("task_manifest_hash") != expected_hash:
        errors.append(f"{task_id}: admission evidence replay hash does not match current task bundle")

    admission = evidence.get("admission", {})
    if admission_status == "verified":
        if not isinstance(admission, dict) or admission.get("decision") != "verified":
            errors.append(f"{task_id}: verified admission evidence must have decision='verified'")
        if not isinstance(admission, dict) or admission.get("verified") is not True:
            errors.append(f"{task_id}: verified admission evidence must have verified=true")
        if evidence.get("baseline", {}).get("status") != "baseline_reproduced":
            errors.append(f"{task_id}: verified admission evidence baseline must be 'baseline_reproduced'")
        gold = evidence.get("gold")
        if not isinstance(gold, dict) or gold.get("status") != "resolved":
            errors.append(f"{task_id}: verified admission evidence gold must be 'resolved'")

    environment = evidence.get("environment", {})
    if task.environment_ref and (
        not isinstance(environment, dict) or environment.get("id") != task.environment_ref
    ):
        errors.append(f"{task_id}: admission evidence environment id must match task environment_ref")
    source = evidence.get("source", {})
    if task.source_ref and (not isinstance(source, dict) or source.get("id") != task.source_ref):
        errors.append(f"{task_id}: admission evidence source id must match task source_ref")


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
