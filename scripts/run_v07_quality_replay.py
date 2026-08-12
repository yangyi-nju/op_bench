#!/usr/bin/env python3
"""Run fresh Baseline/Gold replay for every Task in the final v0.7 Dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.admission import AdmissionRunner  # noqa: E402
from op_bench.environment import EnvironmentManager  # noqa: E402
from op_bench.evaluator import Evaluator  # noqa: E402
from op_bench.factory.artifacts import load_regular_file_bytes  # noqa: E402
from op_bench.factory.quality_admission import (  # noqa: E402
    quality_admission_bundle_hash,
)
from op_bench.integrity import replay_spec_hash  # noqa: E402
from op_bench.progress import ProgressLogger  # noqa: E402
from op_bench.registry import load_resolved_task  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(load_regular_file_bytes(path)).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(public_task_id: str) -> str:
    if re.fullmatch(r"opbench-v07-t[0-9]{4}", public_task_id) is None:
        raise ContractError("public Task ID is not canonical")
    return public_task_id


def _phase_summary(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError("replay phase: expected object")
    return {
        "status": value.get("status"),
        "fail_to_pass_total": value.get("fail_to_pass_total"),
        "fail_to_pass_passed": value.get("fail_to_pass_passed"),
        "pass_to_pass_total": value.get("pass_to_pass_total"),
        "pass_to_pass_passed": value.get("pass_to_pass_passed"),
        "command_count": len(value.get("commands", []))
        if isinstance(value.get("commands"), list)
        else 0,
    }


def _record_from_bundle(
    *,
    task_id: str,
    public_task_id: str,
    task_hash: str,
    source_ref: str,
    environment_ref: str,
    bundle_path: Path,
    attempt_number: int,
) -> dict[str, object]:
    evidence = _load_json(bundle_path / "evidence.json")
    admission = evidence.get("admission")
    baseline = _phase_summary(evidence.get("baseline"))
    gold = _phase_summary(evidence.get("gold"))
    verified = (
        evidence.get("task_id") == task_id
        and evidence.get("task_manifest_hash") == task_hash
        and isinstance(admission, dict)
        and admission.get("decision") == "verified"
        and admission.get("verified") is True
        and baseline is not None
        and baseline["status"] == "baseline_reproduced"
        and gold is not None
        and gold["status"] == "resolved"
    )
    return {
        "task_id": task_id,
        "public_task_id": public_task_id,
        "task_replay_hash": task_hash,
        "source_ref": source_ref,
        "environment_ref": environment_ref,
        "attempt_number": attempt_number,
        "bundle_path": bundle_path.relative_to(ROOT).as_posix(),
        "bundle_hash": quality_admission_bundle_hash(bundle_path),
        "created_at": evidence.get("created_at"),
        "baseline": baseline,
        "gold": gold,
        "decision": admission.get("decision") if isinstance(admission, dict) else None,
        "verified": verified,
        "integrity": "passed" if verified else "failed",
    }


def _failure_record(
    *,
    task_id: str,
    public_task_id: str,
    task_hash: str,
    source_ref: str,
    environment_ref: str,
    attempt_number: int,
    error: Exception,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "public_task_id": public_task_id,
        "task_replay_hash": task_hash,
        "source_ref": source_ref,
        "environment_ref": environment_ref,
        "attempt_number": attempt_number,
        "bundle_path": None,
        "bundle_hash": None,
        "created_at": _now(),
        "baseline": None,
        "gold": None,
        "decision": "infrastructure_invalid",
        "verified": False,
        "integrity": "failed",
        "failure_classification": type(error).__name__,
        "failure_hash": canonical_sha256(
            {"exception_type": type(error).__name__, "message": str(error)}
        ),
    }


def _existing_verified_record(
    task_root: Path,
    *,
    task_id: str,
    public_task_id: str,
    task_hash: str,
    source_ref: str,
    environment_ref: str,
) -> dict[str, object] | None:
    if not task_root.is_dir():
        return None
    for attempt_path in sorted(task_root.glob("attempt-*"), reverse=True):
        if not attempt_path.is_dir() or attempt_path.is_symlink():
            continue
        try:
            attempt_number = int(attempt_path.name.removeprefix("attempt-"))
            record = _record_from_bundle(
                task_id=task_id,
                public_task_id=public_task_id,
                task_hash=task_hash,
                source_ref=source_ref,
                environment_ref=environment_ref,
                bundle_path=attempt_path,
                attempt_number=attempt_number,
            )
        except (ContractError, OSError, ValueError):
            continue
        if record["verified"] is True:
            return record
    return None


def _next_attempt_number(task_root: Path) -> int:
    numbers = []
    if task_root.is_dir():
        for path in task_root.glob("attempt-*" ):
            try:
                numbers.append(int(path.name.removeprefix("attempt-")))
            except ValueError:
                continue
    return max(numbers, default=0) + 1


def _index_payload(
    *,
    dataset_path: Path,
    environment_registry_path: Path,
    source_registry_path: Path,
    created_at: str,
    expected_task_count: int,
    records: dict[str, dict[str, object]],
) -> dict[str, object]:
    ordered = [records[key] for key in sorted(records)]
    payload = {
        "contract_type": "quality_replay_index",
        "schema_version": "v1",
        "release_version": "v0.7",
        "created_at": created_at,
        "dataset_path": dataset_path.relative_to(ROOT).as_posix(),
        "dataset_hash": _file_hash(dataset_path),
        "environment_registry_path": environment_registry_path.relative_to(ROOT).as_posix(),
        "environment_registry_hash": _file_hash(environment_registry_path),
        "source_registry_path": source_registry_path.relative_to(ROOT).as_posix(),
        "source_registry_hash": _file_hash(source_registry_path),
        "task_count": expected_task_count,
        "completed_count": len(ordered),
        "verified_count": sum(record.get("verified") is True for record in ordered),
        "records": ordered,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def validate_quality_replay_index(
    root: Path,
    index_path: Path,
    *,
    release_manifest_path: Path,
) -> list[str]:
    """Validate the exact final replay matrix and rebuild every public record."""

    errors: list[str] = []
    try:
        index = _load_json(index_path)
        release = _load_json(release_manifest_path)
        dataset_ref = release["datasets"]["cumulative"]["dataset"]
        dataset_path = root / str(dataset_ref["relative_path"])
        dataset = _load_json(dataset_path)
        tasks = dataset.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != 50:
            raise ContractError("release Dataset must contain exactly 50 Tasks")
        if _file_hash(dataset_path) != dataset_ref.get("content_hash"):
            raise ContractError("release Dataset hash drift")
        core = dict(index)
        observed_content_hash = core.pop("content_hash", None)
        if observed_content_hash != canonical_sha256(core):
            raise ContractError("replay index content hash drift")
        if (
            index.get("contract_type") != "quality_replay_index"
            or index.get("schema_version") != "v1"
            or index.get("release_version") != "v0.7"
            or index.get("dataset_path") != dataset_path.relative_to(root).as_posix()
            or index.get("dataset_hash") != _file_hash(dataset_path)
            or index.get("task_count") != 50
            or index.get("completed_count") != 50
            or index.get("verified_count") != 50
        ):
            raise ContractError("replay index final counters or Dataset identity mismatch")
        environment_path = root / str(index.get("environment_registry_path"))
        source_path = root / str(index.get("source_registry_path"))
        if (
            index.get("environment_registry_hash") != _file_hash(environment_path)
            or index.get("source_registry_hash") != _file_hash(source_path)
        ):
            raise ContractError("replay registry identity drift")
        expected = {
            str(entry["task_id"]): entry
            for entry in tasks
            if isinstance(entry, dict)
            and isinstance(entry.get("task_id"), str)
        }
        records = index.get("records")
        if not isinstance(records, list) or len(records) != 50:
            raise ContractError("replay index must contain exactly 50 records")
        observed: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ContractError("replay record must be an object")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or task_id not in expected or task_id in observed:
                raise ContractError("replay record Task partition mismatch")
            observed.add(task_id)
            task_entry = expected[task_id]
            task = load_resolved_task(
                root / str(task_entry["task_path"]) / "task.json",
                environment_registry_path=environment_path,
                source_registry_path=source_path,
            )
            public_task_id = task.public_task_id
            if public_task_id is None or record.get("public_task_id") != public_task_id:
                raise ContractError("replay record public Task identity mismatch")
            bundle_value = record.get("bundle_path")
            if not isinstance(bundle_value, str):
                raise ContractError("replay record is missing its evidence bundle")
            rebuilt = _record_from_bundle(
                task_id=task.task_id,
                public_task_id=public_task_id,
                task_hash=replay_spec_hash(task),
                source_ref=task.source_ref,
                environment_ref=task.environment_ref,
                bundle_path=root / bundle_value,
                attempt_number=int(record.get("attempt_number")),
            )
            if rebuilt != record or rebuilt.get("verified") is not True:
                raise ContractError("replay record does not rebuild exactly")
        if observed != set(expected):
            raise ContractError("replay index does not cover the final Dataset")
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/pytorch_v0.7/dataset.json")
    parser.add_argument("--output-dir", default="runs/v0.7_quality_replay")
    parser.add_argument("--environment-registry", default="environments/registry.json")
    parser.add_argument("--source-registry", default="sources/registry.json")
    parser.add_argument("--only-task", action="append")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_path = (ROOT / args.dataset).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    environment_path = (ROOT / args.environment_registry).resolve()
    source_path = (ROOT / args.source_registry).resolve()
    dataset = _load_json(dataset_path)
    tasks = dataset.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 50:
        raise ContractError("quality replay requires the exact 50-Task Dataset")
    selected_ids = set(args.only_task or ())
    selected = [
        entry
        for entry in tasks
        if isinstance(entry, dict)
        and (not selected_ids or str(entry.get("task_id")) in selected_ids)
    ]
    if selected_ids and selected_ids != {
        str(entry.get("task_id")) for entry in selected
    }:
        raise ContractError("--only-task includes an unknown Task")
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            raise ContractError("--max-tasks must be positive")
        selected = selected[: args.max_tasks]

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.json"
    created_at = _now()
    records: dict[str, dict[str, object]] = {}
    if index_path.is_file():
        previous = _load_json(index_path)
        if (
            previous.get("dataset_hash") != _file_hash(dataset_path)
            or previous.get("environment_registry_hash") != _file_hash(environment_path)
            or previous.get("source_registry_hash") != _file_hash(source_path)
        ):
            raise ContractError("existing replay index belongs to different frozen inputs")
        created_at = str(previous.get("created_at"))
        for record in previous.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("task_id"), str):
                records[str(record["task_id"])] = record

    progress = ProgressLogger(enabled=not args.quiet)
    manager = EnvironmentManager(progress=progress)
    evaluator = Evaluator(environment_manager=manager, progress=progress)
    runner = AdmissionRunner(evaluator=evaluator, progress=progress)
    for entry in selected:
        task_path = ROOT / str(entry["task_path"]) / "task.json"
        task = load_resolved_task(
            task_path,
            environment_registry_path=environment_path,
            source_registry_path=source_path,
        )
        if task.public_task_id is None or task.source_ref is None or task.environment_ref is None:
            raise ContractError(f"{task.task_id}: replay identity is incomplete")
        task_hash = replay_spec_hash(task)
        task_root = output_dir / "tasks" / _slug(task.public_task_id)
        existing = _existing_verified_record(
            task_root,
            task_id=task.task_id,
            public_task_id=task.public_task_id,
            task_hash=task_hash,
            source_ref=task.source_ref,
            environment_ref=task.environment_ref,
        )
        if existing is not None:
            records[task.task_id] = existing
        else:
            attempt_number = _next_attempt_number(task_root)
            bundle_path = task_root / f"attempt-{attempt_number:02d}"
            try:
                evidence = runner.run(task)
                runner.write_bundle(evidence, bundle_path)
                records[task.task_id] = _record_from_bundle(
                    task_id=task.task_id,
                    public_task_id=task.public_task_id,
                    task_hash=task_hash,
                    source_ref=task.source_ref,
                    environment_ref=task.environment_ref,
                    bundle_path=bundle_path,
                    attempt_number=attempt_number,
                )
            except Exception as exc:
                records[task.task_id] = _failure_record(
                    task_id=task.task_id,
                    public_task_id=task.public_task_id,
                    task_hash=task_hash,
                    source_ref=task.source_ref,
                    environment_ref=task.environment_ref,
                    attempt_number=attempt_number,
                    error=exc,
                )
        index = _index_payload(
            dataset_path=dataset_path,
            environment_registry_path=environment_path,
            source_registry_path=source_path,
            created_at=created_at,
            expected_task_count=50,
            records=records,
        )
        index_path.write_bytes(canonical_json(index).encode("utf-8"))

    final = _index_payload(
        dataset_path=dataset_path,
        environment_registry_path=environment_path,
        source_registry_path=source_path,
        created_at=created_at,
        expected_task_count=50,
        records=records,
    )
    index_path.write_bytes(canonical_json(final).encode("utf-8"))
    print(
        canonical_json(
            {
                "completed_count": final["completed_count"],
                "index": index_path.relative_to(ROOT).as_posix(),
                "task_count": final["task_count"],
                "verified_count": final["verified_count"],
            }
        )
    )
    return 0 if final["completed_count"] == 50 and final["verified_count"] == 50 else 1


if __name__ == "__main__":
    raise SystemExit(main())
