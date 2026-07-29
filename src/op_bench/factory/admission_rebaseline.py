"""Fail-closed migration of one legacy Admission contract to replay_spec_v1."""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest


_REPLAY_FIELDS = (
    "task_id",
    "version",
    "environment_ref",
    "runtime_tier",
    "source_ref",
    "compatibility",
    "patch_scope",
    "source",
    "environment",
    "evaluation",
    "artifacts",
)
_EXECUTION_FIELDS = (
    "task_id",
    "mode",
    "status",
    "fail_to_pass_total",
    "fail_to_pass_passed",
    "pass_to_pass_total",
    "pass_to_pass_passed",
    "duration_sec",
)


def migrated_admission_payload(
    *,
    task: TaskManifest,
    baseline_admission_bytes: bytes,
) -> dict[str, object]:
    """Return the exact legacy evidence with only its hash contract migrated."""

    baseline = _json_object(baseline_admission_bytes, "baseline admission")
    if baseline.get("task_id") != task.task_id:
        raise ContractError("baseline admission: task_id mismatch")
    selected = copy.deepcopy(baseline)
    selected_hash = replay_spec_hash(task)
    created_at = selected.get("created_at")
    if not isinstance(created_at, str):
        raise ContractError("baseline admission: created_at must be a string")
    selected["task_manifest_hash_kind"] = REPLAY_SPEC_HASH_KIND
    selected["task_manifest_hash"] = selected_hash
    selected["evidence_id"] = (
        f"{task.task_id}:{selected_hash.removeprefix('sha256:')[:12]}:"
        f"{created_at}"
    )
    return selected


def build_admission_contract_rebaseline(
    *,
    task: TaskManifest,
    baseline_commit: str,
    baseline_manifest_bytes: bytes,
    baseline_admission_bytes: bytes,
    baseline_artifact_bytes: Mapping[str, bytes],
) -> dict[str, object]:
    """Prove replay inputs and historical outcomes survived a metadata rewrite."""

    if not isinstance(task, TaskManifest):
        raise ContractError("task: expected TaskManifest")
    if baseline_commit != "4f5addc":
        raise ContractError("baseline_commit: expected frozen 4f5addc baseline")
    baseline_manifest = _json_object(
        baseline_manifest_bytes,
        "baseline manifest",
    )
    current_manifest = _json_object(
        task.task_json_path.read_bytes(),
        "current manifest",
    )
    if baseline_manifest.get("task_id") != task.task_id:
        raise ContractError("baseline manifest: task_id mismatch")

    replay_fields: dict[str, dict[str, object]] = {}
    for field in _REPLAY_FIELDS:
        baseline_value = baseline_manifest.get(field)
        current_value = current_manifest.get(field)
        unchanged = baseline_value == current_value
        replay_fields[field] = {
            "baseline_hash": canonical_sha256(baseline_value),
            "current_hash": canonical_sha256(current_value),
            "unchanged": unchanged,
        }
        if not unchanged:
            raise ContractError(f"replay-spec field drift: {field}")

    baseline_admission = _json_object(
        baseline_admission_bytes,
        "baseline admission",
    )
    if baseline_admission.get("task_id") != task.task_id:
        raise ContractError("baseline admission: task_id mismatch")
    baseline_manifest_hash = _bytes_hash(baseline_manifest_bytes)
    if baseline_admission.get("task_manifest_hash") != baseline_manifest_hash:
        raise ContractError(
            "baseline admission: legacy task_manifest_hash does not bind exact bytes"
        )

    artifact_config = baseline_manifest.get("artifacts")
    current_artifact_config = current_manifest.get("artifacts")
    if not isinstance(artifact_config, Mapping) or not isinstance(
        current_artifact_config,
        Mapping,
    ):
        raise ContractError("manifest artifacts: expected objects")
    expected_artifact_names = tuple(sorted(artifact_config))
    if tuple(sorted(baseline_artifact_bytes)) != expected_artifact_names:
        raise ContractError(
            "baseline artifacts: expected exact configured private artifact set"
        )
    artifacts: dict[str, dict[str, object]] = {}
    for name in expected_artifact_names:
        baseline_path = _safe_relative_path(
            artifact_config[name],
            f"baseline artifacts.{name}",
        )
        current_path = _safe_relative_path(
            current_artifact_config.get(name),
            f"current artifacts.{name}",
        )
        if baseline_path != current_path:
            raise ContractError(f"private artifact path drift: {name}")
        current_bytes = _regular_file_bytes(task.task_dir / baseline_path)
        historical_bytes = baseline_artifact_bytes[name]
        unchanged = historical_bytes == current_bytes
        artifacts[name] = {
            "path": baseline_path.as_posix(),
            "baseline_bytes_sha256": _bytes_hash(historical_bytes),
            "current_bytes_sha256": _bytes_hash(current_bytes),
            "unchanged": unchanged,
        }
        if not unchanged:
            raise ContractError(f"private artifact byte drift: {name}")

    compatibility = _compatibility_proof(
        task=task,
        baseline_manifest=baseline_manifest,
        current_manifest=current_manifest,
        baseline_artifact_bytes=baseline_artifact_bytes,
    )

    runtime_outcomes: dict[str, dict[str, object]] = {}
    for phase in ("baseline", "gold"):
        value = baseline_admission.get(phase)
        if not isinstance(value, Mapping):
            raise ContractError(f"baseline admission: {phase} must be an object")
        selected = {field: value.get(field) for field in _EXECUTION_FIELDS}
        expected_status = (
            "baseline_reproduced" if phase == "baseline" else "resolved"
        )
        if (
            selected["task_id"] != task.task_id
            or selected["mode"] != phase
            or selected["status"] != expected_status
        ):
            raise ContractError(f"baseline admission: {phase} outcome drift")
        runtime_outcomes[phase] = {
            "record_hash": _json_hash(selected),
            "status": selected["status"],
            "fail_to_pass_total": selected["fail_to_pass_total"],
            "fail_to_pass_passed": selected["fail_to_pass_passed"],
            "pass_to_pass_total": selected["pass_to_pass_total"],
            "pass_to_pass_passed": selected["pass_to_pass_passed"],
        }

    migrated = migrated_admission_payload(
        task=task,
        baseline_admission_bytes=baseline_admission_bytes,
    )
    payload: dict[str, object] = {
        "contract_type": "admission_contract_rebaseline",
        "schema_version": "v1",
        "task_id": task.task_id,
        "baseline_commit": baseline_commit,
        "baseline_manifest": {
            "relative_path": f"{task.task_dir.name}/task.json",
            "bytes_sha256": baseline_manifest_hash,
        },
        "old_admission_evidence": {
            "relative_path": "admission/evidence.json",
            "bytes_sha256": _bytes_hash(baseline_admission_bytes),
            "task_manifest_hash": baseline_admission["task_manifest_hash"],
            "task_manifest_hash_kind": "legacy_task_json_bytes",
        },
        "replay_spec_hash_kind": REPLAY_SPEC_HASH_KIND,
        "replay_spec_hash": replay_spec_hash(task),
        "expected_migrated_admission_evidence_hash": _bytes_hash(
            _canonical_bytes(migrated)
        ),
        "replay_spec_fields": replay_fields,
        "private_artifacts": artifacts,
        "compatibility_content": compatibility,
        "historical_runtime_outcomes": runtime_outcomes,
        "proofs": {
            "replay_spec_fields_unchanged": True,
            "private_artifact_bytes_unchanged": True,
            "compatibility_content_unchanged": True,
            "historical_runtime_outcomes_unchanged": True,
        },
        "created_at": "2026-07-29T00:00:00Z",
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _compatibility_proof(
    *,
    task: TaskManifest,
    baseline_manifest: Mapping[str, object],
    current_manifest: Mapping[str, object],
    baseline_artifact_bytes: Mapping[str, bytes],
) -> dict[str, object]:
    baseline = baseline_manifest.get("compatibility")
    current = current_manifest.get("compatibility")
    if baseline != current:
        raise ContractError("compatibility content drift")
    if not isinstance(baseline, Mapping) or "evidence" not in baseline:
        return {
            "configured": False,
            "baseline_bytes_sha256": None,
            "current_bytes_sha256": None,
            "unchanged": True,
        }
    relative = _safe_relative_path(
        baseline["evidence"],
        "compatibility.evidence",
    )
    key = "compatibility_evidence"
    if key not in baseline_artifact_bytes:
        raise ContractError(
            "baseline artifacts: compatibility evidence bytes are required"
        )
    baseline_bytes = baseline_artifact_bytes[key]
    current_bytes = _regular_file_bytes(task.task_dir / relative)
    if baseline_bytes != current_bytes:
        raise ContractError("compatibility evidence byte drift")
    return {
        "configured": True,
        "relative_path": relative.as_posix(),
        "baseline_bytes_sha256": _bytes_hash(baseline_bytes),
        "current_bytes_sha256": _bytes_hash(current_bytes),
        "unchanged": True,
    }


def _json_object(value: bytes, label: str) -> dict[str, object]:
    if not isinstance(value, bytes):
        raise ContractError(f"{label}: expected bytes")
    try:
        selected = json.loads(value.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: invalid JSON") from exc
    if not isinstance(selected, dict):
        raise ContractError(f"{label}: expected object")
    return selected


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ContractError(f"{label}: expected safe relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ContractError(f"{label}: expected safe relative path")
    return relative


def _regular_file_bytes(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ContractError("private artifact: expected regular file")
    return path.read_bytes()


def _bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _bytes_hash(encoded)


def _canonical_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
