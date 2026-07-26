from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.matched_runtime.contracts import CompatibilityEvidence
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path}: expected integer")
    return value


def _relative_json_path(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected string")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or relative.suffix != ".json"
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != value
        or "\\" in value
    ):
        raise ContractError(f"{path}: expected normalized task-relative JSON path")
    return value


def _validate_utc_seconds(value: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ContractError("verified_at: expected UTC RFC3339 seconds")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError("verified_at: expected UTC RFC3339 seconds") from exc
    if parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != value:
        raise ContractError("verified_at: expected UTC RFC3339 seconds")


def validate_matched_runtime_promotion(
    task: TaskManifest,
    compatibility: CompatibilityEvidence,
    admission: Mapping[str, object],
) -> None:
    if not isinstance(task, TaskManifest):
        raise ContractError("task: expected TaskManifest")
    if not isinstance(compatibility, CompatibilityEvidence):
        raise ContractError("compatibility: expected CompatibilityEvidence")
    evidence = _mapping(admission, "admission evidence")
    if compatibility.task_id != task.task_id:
        raise ContractError("task_id mismatch: compatibility evidence")
    if evidence.get("task_id") != task.task_id:
        raise ContractError("task_id mismatch: admission evidence")
    if (
        compatibility.source.source_id != task.source_ref
        or compatibility.source.commit != task.base_commit
    ):
        raise ContractError("source commit mismatch: compatibility evidence")
    source = _mapping(evidence.get("source"), "admission source")
    if source.get("id") != task.source_ref or source.get("base_commit") != task.base_commit:
        raise ContractError("source commit mismatch: admission evidence")
    if compatibility.runtime.environment_id != task.environment_ref:
        raise ContractError("environment mismatch: compatibility evidence")
    environment = _mapping(evidence.get("environment"), "admission environment")
    if (
        environment.get("id") != task.environment_ref
        or environment.get("runtime_tier") != task.runtime_tier
    ):
        raise ContractError("environment mismatch: admission evidence")
    if compatibility.status != "compatible":
        raise ContractError("compatibility status: expected compatible")
    admission_decision = _mapping(evidence.get("admission"), "admission decision")
    if (
        admission_decision.get("decision") != "verified"
        or admission_decision.get("verified") is not True
        or admission_decision.get("failure_classification") is not None
    ):
        raise ContractError("admission decision: expected verified")
    if evidence.get("task_manifest_hash_kind") != REPLAY_SPEC_HASH_KIND:
        raise ContractError("replay hash mismatch: unsupported hash kind")
    expected_replay_hash = replay_spec_hash(task)
    if evidence.get("task_manifest_hash") != expected_replay_hash:
        raise ContractError("replay hash mismatch: admission evidence is stale")
    _validate_test_execution(
        _mapping(evidence.get("baseline"), "baseline"),
        baseline=True,
    )
    _validate_test_execution(
        _mapping(evidence.get("gold"), "gold"),
        baseline=False,
    )
    compatibility_config = _mapping(
        task.data.get("compatibility"),
        "task.compatibility",
    )
    if compatibility_config.get("target_module") != compatibility.source.target_module_path:
        raise ContractError("source commit mismatch: target module drift")


def _validate_test_execution(
    result: Mapping[str, object],
    *,
    baseline: bool,
) -> None:
    phase = "baseline" if baseline else "gold"
    expected_status = "baseline_reproduced" if baseline else "resolved"
    if result.get("status") != expected_status:
        raise ContractError(f"admission decision: {phase} status mismatch")
    fail_total = _integer(
        result.get("fail_to_pass_total"),
        f"{phase}.fail_to_pass_total",
    )
    fail_passed = _integer(
        result.get("fail_to_pass_passed"),
        f"{phase}.fail_to_pass_passed",
    )
    pass_total = _integer(
        result.get("pass_to_pass_total"),
        f"{phase}.pass_to_pass_total",
    )
    pass_passed = _integer(
        result.get("pass_to_pass_passed"),
        f"{phase}.pass_to_pass_passed",
    )
    if fail_total <= 0 or pass_total <= 0:
        raise ContractError("test execution: F2P and P2P totals must be positive")
    if baseline:
        if fail_passed >= fail_total or pass_passed != pass_total:
            raise ContractError(
                "test execution: baseline must fail F2P and preserve P2P"
            )
    elif fail_passed != fail_total or pass_passed != pass_total:
        raise ContractError("test execution: Gold must pass F2P and P2P")


def promote_matched_runtime_task(
    task_path: Path,
    compatibility_path: Path,
    admission_path: Path,
    verified_at: str,
) -> dict[str, object]:
    task_path = Path(task_path).resolve()
    compatibility_path = Path(compatibility_path).resolve()
    admission_path = Path(admission_path).resolve()
    _validate_utc_seconds(verified_at)
    task = TaskManifest.load(task_path)
    compatibility_config = _mapping(
        task.data.get("compatibility"),
        "task.compatibility",
    )
    configured_compatibility_path = _relative_json_path(
        compatibility_config.get("evidence"),
        "task.compatibility.evidence",
    )
    expected_compatibility_path = (
        task.task_dir / configured_compatibility_path
    ).resolve()
    if compatibility_path != expected_compatibility_path:
        raise ContractError("compatibility evidence path mismatch")
    expected_admission_path = (task.task_dir / "admission/evidence.json").resolve()
    if admission_path != expected_admission_path:
        raise ContractError("admission evidence path mismatch")
    try:
        compatibility_payload = json.loads(
            compatibility_path.read_text(encoding="utf-8")
        )
        admission_payload = json.loads(admission_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"promotion evidence could not be loaded: {exc}") from exc
    compatibility = CompatibilityEvidence.from_dict(compatibility_payload)
    admission = _mapping(admission_payload, "admission evidence")
    validate_matched_runtime_promotion(task, compatibility, admission)

    promoted = json.loads(task_path.read_text(encoding="utf-8"))
    promoted["admission"] = {
        "status": "verified",
        "evidence": "admission/evidence.json",
        "compatibility_evidence": configured_compatibility_path,
        "verified_at": verified_at,
    }
    metadata = promoted.setdefault("metadata", {})
    metadata["curation_status"] = "verified"
    metadata["admission_status"] = "verified"
    metadata["source_loading_verified"] = True
    metadata["notes"] = (
        "Restored by v0.7 matched-runtime compatibility and verified Admission; "
        f"compatibility evidence {compatibility.content_hash}."
    )

    from scripts.validate_task import validate_manifest

    errors = validate_manifest(promoted)
    if errors:
        raise ContractError(
            "promoted task manifest is invalid: " + "; ".join(errors)
        )
    encoded = (json.dumps(promoted, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{task_path.name}.",
        suffix=".tmp",
        dir=task_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, task_path)
        directory = os.open(task_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return promoted
