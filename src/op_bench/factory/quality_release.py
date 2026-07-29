"""Formal v0.7 quality validation and historical re-admission accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from op_bench.dataset import DatasetManifest
from op_bench.factory.artifacts import (
    load_canonical_json_artifact,
    load_factory_contract,
    load_regular_file_bytes,
)
from op_bench.factory.complexity import (
    ComplexityEvidence,
    build_complexity_evidence,
)
from op_bench.factory.contracts import FactoryArtifactReference
from op_bench.factory.prompt_quality import (
    PromptQualityEvidence,
    build_private_answer_index,
    build_prompt_quality_evidence,
    validate_prompt_quality_evidence,
)
from op_bench.factory.taxonomy import parse_taxonomy_v2
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt
from op_bench.runtime.legacy import LegacyV05Defaults, full_task_spec_from_v05
from op_bench.runtime.task_view import project_agent_task_view
from op_bench.runtime.validation import ContractError
from op_bench.task import InvalidPublicTaskId, TaskManifest


_PUBLIC_TASK_ID = re.compile(r"opbench-v07-t[0-9]{4}")
_UTC_SECONDS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_ORIGINS = ("retained_historical", "new", "replacement")
_DISPOSITIONS = ("retained", "deferred", "retired")
_QUALITY_FIELDS = (
    "prompt_evidence",
    "complexity_evidence",
    "readmission_evidence",
    "origin",
)


@dataclass(frozen=True)
class QualityTaskRecord:
    task_id: str
    public_task_id: str
    origin: str
    task_path: str
    taxonomy_hash: str
    prompt_evidence: FactoryArtifactReference
    complexity_evidence: FactoryArtifactReference
    admission_evidence: FactoryArtifactReference
    disposition: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ContractError("quality_task_record.task_id: expected string")
        if (
            not isinstance(self.public_task_id, str)
            or _PUBLIC_TASK_ID.fullmatch(self.public_task_id) is None
        ):
            raise ContractError(
                "quality_task_record.public_task_id: expected opaque v0.7 Task ID"
            )
        if self.origin not in _ORIGINS:
            raise ContractError("quality_task_record.origin: unsupported value")
        _safe_relative_path(self.task_path, "quality_task_record.task_path")
        _require_hash(self.taxonomy_hash, "quality_task_record.taxonomy_hash")
        for name in (
            "prompt_evidence",
            "complexity_evidence",
            "admission_evidence",
        ):
            if not isinstance(getattr(self, name), FactoryArtifactReference):
                raise ContractError(
                    f"quality_task_record.{name}: expected FactoryArtifactReference"
                )
        if self.disposition not in _DISPOSITIONS:
            raise ContractError("quality_task_record.disposition: unsupported value")

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
            "taxonomy_hash",
            "prompt_evidence",
            "complexity_evidence",
            "admission_evidence",
            "disposition",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "public_task_id": self.public_task_id,
            "origin": self.origin,
            "task_path": self.task_path,
            "taxonomy_hash": self.taxonomy_hash,
            "prompt_evidence": self.prompt_evidence.to_dict(),
            "complexity_evidence": self.complexity_evidence.to_dict(),
            "admission_evidence": self.admission_evidence.to_dict(),
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_task_record",
    ) -> "QualityTaskRecord":
        data = _exact_mapping(value, path, cls.wire_fields())
        return cls(
            task_id=_string(data["task_id"], f"{path}.task_id"),
            public_task_id=_string(
                data["public_task_id"], f"{path}.public_task_id"
            ),
            origin=_string(data["origin"], f"{path}.origin"),
            task_path=_string(data["task_path"], f"{path}.task_path"),
            taxonomy_hash=_string(
                data["taxonomy_hash"], f"{path}.taxonomy_hash"
            ),
            prompt_evidence=FactoryArtifactReference.from_dict(
                data["prompt_evidence"],
                path=f"{path}.prompt_evidence",
            ),
            complexity_evidence=FactoryArtifactReference.from_dict(
                data["complexity_evidence"],
                path=f"{path}.complexity_evidence",
            ),
            admission_evidence=FactoryArtifactReference.from_dict(
                data["admission_evidence"],
                path=f"{path}.admission_evidence",
            ),
            disposition=_string(
                data["disposition"], f"{path}.disposition"
            ),
        )


@dataclass(frozen=True)
class _AuditArtifact:
    relative_path: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _HistoricalAudit:
    dataset_id: str
    dataset_hash: str
    created_at: str
    records: tuple[QualityTaskRecord, ...]
    artifacts: tuple[_AuditArtifact, ...]


def validate_quality_task(
    root: Path,
    task: TaskManifest,
    *,
    require_verified: bool,
) -> tuple[str, ...]:
    """Return deterministic formal-quality errors for one Task."""

    errors: list[str] = []
    if not isinstance(root, Path):
        return ("root: expected Path",)
    if not isinstance(task, TaskManifest):
        return ("task: expected TaskManifest",)

    taxonomy_hash: str | None = None
    taxonomy_value = task.data.get("taxonomy")
    if taxonomy_value is None:
        if require_verified:
            errors.append("taxonomy: required for formal v0.7")
    else:
        try:
            parse_taxonomy_v2(taxonomy_value)
            taxonomy_hash = canonical_sha256(taxonomy_value)
        except ContractError as exc:
            errors.append(str(exc))

    public_task_id: str | None = None
    try:
        public_task_id = task.public_task_id
    except InvalidPublicTaskId as exc:
        errors.append(str(exc))
    if public_task_id is None and require_verified:
        errors.append("agent_visible.public_task_id: required")

    metadata = task.data.get("metadata")
    difficulty = (
        metadata.get("difficulty") if isinstance(metadata, Mapping) else None
    )
    if difficulty == "easy":
        errors.append("metadata.difficulty: easy is forbidden")
    elif require_verified and difficulty not in ("medium", "hard"):
        errors.append("metadata.difficulty: expected medium or hard")
    if require_verified:
        if task.admission_status != "verified":
            errors.append("admission.status: verified required")
        _, admission_errors = _admission_truth(task)
        errors.extend(admission_errors)
        patch_scope = task.data.get("patch_scope")
        if (
            not isinstance(patch_scope, Mapping)
            or patch_scope.get("mode", "enforced") != "enforced"
            or not task.patch_scope_paths
        ):
            errors.append("patch_scope: private enforced scope required")

    quality = task.data.get("quality")
    if not isinstance(quality, Mapping):
        if require_verified:
            errors.extend(
                f"quality.{name}: required"
                for name in _QUALITY_FIELDS[:3]
            )
            errors.append("quality.origin: required")
        return tuple(errors)

    unknown = sorted(set(quality) - set(_QUALITY_FIELDS))
    if unknown:
        errors.append(f"quality: unknown fields {unknown}")
    origin = quality.get("origin")
    if origin is None:
        if require_verified:
            errors.append("quality.origin: required")
    elif origin not in _ORIGINS:
        errors.append("quality.origin: unsupported value")

    selected_paths: dict[str, Path] = {}
    for field in _QUALITY_FIELDS[:3]:
        value = quality.get(field)
        if value is None:
            if require_verified:
                errors.append(f"quality.{field}: required")
            continue
        try:
            selected_paths[field] = _task_relative_file(
                task.task_dir,
                value,
                f"quality.{field}",
            )
        except ContractError as exc:
            errors.append(str(exc))

    prompt: PromptQualityEvidence | None = None
    prompt_path = selected_paths.get("prompt_evidence")
    if prompt_path is not None:
        try:
            loaded = load_factory_contract(prompt_path)
            if not isinstance(loaded, PromptQualityEvidence):
                raise ContractError(
                    "quality.prompt_evidence: expected prompt_quality contract"
                )
            prompt = loaded
        except ContractError as exc:
            errors.append(_prefixed("quality.prompt_evidence", exc))

    complexity: ComplexityEvidence | None = None
    complexity_path = selected_paths.get("complexity_evidence")
    if complexity_path is not None:
        try:
            loaded = load_factory_contract(complexity_path)
            if not isinstance(loaded, ComplexityEvidence):
                raise ContractError(
                    "quality.complexity_evidence: expected complexity_evidence contract"
                )
            complexity = loaded
        except ContractError as exc:
            errors.append(_prefixed("quality.complexity_evidence", exc))

    view: dict[str, object] | None = None
    private_index = None
    if prompt is not None and public_task_id is not None:
        try:
            view = _quality_agent_task_view(task)
            private_index = _private_answer_index(task)
            validate_prompt_quality_evidence(
                prompt,
                rendered_prompt=render_mcp_prompt(view),
                agent_task_view=view,
                private_index=private_index,
            )
        except (ContractError, OSError, UnicodeDecodeError) as exc:
            errors.append(_prefixed("quality.prompt_evidence", exc))
        if prompt.task_id != task.task_id:
            errors.append("quality.prompt_evidence: task_id mismatch")
        if prompt.public_task_id != public_task_id:
            errors.append("quality.prompt_evidence: public_task_id mismatch")
        if require_verified and prompt.decision != "accepted":
            errors.append("quality.prompt_evidence: accepted decision required")

    if complexity is not None:
        if complexity.task_id != task.task_id:
            errors.append("quality.complexity_evidence: task_id mismatch")
        if difficulty is not None and complexity.difficulty != difficulty:
            errors.append(
                "quality.complexity_evidence: difficulty does not match metadata"
            )
        if require_verified and complexity.decision != "accepted":
            errors.append(
                "quality.complexity_evidence: accepted decision required"
            )

    readmission_path = selected_paths.get("readmission_evidence")
    if readmission_path is not None:
        try:
            readmission = load_canonical_json_artifact(readmission_path)
            _validate_readmission(
                readmission,
                task=task,
                public_task_id=public_task_id,
                origin=origin,
                taxonomy_hash=taxonomy_hash,
                prompt=prompt,
                prompt_relative=quality.get("prompt_evidence"),
                complexity=complexity,
                complexity_relative=quality.get("complexity_evidence"),
                require_verified=require_verified,
            )
        except ContractError as exc:
            errors.append(_prefixed("quality.readmission_evidence", exc))

    return tuple(_ordered_unique(errors))


def build_historical_dispositions(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    created_at: str,
) -> tuple[QualityTaskRecord, ...]:
    """Build one deterministic disposition for every historical Dataset Task."""

    return _build_historical_audit(
        root,
        dataset_path,
        review_root,
        created_at,
    ).records


def write_historical_dispositions(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    output_path: Path,
    created_at: str,
) -> tuple[QualityTaskRecord, ...]:
    """Build and write the canonical task evidence tree and global index."""

    audit = _build_historical_audit(
        root,
        dataset_path,
        review_root,
        created_at,
    )
    output_root = output_path.parent
    for artifact in audit.artifacts:
        _write_canonical(
            output_root / PurePosixPath(artifact.relative_path),
            artifact.payload,
        )
    retained = sum(
        record.disposition == "retained" for record in audit.records
    )
    payload: dict[str, object] = {
        "contract_type": "historical_readmission_index",
        "schema_version": "v1",
        "dataset_id": audit.dataset_id,
        "dataset_hash": audit.dataset_hash,
        "created_at": audit.created_at,
        "task_count": len(audit.records),
        "k": retained,
        "required_candidate_count": 3 * (50 - retained),
        "records": [record.to_dict() for record in audit.records],
    }
    payload["content_hash"] = canonical_sha256(payload)
    _write_canonical(output_path, payload)
    return audit.records


def _build_historical_audit(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    created_at: str,
) -> _HistoricalAudit:
    if not isinstance(root, Path) or not root.is_dir():
        raise ContractError("root: expected repository directory")
    if not isinstance(dataset_path, Path):
        raise ContractError("dataset_path: expected Path")
    if not isinstance(review_root, Path):
        raise ContractError("review_root: expected Path")
    if not isinstance(created_at, str) or _UTC_SECONDS.fullmatch(created_at) is None:
        raise ContractError("created_at: expected UTC RFC3339 seconds")

    dataset_bytes = load_regular_file_bytes(dataset_path)
    try:
        dataset_payload = json.loads(dataset_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("dataset: invalid JSON") from exc
    if not isinstance(dataset_payload, Mapping):
        raise ContractError("dataset: expected JSON object")
    entries = dataset_payload.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ContractError("dataset.tasks: expected non-empty array")
    task_ids = [
        entry.get("task_id") if isinstance(entry, Mapping) else None
        for entry in entries
    ]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ContractError("dataset.tasks: every entry requires task_id")
    if len(set(task_ids)) != len(task_ids):
        raise ContractError("dataset.tasks: duplicate task_id")

    loaded_tasks = DatasetManifest.load(dataset_path).load_tasks()
    by_id = {task.task_id: task for task in loaded_tasks}
    if set(by_id) != set(task_ids):
        raise ContractError("dataset.tasks: loaded Task identities do not match")

    records: list[QualityTaskRecord] = []
    artifacts: list[_AuditArtifact] = []
    accepted_fingerprints: set[str] = set()
    for index, entry_value in enumerate(entries, start=1):
        assert isinstance(entry_value, Mapping)
        task_id = str(entry_value["task_id"])
        public_task_id = f"opbench-v07-t{index:04d}"
        original = by_id[task_id]
        data = dict(original.data)
        agent_visible = dict(
            data.get("agent_visible")
            if isinstance(data.get("agent_visible"), Mapping)
            else {}
        )
        agent_visible["public_task_id"] = public_task_id
        data["agent_visible"] = agent_visible
        task = TaskManifest(task_dir=original.task_dir, data=data)
        task_path = _repo_relative(root, task.task_dir, "task_path")
        artifact_prefix = f"tasks/{public_task_id}/quality"

        errors: list[str] = []
        retirement = False
        taxonomy_value = task.data.get("taxonomy")
        if taxonomy_value is None:
            taxonomy_hash = canonical_sha256(
                {"task_id": task_id, "taxonomy": "missing"}
            )
            errors.append("taxonomy: missing")
        else:
            try:
                taxonomy_hash = _json_hash(taxonomy_value)
                parse_taxonomy_v2(taxonomy_value)
            except ContractError as exc:
                errors.append(str(exc))

        view: dict[str, object] | None = None
        private_index = None
        try:
            view = _quality_agent_task_view(task)
            private_index = _private_answer_index(task)
        except (ContractError, OSError, UnicodeDecodeError) as exc:
            errors.append(f"agent_task_view: {exc}")

        prompt_review_path = _review_path(
            review_root,
            task_id,
            "prompt",
        )
        prompt_review, prompt_review_hash, prompt_review_error = _review_json(
            prompt_review_path,
            "prompt review",
        )
        prompt_evidence: PromptQualityEvidence | None = None
        prompt_errors: list[str] = []
        if prompt_review_error is not None:
            prompt_errors.append(prompt_review_error)
        elif view is None or private_index is None:
            prompt_errors.append("agent_task_view unavailable")
        else:
            try:
                prompt_evidence = _prompt_evidence_from_review(
                    prompt_review,
                    task=task,
                    public_task_id=public_task_id,
                    view=view,
                    private_index=private_index,
                    created_at=created_at,
                )
                if prompt_evidence.decision == "rejected":
                    retirement = True
                elif prompt_evidence.decision != "accepted":
                    prompt_errors.append(
                        f"decision is {prompt_evidence.decision}"
                    )
            except ContractError as exc:
                prompt_errors.append(str(exc))
                if (
                    isinstance(prompt_review, Mapping)
                    and prompt_review.get("decision") == "rejected"
                ):
                    retirement = True
        errors.extend(f"prompt: {error}" for error in prompt_errors)
        prompt_payload = (
            prompt_evidence.to_dict()
            if prompt_evidence is not None
            else _audit_payload(
                "historical_prompt_review",
                task_id=task_id,
                public_task_id=public_task_id,
                review_hash=prompt_review_hash,
                errors=prompt_errors,
                created_at=created_at,
            )
        )
        prompt_ref = FactoryArtifactReference(
            artifact_type=str(prompt_payload["contract_type"]),
            artifact_id=task_id,
            content_hash=str(prompt_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/prompt.json",
        )
        artifacts.append(_AuditArtifact(prompt_ref.relative_path, prompt_payload))

        complexity_review_path = _review_path(
            review_root,
            task_id,
            "complexity",
        )
        (
            complexity_review,
            complexity_review_hash,
            complexity_review_error,
        ) = _review_json(complexity_review_path, "complexity review")
        complexity: ComplexityEvidence | None = None
        complexity_errors: list[str] = []
        if complexity_review_error is not None:
            complexity_errors.append(complexity_review_error)
        else:
            try:
                complexity = _complexity_from_review(
                    complexity_review,
                    task_id=task_id,
                    created_at=created_at,
                )
                if complexity.decision == "rejected":
                    retirement = True
                elif complexity.decision != "accepted":
                    complexity_errors.append(
                        f"decision is {complexity.decision}"
                    )
                difficulty = (
                    task.data.get("metadata", {}).get("difficulty")
                    if isinstance(task.data.get("metadata"), Mapping)
                    else None
                )
                if complexity.difficulty != difficulty:
                    complexity_errors.append(
                        "difficulty does not match Task metadata"
                    )
            except ContractError as exc:
                complexity_errors.append(str(exc))
                if (
                    isinstance(complexity_review, Mapping)
                    and complexity_review.get("decision") == "rejected"
                ):
                    retirement = True
        errors.extend(
            f"complexity: {error}" for error in complexity_errors
        )
        complexity_payload = (
            complexity.to_dict()
            if complexity is not None
            else _audit_payload(
                "historical_complexity_review",
                task_id=task_id,
                public_task_id=public_task_id,
                review_hash=complexity_review_hash,
                errors=complexity_errors,
                created_at=created_at,
            )
        )
        complexity_ref = FactoryArtifactReference(
            artifact_type=str(complexity_payload["contract_type"]),
            artifact_id=task_id,
            content_hash=str(complexity_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/complexity.json",
        )
        artifacts.append(
            _AuditArtifact(complexity_ref.relative_path, complexity_payload)
        )

        admission_hash, admission_errors = _admission_truth(task)
        errors.extend(admission_errors)
        if (
            task.metadata_source_loading_verified is not True
            and task.runtime_tier != "cpu_source_snapshot_fuller"
        ):
            errors.append("runtime: source loading truth is missing")
        if (
            not retirement
            and not errors
            and complexity is not None
            and complexity.decision == "accepted"
        ):
            if complexity.duplicate_fingerprint in accepted_fingerprints:
                errors.append(
                    "complexity: duplicate fingerprint already retained"
                )
                retirement = True
            else:
                accepted_fingerprints.add(
                    complexity.duplicate_fingerprint
                )

        disposition = (
            "retired"
            if retirement
            else "deferred"
            if errors
            else "retained"
        )
        readmission_payload: dict[str, object] = {
            "contract_type": "quality_readmission",
            "schema_version": "v1",
            "task_id": task_id,
            "public_task_id": public_task_id,
            "origin": "retained_historical",
            "disposition": disposition,
            "taxonomy_hash": taxonomy_hash,
            "prompt_evidence": prompt_ref.to_dict(),
            "complexity_evidence": complexity_ref.to_dict(),
            "admission_evidence_hash": admission_hash,
            "review_input_hashes": {
                "prompt": prompt_review_hash,
                "complexity": complexity_review_hash,
            },
            "errors": list(_ordered_unique(errors)),
            "created_at": created_at,
        }
        readmission_payload["content_hash"] = canonical_sha256(
            readmission_payload
        )
        readmission_ref = FactoryArtifactReference(
            artifact_type="quality_readmission",
            artifact_id=task_id,
            content_hash=str(readmission_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/readmission.json",
        )
        artifacts.append(
            _AuditArtifact(
                readmission_ref.relative_path,
                readmission_payload,
            )
        )
        records.append(
            QualityTaskRecord(
                task_id=task_id,
                public_task_id=public_task_id,
                origin="retained_historical",
                task_path=task_path,
                taxonomy_hash=taxonomy_hash,
                prompt_evidence=prompt_ref,
                complexity_evidence=complexity_ref,
                admission_evidence=readmission_ref,
                disposition=disposition,
            )
        )

    return _HistoricalAudit(
        dataset_id=_string(
            dataset_payload.get("dataset_id"),
            "dataset.dataset_id",
        ),
        dataset_hash=canonical_sha256(dataset_payload),
        created_at=created_at,
        records=tuple(records),
        artifacts=tuple(artifacts),
    )


def _quality_agent_task_view(task: TaskManifest) -> dict[str, object]:
    spec = full_task_spec_from_v05(task)
    defaults = LegacyV05Defaults.standard()
    capability = replace(
        defaults.capability_policy,
        policy_id="opbench-v0.7-repository-root-v1",
        writable_paths=(".",),
        registered_tests=tuple(
            sorted(selector.selector_id for selector in spec.public_tests)
        ),
    )
    return project_agent_task_view(
        spec,
        capability,
        defaults.budget_policy,
    ).to_dict()


def _private_answer_index(task: TaskManifest):
    artifacts = task.data.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("artifacts: expected object")
    gold_path = _task_relative_file(
        task.task_dir,
        artifacts.get("gold_patch"),
        "artifacts.gold_patch",
    )
    hidden_value = artifacts.get(
        "hidden_test_patch",
        artifacts.get("test_patch"),
    )
    hidden_path = _task_relative_file(
        task.task_dir,
        hidden_value,
        "artifacts.hidden_test_patch",
    )
    spec = full_task_spec_from_v05(task)
    return build_private_answer_index(
        gold_patch=load_regular_file_bytes(gold_path).decode("utf-8"),
        hidden_test_patch=load_regular_file_bytes(hidden_path).decode("utf-8"),
        patch_scope=tuple(task.patch_scope_paths),
        hidden_selectors=tuple(
            selector.selector_id for selector in spec.hidden_tests
        ),
    )


def _validate_readmission(
    value: Mapping[str, object],
    *,
    task: TaskManifest,
    public_task_id: str | None,
    origin: object,
    taxonomy_hash: str | None,
    prompt: PromptQualityEvidence | None,
    prompt_relative: object,
    complexity: ComplexityEvidence | None,
    complexity_relative: object,
    require_verified: bool,
) -> None:
    required = (
        "contract_type",
        "schema_version",
        "task_id",
        "public_task_id",
        "origin",
        "disposition",
        "taxonomy_hash",
        "prompt_evidence",
        "complexity_evidence",
        "admission_evidence_hash",
        "created_at",
        "content_hash",
    )
    missing = sorted(set(required) - set(value))
    if missing:
        raise ContractError(f"missing fields {missing}")
    if value["contract_type"] != "quality_readmission":
        raise ContractError("contract_type: expected 'quality_readmission'")
    if value["schema_version"] != "v1":
        raise ContractError("schema_version: expected 'v1'")
    stored_hash = value["content_hash"]
    _require_hash(stored_hash, "content_hash")
    if stored_hash != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_hash"}
    ):
        raise ContractError("content_hash: payload hash mismatch")
    if value["task_id"] != task.task_id:
        raise ContractError("task_id mismatch")
    if value["public_task_id"] != public_task_id:
        raise ContractError("public_task_id mismatch")
    if value["origin"] != origin:
        raise ContractError("origin mismatch")
    if value["taxonomy_hash"] != taxonomy_hash:
        raise ContractError("taxonomy_hash mismatch")
    if value["disposition"] not in _DISPOSITIONS:
        raise ContractError("disposition: unsupported value")
    if require_verified and value["disposition"] != "retained":
        raise ContractError("disposition: retained required")
    if prompt is not None:
        _validate_embedded_reference(
            value["prompt_evidence"],
            artifact_type=prompt.contract_type,
            artifact_id=task.task_id,
            content_hash=prompt.content_hash,
            relative_path=prompt_relative,
            path="prompt_evidence",
        )
    if complexity is not None:
        _validate_embedded_reference(
            value["complexity_evidence"],
            artifact_type=complexity.contract_type,
            artifact_id=task.task_id,
            content_hash=complexity.content_hash,
            relative_path=complexity_relative,
            path="complexity_evidence",
        )
    admission = task.data.get("admission")
    if not isinstance(admission, Mapping):
        raise ContractError("Task admission evidence is missing")
    admission_path = _task_relative_file(
        task.task_dir,
        admission.get("evidence"),
        "admission.evidence",
    )
    admission_bytes = load_regular_file_bytes(admission_path)
    try:
        json.loads(admission_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("admission evidence: invalid JSON") from exc
    expected_admission_hash = _bytes_hash(admission_bytes)
    if value["admission_evidence_hash"] != expected_admission_hash:
        raise ContractError("admission_evidence_hash mismatch")


def _validate_embedded_reference(
    value: object,
    *,
    artifact_type: str,
    artifact_id: str,
    content_hash: str,
    relative_path: object,
    path: str,
) -> None:
    reference = FactoryArtifactReference.from_dict(value, path=path)
    if reference.artifact_type != artifact_type:
        raise ContractError(f"{path}: artifact_type mismatch")
    if reference.artifact_id != artifact_id:
        raise ContractError(f"{path}: artifact_id mismatch")
    if reference.content_hash != content_hash:
        raise ContractError(f"{path}: content_hash mismatch")
    if reference.relative_path != relative_path:
        raise ContractError(f"{path}: relative_path mismatch")


def _prompt_evidence_from_review(
    review: object,
    *,
    task: TaskManifest,
    public_task_id: str,
    view: Mapping[str, object],
    private_index,
    created_at: str,
) -> PromptQualityEvidence:
    if not isinstance(review, Mapping):
        raise ContractError("prompt review: expected object")
    if review.get("contract_type") == PromptQualityEvidence.contract_type:
        evidence = PromptQualityEvidence.from_dict(review)
        validate_prompt_quality_evidence(
            evidence,
            rendered_prompt=render_mcp_prompt(view),
            agent_task_view=view,
            private_index=private_index,
        )
        if evidence.task_id != task.task_id:
            raise ContractError("prompt review: task_id mismatch")
        if evidence.public_task_id != public_task_id:
            raise ContractError("prompt review: public_task_id mismatch")
        return evidence
    required = ("blind_review", "semantic_review", "decision")
    missing = sorted(set(required) - set(review))
    if missing:
        raise ContractError(f"prompt review: missing fields {missing}")
    return build_prompt_quality_evidence(
        task_id=task.task_id,
        public_task_id=public_task_id,
        rendered_prompt=render_mcp_prompt(view),
        agent_task_view=view,
        private_index=private_index,
        scanner_version=str(
            review.get("scanner_version", "prompt-overlap-v1")
        ),
        blind_review=_mapping(review["blind_review"], "blind_review"),
        semantic_review=_mapping(
            review["semantic_review"],
            "semantic_review",
        ),
        decision=_string(review["decision"], "decision"),
        created_at=str(review.get("created_at", created_at)),
    )


def _complexity_from_review(
    review: object,
    *,
    task_id: str,
    created_at: str,
) -> ComplexityEvidence:
    if not isinstance(review, Mapping):
        raise ContractError("complexity review: expected object")
    if review.get("contract_type") == ComplexityEvidence.contract_type:
        evidence = ComplexityEvidence.from_dict(review)
        if evidence.task_id != task_id:
            raise ContractError("complexity review: task_id mismatch")
        return evidence
    required = (
        "localization",
        "diagnosis",
        "repair_regression",
        "dimension_evidence",
        "hard_rejections",
        "risk_signals",
        "duplicate_fingerprint",
        "duplicate_decision",
        "blind_pilot",
        "second_review",
        "reviewer",
    )
    missing = sorted(set(required) - set(review))
    if missing:
        raise ContractError(f"complexity review: missing fields {missing}")
    return build_complexity_evidence(
        task_id=task_id,
        localization=review["localization"],  # type: ignore[arg-type]
        diagnosis=review["diagnosis"],  # type: ignore[arg-type]
        repair_regression=review["repair_regression"],  # type: ignore[arg-type]
        dimension_evidence=_mapping(
            review["dimension_evidence"],
            "dimension_evidence",
        ),
        hard_rejections=tuple(
            _list(review["hard_rejections"], "hard_rejections")
        ),
        risk_signals=tuple(
            _list(review["risk_signals"], "risk_signals")
        ),
        duplicate_fingerprint=_string(
            review["duplicate_fingerprint"],
            "duplicate_fingerprint",
        ),
        duplicate_decision=_string(
            review["duplicate_decision"],
            "duplicate_decision",
        ),
        blind_pilot=(
            None
            if review["blind_pilot"] is None
            else _mapping(review["blind_pilot"], "blind_pilot")
        ),
        second_review=review["second_review"],  # type: ignore[arg-type]
        reviewer=_string(review["reviewer"], "reviewer"),
        reviewed_at=str(review.get("reviewed_at", created_at)),
    )


def _admission_truth(task: TaskManifest) -> tuple[str, tuple[str, ...]]:
    errors: list[str] = []
    admission = task.data.get("admission")
    if not isinstance(admission, Mapping):
        return (
            canonical_sha256(
                {"task_id": task.task_id, "admission": "missing"}
            ),
            ("admission: evidence is missing",),
        )
    try:
        path = _task_relative_file(
            task.task_dir,
            admission.get("evidence"),
            "admission.evidence",
        )
        admission_bytes = load_regular_file_bytes(path)
        payload = json.loads(admission_bytes.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            canonical_sha256(
                {"task_id": task.task_id, "admission": "invalid"}
            ),
            (f"admission: {exc}",),
        )
    if not isinstance(payload, Mapping):
        return (
            _bytes_hash(admission_bytes),
            ("admission: evidence must be an object",),
        )
    decision = payload.get("admission")
    if (
        not isinstance(decision, Mapping)
        or decision.get("verified") is not True
        or decision.get("decision") != "verified"
    ):
        errors.append("admission: verified decision is missing")
    baseline = payload.get("baseline")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("status") != "baseline_reproduced"
    ):
        errors.append("runtime: baseline truth is missing")
    gold = payload.get("gold")
    if not isinstance(gold, Mapping) or gold.get("status") != "resolved":
        errors.append("runtime: Gold truth is missing")
    return _bytes_hash(admission_bytes), tuple(errors)


def _review_path(root: Path, task_id: str, kind: str) -> Path:
    candidates = (
        root / task_id / f"{kind}.json",
        root / kind / f"{task_id}.json",
        root / f"{task_id}.{kind}.json",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def _review_json(
    path: Path,
    label: str,
) -> tuple[object | None, str, str | None]:
    if not path.exists():
        return (
            None,
            canonical_sha256({"status": "missing", "label": label}),
            f"{label} is missing",
        )
    try:
        content = load_regular_file_bytes(path)
        value = json.loads(content.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            canonical_sha256({"status": "invalid", "label": label}),
            f"{label} is invalid: {exc}",
        )
    return value, _json_hash(value), None


def _audit_payload(
    contract_type: str,
    *,
    task_id: str,
    public_task_id: str,
    review_hash: str,
    errors: list[str],
    created_at: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": contract_type,
        "schema_version": "v1",
        "task_id": task_id,
        "public_task_id": public_task_id,
        "review_hash": review_hash,
        "status": "deferred",
        "errors": list(_ordered_unique(errors)),
        "created_at": created_at,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _task_relative_file(
    task_root: Path,
    value: object,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}: expected non-empty task-relative path")
    if "\\" in value:
        raise ContractError(f"{label}: expected normalized relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ContractError(f"{label}: path escapes task root")
    root = task_root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label}: symlinks are not allowed")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{label}: artifact file is unavailable") from exc
    if root != resolved and root not in resolved.parents:
        raise ContractError(f"{label}: path escapes task root")
    if not resolved.is_file():
        raise ContractError(f"{label}: expected regular file")
    return resolved


def _repo_relative(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"{label}: path is outside repository") from exc
    return _safe_relative_path(relative, label)


def _safe_relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != text
    ):
        raise ContractError(f"{label}: expected safe relative path")
    return text


def _exact_mapping(
    value: object,
    path: str,
    fields: tuple[str, ...],
) -> Mapping[str, object]:
    data = _mapping(value, path)
    missing = sorted(set(fields) - set(data))
    unknown = sorted(set(data) - set(fields))
    if missing:
        raise ContractError(f"{path}: missing fields {missing}")
    if unknown:
        raise ContractError(f"{path}: unknown fields {unknown}")
    return data


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{path}: expected array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path}: expected non-empty string")
    return value


def _require_hash(value: object, path: str) -> str:
    text = _string(value, path)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ContractError(f"{path}: expected sha256 digest")
    return text


def _bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_hash(value: object) -> str:
    try:
        return canonical_sha256(value)
    except ContractError:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return _bytes_hash(encoded)


def _prefixed(path: str, exc: BaseException) -> str:
    text = str(exc)
    return text if text.startswith(f"{path}:") else f"{path}: {text}"


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(value).encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ContractError("output path: expected regular file")
        if path.read_bytes() == content:
            return
    path.write_bytes(content)


__all__ = [
    "QualityTaskRecord",
    "build_historical_dispositions",
    "validate_quality_task",
    "write_historical_dispositions",
]
