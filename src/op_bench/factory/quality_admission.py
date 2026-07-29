"""Canonical v0.7 expansion admission artifacts.

The contracts in this module deliberately keep review authorization separate
from task construction.  A deferred screening decision can only become build
eligible through an independently reviewed, source-bound reassessment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Callable, ClassVar

from op_bench.admission import AdmissionRunner
from op_bench.factory.artifacts import (
    load_canonical_json_artifact,
    load_factory_contract,
    load_regular_file_bytes,
)
from op_bench.factory.contracts import FactoryArtifactReference
from op_bench.factory.quality_release import (
    QualityCandidateDecision,
    QualityCandidateRecord,
    quality_prompt_source_inputs,
    validate_candidate_index,
    validate_historical_index,
    validate_quality_task,
)
from op_bench.factory.prompt_quality import (
    PromptQualityEvidence,
    validate_prompt_quality_evidence,
)
from op_bench.integrity import replay_spec_hash
from op_bench.registry import load_resolved_task
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_mapping,
    require_str,
)
from op_bench.task import InvalidPublicTaskId, TaskManifest


_HASH_PATTERN = r"sha256:[0-9a-f]{64}"
_CANDIDATE_ID_PATTERN = r"quality-candidate:v1:[0-9a-f]{64}"
_DECISION_ID_PATTERN = r"quality-decision:v1:[0-9a-f]{64}"
_UTC_SECONDS_PATTERN = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_REASSESSMENT_DECISIONS = ("accepted_for_build", "rejected")
_REASON_RESOLUTIONS = ("resolved", "confirmed_blocker")
_SCREENING_DISPOSITIONS = (
    "accepted_for_build",
    "deferred_for_review",
    "hard_rejected",
)
_ACCEPTED_INDEX_STATUSES = ("building", "complete")
_EXPANSION_ORIGINS = ("new", "replacement")
_PUBLIC_TASK_ID_PATTERN = r"opbench-v07-t[0-9]{4}"
_OFFICIAL_ACCEPTED_INDEX = "factory/v0.7/p8/accepted_tasks.json"
_OFFICIAL_HISTORICAL_INDEX = (
    "factory/v0.7/p7/historical_readmission.json"
)
_OFFICIAL_CANDIDATE_INDEX = (
    "factory/v0.7/p8/screening/screening_index.json"
)
_PREFLIGHT_STATUSES = ("not_run", "passed", "failed")


def _utc_seconds(value: object, path: str) -> str:
    text = require_str(value, path, pattern=_UTC_SECONDS_PATTERN)
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: invalid UTC timestamp") from exc
    return text


def _canonical_string_tuple(
    value: object,
    path: str,
) -> tuple[str, ...]:
    items = require_list(value, path)
    if not items:
        raise ContractError(f"{path}: must contain at least one value")
    result = tuple(
        require_str(item, f"{path}[{index}]")
        for index, item in enumerate(items)
    )
    if result != tuple(sorted(set(result))):
        raise ContractError(f"{path}: expected sorted unique values")
    return result


@dataclass(frozen=True)
class QualityCandidateReasonResolution:
    """Resolution of one exact preliminary review reason."""

    reason: str
    resolution: str
    evidence: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("reason", "resolution", "evidence")

    def __post_init__(self) -> None:
        require_str(self.reason, "reason_resolution.reason")
        require_enum(
            self.resolution,
            "reason_resolution.resolution",
            _REASON_RESOLUTIONS,
        )
        require_str(self.evidence, "reason_resolution.evidence")

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "resolution": self.resolution,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate_reason_resolution",
    ) -> "QualityCandidateReasonResolution":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            reason=require_str(data["reason"], f"{path}.reason"),
            resolution=require_enum(
                data["resolution"],
                f"{path}.resolution",
                _REASON_RESOLUTIONS,
            ),
            evidence=require_str(data["evidence"], f"{path}.evidence"),
        )


@dataclass(frozen=True)
class QualityCandidateReassessment:
    """Independent authorization for a deferred candidate."""

    contract_type: ClassVar[str] = "quality_candidate_reassessment"
    schema_version: ClassVar[str] = "v1"

    pr_number: int
    candidate_id: str
    candidate_hash: str
    decision_id: str
    decision_hash: str
    deferred_reasons: tuple[str, ...]
    reason_resolutions: tuple[QualityCandidateReasonResolution, ...]
    reviewer: str
    reviewed_at: str
    decision: str
    rationale: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "pr_number",
            "candidate_id",
            "candidate_hash",
            "decision_id",
            "decision_hash",
            "deferred_reasons",
            "reason_resolutions",
            "reviewer",
            "reviewed_at",
            "decision",
            "rationale",
            "content_hash",
        )

    def __post_init__(self) -> None:
        require_int(self.pr_number, "candidate_reassessment.pr_number", minimum=1)
        require_str(
            self.candidate_id,
            "candidate_reassessment.candidate_id",
            pattern=_CANDIDATE_ID_PATTERN,
        )
        require_str(
            self.candidate_hash,
            "candidate_reassessment.candidate_hash",
            pattern=_HASH_PATTERN,
        )
        require_str(
            self.decision_id,
            "candidate_reassessment.decision_id",
            pattern=_DECISION_ID_PATTERN,
        )
        require_str(
            self.decision_hash,
            "candidate_reassessment.decision_hash",
            pattern=_HASH_PATTERN,
        )
        if self.deferred_reasons != tuple(
            sorted(set(self.deferred_reasons))
        ) or not self.deferred_reasons:
            raise ContractError(
                "candidate_reassessment.deferred_reasons: "
                "expected sorted unique values"
            )
        for reason in self.deferred_reasons:
            require_str(reason, "candidate_reassessment.deferred_reasons[]")
        for resolution in self.reason_resolutions:
            if not isinstance(
                resolution, QualityCandidateReasonResolution
            ):
                raise ContractError(
                    "candidate_reassessment.reason_resolutions: "
                    "expected reason resolution"
                )
        resolution_reasons = tuple(
            resolution.reason for resolution in self.reason_resolutions
        )
        if resolution_reasons != self.deferred_reasons:
            raise ContractError(
                "candidate_reassessment.reason_resolutions: "
                "must resolve every exact deferred reason once and in order"
            )
        require_str(self.reviewer, "candidate_reassessment.reviewer")
        _utc_seconds(
            self.reviewed_at,
            "candidate_reassessment.reviewed_at",
        )
        decision = require_enum(
            self.decision,
            "candidate_reassessment.decision",
            _REASSESSMENT_DECISIONS,
        )
        has_blocker = any(
            resolution.resolution == "confirmed_blocker"
            for resolution in self.reason_resolutions
        )
        if decision == "accepted_for_build" and has_blocker:
            raise ContractError(
                "candidate_reassessment: accepted reassessment cannot "
                "contain a confirmed blocker"
            )
        if decision == "rejected" and not has_blocker:
            raise ContractError(
                "candidate_reassessment: rejected reassessment must "
                "contain a confirmed blocker"
            )
        require_str(self.rationale, "candidate_reassessment.rationale")

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "pr_number": self.pr_number,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "decision_id": self.decision_id,
            "decision_hash": self.decision_hash,
            "deferred_reasons": list(self.deferred_reasons),
            "reason_resolutions": [
                resolution.to_dict()
                for resolution in self.reason_resolutions
            ],
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "decision": self.decision,
            "rationale": self.rationale,
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate_reassessment",
    ) -> "QualityCandidateReassessment":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        reasons = _canonical_string_tuple(
            data["deferred_reasons"],
            f"{path}.deferred_reasons",
        )
        resolution_values = require_list(
            data["reason_resolutions"],
            f"{path}.reason_resolutions",
        )
        reassessment = cls(
            pr_number=require_int(
                data["pr_number"], f"{path}.pr_number", minimum=1
            ),
            candidate_id=require_str(
                data["candidate_id"],
                f"{path}.candidate_id",
                pattern=_CANDIDATE_ID_PATTERN,
            ),
            candidate_hash=require_str(
                data["candidate_hash"],
                f"{path}.candidate_hash",
                pattern=_HASH_PATTERN,
            ),
            decision_id=require_str(
                data["decision_id"],
                f"{path}.decision_id",
                pattern=_DECISION_ID_PATTERN,
            ),
            decision_hash=require_str(
                data["decision_hash"],
                f"{path}.decision_hash",
                pattern=_HASH_PATTERN,
            ),
            deferred_reasons=reasons,
            reason_resolutions=tuple(
                QualityCandidateReasonResolution.from_dict(
                    item,
                    path=f"{path}.reason_resolutions[{index}]",
                )
                for index, item in enumerate(resolution_values)
            ),
            reviewer=require_str(
                data["reviewer"], f"{path}.reviewer"
            ),
            reviewed_at=_utc_seconds(
                data["reviewed_at"],
                f"{path}.reviewed_at",
            ),
            decision=require_enum(
                data["decision"],
                f"{path}.decision",
                _REASSESSMENT_DECISIONS,
            ),
            rationale=require_str(
                data["rationale"], f"{path}.rationale"
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=_HASH_PATTERN,
        )
        if stored_hash != reassessment.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected "
                f"{reassessment.content_hash!r}"
            )
        return reassessment


def _relative_posix_path(
    value: object,
    path: str,
    *,
    suffix: str | None = None,
) -> str:
    text = require_str(value, path)
    if "\\" in text:
        raise ContractError(f"{path}: expected normalized POSIX path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != text
    ):
        raise ContractError(f"{path}: expected normalized relative path")
    if suffix is not None and not text.endswith(suffix):
        raise ContractError(f"{path}: expected {suffix!r} suffix")
    return text


def _task_path(value: object, path: str) -> str:
    text = _relative_posix_path(value, path)
    parts = PurePosixPath(text).parts
    if len(parts) != 3 or parts[:2] != ("tasks", "pytorch"):
        raise ContractError(f"{path}: expected tasks/pytorch/<task>")
    return text


@dataclass(frozen=True)
class QualityAcceptedTaskRecord:
    """One expansion Task authorized by the exact screening funnel."""

    screening_record_index: int
    pr_number: int
    candidate_id: str
    candidate_hash: str
    decision_id: str
    decision_hash: str
    screening_disposition: str
    reassessment: FactoryArtifactReference | None
    task_id: str
    public_task_id: str
    origin: str
    task_path: str
    task_manifest_hash: str
    replay_spec_hash: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "screening_record_index",
            "pr_number",
            "candidate_id",
            "candidate_hash",
            "decision_id",
            "decision_hash",
            "screening_disposition",
            "reassessment",
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
            "task_manifest_hash",
            "replay_spec_hash",
        )

    def __post_init__(self) -> None:
        require_int(
            self.screening_record_index,
            "accepted_task.screening_record_index",
            minimum=0,
        )
        require_int(self.pr_number, "accepted_task.pr_number", minimum=1)
        require_str(
            self.candidate_id,
            "accepted_task.candidate_id",
            pattern=_CANDIDATE_ID_PATTERN,
        )
        require_str(
            self.candidate_hash,
            "accepted_task.candidate_hash",
            pattern=_HASH_PATTERN,
        )
        require_str(
            self.decision_id,
            "accepted_task.decision_id",
            pattern=_DECISION_ID_PATTERN,
        )
        require_str(
            self.decision_hash,
            "accepted_task.decision_hash",
            pattern=_HASH_PATTERN,
        )
        require_enum(
            self.screening_disposition,
            "accepted_task.screening_disposition",
            _SCREENING_DISPOSITIONS,
        )
        if self.reassessment is not None and not isinstance(
            self.reassessment, FactoryArtifactReference
        ):
            raise ContractError(
                "accepted_task.reassessment: expected artifact reference"
            )
        require_str(self.task_id, "accepted_task.task_id")
        require_str(
            self.public_task_id,
            "accepted_task.public_task_id",
            pattern=_PUBLIC_TASK_ID_PATTERN,
        )
        require_enum(
            self.origin, "accepted_task.origin", _EXPANSION_ORIGINS
        )
        _task_path(self.task_path, "accepted_task.task_path")
        require_str(
            self.task_manifest_hash,
            "accepted_task.task_manifest_hash",
            pattern=_HASH_PATTERN,
        )
        require_str(
            self.replay_spec_hash,
            "accepted_task.replay_spec_hash",
            pattern=_HASH_PATTERN,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "screening_record_index": self.screening_record_index,
            "pr_number": self.pr_number,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "decision_id": self.decision_id,
            "decision_hash": self.decision_hash,
            "screening_disposition": self.screening_disposition,
            "reassessment": (
                None
                if self.reassessment is None
                else self.reassessment.to_dict()
            ),
            "task_id": self.task_id,
            "public_task_id": self.public_task_id,
            "origin": self.origin,
            "task_path": self.task_path,
            "task_manifest_hash": self.task_manifest_hash,
            "replay_spec_hash": self.replay_spec_hash,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_accepted_task_record",
    ) -> "QualityAcceptedTaskRecord":
        data = require_exact_fields(value, path, cls.wire_fields())
        reassessment_value = data["reassessment"]
        reassessment = (
            None
            if reassessment_value is None
            else FactoryArtifactReference.from_dict(
                reassessment_value,
                path=f"{path}.reassessment",
            )
        )
        return cls(
            screening_record_index=require_int(
                data["screening_record_index"],
                f"{path}.screening_record_index",
                minimum=0,
            ),
            pr_number=require_int(
                data["pr_number"], f"{path}.pr_number", minimum=1
            ),
            candidate_id=require_str(
                data["candidate_id"],
                f"{path}.candidate_id",
                pattern=_CANDIDATE_ID_PATTERN,
            ),
            candidate_hash=require_str(
                data["candidate_hash"],
                f"{path}.candidate_hash",
                pattern=_HASH_PATTERN,
            ),
            decision_id=require_str(
                data["decision_id"],
                f"{path}.decision_id",
                pattern=_DECISION_ID_PATTERN,
            ),
            decision_hash=require_str(
                data["decision_hash"],
                f"{path}.decision_hash",
                pattern=_HASH_PATTERN,
            ),
            screening_disposition=require_enum(
                data["screening_disposition"],
                f"{path}.screening_disposition",
                _SCREENING_DISPOSITIONS,
            ),
            reassessment=reassessment,
            task_id=require_str(data["task_id"], f"{path}.task_id"),
            public_task_id=require_str(
                data["public_task_id"],
                f"{path}.public_task_id",
                pattern=_PUBLIC_TASK_ID_PATTERN,
            ),
            origin=require_enum(
                data["origin"], f"{path}.origin", _EXPANSION_ORIGINS
            ),
            task_path=_task_path(
                data["task_path"], f"{path}.task_path"
            ),
            task_manifest_hash=require_str(
                data["task_manifest_hash"],
                f"{path}.task_manifest_hash",
                pattern=_HASH_PATTERN,
            ),
            replay_spec_hash=require_str(
                data["replay_spec_hash"],
                f"{path}.replay_spec_hash",
                pattern=_HASH_PATTERN,
            ),
        )


@dataclass(frozen=True)
class QualityAcceptedTaskIndex:
    """Canonical ordered set of screening-authorized expansion Tasks."""

    contract_type: ClassVar[str] = "quality_accepted_task_index"
    schema_version: ClassVar[str] = "v1"

    created_at: str
    historical_index_path: str
    historical_index_hash: str
    retained_count: int
    required_task_count: int
    candidate_index_path: str
    candidate_index_hash: str
    status: str
    task_count: int
    tasks: tuple[QualityAcceptedTaskRecord, ...]

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "created_at",
            "historical_index_path",
            "historical_index_hash",
            "retained_count",
            "required_task_count",
            "candidate_index_path",
            "candidate_index_hash",
            "status",
            "task_count",
            "tasks",
            "content_hash",
        )

    def __post_init__(self) -> None:
        _utc_seconds(
            self.created_at,
            "accepted_index.created_at",
        )
        _relative_posix_path(
            self.historical_index_path,
            "accepted_index.historical_index_path",
            suffix=".json",
        )
        require_str(
            self.historical_index_hash,
            "accepted_index.historical_index_hash",
            pattern=_HASH_PATTERN,
        )
        require_int(
            self.retained_count,
            "accepted_index.retained_count",
            minimum=0,
        )
        if self.retained_count > 25:
            raise ContractError(
                "accepted_index.retained_count: must be <= 25"
            )
        require_int(
            self.required_task_count,
            "accepted_index.required_task_count",
            minimum=1,
        )
        if self.required_task_count != 50 - self.retained_count:
            raise ContractError(
                "accepted_index.required_task_count: expected 50 - "
                "retained_count"
            )
        _relative_posix_path(
            self.candidate_index_path,
            "accepted_index.candidate_index_path",
            suffix=".json",
        )
        require_str(
            self.candidate_index_hash,
            "accepted_index.candidate_index_hash",
            pattern=_HASH_PATTERN,
        )
        require_enum(
            self.status,
            "accepted_index.status",
            _ACCEPTED_INDEX_STATUSES,
        )
        require_int(
            self.task_count, "accepted_index.task_count", minimum=1
        )
        if self.task_count != len(self.tasks):
            raise ContractError(
                "accepted_index.task_count: must equal tasks length"
            )
        if self.status == "building" and not (
            1 <= self.task_count < self.required_task_count
        ):
            raise ContractError(
                "accepted_index.building: expected 1 through "
                "required_task_count - 1 Tasks"
            )
        if (
            self.status == "complete"
            and self.task_count != self.required_task_count
        ):
            raise ContractError(
                "accepted_index.complete: expected exactly "
                "required_task_count Tasks"
            )
        for record in self.tasks:
            if not isinstance(record, QualityAcceptedTaskRecord):
                raise ContractError(
                    "accepted_index.tasks: expected accepted Task record"
                )
        screening_positions = tuple(
            record.screening_record_index for record in self.tasks
        )
        if screening_positions != tuple(
            sorted(set(screening_positions))
        ):
            raise ContractError(
                "accepted_index.tasks.screening_record_index: "
                "expected strict screening order"
            )
        for label, values in (
            ("pr_number", tuple(record.pr_number for record in self.tasks)),
            ("candidate_id", tuple(record.candidate_id for record in self.tasks)),
            ("decision_id", tuple(record.decision_id for record in self.tasks)),
            ("task_id", tuple(record.task_id for record in self.tasks)),
            (
                "public_task_id",
                tuple(record.public_task_id for record in self.tasks),
            ),
            ("task_path", tuple(record.task_path for record in self.tasks)),
        ):
            if len(values) != len(set(values)):
                raise ContractError(
                    f"accepted_index.tasks.{label}: duplicate"
                )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "historical_index_path": self.historical_index_path,
            "historical_index_hash": self.historical_index_hash,
            "retained_count": self.retained_count,
            "required_task_count": self.required_task_count,
            "candidate_index_path": self.candidate_index_path,
            "candidate_index_hash": self.candidate_index_hash,
            "status": self.status,
            "task_count": self.task_count,
            "tasks": [task.to_dict() for task in self.tasks],
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_accepted_task_index",
    ) -> "QualityAcceptedTaskIndex":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        tasks_value = require_list(data["tasks"], f"{path}.tasks")
        index = cls(
            created_at=_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
            historical_index_path=_relative_posix_path(
                data["historical_index_path"],
                f"{path}.historical_index_path",
                suffix=".json",
            ),
            historical_index_hash=require_str(
                data["historical_index_hash"],
                f"{path}.historical_index_hash",
                pattern=_HASH_PATTERN,
            ),
            retained_count=require_int(
                data["retained_count"],
                f"{path}.retained_count",
                minimum=0,
            ),
            required_task_count=require_int(
                data["required_task_count"],
                f"{path}.required_task_count",
                minimum=1,
            ),
            candidate_index_path=_relative_posix_path(
                data["candidate_index_path"],
                f"{path}.candidate_index_path",
                suffix=".json",
            ),
            candidate_index_hash=require_str(
                data["candidate_index_hash"],
                f"{path}.candidate_index_hash",
                pattern=_HASH_PATTERN,
            ),
            status=require_enum(
                data["status"],
                f"{path}.status",
                _ACCEPTED_INDEX_STATUSES,
            ),
            task_count=require_int(
                data["task_count"], f"{path}.task_count", minimum=1
            ),
            tasks=tuple(
                QualityAcceptedTaskRecord.from_dict(
                    item, path=f"{path}.tasks[{position}]"
                )
                for position, item in enumerate(tasks_value)
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=_HASH_PATTERN,
        )
        if stored_hash != index.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {index.content_hash!r}"
            )
        return index


def _root_path(root: Path, relative: str, path: str) -> Path:
    if not isinstance(root, Path) or not root.is_dir() or root.is_symlink():
        raise ContractError("root: expected real repository directory")
    normalized = _relative_posix_path(relative, path)
    current = root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{path}: symlink is forbidden")
    return current


def _content_hash(
    value: Mapping[str, object],
    path: str,
) -> str:
    stored = require_str(
        value.get("content_hash"),
        f"{path}.content_hash",
        pattern=_HASH_PATTERN,
    )
    computed = canonical_sha256(
        {
            key: item
            for key, item in value.items()
            if key != "content_hash"
        }
    )
    if stored != computed:
        raise ContractError(f"{path}.content_hash: payload mismatch")
    return stored


def _load_screening_artifacts(
    *,
    candidate_index_path: Path,
    screening_entry: Mapping[str, object],
    record: QualityAcceptedTaskRecord,
) -> tuple[QualityCandidateRecord, QualityCandidateDecision]:
    entry_path = (
        f"candidate_index.records[{record.screening_record_index}]"
    )
    entry = require_exact_fields(
        screening_entry,
        entry_path,
        ("pr_number", "candidate", "decision", "disposition"),
    )
    if entry["pr_number"] != record.pr_number:
        raise ContractError(f"{entry_path}.pr_number: accepted mismatch")
    if entry["disposition"] != record.screening_disposition:
        raise ContractError(f"{entry_path}.disposition: accepted mismatch")
    candidate_reference = FactoryArtifactReference.from_dict(
        entry["candidate"], path=f"{entry_path}.candidate"
    )
    decision_reference = FactoryArtifactReference.from_dict(
        entry["decision"], path=f"{entry_path}.decision"
    )
    candidate_path = _root_path(
        candidate_index_path.parent,
        candidate_reference.relative_path,
        f"{entry_path}.candidate.relative_path",
    )
    decision_path = _root_path(
        candidate_index_path.parent,
        decision_reference.relative_path,
        f"{entry_path}.decision.relative_path",
    )
    candidate = load_factory_contract(candidate_path)
    decision = load_factory_contract(decision_path)
    if not isinstance(candidate, QualityCandidateRecord):
        raise ContractError(f"{entry_path}.candidate: wrong contract")
    if not isinstance(decision, QualityCandidateDecision):
        raise ContractError(f"{entry_path}.decision: wrong contract")
    if (
        candidate_reference.artifact_type != candidate.contract_type
        or candidate_reference.artifact_id != candidate.candidate_id
        or candidate_reference.content_hash != candidate.content_hash
    ):
        raise ContractError(f"{entry_path}.candidate: reference mismatch")
    if (
        decision_reference.artifact_type != decision.contract_type
        or decision_reference.artifact_id != decision.decision_id
        or decision_reference.content_hash != decision.content_hash
    ):
        raise ContractError(f"{entry_path}.decision: reference mismatch")
    if (
        candidate.pr_number != record.pr_number
        or candidate.candidate_id != record.candidate_id
        or candidate.content_hash != record.candidate_hash
    ):
        raise ContractError(f"{entry_path}.candidate: accepted mismatch")
    if (
        decision.decision_id != record.decision_id
        or decision.content_hash != record.decision_hash
        or decision.candidate_id != candidate.candidate_id
        or decision.candidate_hash != candidate.content_hash
    ):
        raise ContractError(f"{entry_path}.decision: accepted mismatch")
    if (
        candidate.candidate_status != record.screening_disposition
        or decision.disposition != record.screening_disposition
    ):
        raise ContractError(f"{entry_path}.disposition: source mismatch")
    return candidate, decision


def _validate_reassessment_binding(
    root: Path,
    record: QualityAcceptedTaskRecord,
    candidate: QualityCandidateRecord,
    decision: QualityCandidateDecision,
) -> None:
    reference = record.reassessment
    if record.screening_disposition == "accepted_for_build":
        if reference is not None:
            raise ContractError(
                "accepted_task.reassessment: direct acceptance must be null"
            )
        return
    if record.screening_disposition == "hard_rejected":
        raise ContractError(
            "accepted_task.screening_disposition: hard_rejected is forbidden"
        )
    if reference is None:
        raise ContractError(
            "accepted_task.reassessment: deferred candidate requires review"
        )
    if reference.artifact_type != QualityCandidateReassessment.contract_type:
        raise ContractError(
            "accepted_task.reassessment.artifact_type: mismatch"
        )
    reassessment_path = _root_path(
        root,
        reference.relative_path,
        "accepted_task.reassessment.relative_path",
    )
    reassessment = QualityCandidateReassessment.from_dict(
        load_canonical_json_artifact(reassessment_path)
    )
    expected_artifact_id = (
        "quality-reassessment:v1:"
        + reassessment.content_hash.removeprefix("sha256:")
    )
    if (
        reference.artifact_id != expected_artifact_id
        or reference.content_hash != reassessment.content_hash
    ):
        raise ContractError("accepted_task.reassessment: reference mismatch")
    if (
        reassessment.pr_number != record.pr_number
        or reassessment.candidate_id != candidate.candidate_id
        or reassessment.candidate_hash != candidate.content_hash
        or reassessment.decision_id != decision.decision_id
        or reassessment.decision_hash != decision.content_hash
    ):
        raise ContractError("accepted_task.reassessment: source mismatch")
    if reassessment.deferred_reasons != decision.preliminary_review_reasons:
        raise ContractError(
            "accepted_task.reassessment.deferred_reasons: "
            "screening mismatch"
        )
    if reassessment.decision != "accepted_for_build":
        raise ContractError(
            "accepted_task.reassessment.decision: candidate not accepted"
        )


def _validate_task_binding(
    root: Path,
    record: QualityAcceptedTaskRecord,
    candidate: QualityCandidateRecord,
) -> None:
    task_dir = _root_path(
        root, record.task_path, "accepted_task.task_path"
    )
    if not task_dir.is_dir():
        raise ContractError("accepted_task.task_path: expected directory")
    manifest_path = task_dir / "task.json"
    manifest_data = dict(load_canonical_json_artifact(manifest_path))
    if canonical_sha256(manifest_data) != record.task_manifest_hash:
        raise ContractError(
            "accepted_task.task_manifest_hash: manifest mismatch"
        )
    task = TaskManifest.load(manifest_path)
    if task.task_id != record.task_id:
        raise ContractError("accepted_task.task_id: manifest mismatch")
    try:
        public_task_id = task.public_task_id
    except InvalidPublicTaskId as exc:
        raise ContractError(f"accepted_task.public_task_id: {exc}") from exc
    if public_task_id != record.public_task_id:
        raise ContractError("accepted_task.public_task_id: manifest mismatch")
    quality = require_mapping(
        manifest_data.get("quality"), "accepted_task.manifest.quality"
    )
    if quality.get("origin") != record.origin:
        raise ContractError("accepted_task.origin: manifest mismatch")
    source = require_mapping(
        manifest_data.get("source"), "accepted_task.manifest.source"
    )
    expected_source = {
        "repo": candidate.repository,
        "pr_number": candidate.pr_number,
        "base_commit": candidate.base_commit,
        "merge_commit": candidate.merge_commit,
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise ContractError(
                f"accepted_task.source.{field}: candidate mismatch"
            )
    if replay_spec_hash(task) != record.replay_spec_hash:
        raise ContractError(
            "accepted_task.replay_spec_hash: manifest mismatch"
        )


def load_quality_accepted_task_index(
    root: Path,
    index_path: Path,
    *,
    require_complete: bool = False,
) -> QualityAcceptedTaskIndex:
    """Load and cross-bind a building or final expansion Task index."""

    if require_complete:
        expected = root / PurePosixPath(_OFFICIAL_ACCEPTED_INDEX)
        if index_path.absolute() != expected.absolute():
            raise ContractError(
                "accepted_index: official validation requires "
                f"{_OFFICIAL_ACCEPTED_INDEX}"
            )
    index = QualityAcceptedTaskIndex.from_dict(
        load_canonical_json_artifact(index_path)
    )
    if require_complete and index.status != "complete":
        raise ContractError(
            "accepted_index: official validation requires complete status"
        )
    if require_complete and (
        index.historical_index_path != _OFFICIAL_HISTORICAL_INDEX
        or index.candidate_index_path != _OFFICIAL_CANDIDATE_INDEX
    ):
        raise ContractError(
            "accepted_index: official source index paths are required"
        )

    historical_path = _root_path(
        root,
        index.historical_index_path,
        "accepted_index.historical_index_path",
    )
    historical = load_canonical_json_artifact(historical_path)
    if historical.get("contract_type") != "historical_readmission_index":
        raise ContractError("historical_index.contract_type: mismatch")
    if historical.get("schema_version") != "v1":
        raise ContractError("historical_index.schema_version: mismatch")
    historical_hash = _content_hash(historical, "historical_index")
    if historical_hash != index.historical_index_hash:
        raise ContractError("accepted_index.historical_index_hash: mismatch")
    if historical.get("k") != index.retained_count:
        raise ContractError("accepted_index.retained_count: mismatch")

    candidate_index_path = _root_path(
        root,
        index.candidate_index_path,
        "accepted_index.candidate_index_path",
    )
    candidate_index = load_canonical_json_artifact(candidate_index_path)
    if (
        candidate_index.get("contract_type")
        != "quality_candidate_screening_index"
    ):
        raise ContractError("candidate_index.contract_type: mismatch")
    if candidate_index.get("schema_version") != "v1":
        raise ContractError("candidate_index.schema_version: mismatch")
    candidate_hash = _content_hash(candidate_index, "candidate_index")
    if candidate_hash != index.candidate_index_hash:
        raise ContractError("accepted_index.candidate_index_hash: mismatch")
    if (
        candidate_index.get("historical_index_hash")
        != historical_hash
    ):
        raise ContractError(
            "candidate_index.historical_index_hash: accepted source mismatch"
        )
    if candidate_index.get("historical_k") != index.retained_count:
        raise ContractError(
            "candidate_index.historical_k: retained count mismatch"
        )
    screening_records = require_list(
        candidate_index.get("records"), "candidate_index.records"
    )

    for record in index.tasks:
        if record.screening_record_index >= len(screening_records):
            raise ContractError(
                "accepted_task.screening_record_index: out of range"
            )
        screening_entry = require_mapping(
            screening_records[record.screening_record_index],
            (
                "candidate_index.records"
                f"[{record.screening_record_index}]"
            ),
        )
        candidate, decision = _load_screening_artifacts(
            candidate_index_path=candidate_index_path,
            screening_entry=screening_entry,
            record=record,
        )
        _validate_reassessment_binding(
            root, record, candidate, decision
        )
        _validate_task_binding(root, record, candidate)

    if require_complete:
        historical_errors = validate_historical_index(root, historical_path)
        if historical_errors:
            raise ContractError(
                "accepted_index.historical_index: "
                + "; ".join(historical_errors)
            )
        candidate_errors = validate_candidate_index(
            root, candidate_index_path, require_minimum=True
        )
        if candidate_errors:
            raise ContractError(
                "accepted_index.candidate_index: "
                + "; ".join(candidate_errors)
            )
    return index


def validate_quality_accepted_task_index(
    root: Path,
    index_path: Path,
) -> tuple[str, ...]:
    """Validate only the fixed, complete public expansion index."""

    try:
        load_quality_accepted_task_index(
            root, index_path, require_complete=True
        )
    except (ContractError, OSError, ValueError) as exc:
        return (str(exc),)
    return ()


def validate_quality_admission_prompt(
    task: TaskManifest,
) -> PromptQualityEvidence:
    """Revalidate stored Prompt evidence from the Task's exact live inputs."""

    if not isinstance(task, TaskManifest):
        raise ContractError("task: expected TaskManifest")
    quality = require_mapping(
        task.data.get("quality"), "quality_admission.task.quality"
    )
    prompt_relative = _relative_posix_path(
        quality.get("prompt_evidence"),
        "quality_admission.task.quality.prompt_evidence",
        suffix=".json",
    )
    prompt_path = _root_path(
        task.task_dir,
        prompt_relative,
        "quality_admission.task.quality.prompt_evidence",
    )
    loaded = load_factory_contract(prompt_path)
    if not isinstance(loaded, PromptQualityEvidence):
        raise ContractError(
            "quality_admission.prompt: expected PromptQualityEvidence"
        )
    agent_task_view, private_index = quality_prompt_source_inputs(task)
    validate_prompt_quality_evidence(
        loaded,
        rendered_prompt=render_mcp_prompt(agent_task_view),
        agent_task_view=agent_task_view,
        private_index=private_index,
    )
    if loaded.task_id != task.task_id:
        raise ContractError("quality_admission.prompt.task_id: mismatch")
    try:
        public_task_id = task.public_task_id
    except InvalidPublicTaskId as exc:
        raise ContractError(
            f"quality_admission.prompt.public_task_id: {exc}"
        ) from exc
    if loaded.public_task_id != public_task_id:
        raise ContractError(
            "quality_admission.prompt.public_task_id: mismatch"
        )
    if loaded.decision != "accepted":
        raise ContractError(
            "quality_admission.prompt.decision: accepted required"
        )
    return loaded


def _optional_str(
    value: object,
    path: str,
    *,
    pattern: str | None = None,
) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=pattern)


def _error_tuple(value: object, path: str) -> tuple[str, ...]:
    items = require_list(value, path)
    result = tuple(
        require_str(item, f"{path}[{index}]")
        for index, item in enumerate(items)
    )
    if len(result) != len(set(result)):
        raise ContractError(f"{path}: duplicate error")
    return result


@dataclass(frozen=True)
class QualityAdmissionResultRecord:
    """Canonical outcome of every Task 9 admission gate."""

    screening_record_index: int
    pr_number: int
    task_id: str
    public_task_id: str
    origin: str
    task_path: str
    task_manifest_hash: str
    replay_spec_hash: str
    prompt_evidence_hash: str | None
    prompt_errors: tuple[str, ...]
    preflight_status: str
    preflight_messages_hash: str
    admission_decision: str | None
    admission_verified: bool
    baseline_status: str | None
    gold_status: str | None
    admission_evidence_hash: str | None
    final_quality_errors: tuple[str, ...]
    verified: bool

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "screening_record_index",
            "pr_number",
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
            "task_manifest_hash",
            "replay_spec_hash",
            "prompt_evidence_hash",
            "prompt_errors",
            "preflight_status",
            "preflight_messages_hash",
            "admission_decision",
            "admission_verified",
            "baseline_status",
            "gold_status",
            "admission_evidence_hash",
            "final_quality_errors",
            "verified",
        )

    def __post_init__(self) -> None:
        require_int(
            self.screening_record_index,
            "admission_result.screening_record_index",
            minimum=0,
        )
        require_int(
            self.pr_number, "admission_result.pr_number", minimum=1
        )
        require_str(self.task_id, "admission_result.task_id")
        require_str(
            self.public_task_id,
            "admission_result.public_task_id",
            pattern=_PUBLIC_TASK_ID_PATTERN,
        )
        require_enum(
            self.origin, "admission_result.origin", _EXPANSION_ORIGINS
        )
        _task_path(self.task_path, "admission_result.task_path")
        for value, path in (
            (self.task_manifest_hash, "admission_result.task_manifest_hash"),
            (self.replay_spec_hash, "admission_result.replay_spec_hash"),
            (
                self.preflight_messages_hash,
                "admission_result.preflight_messages_hash",
            ),
        ):
            require_str(value, path, pattern=_HASH_PATTERN)
        _optional_str(
            self.prompt_evidence_hash,
            "admission_result.prompt_evidence_hash",
            pattern=_HASH_PATTERN,
        )
        if not isinstance(self.prompt_errors, tuple) or len(
            self.prompt_errors
        ) != len(set(self.prompt_errors)):
            raise ContractError(
                "admission_result.prompt_errors: expected unique tuple"
            )
        for error in self.prompt_errors:
            require_str(error, "admission_result.prompt_errors[]")
        require_enum(
            self.preflight_status,
            "admission_result.preflight_status",
            _PREFLIGHT_STATUSES,
        )
        _optional_str(
            self.admission_decision,
            "admission_result.admission_decision",
        )
        require_bool(
            self.admission_verified,
            "admission_result.admission_verified",
        )
        _optional_str(
            self.baseline_status, "admission_result.baseline_status"
        )
        _optional_str(self.gold_status, "admission_result.gold_status")
        _optional_str(
            self.admission_evidence_hash,
            "admission_result.admission_evidence_hash",
            pattern=_HASH_PATTERN,
        )
        if not isinstance(self.final_quality_errors, tuple) or len(
            self.final_quality_errors
        ) != len(set(self.final_quality_errors)):
            raise ContractError(
                "admission_result.final_quality_errors: "
                "expected unique tuple"
            )
        for error in self.final_quality_errors:
            require_str(error, "admission_result.final_quality_errors[]")
        require_bool(self.verified, "admission_result.verified")

        prompt_passed = (
            self.prompt_evidence_hash is not None
            and not self.prompt_errors
        )
        preflight_passed = self.preflight_status == "passed"
        admission_passed = (
            self.admission_decision == "verified"
            and self.admission_verified
            and self.baseline_status == "baseline_reproduced"
            and self.gold_status == "resolved"
            and self.admission_evidence_hash is not None
        )
        quality_passed = not self.final_quality_errors
        expected_verified = (
            prompt_passed
            and preflight_passed
            and admission_passed
            and quality_passed
        )
        if self.verified != expected_verified:
            raise ContractError(
                "admission_result.verified: must equal all four gates"
            )
        if self.prompt_errors and self.preflight_status != "not_run":
            raise ContractError(
                "admission_result.preflight_status: "
                "Prompt failure requires not_run"
            )
        if self.preflight_status != "passed" and (
            self.admission_decision is not None
            or self.baseline_status is not None
            or self.gold_status is not None
            or self.admission_evidence_hash is not None
            or self.admission_verified
        ):
            raise ContractError(
                "admission_result: Admission cannot run before preflight"
            )
        if self.admission_decision is None and (
            self.baseline_status is not None
            or self.gold_status is not None
            or self.admission_evidence_hash is not None
            or self.admission_verified
        ):
            raise ContractError(
                "admission_result: partial Admission outcome"
            )
        if self.admission_verified != (
            self.admission_decision == "verified"
        ):
            raise ContractError(
                "admission_result.admission_verified: decision mismatch"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "screening_record_index": self.screening_record_index,
            "pr_number": self.pr_number,
            "task_id": self.task_id,
            "public_task_id": self.public_task_id,
            "origin": self.origin,
            "task_path": self.task_path,
            "task_manifest_hash": self.task_manifest_hash,
            "replay_spec_hash": self.replay_spec_hash,
            "prompt_evidence_hash": self.prompt_evidence_hash,
            "prompt_errors": list(self.prompt_errors),
            "preflight_status": self.preflight_status,
            "preflight_messages_hash": self.preflight_messages_hash,
            "admission_decision": self.admission_decision,
            "admission_verified": self.admission_verified,
            "baseline_status": self.baseline_status,
            "gold_status": self.gold_status,
            "admission_evidence_hash": self.admission_evidence_hash,
            "final_quality_errors": list(self.final_quality_errors),
            "verified": self.verified,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_admission_result_record",
    ) -> "QualityAdmissionResultRecord":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            screening_record_index=require_int(
                data["screening_record_index"],
                f"{path}.screening_record_index",
                minimum=0,
            ),
            pr_number=require_int(
                data["pr_number"], f"{path}.pr_number", minimum=1
            ),
            task_id=require_str(data["task_id"], f"{path}.task_id"),
            public_task_id=require_str(
                data["public_task_id"],
                f"{path}.public_task_id",
                pattern=_PUBLIC_TASK_ID_PATTERN,
            ),
            origin=require_enum(
                data["origin"], f"{path}.origin", _EXPANSION_ORIGINS
            ),
            task_path=_task_path(
                data["task_path"], f"{path}.task_path"
            ),
            task_manifest_hash=require_str(
                data["task_manifest_hash"],
                f"{path}.task_manifest_hash",
                pattern=_HASH_PATTERN,
            ),
            replay_spec_hash=require_str(
                data["replay_spec_hash"],
                f"{path}.replay_spec_hash",
                pattern=_HASH_PATTERN,
            ),
            prompt_evidence_hash=_optional_str(
                data["prompt_evidence_hash"],
                f"{path}.prompt_evidence_hash",
                pattern=_HASH_PATTERN,
            ),
            prompt_errors=_error_tuple(
                data["prompt_errors"], f"{path}.prompt_errors"
            ),
            preflight_status=require_enum(
                data["preflight_status"],
                f"{path}.preflight_status",
                _PREFLIGHT_STATUSES,
            ),
            preflight_messages_hash=require_str(
                data["preflight_messages_hash"],
                f"{path}.preflight_messages_hash",
                pattern=_HASH_PATTERN,
            ),
            admission_decision=_optional_str(
                data["admission_decision"],
                f"{path}.admission_decision",
            ),
            admission_verified=require_bool(
                data["admission_verified"],
                f"{path}.admission_verified",
            ),
            baseline_status=_optional_str(
                data["baseline_status"], f"{path}.baseline_status"
            ),
            gold_status=_optional_str(
                data["gold_status"], f"{path}.gold_status"
            ),
            admission_evidence_hash=_optional_str(
                data["admission_evidence_hash"],
                f"{path}.admission_evidence_hash",
                pattern=_HASH_PATTERN,
            ),
            final_quality_errors=_error_tuple(
                data["final_quality_errors"],
                f"{path}.final_quality_errors",
            ),
            verified=require_bool(data["verified"], f"{path}.verified"),
        )


@dataclass(frozen=True)
class QualityAdmissionResultIndex:
    """Canonical ordered results bound to accepted Tasks and registries."""

    contract_type: ClassVar[str] = "quality_admission_result_index"
    schema_version: ClassVar[str] = "v1"

    created_at: str
    accepted_index_path: str
    accepted_index_hash: str
    environment_registry_path: str
    environment_registry_hash: str
    source_registry_path: str
    source_registry_hash: str
    task_count: int
    verified_count: int
    results: tuple[QualityAdmissionResultRecord, ...]

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "created_at",
            "accepted_index_path",
            "accepted_index_hash",
            "environment_registry_path",
            "environment_registry_hash",
            "source_registry_path",
            "source_registry_hash",
            "task_count",
            "verified_count",
            "results",
            "content_hash",
        )

    def __post_init__(self) -> None:
        _utc_seconds(
            self.created_at,
            "admission_results.created_at",
        )
        for value, path in (
            (
                self.accepted_index_path,
                "admission_results.accepted_index_path",
            ),
            (
                self.environment_registry_path,
                "admission_results.environment_registry_path",
            ),
            (
                self.source_registry_path,
                "admission_results.source_registry_path",
            ),
        ):
            _relative_posix_path(value, path, suffix=".json")
        for value, path in (
            (
                self.accepted_index_hash,
                "admission_results.accepted_index_hash",
            ),
            (
                self.environment_registry_hash,
                "admission_results.environment_registry_hash",
            ),
            (
                self.source_registry_hash,
                "admission_results.source_registry_hash",
            ),
        ):
            require_str(value, path, pattern=_HASH_PATTERN)
        require_int(
            self.task_count, "admission_results.task_count", minimum=1
        )
        require_int(
            self.verified_count,
            "admission_results.verified_count",
            minimum=0,
        )
        if self.task_count != len(self.results):
            raise ContractError(
                "admission_results.task_count: must equal results length"
            )
        actual_verified = sum(result.verified for result in self.results)
        if self.verified_count != actual_verified:
            raise ContractError(
                "admission_results.verified_count: mismatch"
            )
        for result in self.results:
            if not isinstance(result, QualityAdmissionResultRecord):
                raise ContractError(
                    "admission_results.results: expected result record"
                )
        positions = tuple(
            result.screening_record_index for result in self.results
        )
        if positions != tuple(sorted(set(positions))):
            raise ContractError(
                "admission_results.results: expected strict screening order"
            )
        for label, values in (
            ("task_id", tuple(result.task_id for result in self.results)),
            (
                "public_task_id",
                tuple(result.public_task_id for result in self.results),
            ),
            ("task_path", tuple(result.task_path for result in self.results)),
        ):
            if len(values) != len(set(values)):
                raise ContractError(
                    f"admission_results.results.{label}: duplicate"
                )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "accepted_index_path": self.accepted_index_path,
            "accepted_index_hash": self.accepted_index_hash,
            "environment_registry_path": self.environment_registry_path,
            "environment_registry_hash": self.environment_registry_hash,
            "source_registry_path": self.source_registry_path,
            "source_registry_hash": self.source_registry_hash,
            "task_count": self.task_count,
            "verified_count": self.verified_count,
            "results": [result.to_dict() for result in self.results],
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_admission_result_index",
    ) -> "QualityAdmissionResultIndex":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        results_value = require_list(data["results"], f"{path}.results")
        index = cls(
            created_at=_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
            accepted_index_path=_relative_posix_path(
                data["accepted_index_path"],
                f"{path}.accepted_index_path",
                suffix=".json",
            ),
            accepted_index_hash=require_str(
                data["accepted_index_hash"],
                f"{path}.accepted_index_hash",
                pattern=_HASH_PATTERN,
            ),
            environment_registry_path=_relative_posix_path(
                data["environment_registry_path"],
                f"{path}.environment_registry_path",
                suffix=".json",
            ),
            environment_registry_hash=require_str(
                data["environment_registry_hash"],
                f"{path}.environment_registry_hash",
                pattern=_HASH_PATTERN,
            ),
            source_registry_path=_relative_posix_path(
                data["source_registry_path"],
                f"{path}.source_registry_path",
                suffix=".json",
            ),
            source_registry_hash=require_str(
                data["source_registry_hash"],
                f"{path}.source_registry_hash",
                pattern=_HASH_PATTERN,
            ),
            task_count=require_int(
                data["task_count"], f"{path}.task_count", minimum=1
            ),
            verified_count=require_int(
                data["verified_count"],
                f"{path}.verified_count",
                minimum=0,
            ),
            results=tuple(
                QualityAdmissionResultRecord.from_dict(
                    item, path=f"{path}.results[{position}]"
                )
                for position, item in enumerate(results_value)
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=_HASH_PATTERN,
        )
        if stored_hash != index.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {index.content_hash!r}"
            )
        return index


def _bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _repository_relative(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.absolute().relative_to(root.absolute())
    except ValueError as exc:
        raise ContractError(
            f"{label}: expected path inside repository root"
        ) from exc
    return _relative_posix_path(
        relative.as_posix(), label, suffix=".json"
    )


def _write_canonical_file(
    path: Path,
    value: object,
    *,
    root: Path,
) -> None:
    absolute_path = path.absolute()
    absolute_root = root.absolute()
    try:
        absolute_path.relative_to(absolute_root)
    except ValueError as exc:
        raise ContractError("output path: outside write root") from exc
    current = absolute_path
    while current != absolute_root:
        if current.is_symlink():
            raise ContractError("output path: symlink is forbidden")
        current = current.parent
    if absolute_root.is_symlink():
        raise ContractError("output root: symlink is forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _rebind_readmission(
    task: TaskManifest,
    admission_evidence_path: Path,
) -> None:
    quality = require_mapping(
        task.data.get("quality"), "quality_admission.task.quality"
    )
    relative = _relative_posix_path(
        quality.get("readmission_evidence"),
        "quality_admission.task.quality.readmission_evidence",
        suffix=".json",
    )
    path = _root_path(
        task.task_dir,
        relative,
        "quality_admission.task.quality.readmission_evidence",
    )
    payload = dict(load_canonical_json_artifact(path))
    payload["admission_evidence_hash"] = _bytes_hash(
        load_regular_file_bytes(admission_evidence_path)
    )
    payload["content_hash"] = canonical_sha256(
        {
            key: item
            for key, item in payload.items()
            if key != "content_hash"
        }
    )
    _write_canonical_file(path, payload, root=task.task_dir)


def _not_run_errors(gate: str) -> tuple[str, ...]:
    return (f"quality_validation: not run because {gate} failed",)


def run_quality_admission(
    *,
    root: Path,
    accepted_index_path: Path,
    output_path: Path,
    environment_registry_path: Path,
    source_registry_path: Path,
    created_at: str,
    preflight: Callable[[Path], tuple[bool, list[str]]] | None = None,
    admission_runner: AdmissionRunner | None = None,
) -> QualityAdmissionResultIndex:
    """Run the existing gates for every Task path in the accepted index."""

    _utc_seconds(
        created_at,
        "quality_admission.created_at",
    )
    _repository_relative(
        root, output_path, "quality_admission.output_path"
    )
    accepted = load_quality_accepted_task_index(
        root, accepted_index_path, require_complete=False
    )
    accepted_relative = _repository_relative(
        root, accepted_index_path, "quality_admission.accepted_index_path"
    )
    environment_relative = _repository_relative(
        root,
        environment_registry_path,
        "quality_admission.environment_registry_path",
    )
    source_relative = _repository_relative(
        root,
        source_registry_path,
        "quality_admission.source_registry_path",
    )
    environment_bytes = load_regular_file_bytes(environment_registry_path)
    source_bytes = load_regular_file_bytes(source_registry_path)

    if preflight is None:
        from scripts.preflight_task import preflight_task

        preflight = preflight_task
    if admission_runner is None:
        parsed = datetime.fromisoformat(
            created_at.removesuffix("Z") + "+00:00"
        )
        admission_runner = AdmissionRunner(
            now=lambda: parsed.astimezone(timezone.utc)
        )

    results: list[QualityAdmissionResultRecord] = []
    for accepted_record in accepted.tasks:
        task_dir = _root_path(
            root,
            accepted_record.task_path,
            "quality_admission.accepted_task.task_path",
        )
        prompt_hash: str | None = None
        prompt_errors: tuple[str, ...] = ()
        task: TaskManifest | None = None
        try:
            task = load_resolved_task(
                task_dir / "task.json",
                environment_registry_path=environment_registry_path,
                source_registry_path=source_registry_path,
            )
            prompt = validate_quality_admission_prompt(task)
            prompt_hash = prompt.content_hash
        except (ContractError, OSError, UnicodeDecodeError, ValueError) as exc:
            prompt_errors = (f"quality_admission.task: {exc}",)

        preflight_status = "not_run"
        preflight_messages: list[str] = []
        if not prompt_errors:
            try:
                preflight_ok, preflight_messages = preflight(task_dir)
                preflight_status = "passed" if preflight_ok else "failed"
            except (OSError, RuntimeError, ValueError) as exc:
                preflight_status = "failed"
                preflight_messages = [f"preflight exception: {exc}"]
        preflight_messages_hash = canonical_sha256(preflight_messages)

        admission_decision: str | None = None
        admission_verified = False
        baseline_status: str | None = None
        gold_status: str | None = None
        admission_hash: str | None = None
        final_errors: tuple[str, ...]
        if prompt_errors:
            final_errors = _not_run_errors("Prompt")
        elif preflight_status != "passed":
            final_errors = _not_run_errors("preflight")
        else:
            assert task is not None
            try:
                evidence = admission_runner.run(task)
                admission_decision = evidence.decision
                admission_verified = evidence.verified
                baseline_status = str(evidence.baseline["status"])
                gold_status = (
                    None
                    if evidence.gold is None
                    else str(evidence.gold["status"])
                )
                stable_path = admission_runner.write_task_evidence(
                    task, evidence
                )
                admission_hash = _bytes_hash(
                    load_regular_file_bytes(stable_path)
                )
                _rebind_readmission(task, stable_path)
                final_errors = tuple(
                    validate_quality_task(
                        root, task, require_verified=True
                    )
                )
            except (
                ContractError,
                OSError,
                RuntimeError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                final_errors = (f"quality_admission: {exc}",)

        verified = (
            prompt_hash is not None
            and not prompt_errors
            and preflight_status == "passed"
            and admission_decision == "verified"
            and admission_verified
            and baseline_status == "baseline_reproduced"
            and gold_status == "resolved"
            and admission_hash is not None
            and not final_errors
        )
        results.append(
            QualityAdmissionResultRecord(
                screening_record_index=(
                    accepted_record.screening_record_index
                ),
                pr_number=accepted_record.pr_number,
                task_id=accepted_record.task_id,
                public_task_id=accepted_record.public_task_id,
                origin=accepted_record.origin,
                task_path=accepted_record.task_path,
                task_manifest_hash=accepted_record.task_manifest_hash,
                replay_spec_hash=accepted_record.replay_spec_hash,
                prompt_evidence_hash=prompt_hash,
                prompt_errors=prompt_errors,
                preflight_status=preflight_status,
                preflight_messages_hash=preflight_messages_hash,
                admission_decision=admission_decision,
                admission_verified=admission_verified,
                baseline_status=baseline_status,
                gold_status=gold_status,
                admission_evidence_hash=admission_hash,
                final_quality_errors=tuple(dict.fromkeys(final_errors)),
                verified=verified,
            )
        )

    result_index = QualityAdmissionResultIndex(
        created_at=created_at,
        accepted_index_path=accepted_relative,
        accepted_index_hash=accepted.content_hash,
        environment_registry_path=environment_relative,
        environment_registry_hash=_bytes_hash(environment_bytes),
        source_registry_path=source_relative,
        source_registry_hash=_bytes_hash(source_bytes),
        task_count=len(results),
        verified_count=sum(result.verified for result in results),
        results=tuple(results),
    )
    _write_canonical_file(
        output_path, result_index.to_dict(), root=root
    )
    return result_index


def load_quality_admission_result_index(
    root: Path,
    result_path: Path,
    accepted_index_path: Path,
) -> QualityAdmissionResultIndex:
    """Load result bytes and rebind every outcome to its exact inputs."""

    result = QualityAdmissionResultIndex.from_dict(
        load_canonical_json_artifact(result_path)
    )
    expected_accepted_relative = _repository_relative(
        root,
        accepted_index_path,
        "admission_results.accepted_index_path",
    )
    if result.accepted_index_path != expected_accepted_relative:
        raise ContractError(
            "admission_results.accepted_index_path: requested mismatch"
        )
    accepted = load_quality_accepted_task_index(
        root, accepted_index_path, require_complete=False
    )
    if result.accepted_index_hash != accepted.content_hash:
        raise ContractError(
            "admission_results.accepted_index_hash: mismatch"
        )
    environment_registry_path = _root_path(
        root,
        result.environment_registry_path,
        "admission_results.environment_registry_path",
    )
    source_registry_path = _root_path(
        root,
        result.source_registry_path,
        "admission_results.source_registry_path",
    )
    if result.environment_registry_hash != _bytes_hash(
        load_regular_file_bytes(environment_registry_path)
    ):
        raise ContractError(
            "admission_results.environment_registry_hash: mismatch"
        )
    if result.source_registry_hash != _bytes_hash(
        load_regular_file_bytes(source_registry_path)
    ):
        raise ContractError(
            "admission_results.source_registry_hash: mismatch"
        )
    if len(result.results) != len(accepted.tasks):
        raise ContractError("admission_results: accepted count mismatch")
    for position, (outcome, authorized) in enumerate(
        zip(result.results, accepted.tasks)
    ):
        for field in (
            "screening_record_index",
            "pr_number",
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
            "task_manifest_hash",
            "replay_spec_hash",
        ):
            if getattr(outcome, field) != getattr(authorized, field):
                raise ContractError(
                    f"admission_results.results[{position}].{field}: "
                    "accepted mismatch"
                )
        task_dir = _root_path(
            root,
            outcome.task_path,
            f"admission_results.results[{position}].task_path",
        )
        if outcome.prompt_evidence_hash is not None:
            task = TaskManifest.load(task_dir / "task.json")
            quality = require_mapping(
                task.data.get("quality"),
                f"admission_results.results[{position}].quality",
            )
            prompt_path = _root_path(
                task_dir,
                _relative_posix_path(
                    quality.get("prompt_evidence"),
                    (
                        f"admission_results.results[{position}]"
                        ".prompt_evidence"
                    ),
                    suffix=".json",
                ),
                (
                    f"admission_results.results[{position}]"
                    ".prompt_evidence"
                ),
            )
            prompt = load_factory_contract(prompt_path)
            if (
                not isinstance(prompt, PromptQualityEvidence)
                or prompt.content_hash != outcome.prompt_evidence_hash
            ):
                raise ContractError(
                    f"admission_results.results[{position}]"
                    ".prompt_evidence_hash: mismatch"
                )
        if outcome.admission_evidence_hash is not None:
            stable_path = task_dir / "admission/evidence.json"
            stable_bytes = load_regular_file_bytes(stable_path)
            if outcome.admission_evidence_hash != _bytes_hash(stable_bytes):
                raise ContractError(
                    f"admission_results.results[{position}]"
                    ".admission_evidence_hash: mismatch"
                )
            try:
                stable_value = json.loads(stable_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"admission_results.results[{position}]"
                    ".admission_evidence: invalid JSON"
                ) from exc
            stable = require_mapping(
                stable_value,
                f"admission_results.results[{position}]"
                ".admission_evidence",
            )
            admission = require_mapping(
                stable.get("admission"),
                f"admission_results.results[{position}]"
                ".admission_evidence.admission",
            )
            baseline = require_mapping(
                stable.get("baseline"),
                f"admission_results.results[{position}]"
                ".admission_evidence.baseline",
            )
            gold_value = stable.get("gold")
            gold = (
                None
                if gold_value is None
                else require_mapping(
                    gold_value,
                    f"admission_results.results[{position}]"
                    ".admission_evidence.gold",
                )
            )
            expected_truth = {
                "task_id": outcome.task_id,
                "admission_decision": outcome.admission_decision,
                "admission_verified": outcome.admission_verified,
                "baseline_status": outcome.baseline_status,
                "gold_status": outcome.gold_status,
            }
            actual_truth = {
                "task_id": stable.get("task_id"),
                "admission_decision": admission.get("decision"),
                "admission_verified": admission.get("verified"),
                "baseline_status": baseline.get("status"),
                "gold_status": (
                    None if gold is None else gold.get("status")
                ),
            }
            if actual_truth != expected_truth:
                raise ContractError(
                    f"admission_results.results[{position}]"
                    ".admission_evidence: outcome mismatch"
                )
    return result
