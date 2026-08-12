"""Deterministic complexity and semantic-duplicate evidence for Factory admission."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import re
from types import MappingProxyType
from typing import ClassVar

from op_bench.factory.taxonomy import TaskTaxonomyV2
from op_bench.runtime.canonical import JsonValue, canonical_sha256
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


HARD_REJECTIONS = (
    "mechanical_after_sanitization",
    "no_public_contract_impact",
    "non_behavioral_change",
    "semantic_duplicate",
    "standard_admission_failure",
)
RISK_SIGNALS = (
    "gold_patch_lte_4_lines",
    "gold_patch_single_file",
    "single_hidden_f2p",
    "reference_agent_all_resolved",
    "short_repair_time",
    "low_tool_or_read_count",
)
_DIMENSIONS = ("localization", "diagnosis", "repair_regression")
_SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
_UTC_SECONDS_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)


def _validate_utc_seconds(value: object, path: str) -> str:
    text = require_str(value, path)
    if _UTC_SECONDS_PATTERN.fullmatch(text) is None:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: invalid UTC timestamp") from exc
    return text


def _score(value: object, path: str) -> int:
    score = require_int(value, path, minimum=0)
    if score > 2:
        raise ContractError(f"{path}: must be <= 2")
    return score


def _registry_tuple(
    value: object,
    *,
    path: str,
    registry: tuple[str, ...],
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ContractError(f"{path}: expected tuple")
    seen: set[str] = set()
    for index, item in enumerate(value):
        code = require_enum(item, f"{path}[{index}]", registry)
        if code in seen:
            raise ContractError(f"{path}: duplicate value {code!r}")
        seen.add(code)
    return tuple(code for code in registry if code in seen)


def _registry_list(
    value: object,
    *,
    path: str,
    registry: tuple[str, ...],
) -> tuple[str, ...]:
    items = require_list(value, path)
    return _registry_tuple(tuple(items), path=path, registry=registry)


def _dimension_evidence(value: object, *, path: str) -> Mapping[str, str]:
    data = require_exact_fields(value, path, _DIMENSIONS)
    normalized: dict[str, str] = {}
    for dimension in _DIMENSIONS:
        written = require_str(data[dimension], f"{path}.{dimension}")
        if not written.strip():
            raise ContractError(f"{path}.{dimension}: expected written evidence")
        normalized[dimension] = written.strip()
    return MappingProxyType(normalized)


def _blind_pilot(value: object, *, path: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    data = require_mapping(value, path)
    canonical_sha256(data)
    require_enum(data.get("decision"), f"{path}.decision", ("accepted", "deferred", "rejected"))
    require_bool(data.get("counts_toward_final"), f"{path}.counts_toward_final")
    return MappingProxyType(data)


def _accepted_pilot(blind_pilot: Mapping[str, object] | None) -> bool:
    return (
        blind_pilot is not None
        and blind_pilot["decision"] == "accepted"
        and blind_pilot["counts_toward_final"] is False
    )


def _derived_decision(
    *,
    total: int,
    hard_rejections: tuple[str, ...],
    blind_pilot: Mapping[str, object] | None,
    second_review: bool,
) -> tuple[str, str | None]:
    if hard_rejections or total <= 3:
        return "rejected", None
    if total == 4:
        if _accepted_pilot(blind_pilot) and second_review:
            return "accepted", "medium"
        return "deferred", None
    return "accepted", "hard"


def semantic_duplicate_fingerprint(
    taxonomy: TaskTaxonomyV2,
    behavior_tokens: Sequence[str],
) -> str:
    """Hash normalized behavior facts without identifiers or answer-side provenance."""

    if not isinstance(taxonomy, TaskTaxonomyV2):
        raise ContractError("taxonomy: expected TaskTaxonomyV2")
    if isinstance(behavior_tokens, str) or not isinstance(behavior_tokens, Sequence):
        raise ContractError("behavior_tokens: expected string sequence")
    tokens = sorted(
        {
            require_str(token, f"behavior_tokens[{index}]")
            for index, token in enumerate(behavior_tokens)
        }
    )
    return canonical_sha256(
        {
            "contract_family": taxonomy.contract_family,
            "failure_type": taxonomy.failure_type,
            "triggers": list(taxonomy.trigger_tags),
            "contexts": {
                "devices": list(taxonomy.execution_context.devices),
                "modes": list(taxonomy.execution_context.modes),
                "phases": list(taxonomy.execution_context.phases),
                "distributed": taxonomy.execution_context.distributed,
            },
            "behavior_tokens": tokens,
        }
    )


@dataclass(frozen=True, init=False)
class ComplexityEvidence:
    contract_type: ClassVar[str] = "complexity_evidence"
    schema_version: ClassVar[str] = "v1"

    task_id: str
    localization: int
    diagnosis: int
    repair_regression: int
    dimension_evidence: Mapping[str, str]
    hard_rejections: tuple[str, ...]
    risk_signals: tuple[str, ...]
    duplicate_fingerprint: str
    duplicate_decision: str
    blind_pilot: Mapping[str, object] | None
    second_review: bool
    reviewer: str
    reviewed_at: str
    total: int
    difficulty: str | None
    decision: str
    content_hash: str = ""

    def __init__(
        self,
        *,
        task_id: str,
        localization: int,
        diagnosis: int,
        repair_regression: int,
        dimension_evidence: Mapping[str, str],
        hard_rejections: tuple[str, ...],
        risk_signals: tuple[str, ...],
        duplicate_fingerprint: str,
        duplicate_decision: str,
        blind_pilot: Mapping[str, object] | None,
        second_review: bool,
        reviewer: str,
        reviewed_at: str,
    ) -> None:
        """Construct evidence with all admission fields derived from review inputs."""

        hard = _registry_tuple(
            hard_rejections,
            path="complexity_evidence.hard_rejections",
            registry=HARD_REJECTIONS,
        )
        duplicate = require_enum(
            duplicate_decision,
            "complexity_evidence.duplicate_decision",
            ("distinct", "duplicate"),
        )
        if duplicate == "duplicate":
            hard = tuple(
                code for code in HARD_REJECTIONS if code in {*hard, "semantic_duplicate"}
            )
        localization_score = _score(localization, "complexity_evidence.localization")
        diagnosis_score = _score(diagnosis, "complexity_evidence.diagnosis")
        repair_score = _score(
            repair_regression,
            "complexity_evidence.repair_regression",
        )
        pilot = _blind_pilot(blind_pilot, path="complexity_evidence.blind_pilot")
        review_complete = require_bool(second_review, "complexity_evidence.second_review")
        total = localization_score + diagnosis_score + repair_score
        decision, difficulty = _derived_decision(
            total=total,
            hard_rejections=hard,
            blind_pilot=pilot,
            second_review=review_complete,
        )
        for name, value in (
            ("task_id", task_id),
            ("localization", localization_score),
            ("diagnosis", diagnosis_score),
            ("repair_regression", repair_score),
            (
                "dimension_evidence",
                _dimension_evidence(dimension_evidence, path="complexity_evidence.dimension_evidence"),
            ),
            ("hard_rejections", hard),
            (
                "risk_signals",
                _registry_tuple(
                    risk_signals,
                    path="complexity_evidence.risk_signals",
                    registry=RISK_SIGNALS,
                ),
            ),
            ("duplicate_fingerprint", duplicate_fingerprint),
            ("duplicate_decision", duplicate),
            ("blind_pilot", pilot),
            ("second_review", review_complete),
            ("reviewer", reviewer),
            ("reviewed_at", reviewed_at),
            ("total", total),
            ("difficulty", difficulty),
            ("decision", decision),
            ("content_hash", ""),
        ):
            object.__setattr__(self, name, value)
        self._validate_stored()

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "task_id",
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
            "reviewed_at",
            "total",
            "difficulty",
            "decision",
            "content_hash",
        )

    def _validate_stored(self) -> None:
        require_str(self.task_id, "complexity_evidence.task_id")
        localization = _score(self.localization, "complexity_evidence.localization")
        diagnosis = _score(self.diagnosis, "complexity_evidence.diagnosis")
        repair = _score(self.repair_regression, "complexity_evidence.repair_regression")
        evidence = _dimension_evidence(
            self.dimension_evidence,
            path="complexity_evidence.dimension_evidence",
        )
        hard = _registry_tuple(
            self.hard_rejections,
            path="complexity_evidence.hard_rejections",
            registry=HARD_REJECTIONS,
        )
        risks = _registry_tuple(
            self.risk_signals,
            path="complexity_evidence.risk_signals",
            registry=RISK_SIGNALS,
        )
        fingerprint = require_str(
            self.duplicate_fingerprint,
            "complexity_evidence.duplicate_fingerprint",
            pattern=_SHA256_PATTERN,
        )
        duplicate = require_enum(
            self.duplicate_decision,
            "complexity_evidence.duplicate_decision",
            ("distinct", "duplicate"),
        )
        if duplicate == "duplicate" and "semantic_duplicate" not in hard:
            raise ContractError(
                "complexity_evidence.hard_rejections: duplicate requires semantic_duplicate"
            )
        if duplicate == "distinct" and "semantic_duplicate" in hard:
            raise ContractError(
                "complexity_evidence.hard_rejections: semantic_duplicate requires duplicate"
            )
        pilot = _blind_pilot(self.blind_pilot, path="complexity_evidence.blind_pilot")
        second_review = require_bool(
            self.second_review,
            "complexity_evidence.second_review",
        )
        require_str(self.reviewer, "complexity_evidence.reviewer")
        reviewed_at = _validate_utc_seconds(
            self.reviewed_at,
            "complexity_evidence.reviewed_at",
        )
        total = localization + diagnosis + repair
        if self.total != total:
            raise ContractError("complexity_evidence.total: does not match scores")
        decision, difficulty = _derived_decision(
            total=total,
            hard_rejections=hard,
            blind_pilot=pilot,
            second_review=second_review,
        )
        if self.decision != decision:
            raise ContractError("complexity_evidence.decision: does not match admission gates")
        if self.difficulty != difficulty:
            raise ContractError("complexity_evidence.difficulty: does not match admission gates")
        object.__setattr__(self, "dimension_evidence", evidence)
        object.__setattr__(self, "hard_rejections", hard)
        object.__setattr__(self, "risk_signals", risks)
        object.__setattr__(self, "blind_pilot", pilot)
        object.__setattr__(self, "second_review", second_review)
        object.__setattr__(self, "reviewed_at", reviewed_at)
        object.__setattr__(self, "duplicate_fingerprint", fingerprint)
        expected_hash = canonical_sha256(self._payload_without_hash())
        if self.content_hash == "":
            object.__setattr__(self, "content_hash", expected_hash)
        elif require_str(
            self.content_hash,
            "complexity_evidence.content_hash",
            pattern=_SHA256_PATTERN,
        ) != expected_hash:
            raise ContractError("complexity_evidence.content_hash: payload hash mismatch")

    def _payload_without_hash(self) -> dict[str, JsonValue]:
        return {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "localization": self.localization,
            "diagnosis": self.diagnosis,
            "repair_regression": self.repair_regression,
            "dimension_evidence": dict(self.dimension_evidence),
            "hard_rejections": list(self.hard_rejections),
            "risk_signals": list(self.risk_signals),
            "duplicate_fingerprint": self.duplicate_fingerprint,
            "duplicate_decision": self.duplicate_decision,
            "blind_pilot": None if self.blind_pilot is None else dict(self.blind_pilot),
            "second_review": self.second_review,
            "reviewer": self.reviewer,
            "reviewed_at": self.reviewed_at,
            "total": self.total,
            "difficulty": self.difficulty,
            "decision": self.decision,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        payload = self._payload_without_hash()
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "complexity_evidence",
    ) -> "ComplexityEvidence":
        data = require_exact_fields(value, path, cls.wire_fields())
        if require_str(data["contract_type"], f"{path}.contract_type") != cls.contract_type:
            raise ContractError(f"{path}.contract_type: expected {cls.contract_type!r}")
        if require_str(data["schema_version"], f"{path}.schema_version") != cls.schema_version:
            raise ContractError(f"{path}.schema_version: expected {cls.schema_version!r}")
        evidence = object.__new__(cls)
        for name, item in (
            ("task_id", require_str(data["task_id"], f"{path}.task_id")),
            ("localization", _score(data["localization"], f"{path}.localization")),
            ("diagnosis", _score(data["diagnosis"], f"{path}.diagnosis")),
            (
                "repair_regression",
                _score(data["repair_regression"], f"{path}.repair_regression"),
            ),
            ("dimension_evidence", _dimension_evidence(data["dimension_evidence"], path=f"{path}.dimension_evidence")),
            ("hard_rejections", _registry_list(data["hard_rejections"], path=f"{path}.hard_rejections", registry=HARD_REJECTIONS)),
            ("risk_signals", _registry_list(data["risk_signals"], path=f"{path}.risk_signals", registry=RISK_SIGNALS)),
            (
                "duplicate_fingerprint",
                require_str(data["duplicate_fingerprint"], f"{path}.duplicate_fingerprint"),
            ),
            ("duplicate_decision", require_str(data["duplicate_decision"], f"{path}.duplicate_decision")),
            ("blind_pilot", _blind_pilot(data["blind_pilot"], path=f"{path}.blind_pilot")),
            ("second_review", require_bool(data["second_review"], f"{path}.second_review")),
            ("reviewer", require_str(data["reviewer"], f"{path}.reviewer")),
            ("reviewed_at", _validate_utc_seconds(data["reviewed_at"], f"{path}.reviewed_at")),
            ("total", require_int(data["total"], f"{path}.total", minimum=0)),
            ("difficulty", data["difficulty"]),
            ("decision", require_str(data["decision"], f"{path}.decision")),
            ("content_hash", require_str(data["content_hash"], f"{path}.content_hash")),
        ):
            object.__setattr__(evidence, name, item)
        evidence._validate_stored()
        return evidence


def build_complexity_evidence(
    *,
    task_id: str,
    localization: int,
    diagnosis: int,
    repair_regression: int,
    dimension_evidence: Mapping[str, str],
    hard_rejections: tuple[str, ...],
    risk_signals: tuple[str, ...],
    duplicate_fingerprint: str,
    duplicate_decision: str,
    blind_pilot: Mapping[str, object] | None,
    second_review: bool,
    reviewer: str,
    reviewed_at: str,
) -> ComplexityEvidence:
    """Build deterministic evidence; agent-resolution inputs are intentionally absent."""

    return ComplexityEvidence(
        task_id=task_id,
        localization=localization,
        diagnosis=diagnosis,
        repair_regression=repair_regression,
        dimension_evidence=dimension_evidence,
        hard_rejections=hard_rejections,
        risk_signals=risk_signals,
        duplicate_fingerprint=duplicate_fingerprint,
        duplicate_decision=duplicate_decision,
        blind_pilot=blind_pilot,
        second_review=second_review,
        reviewer=reviewer,
        reviewed_at=reviewed_at,
    )
