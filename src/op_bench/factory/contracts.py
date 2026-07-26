from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Any, ClassVar

from op_bench.runtime.canonical import JsonValue, canonical_json, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
    require_str_tuple,
)


SCHEMA_VERSION = "v1"
SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
GIT_COMMIT_PATTERN = r"[0-9a-f]{40}"
CANDIDATE_ID_PATTERN = r"candidate:v1:[0-9a-f]{64}"
IDENTIFIER_PATTERN = r"[a-z0-9][a-z0-9._-]*"
UTC_SECONDS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

DISCOVERY_SOURCES = ("fixture", "git_log", "github_pr_list")
PROBLEM_DIMENSIONS = ("boundary", "precision")
CHANGE_KINDS = ("bugfix", "refactor", "cleanup", "feature")
FINDING_SEVERITIES = ("warning", "defer", "reject")
DECISION_DISPOSITIONS = ("accepted", "deferred", "rejected")
DECISION_SOURCES = ("automation", "human_review")
REASON_CODE_NAMESPACES = (
    "metadata",
    "window",
    "change",
    "test",
    "source",
    "runtime",
    "taxonomy",
    "duplicate",
    "review",
)
FACTORY_EVIDENCE_TYPES = (
    "screening_decision",
    "task_bundle",
    "preflight",
    "baseline",
    "gold",
    "human_review",
    "integrity",
)
FACTORY_ADMISSION_STATES = (
    "discovered",
    "deferred",
    "screened",
    "bundled",
    "preflight_passed",
    "baseline_reproduced",
    "gold_resolved",
    "reviewed",
    "verified",
    "rejected",
    "deprecated",
)
FACTORY_ACTOR_KINDS = ("automation", "human")
ADMISSION_STAGE_EVIDENCE = {
    "discovered": (),
    "screened": ("screening_decision",),
    "bundled": ("screening_decision", "task_bundle"),
    "preflight_passed": (
        "screening_decision",
        "task_bundle",
        "preflight",
    ),
    "baseline_reproduced": (
        "screening_decision",
        "task_bundle",
        "preflight",
        "baseline",
    ),
    "gold_resolved": (
        "screening_decision",
        "task_bundle",
        "preflight",
        "baseline",
        "gold",
    ),
    "reviewed": (
        "screening_decision",
        "task_bundle",
        "preflight",
        "baseline",
        "gold",
        "human_review",
    ),
    "verified": (
        "screening_decision",
        "task_bundle",
        "preflight",
        "baseline",
        "gold",
        "human_review",
        "integrity",
    ),
}
PROBLEM_SUBCLASSES = {
    "boundary": ("B1", "B2", "B3", "B4", "B5"),
    "precision": ("P1", "P2", "P3", "P4", "P5"),
}
MAX_TITLE_LENGTH = 300
MAX_DESCRIPTION_LENGTH = 4_000


def factory_content_hash(payload: Mapping[str, object]) -> str:
    """Hash a Factory artifact without its self-referential content_hash."""

    without_hash = {key: value for key, value in payload.items() if key != "content_hash"}
    return canonical_sha256(without_hash)


def _validate_relative_path(value: object, path: str) -> str:
    text = require_str(value, path)
    if "\\" in text:
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    relative = PurePosixPath(text)
    if relative.is_absolute() or not relative.parts or any(
        part in ("", ".", "..") for part in relative.parts
    ):
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    if relative.as_posix() != text:
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    return text


def _validate_utc_seconds(value: object, path: str) -> str:
    text = require_str(value, path)
    if UTC_SECONDS_PATTERN.fullmatch(text) is None:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds") from exc
    return text


def _validate_optional_utc_seconds(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _validate_utc_seconds(value, path)


def _validate_optional_commit(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=GIT_COMMIT_PATTERN)


def _require_bounded_str(
    value: object,
    path: str,
    *,
    maximum: int,
) -> str:
    text = require_str(value, path)
    if len(text) > maximum:
        raise ContractError(f"{path}: must contain at most {maximum} characters")
    return text


def _freeze_json_value(value: JsonValue, path: str) -> JsonValue:
    canonical_json(value)
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: object keys must be strings")
            frozen[key] = _freeze_json_value(item, f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    return value


def _wire_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _wire_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire_json_value(item) for item in value]
    if isinstance(value, list):
        return [_wire_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class FactoryArtifactReference:
    artifact_type: str
    artifact_id: str
    content_hash: str
    relative_path: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("artifact_type", "artifact_id", "content_hash", "relative_path")

    def __post_init__(self) -> None:
        require_str(self.artifact_type, "artifact_type", pattern=IDENTIFIER_PATTERN)
        require_str(self.artifact_id, "artifact_id")
        require_str(self.content_hash, "content_hash", pattern=SHA256_PATTERN)
        _validate_relative_path(self.relative_path, "relative_path")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "content_hash": self.content_hash,
            "relative_path": self.relative_path,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "artifact_reference",
    ) -> "FactoryArtifactReference":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            artifact_type=require_str(
                data["artifact_type"],
                f"{path}.artifact_type",
                pattern=IDENTIFIER_PATTERN,
            ),
            artifact_id=require_str(data["artifact_id"], f"{path}.artifact_id"),
            content_hash=require_str(
                data["content_hash"],
                f"{path}.content_hash",
                pattern=SHA256_PATTERN,
            ),
            relative_path=_validate_relative_path(
                data["relative_path"],
                f"{path}.relative_path",
            ),
        )


@dataclass(frozen=True)
class ChangedFile:
    path: str
    additions: int
    deletions: int
    is_test: bool

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("path", "additions", "deletions", "is_test")

    def __post_init__(self) -> None:
        _validate_relative_path(self.path, "path")
        require_int(self.additions, "additions", minimum=0)
        require_int(self.deletions, "deletions", minimum=0)
        require_bool(self.is_test, "is_test")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "path": self.path,
            "additions": self.additions,
            "deletions": self.deletions,
            "is_test": self.is_test,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "changed_file") -> "ChangedFile":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            path=_validate_relative_path(data["path"], f"{path}.path"),
            additions=require_int(data["additions"], f"{path}.additions", minimum=0),
            deletions=require_int(data["deletions"], f"{path}.deletions", minimum=0),
            is_test=require_bool(data["is_test"], f"{path}.is_test"),
        )


@dataclass(frozen=True)
class CandidateRecord:
    contract_type: ClassVar[str] = "factory_candidate"
    schema_version: ClassVar[str] = SCHEMA_VERSION

    candidate_id: str
    framework: str
    repository: str
    pr_number: int
    pr_url: str
    base_commit: str | None
    merge_commit: str | None
    author_date: str | None
    merge_date: str | None
    title: str
    description: str
    changed_files: tuple[ChangedFile, ...]
    total_files: int
    total_changed_lines: int
    discovery_source: str
    keyword_pack_id: str
    matched_keyword_ids: tuple[str, ...]
    proposed_dimension: str
    proposed_subclass: str
    raw_metadata: FactoryArtifactReference
    created_at: str
    change_kind: str = "bugfix"
    external_test: FactoryArtifactReference | None = None
    environment_freeze: FactoryArtifactReference | None = None
    source_available: bool = True
    runtime_supported: bool = True

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "candidate_id",
            "framework",
            "repository",
            "pr_number",
            "pr_url",
            "base_commit",
            "merge_commit",
            "author_date",
            "merge_date",
            "title",
            "description",
            "changed_files",
            "total_files",
            "total_changed_lines",
            "discovery_source",
            "keyword_pack_id",
            "matched_keyword_ids",
            "proposed_dimension",
            "proposed_subclass",
            "raw_metadata",
            "created_at",
            "change_kind",
            "external_test",
            "environment_freeze",
            "source_available",
            "runtime_supported",
            "content_hash",
        )

    @classmethod
    def candidate_id_for(
        cls,
        *,
        repository: str,
        pr_number: int,
        base_commit: str | None,
        merge_commit: str | None,
    ) -> str:
        digest = canonical_sha256(
            {
                "repository": repository,
                "pr_number": pr_number,
                "base_commit": base_commit,
                "merge_commit": merge_commit,
            }
        )
        return "candidate:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        require_str(self.candidate_id, "candidate_id", pattern=CANDIDATE_ID_PATTERN)
        require_str(self.framework, "framework", pattern=IDENTIFIER_PATTERN)
        require_str(self.repository, "repository", pattern=r"[^/\s]+/[^/\s]+")
        require_int(self.pr_number, "pr_number", minimum=1)
        expected_url = f"https://github.com/{self.repository}/pull/{self.pr_number}"
        if self.pr_url != expected_url:
            raise ContractError(f"pr_url: expected {expected_url!r}")
        _validate_optional_commit(self.base_commit, "base_commit")
        _validate_optional_commit(self.merge_commit, "merge_commit")
        _validate_optional_utc_seconds(self.author_date, "author_date")
        _validate_optional_utc_seconds(self.merge_date, "merge_date")
        _require_bounded_str(self.title, "title", maximum=MAX_TITLE_LENGTH)
        _require_bounded_str(
            self.description,
            "description",
            maximum=MAX_DESCRIPTION_LENGTH,
        )
        if not isinstance(self.changed_files, tuple) or not self.changed_files:
            raise ContractError("changed_files: expected non-empty tuple")
        paths: set[str] = set()
        for index, changed_file in enumerate(self.changed_files):
            if not isinstance(changed_file, ChangedFile):
                raise ContractError(
                    f"changed_files[{index}]: expected ChangedFile"
                )
            if changed_file.path in paths:
                raise ContractError(
                    f"changed_files: duplicate path {changed_file.path!r}"
                )
            paths.add(changed_file.path)
        require_int(self.total_files, "total_files", minimum=1)
        if self.total_files != len(self.changed_files):
            raise ContractError(
                "total_files: must match the number of changed_files"
            )
        require_int(self.total_changed_lines, "total_changed_lines", minimum=0)
        actual_changed_lines = sum(
            item.additions + item.deletions for item in self.changed_files
        )
        if self.total_changed_lines != actual_changed_lines:
            raise ContractError(
                "total_changed_lines: must match changed-file additions and deletions"
            )
        require_enum(
            self.discovery_source,
            "discovery_source",
            DISCOVERY_SOURCES,
        )
        require_str(
            self.keyword_pack_id,
            "keyword_pack_id",
            pattern=IDENTIFIER_PATTERN,
        )
        if not isinstance(self.matched_keyword_ids, tuple):
            raise ContractError("matched_keyword_ids: expected tuple")
        for index, keyword_id in enumerate(self.matched_keyword_ids):
            require_str(
                keyword_id,
                f"matched_keyword_ids[{index}]",
                pattern=IDENTIFIER_PATTERN,
            )
        if len(set(self.matched_keyword_ids)) != len(self.matched_keyword_ids):
            raise ContractError("matched_keyword_ids: duplicate value")
        if tuple(sorted(self.matched_keyword_ids)) != self.matched_keyword_ids:
            raise ContractError("matched_keyword_ids: expected sorted values")
        require_enum(
            self.proposed_dimension,
            "proposed_dimension",
            PROBLEM_DIMENSIONS,
        )
        require_enum(
            self.proposed_subclass,
            "proposed_subclass",
            PROBLEM_SUBCLASSES[self.proposed_dimension],
        )
        if not isinstance(self.raw_metadata, FactoryArtifactReference):
            raise ContractError("raw_metadata: expected FactoryArtifactReference")
        _validate_utc_seconds(self.created_at, "created_at")
        require_enum(self.change_kind, "change_kind", CHANGE_KINDS)
        for value, path in (
            (self.external_test, "external_test"),
            (self.environment_freeze, "environment_freeze"),
        ):
            if value is not None and not isinstance(value, FactoryArtifactReference):
                raise ContractError(
                    f"{path}: expected FactoryArtifactReference or null"
                )
        require_bool(self.source_available, "source_available")
        require_bool(self.runtime_supported, "runtime_supported")
        expected_id = self.candidate_id_for(
            repository=self.repository,
            pr_number=self.pr_number,
            base_commit=self.base_commit,
            merge_commit=self.merge_commit,
        )
        if self.candidate_id != expected_id:
            raise ContractError(
                f"candidate_id: expected derived identity {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return factory_content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "framework": self.framework,
            "repository": self.repository,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "base_commit": self.base_commit,
            "merge_commit": self.merge_commit,
            "author_date": self.author_date,
            "merge_date": self.merge_date,
            "title": self.title,
            "description": self.description,
            "changed_files": [item.to_dict() for item in self.changed_files],
            "total_files": self.total_files,
            "total_changed_lines": self.total_changed_lines,
            "discovery_source": self.discovery_source,
            "keyword_pack_id": self.keyword_pack_id,
            "matched_keyword_ids": list(self.matched_keyword_ids),
            "proposed_dimension": self.proposed_dimension,
            "proposed_subclass": self.proposed_subclass,
            "raw_metadata": self.raw_metadata.to_dict(),
            "created_at": self.created_at,
            "change_kind": self.change_kind,
            "external_test": (
                None if self.external_test is None else self.external_test.to_dict()
            ),
            "environment_freeze": (
                None
                if self.environment_freeze is None
                else self.environment_freeze.to_dict()
            ),
            "source_available": self.source_available,
            "runtime_supported": self.runtime_supported,
        }
        if include_hash:
            payload["content_hash"] = factory_content_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, value: object, *, path: str = "factory_candidate") -> "CandidateRecord":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        changed_files = require_list(data["changed_files"], f"{path}.changed_files")
        candidate = cls(
            candidate_id=require_str(
                data["candidate_id"],
                f"{path}.candidate_id",
                pattern=CANDIDATE_ID_PATTERN,
            ),
            framework=require_str(data["framework"], f"{path}.framework"),
            repository=require_str(data["repository"], f"{path}.repository"),
            pr_number=require_int(data["pr_number"], f"{path}.pr_number", minimum=1),
            pr_url=require_str(data["pr_url"], f"{path}.pr_url"),
            base_commit=_validate_optional_commit(
                data["base_commit"],
                f"{path}.base_commit",
            ),
            merge_commit=_validate_optional_commit(
                data["merge_commit"],
                f"{path}.merge_commit",
            ),
            author_date=_validate_optional_utc_seconds(
                data["author_date"],
                f"{path}.author_date",
            ),
            merge_date=_validate_optional_utc_seconds(
                data["merge_date"],
                f"{path}.merge_date",
            ),
            title=_require_bounded_str(
                data["title"],
                f"{path}.title",
                maximum=MAX_TITLE_LENGTH,
            ),
            description=_require_bounded_str(
                data["description"],
                f"{path}.description",
                maximum=MAX_DESCRIPTION_LENGTH,
            ),
            changed_files=tuple(
                ChangedFile.from_dict(
                    item,
                    path=f"{path}.changed_files[{index}]",
                )
                for index, item in enumerate(changed_files)
            ),
            total_files=require_int(
                data["total_files"],
                f"{path}.total_files",
                minimum=1,
            ),
            total_changed_lines=require_int(
                data["total_changed_lines"],
                f"{path}.total_changed_lines",
                minimum=0,
            ),
            discovery_source=require_enum(
                data["discovery_source"],
                f"{path}.discovery_source",
                DISCOVERY_SOURCES,
            ),
            keyword_pack_id=require_str(
                data["keyword_pack_id"],
                f"{path}.keyword_pack_id",
                pattern=IDENTIFIER_PATTERN,
            ),
            matched_keyword_ids=require_str_tuple(
                data["matched_keyword_ids"],
                f"{path}.matched_keyword_ids",
            ),
            proposed_dimension=require_enum(
                data["proposed_dimension"],
                f"{path}.proposed_dimension",
                PROBLEM_DIMENSIONS,
            ),
            proposed_subclass=require_str(
                data["proposed_subclass"],
                f"{path}.proposed_subclass",
            ),
            raw_metadata=FactoryArtifactReference.from_dict(
                data["raw_metadata"],
                path=f"{path}.raw_metadata",
            ),
            created_at=_validate_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
            change_kind=require_enum(
                data["change_kind"],
                f"{path}.change_kind",
                CHANGE_KINDS,
            ),
            external_test=(
                None
                if data["external_test"] is None
                else FactoryArtifactReference.from_dict(
                    data["external_test"],
                    path=f"{path}.external_test",
                )
            ),
            environment_freeze=(
                None
                if data["environment_freeze"] is None
                else FactoryArtifactReference.from_dict(
                    data["environment_freeze"],
                    path=f"{path}.environment_freeze",
                )
            ),
            source_available=require_bool(
                data["source_available"],
                f"{path}.source_available",
            ),
            runtime_supported=require_bool(
                data["runtime_supported"],
                f"{path}.runtime_supported",
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=SHA256_PATTERN,
        )
        if stored_hash != candidate.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {candidate.content_hash!r}"
            )
        return candidate


@dataclass(frozen=True)
class ScreeningFinding:
    code: str
    severity: str
    rule_id: str
    message: str
    observed: JsonValue
    expected: Mapping[str, JsonValue]

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "code",
            "severity",
            "rule_id",
            "message",
            "observed",
            "expected",
        )

    def __post_init__(self) -> None:
        namespaces = "|".join(REASON_CODE_NAMESPACES)
        require_str(
            self.code,
            "code",
            pattern=rf"(?:{namespaces})\.[a-z0-9][a-z0-9._-]*",
        )
        require_enum(self.severity, "severity", FINDING_SEVERITIES)
        require_str(self.rule_id, "rule_id", pattern=IDENTIFIER_PATTERN)
        _require_bounded_str(self.message, "message", maximum=500)
        object.__setattr__(
            self,
            "observed",
            _freeze_json_value(self.observed, "observed"),
        )
        if not isinstance(self.expected, Mapping):
            raise ContractError("expected: expected object")
        object.__setattr__(
            self,
            "expected",
            _freeze_json_value(dict(self.expected), "expected"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
            "observed": _wire_json_value(self.observed),
            "expected": _wire_json_value(self.expected),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "screening_finding",
    ) -> "ScreeningFinding":
        data = require_exact_fields(value, path, cls.wire_fields())
        expected = data["expected"]
        if not isinstance(expected, Mapping):
            raise ContractError(f"{path}.expected: expected object")
        return cls(
            code=require_str(data["code"], f"{path}.code"),
            severity=require_enum(
                data["severity"],
                f"{path}.severity",
                FINDING_SEVERITIES,
            ),
            rule_id=require_str(data["rule_id"], f"{path}.rule_id"),
            message=require_str(data["message"], f"{path}.message"),
            observed=data["observed"],
            expected=dict(expected),
        )


_FINDING_SEVERITY_RANK = {"reject": 0, "defer": 1, "warning": 2}


def finding_sort_key(finding: ScreeningFinding) -> tuple[int, str, str]:
    return (
        _FINDING_SEVERITY_RANK[finding.severity],
        finding.code,
        finding.rule_id,
    )


def derived_disposition(findings: tuple[ScreeningFinding, ...]) -> str:
    severities = {finding.severity for finding in findings}
    if "reject" in severities:
        return "rejected"
    if "defer" in severities:
        return "deferred"
    return "accepted"


@dataclass(frozen=True)
class DecisionRecord:
    contract_type: ClassVar[str] = "factory_decision"
    schema_version: ClassVar[str] = SCHEMA_VERSION

    decision_id: str
    candidate_id: str
    candidate_content_hash: str
    rule_set_id: str
    rule_set_hash: str
    target_dimension: str
    target_subclass: str
    findings: tuple[ScreeningFinding, ...]
    disposition: str
    decision_source: str
    prior_decision: FactoryArtifactReference | None
    created_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "decision_id",
            "candidate_id",
            "candidate_content_hash",
            "rule_set_id",
            "rule_set_hash",
            "target_dimension",
            "target_subclass",
            "findings",
            "disposition",
            "decision_source",
            "prior_decision",
            "created_at",
            "content_hash",
        )

    @classmethod
    def decision_id_for(
        cls,
        *,
        candidate_id: str,
        candidate_content_hash: str,
        rule_set_hash: str,
        decision_source: str,
        prior_decision_hash: str | None,
    ) -> str:
        digest = canonical_sha256(
            {
                "candidate_id": candidate_id,
                "candidate_content_hash": candidate_content_hash,
                "rule_set_hash": rule_set_hash,
                "decision_source": decision_source,
                "prior_decision_hash": prior_decision_hash,
            }
        )
        return "decision:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        require_str(
            self.decision_id,
            "decision_id",
            pattern=r"decision:v1:[0-9a-f]{64}",
        )
        require_str(
            self.candidate_id,
            "candidate_id",
            pattern=CANDIDATE_ID_PATTERN,
        )
        require_str(
            self.candidate_content_hash,
            "candidate_content_hash",
            pattern=SHA256_PATTERN,
        )
        require_str(
            self.rule_set_id,
            "rule_set_id",
            pattern=r"screening:v1:[0-9a-f]{64}",
        )
        require_str(self.rule_set_hash, "rule_set_hash", pattern=SHA256_PATTERN)
        require_enum(
            self.target_dimension,
            "target_dimension",
            PROBLEM_DIMENSIONS,
        )
        require_enum(
            self.target_subclass,
            "target_subclass",
            PROBLEM_SUBCLASSES[self.target_dimension],
        )
        if not isinstance(self.findings, tuple):
            raise ContractError("findings: expected tuple")
        for index, finding in enumerate(self.findings):
            if not isinstance(finding, ScreeningFinding):
                raise ContractError(
                    f"findings[{index}]: expected ScreeningFinding"
                )
        if tuple(sorted(self.findings, key=finding_sort_key)) != self.findings:
            raise ContractError("findings: expected deterministic severity/code order")
        finding_keys = tuple(
            (finding.code, finding.rule_id) for finding in self.findings
        )
        if len(set(finding_keys)) != len(finding_keys):
            raise ContractError("findings: duplicate code and rule_id")
        require_enum(
            self.disposition,
            "disposition",
            DECISION_DISPOSITIONS,
        )
        expected_disposition = derived_disposition(self.findings)
        if self.disposition != expected_disposition:
            raise ContractError(
                "disposition: does not match the highest finding severity"
            )
        require_enum(
            self.decision_source,
            "decision_source",
            DECISION_SOURCES,
        )
        if self.prior_decision is not None:
            if not isinstance(self.prior_decision, FactoryArtifactReference):
                raise ContractError(
                    "prior_decision: expected FactoryArtifactReference or null"
                )
            if self.prior_decision.artifact_type != "factory_decision":
                raise ContractError(
                    "prior_decision.artifact_type: expected 'factory_decision'"
                )
        _validate_utc_seconds(self.created_at, "created_at")
        expected_id = self.decision_id_for(
            candidate_id=self.candidate_id,
            candidate_content_hash=self.candidate_content_hash,
            rule_set_hash=self.rule_set_hash,
            decision_source=self.decision_source,
            prior_decision_hash=(
                None
                if self.prior_decision is None
                else self.prior_decision.content_hash
            ),
        )
        if self.decision_id != expected_id:
            raise ContractError(
                f"decision_id: expected derived identity {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return factory_content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "candidate_content_hash": self.candidate_content_hash,
            "rule_set_id": self.rule_set_id,
            "rule_set_hash": self.rule_set_hash,
            "target_dimension": self.target_dimension,
            "target_subclass": self.target_subclass,
            "findings": [finding.to_dict() for finding in self.findings],
            "disposition": self.disposition,
            "decision_source": self.decision_source,
            "prior_decision": (
                None
                if self.prior_decision is None
                else self.prior_decision.to_dict()
            ),
            "created_at": self.created_at,
        }
        if include_hash:
            payload["content_hash"] = factory_content_hash(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "factory_decision",
    ) -> "DecisionRecord":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        finding_values = require_list(data["findings"], f"{path}.findings")
        decision = cls(
            decision_id=require_str(data["decision_id"], f"{path}.decision_id"),
            candidate_id=require_str(
                data["candidate_id"],
                f"{path}.candidate_id",
            ),
            candidate_content_hash=require_str(
                data["candidate_content_hash"],
                f"{path}.candidate_content_hash",
            ),
            rule_set_id=require_str(data["rule_set_id"], f"{path}.rule_set_id"),
            rule_set_hash=require_str(
                data["rule_set_hash"],
                f"{path}.rule_set_hash",
            ),
            target_dimension=require_str(
                data["target_dimension"],
                f"{path}.target_dimension",
            ),
            target_subclass=require_str(
                data["target_subclass"],
                f"{path}.target_subclass",
            ),
            findings=tuple(
                ScreeningFinding.from_dict(
                    item,
                    path=f"{path}.findings[{index}]",
                )
                for index, item in enumerate(finding_values)
            ),
            disposition=require_str(
                data["disposition"],
                f"{path}.disposition",
            ),
            decision_source=require_str(
                data["decision_source"],
                f"{path}.decision_source",
            ),
            prior_decision=(
                None
                if data["prior_decision"] is None
                else FactoryArtifactReference.from_dict(
                    data["prior_decision"],
                    path=f"{path}.prior_decision",
                )
            ),
            created_at=_validate_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=SHA256_PATTERN,
        )
        if stored_hash != decision.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {decision.content_hash!r}"
            )
        return decision


@dataclass(frozen=True)
class FactoryEvidence:
    evidence_type: str
    reference: FactoryArtifactReference
    claims: Mapping[str, JsonValue]

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("evidence_type", "reference", "claims")

    def __post_init__(self) -> None:
        require_enum(
            self.evidence_type,
            "evidence_type",
            FACTORY_EVIDENCE_TYPES,
        )
        if not isinstance(self.reference, FactoryArtifactReference):
            raise ContractError("reference: expected FactoryArtifactReference")
        if not isinstance(self.claims, Mapping):
            raise ContractError("claims: expected object")
        object.__setattr__(
            self,
            "claims",
            _freeze_json_value(dict(self.claims), "claims"),
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "evidence_type": self.evidence_type,
            "reference": self.reference.to_dict(),
            "claims": _wire_json_value(self.claims),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "factory_evidence",
    ) -> "FactoryEvidence":
        data = require_exact_fields(value, path, cls.wire_fields())
        claims = data["claims"]
        if not isinstance(claims, Mapping):
            raise ContractError(f"{path}.claims: expected object")
        return cls(
            evidence_type=require_enum(
                data["evidence_type"],
                f"{path}.evidence_type",
                FACTORY_EVIDENCE_TYPES,
            ),
            reference=FactoryArtifactReference.from_dict(
                data["reference"],
                path=f"{path}.reference",
            ),
            claims=dict(claims),
        )


def validate_admission_evidence(
    state: str,
    evidence: tuple[FactoryEvidence, ...],
) -> None:
    required = ADMISSION_STAGE_EVIDENCE.get(state, ())
    by_type = {item.evidence_type: item for item in evidence}
    missing = tuple(item for item in required if item not in by_type)
    if missing:
        raise ContractError(
            f"evidence: {state} requires {', '.join(missing)}"
        )
    expected_statuses = {
        "screening_decision": "accepted",
        "task_bundle": "complete",
        "preflight": "passed",
        "baseline": "failed_as_expected",
        "gold": "resolved",
        "human_review": "approved",
        "integrity": "passed",
    }
    for evidence_type in required:
        actual_status = by_type[evidence_type].claims.get("status")
        expected_status = expected_statuses[evidence_type]
        if actual_status != expected_status:
            raise ContractError(
                f"{evidence_type}: expected status {expected_status!r}"
            )
    if "baseline" in required:
        for name in ("source_hash", "runtime_hash", "selector_hash"):
            require_str(
                by_type["baseline"].claims.get(name),
                f"baseline.{name}",
                pattern=SHA256_PATTERN,
            )
    if "gold" in required:
        for name in ("source_hash", "runtime_hash", "selector_hash"):
            baseline_value = by_type["baseline"].claims.get(name)
            gold_value = require_str(
                by_type["gold"].claims.get(name),
                f"gold.{name}",
                pattern=SHA256_PATTERN,
            )
            if gold_value != baseline_value:
                label = name.removesuffix("_hash")
                raise ContractError(
                    f"gold {label}: must match baseline {label}"
                )


@dataclass(frozen=True)
class FactoryAdmissionRecord:
    contract_type: ClassVar[str] = "factory_admission"
    schema_version: ClassVar[str] = SCHEMA_VERSION

    admission_id: str
    candidate: FactoryArtifactReference
    decision: FactoryArtifactReference | None
    task: FactoryArtifactReference | None
    state: str
    previous_record_hash: str | None
    evidence: tuple[FactoryEvidence, ...]
    transition_reason: str
    actor_kind: str
    created_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "admission_id",
            "candidate",
            "decision",
            "task",
            "state",
            "previous_record_hash",
            "evidence",
            "transition_reason",
            "actor_kind",
            "created_at",
            "content_hash",
        )

    @classmethod
    def admission_id_for(
        cls,
        *,
        candidate: FactoryArtifactReference,
        decision: FactoryArtifactReference | None,
        task: FactoryArtifactReference | None,
        state: str,
        previous_record_hash: str | None,
        evidence: tuple[FactoryEvidence, ...],
    ) -> str:
        digest = canonical_sha256(
            {
                "candidate": candidate.to_dict(),
                "decision": None if decision is None else decision.to_dict(),
                "task": None if task is None else task.to_dict(),
                "state": state,
                "previous_record_hash": previous_record_hash,
                "evidence": [item.to_dict() for item in evidence],
            }
        )
        return "admission:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        require_str(
            self.admission_id,
            "admission_id",
            pattern=r"admission:v1:[0-9a-f]{64}",
        )
        if not isinstance(self.candidate, FactoryArtifactReference):
            raise ContractError("candidate: expected FactoryArtifactReference")
        if self.candidate.artifact_type != "factory_candidate":
            raise ContractError(
                "candidate.artifact_type: expected 'factory_candidate'"
            )
        if self.decision is not None:
            if not isinstance(self.decision, FactoryArtifactReference):
                raise ContractError(
                    "decision: expected FactoryArtifactReference or null"
                )
            if self.decision.artifact_type != "factory_decision":
                raise ContractError(
                    "decision.artifact_type: expected 'factory_decision'"
                )
        if self.task is not None and not isinstance(
            self.task,
            FactoryArtifactReference,
        ):
            raise ContractError("task: expected FactoryArtifactReference or null")
        require_enum(self.state, "state", FACTORY_ADMISSION_STATES)
        if self.previous_record_hash is not None:
            require_str(
                self.previous_record_hash,
                "previous_record_hash",
                pattern=SHA256_PATTERN,
            )
        if not isinstance(self.evidence, tuple):
            raise ContractError("evidence: expected tuple")
        for index, item in enumerate(self.evidence):
            if not isinstance(item, FactoryEvidence):
                raise ContractError(
                    f"evidence[{index}]: expected FactoryEvidence"
                )
        evidence_types = tuple(item.evidence_type for item in self.evidence)
        if evidence_types != tuple(sorted(evidence_types)):
            raise ContractError("evidence: expected sorted evidence types")
        if len(set(evidence_types)) != len(evidence_types):
            raise ContractError("evidence: duplicate evidence type")
        if self.state in ADMISSION_STAGE_EVIDENCE:
            validate_admission_evidence(self.state, self.evidence)
        if self.state not in ("discovered", "deferred") and self.decision is None:
            raise ContractError(f"decision: required for state {self.state!r}")
        if self.state in (
            "bundled",
            "preflight_passed",
            "baseline_reproduced",
            "gold_resolved",
            "reviewed",
            "verified",
        ) and self.task is None:
            raise ContractError(f"task: required for state {self.state!r}")
        _require_bounded_str(
            self.transition_reason,
            "transition_reason",
            maximum=500,
        )
        require_enum(self.actor_kind, "actor_kind", FACTORY_ACTOR_KINDS)
        _validate_utc_seconds(self.created_at, "created_at")
        expected_id = self.admission_id_for(
            candidate=self.candidate,
            decision=self.decision,
            task=self.task,
            state=self.state,
            previous_record_hash=self.previous_record_hash,
            evidence=self.evidence,
        )
        if self.admission_id != expected_id:
            raise ContractError(
                f"admission_id: expected derived identity {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return factory_content_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "admission_id": self.admission_id,
            "candidate": self.candidate.to_dict(),
            "decision": (
                None if self.decision is None else self.decision.to_dict()
            ),
            "task": None if self.task is None else self.task.to_dict(),
            "state": self.state,
            "previous_record_hash": self.previous_record_hash,
            "evidence": [item.to_dict() for item in self.evidence],
            "transition_reason": self.transition_reason,
            "actor_kind": self.actor_kind,
            "created_at": self.created_at,
        }
        if include_hash:
            payload["content_hash"] = factory_content_hash(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "factory_admission",
    ) -> "FactoryAdmissionRecord":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        evidence_values = require_list(data["evidence"], f"{path}.evidence")
        previous_hash = data["previous_record_hash"]
        if previous_hash is not None:
            previous_hash = require_str(
                previous_hash,
                f"{path}.previous_record_hash",
                pattern=SHA256_PATTERN,
            )
        record = cls(
            admission_id=require_str(
                data["admission_id"],
                f"{path}.admission_id",
            ),
            candidate=FactoryArtifactReference.from_dict(
                data["candidate"],
                path=f"{path}.candidate",
            ),
            decision=(
                None
                if data["decision"] is None
                else FactoryArtifactReference.from_dict(
                    data["decision"],
                    path=f"{path}.decision",
                )
            ),
            task=(
                None
                if data["task"] is None
                else FactoryArtifactReference.from_dict(
                    data["task"],
                    path=f"{path}.task",
                )
            ),
            state=require_str(data["state"], f"{path}.state"),
            previous_record_hash=previous_hash,
            evidence=tuple(
                FactoryEvidence.from_dict(
                    item,
                    path=f"{path}.evidence[{index}]",
                )
                for index, item in enumerate(evidence_values)
            ),
            transition_reason=require_str(
                data["transition_reason"],
                f"{path}.transition_reason",
            ),
            actor_kind=require_str(
                data["actor_kind"],
                f"{path}.actor_kind",
            ),
            created_at=_validate_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=SHA256_PATTERN,
        )
        if stored_hash != record.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {record.content_hash!r}"
            )
        return record
