from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import Any, ClassVar

from op_bench.runtime.canonical import JsonValue, canonical_sha256
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
    base_commit: str
    merge_commit: str
    author_date: str
    merge_date: str
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
            "content_hash",
        )

    @classmethod
    def candidate_id_for(
        cls,
        *,
        repository: str,
        pr_number: int,
        base_commit: str,
        merge_commit: str,
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
        require_str(self.base_commit, "base_commit", pattern=GIT_COMMIT_PATTERN)
        require_str(self.merge_commit, "merge_commit", pattern=GIT_COMMIT_PATTERN)
        _validate_utc_seconds(self.author_date, "author_date")
        _validate_utc_seconds(self.merge_date, "merge_date")
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
            base_commit=require_str(
                data["base_commit"],
                f"{path}.base_commit",
                pattern=GIT_COMMIT_PATTERN,
            ),
            merge_commit=require_str(
                data["merge_commit"],
                f"{path}.merge_commit",
                pattern=GIT_COMMIT_PATTERN,
            ),
            author_date=_validate_utc_seconds(
                data["author_date"],
                f"{path}.author_date",
            ),
            merge_date=_validate_utc_seconds(
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
