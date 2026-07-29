from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_int,
    require_list,
    require_mapping,
    require_str,
)


SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
COHORT_ID_PATTERN = r"cohort:v1:[0-9a-f]{64}"
RELEASE_ID_PATTERN = r"release:v1:[0-9a-f]{64}"
BASELINE_COMMIT = "4f5addc"
DATASET_ROLES = ("boundary", "cumulative", "precision")


@dataclass(frozen=True)
class PreQualityArchive:
    schema_version: str
    archive_id: str
    baseline_commit: str
    task_count: int
    dataset_hashes: Mapping[str, str]
    release_id: str
    release_hash: str
    validation_dataset_digest: str
    validation_attempts: int
    cohort_ids: tuple[str, ...]
    limitations: tuple[str, ...]
    content_hash: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "schema_version",
            "archive_id",
            "baseline_commit",
            "task_count",
            "dataset_hashes",
            "release_id",
            "release_hash",
            "validation_dataset_digest",
            "validation_attempts",
            "cohort_ids",
            "limitations",
            "content_hash",
        )

    @classmethod
    def from_dict(cls, value: object) -> "PreQualityArchive":
        data = require_exact_fields(
            value,
            "pre_quality_archive",
            cls.wire_fields(),
        )
        schema_version = require_str(
            data["schema_version"],
            "pre_quality_archive.schema_version",
        )
        if schema_version != "v1":
            raise ContractError("pre_quality_archive.schema_version: expected 'v1'")
        archive_id = require_str(
            data["archive_id"], "pre_quality_archive.archive_id"
        )
        if archive_id != "v0.7-pre-quality":
            raise ContractError(
                "pre_quality_archive.archive_id: expected 'v0.7-pre-quality'"
            )
        baseline_commit = require_str(
            data["baseline_commit"],
            "pre_quality_archive.baseline_commit",
        )
        if baseline_commit != BASELINE_COMMIT:
            raise ContractError(
                "pre_quality_archive.baseline_commit: expected '4f5addc'"
            )

        dataset_hashes_value = require_mapping(
            data["dataset_hashes"], "pre_quality_archive.dataset_hashes"
        )
        if set(dataset_hashes_value) != set(DATASET_ROLES):
            raise ContractError(
                "pre_quality_archive.dataset_hashes: expected boundary, "
                "cumulative, and precision hashes"
            )
        dataset_hashes = {
            role: require_str(
                dataset_hashes_value[role],
                f"pre_quality_archive.dataset_hashes.{role}",
                pattern=SHA256_PATTERN,
            )
            for role in DATASET_ROLES
        }
        cohort_ids = _require_sorted_unique_strings(
            data["cohort_ids"],
            "pre_quality_archive.cohort_ids",
            pattern=COHORT_ID_PATTERN,
        )
        limitations = _require_unique_strings(
            data["limitations"],
            "pre_quality_archive.limitations",
        )
        stored_hash = require_str(
            data["content_hash"],
            "pre_quality_archive.content_hash",
            pattern=SHA256_PATTERN,
        )
        payload = {
            field: data[field]
            for field in cls.wire_fields()
            if field != "content_hash"
        }
        expected_hash = canonical_sha256(payload)
        if stored_hash != expected_hash:
            raise ContractError("pre_quality_archive.content_hash: payload hash mismatch")

        return cls(
            schema_version=schema_version,
            archive_id=archive_id,
            baseline_commit=baseline_commit,
            task_count=require_int(
                data["task_count"], "pre_quality_archive.task_count", minimum=1
            ),
            dataset_hashes=MappingProxyType(dataset_hashes),
            release_id=require_str(
                data["release_id"],
                "pre_quality_archive.release_id",
                pattern=RELEASE_ID_PATTERN,
            ),
            release_hash=require_str(
                data["release_hash"],
                "pre_quality_archive.release_hash",
                pattern=SHA256_PATTERN,
            ),
            validation_dataset_digest=require_str(
                data["validation_dataset_digest"],
                "pre_quality_archive.validation_dataset_digest",
                pattern=SHA256_PATTERN,
            ),
            validation_attempts=require_int(
                data["validation_attempts"],
                "pre_quality_archive.validation_attempts",
                minimum=1,
            ),
            cohort_ids=cohort_ids,
            limitations=limitations,
            content_hash=stored_hash,
        )


def _require_sorted_unique_strings(
    value: object,
    path: str,
    *,
    pattern: str | None,
) -> tuple[str, ...]:
    items = require_list(value, path)
    if not items:
        raise ContractError(f"{path}: expected at least one value")
    result = tuple(
        require_str(item, f"{path}[{index}]", pattern=pattern)
        for index, item in enumerate(items)
    )
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise ContractError(f"{path}: expected sorted, unique values")
    return result


def _require_unique_strings(value: object, path: str) -> tuple[str, ...]:
    items = require_list(value, path)
    if not items:
        raise ContractError(f"{path}: expected at least one value")
    result = tuple(
        require_str(item, f"{path}[{index}]")
        for index, item in enumerate(items)
    )
    if len(set(result)) != len(result):
        raise ContractError(f"{path}: expected unique values")
    return result


def load_pre_quality_archive(path: Path) -> PreQualityArchive:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    encoded = canonical_json(value).encode("utf-8")
    if raw not in (encoded, encoded + b"\n"):
        raise ContractError(f"{path}: JSON is not canonical")
    return PreQualityArchive.from_dict(value)
