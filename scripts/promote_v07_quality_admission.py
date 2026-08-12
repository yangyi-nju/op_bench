#!/usr/bin/env python3
"""Promote verified 7+29 v0.7 quality Admission results to formal 36.

The command validates both source result indexes before touching official
artifacts.  It then installs the expanded source Registry, complete accepted
index, and merged result index as one rollback-protected operation.  It never
runs Admission and refuses partial or unverified inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import (  # noqa: E402
    load_regular_file_bytes,
)
from op_bench.factory.quality_admission import (  # noqa: E402
    QualityAcceptedTaskIndex,
    QualityAdmissionResultIndex,
    load_quality_accepted_task_index,
    load_quality_admission_result_index,
)
from op_bench.factory.quality_release import quality_bytes_hash  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


OFFICIAL_ACCEPTED = "factory/v0.7/p8/accepted_tasks.json"
OFFICIAL_RESULTS = "factory/v0.7/p8/admission_results.json"
OFFICIAL_SOURCE_REGISTRY = "sources/registry.json"
OFFICIAL_ENVIRONMENT_REGISTRY = "environments/registry.json"
DEFAULT_STAGING_ACCEPTED = (
    "runs/v0.7_quality_admission_staging/accepted_tasks_29.json"
)
DEFAULT_STAGING_RESULTS = (
    "runs/v0.7_quality_admission_staging/admission_results_29.json"
)
DEFAULT_STAGING_SOURCE_REGISTRY = (
    "runs/v0.7_quality_admission_staging/source_registry.json"
)


def _rooted(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _write_canonical(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _write_pretty_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _semantic_source(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key != "local_path"}


def _promoted_source_registry(
    *,
    official_path: Path,
    staging_path: Path,
) -> dict[str, object]:
    official = _load_json_object(official_path)
    staging = _load_json_object(staging_path)
    if official.get("version") != staging.get("version"):
        raise ContractError("source registry: version mismatch")
    official_sources = official.get("sources")
    staging_sources = staging.get("sources")
    if not isinstance(official_sources, list) or not isinstance(
        staging_sources, list
    ):
        raise ContractError("source registry: sources must be lists")
    if len(staging_sources) != 72:
        raise ContractError("staging source registry: expected exactly 72")
    official_by_id = {
        item.get("id"): item
        for item in official_sources
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    staging_by_id = {
        item.get("id"): item
        for item in staging_sources
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    if len(official_by_id) != len(official_sources):
        raise ContractError("official source registry: duplicate or invalid id")
    if len(staging_by_id) != len(staging_sources):
        raise ContractError("staging source registry: duplicate or invalid id")
    if not set(official_by_id).issubset(staging_by_id):
        raise ContractError("staging source registry: official source missing")
    for source_id, official_source in official_by_id.items():
        if _semantic_source(official_source) != _semantic_source(
            staging_by_id[source_id]
        ):
            raise ContractError(
                f"staging source registry: official source drift {source_id}"
            )

    promoted_sources: list[dict[str, object]] = []
    for index, raw_source in enumerate(staging_sources):
        if not isinstance(raw_source, dict):
            raise ContractError(
                f"staging source registry: sources[{index}] must be an object"
            )
        source = dict(raw_source)
        local_path = source.get("local_path")
        if not isinstance(local_path, str):
            raise ContractError(
                f"staging source registry: sources[{index}].local_path"
            )
        absolute = (staging_path.parent / local_path).resolve()
        if not absolute.is_dir() or absolute.is_symlink():
            raise ContractError(
                f"staging source registry: unavailable local source {source['id']}"
            )
        source["local_path"] = Path(
            os.path.relpath(absolute, official_path.parent.resolve())
        ).as_posix()
        promoted_sources.append(source)
    return {"version": staging["version"], "sources": promoted_sources}


def _require_fully_verified(
    index: QualityAdmissionResultIndex,
    *,
    expected_count: int,
    label: str,
) -> None:
    if index.task_count != expected_count:
        raise ContractError(f"{label}: expected {expected_count} results")
    if index.verified_count != expected_count or not all(
        result.verified for result in index.results
    ):
        raise ContractError(f"{label}: every result must be verified")


def promote(
    *,
    staging_accepted_path: Path,
    staging_results_path: Path,
    staging_source_registry_path: Path,
    created_at: str,
) -> tuple[QualityAcceptedTaskIndex, QualityAdmissionResultIndex]:
    official_accepted_path = ROOT / OFFICIAL_ACCEPTED
    official_results_path = ROOT / OFFICIAL_RESULTS
    official_source_path = ROOT / OFFICIAL_SOURCE_REGISTRY
    official_environment_path = ROOT / OFFICIAL_ENVIRONMENT_REGISTRY

    old_accepted = load_quality_accepted_task_index(
        ROOT, official_accepted_path
    )
    old_results = load_quality_admission_result_index(
        ROOT,
        official_results_path,
        official_accepted_path,
        require_private_bundles=False,
    )
    staging_accepted = load_quality_accepted_task_index(
        ROOT, staging_accepted_path
    )
    staging_results = load_quality_admission_result_index(
        ROOT,
        staging_results_path,
        staging_accepted_path,
        require_private_bundles=True,
    )
    _require_fully_verified(old_results, expected_count=7, label="official")
    _require_fully_verified(
        staging_results, expected_count=29, label="staging"
    )
    if old_accepted.task_count != 7 or staging_accepted.task_count != 29:
        raise ContractError("accepted indexes: expected exact 7+29 composition")
    if (
        old_accepted.historical_index_hash
        != staging_accepted.historical_index_hash
        or old_accepted.candidate_index_hash
        != staging_accepted.candidate_index_hash
    ):
        raise ContractError("accepted indexes: source funnel mismatch")

    tasks = tuple(
        sorted(
            (*old_accepted.tasks, *staging_accepted.tasks),
            key=lambda item: item.screening_record_index,
        )
    )
    outcomes = tuple(
        sorted(
            (*old_results.results, *staging_results.results),
            key=lambda item: item.screening_record_index,
        )
    )
    if len(tasks) != 36 or len(outcomes) != 36:
        raise ContractError("promotion: expected exact 36-task union")
    for label, values in (
        ("screening index", [item.screening_record_index for item in tasks]),
        ("task id", [item.task_id for item in tasks]),
        ("public task id", [item.public_task_id for item in tasks]),
        ("PR", [item.pr_number for item in tasks]),
    ):
        if len(values) != len(set(values)):
            raise ContractError(f"promotion: duplicate {label}")

    promoted_registry = _promoted_source_registry(
        official_path=official_source_path,
        staging_path=staging_source_registry_path,
    )
    accepted = replace(
        old_accepted,
        created_at=created_at,
        status="complete",
        task_count=36,
        tasks=tasks,
    )

    protected = (
        official_source_path,
        official_accepted_path,
        official_results_path,
    )
    previous = {path: path.read_bytes() for path in protected}
    try:
        _write_pretty_json(official_source_path, promoted_registry)
        _write_canonical(official_accepted_path, accepted.to_dict())
        results = QualityAdmissionResultIndex(
            created_at=created_at,
            accepted_index_path=OFFICIAL_ACCEPTED,
            accepted_index_hash=accepted.content_hash,
            environment_registry_path=OFFICIAL_ENVIRONMENT_REGISTRY,
            environment_registry_hash=quality_bytes_hash(
                load_regular_file_bytes(official_environment_path)
            ),
            source_registry_path=OFFICIAL_SOURCE_REGISTRY,
            source_registry_hash=quality_bytes_hash(
                load_regular_file_bytes(official_source_path)
            ),
            task_count=36,
            verified_count=36,
            results=outcomes,
        )
        _write_canonical(official_results_path, results.to_dict())
        loaded_accepted = load_quality_accepted_task_index(
            ROOT, official_accepted_path, require_complete=True
        )
        loaded_results = load_quality_admission_result_index(
            ROOT,
            official_results_path,
            official_accepted_path,
            require_private_bundles=False,
        )
        if (
            loaded_accepted.content_hash != accepted.content_hash
            or loaded_results.content_hash != results.content_hash
        ):
            raise ContractError("promotion: final artifacts did not round-trip")
    except BaseException:
        for path, content in previous.items():
            path.write_bytes(content)
        raise
    return accepted, results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging-accepted", default=DEFAULT_STAGING_ACCEPTED
    )
    parser.add_argument("--staging-results", default=DEFAULT_STAGING_RESULTS)
    parser.add_argument(
        "--staging-source-registry",
        default=DEFAULT_STAGING_SOURCE_REGISTRY,
    )
    parser.add_argument("--created-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        accepted, results = promote(
            staging_accepted_path=_rooted(args.staging_accepted),
            staging_results_path=_rooted(args.staging_results),
            staging_source_registry_path=_rooted(
                args.staging_source_registry
            ),
            created_at=args.created_at,
        )
    except (ContractError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"quality Admission promotion failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "accepted_index_hash": accepted.content_hash,
                "result_index_hash": results.content_hash,
                "task_count": accepted.task_count,
                "verified_count": results.verified_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
