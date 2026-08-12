#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

from op_bench.factory.artifacts import FactoryArtifactStore
from op_bench.factory.contracts import (
    DISCOVERY_SOURCES,
    CandidateRecord,
    ChangedFile,
    FactoryArtifactReference,
    factory_content_hash,
)
from op_bench.factory.screening import (
    V07_BOUNDARY_SCREENING_V1,
    screen_candidate,
)
from op_bench.factory.taxonomy import match_keyword_packs
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
)


_LEGACY_CAPTURE_FIELDS = (
    "repository",
    "pr_number",
    "base_commit",
    "merge_commit",
    "author_date",
    "merge_date",
    "title",
    "description",
    "changed_files",
    "proposed_subclass",
    "change_kind",
    "source_available",
    "runtime_supported",
)
_REAL_CAPTURE_FIELDS = _LEGACY_CAPTURE_FIELDS + (
    "discovery_source",
    "external_test",
    "environment_freeze",
)
_REAL_ONLY_CAPTURE_FIELDS = frozenset(
    set(_REAL_CAPTURE_FIELDS) - set(_LEGACY_CAPTURE_FIELDS)
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen local Boundary candidate captures.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _optional_commit(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=r"[0-9a-f]{40}")


def _optional_timestamp(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path)


def _optional_reference(
    value: object,
    path: str,
) -> FactoryArtifactReference | None:
    if value is None:
        return None
    return FactoryArtifactReference.from_dict(value, path=path)


def _candidate_from_capture(
    value: object,
    *,
    index: int,
    created_at: str,
) -> CandidateRecord:
    path = f"candidates[{index}]"
    capture_fields = _LEGACY_CAPTURE_FIELDS
    if isinstance(value, Mapping) and any(
        field in value for field in _REAL_ONLY_CAPTURE_FIELDS
    ):
        capture_fields = _REAL_CAPTURE_FIELDS
    data = require_exact_fields(value, path, capture_fields)
    is_real_capture = capture_fields == _REAL_CAPTURE_FIELDS
    repository = require_str(data["repository"], f"{path}.repository")
    pr_number = require_int(data["pr_number"], f"{path}.pr_number", minimum=1)
    base_commit = _optional_commit(data["base_commit"], f"{path}.base_commit")
    merge_commit = _optional_commit(
        data["merge_commit"],
        f"{path}.merge_commit",
    )
    changed_values = require_list(
        data["changed_files"],
        f"{path}.changed_files",
    )
    changed_files = tuple(
        ChangedFile.from_dict(
            item,
            path=f"{path}.changed_files[{file_index}]",
        )
        for file_index, item in enumerate(changed_values)
    )
    title = require_str(data["title"], f"{path}.title")
    description = require_str(data["description"], f"{path}.description")
    proposed_subclass = require_enum(
        data["proposed_subclass"],
        f"{path}.proposed_subclass",
        ("B1", "B2", "B3", "B4", "B5"),
    )
    matched_pack_ids = match_keyword_packs(
        f"{title}\n{description}",
        tuple(item.path for item in changed_files),
    )
    raw_hash = canonical_sha256(data)
    candidate = CandidateRecord(
        candidate_id=CandidateRecord.candidate_id_for(
            repository=repository,
            pr_number=pr_number,
            base_commit=base_commit,
            merge_commit=merge_commit,
        ),
        framework="pytorch",
        repository=repository,
        pr_number=pr_number,
        pr_url=f"https://github.com/{repository}/pull/{pr_number}",
        base_commit=base_commit,
        merge_commit=merge_commit,
        author_date=_optional_timestamp(
            data["author_date"],
            f"{path}.author_date",
        ),
        merge_date=_optional_timestamp(
            data["merge_date"],
            f"{path}.merge_date",
        ),
        title=title,
        description=description,
        changed_files=changed_files,
        total_files=len(changed_files),
        total_changed_lines=sum(
            item.additions + item.deletions for item in changed_files
        ),
        discovery_source=(
            require_enum(
                data["discovery_source"],
                f"{path}.discovery_source",
                DISCOVERY_SOURCES,
            )
            if is_real_capture
            else "fixture"
        ),
        keyword_pack_id=f"boundary-{proposed_subclass.lower()}-v1",
        matched_keyword_ids=tuple(sorted(matched_pack_ids)),
        proposed_dimension="boundary",
        proposed_subclass=proposed_subclass,
        raw_metadata=FactoryArtifactReference(
            artifact_type="candidate_raw_metadata",
            artifact_id=f"pr:{repository}#{pr_number}",
            content_hash=raw_hash,
            relative_path=f"raw/pr-{pr_number}.json",
        ),
        created_at=created_at,
        change_kind=require_enum(
            data["change_kind"],
            f"{path}.change_kind",
            ("bugfix", "refactor", "cleanup", "feature"),
        ),
        external_test=(
            _optional_reference(
                data["external_test"],
                f"{path}.external_test",
            )
            if is_real_capture
            else None
        ),
        environment_freeze=(
            _optional_reference(
                data["environment_freeze"],
                f"{path}.environment_freeze",
            )
            if is_real_capture
            else None
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
    return CandidateRecord.from_dict(candidate.to_dict())


def load_candidate_captures(
    path: Path,
    created_at: str,
) -> tuple[CandidateRecord, ...]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input is not readable JSON") from exc
    values = require_list(value, "candidates")
    if not values:
        raise ContractError("candidates: expected at least one entry")
    candidates = tuple(
        _candidate_from_capture(item, index=index, created_at=created_at)
        for index, item in enumerate(values)
    )
    identities = tuple(item.candidate_id for item in candidates)
    if len(set(identities)) != len(identities):
        raise ContractError("candidates: duplicate candidate identity")
    return tuple(
        sorted(
            candidates,
            key=lambda item: (item.repository, item.pr_number, item.candidate_id),
        )
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidates = load_candidate_captures(args.input, args.created_at)
        decisions = tuple(screen_candidate(item) for item in candidates)
    except (ValueError, ContractError) as exc:
        print(f"[contract_invalid] {exc}", file=sys.stderr)
        return 2

    counts = {"accepted": 0, "deferred": 0, "rejected": 0}
    decision_entries: list[dict[str, object]] = []
    try:
        with FactoryArtifactStore(args.output_dir) as store:
            for selected, decision in zip(candidates, decisions):
                candidate_reference = store.write_contract(
                    f"candidates/pr-{selected.pr_number}.json",
                    selected,
                )
                decision_reference = store.write_contract(
                    f"decisions/pr-{selected.pr_number}.json",
                    decision,
                )
                counts[decision.disposition] += 1
                decision_entries.append(
                    {
                        "candidate": candidate_reference.to_dict(),
                        "decision": decision_reference.to_dict(),
                        "disposition": decision.disposition,
                        "finding_codes": [
                            finding.code for finding in decision.findings
                        ],
                        "pr_number": selected.pr_number,
                        "target_subclass": decision.target_subclass,
                    }
                )
            index: dict[str, object] = {
                "contract_type": "factory_screening_index",
                "schema_version": "v1",
                "created_at": args.created_at,
                "rule_set_id": V07_BOUNDARY_SCREENING_V1.rule_set_id,
                "rule_set_hash": V07_BOUNDARY_SCREENING_V1.rule_set_hash,
                "counts": counts,
                "decisions": decision_entries,
            }
            index["content_hash"] = factory_content_hash(index)
            store.write_json(
                "screening_index.json",
                index,
                artifact_type="factory_screening_index",
                artifact_id=(
                    "screening-index:v1:"
                    + str(index["content_hash"]).removeprefix("sha256:")
                ),
            )
    except (ContractError, OSError) as exc:
        print(f"[artifact_unsafe] {exc}", file=sys.stderr)
        return 1

    print(
        canonical_json(
            {
                "counts": counts,
                "rule_set_id": V07_BOUNDARY_SCREENING_V1.rule_set_id,
                "status": "screened",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
