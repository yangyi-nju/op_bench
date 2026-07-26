#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from op_bench.factory.artifacts import FactoryArtifactStore
from op_bench.factory.contracts import (
    DatasetFreezeEntry,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
)
from op_bench.factory.freeze import (
    build_freeze_manifest,
    rebuild_dataset_manifest,
)
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_list,
    require_mapping,
    require_str,
)


_REQUEST_FIELDS = (
    "dataset_id",
    "dataset_version",
    "base_dataset",
    "factory_protocol",
    "screening_rule_set",
    "exclusion_index",
    "registries",
    "entries",
    "admissions",
    "reference_hashes",
    "created_at",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic local OpBench Dataset Freeze.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _load_request(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input is not readable JSON") from exc
    return require_exact_fields(value, "freeze_request", _REQUEST_FIELDS)


def _build(data: dict[str, object]):
    entries = tuple(
        DatasetFreezeEntry.from_dict(
            value,
            path=f"freeze_request.entries[{index}]",
        )
        for index, value in enumerate(
            require_list(data["entries"], "freeze_request.entries")
        )
    )
    admissions = tuple(
        FactoryAdmissionRecord.from_dict(
            value,
            path=f"freeze_request.admissions[{index}]",
        )
        for index, value in enumerate(
            require_list(data["admissions"], "freeze_request.admissions")
        )
    )
    registries = require_mapping(
        data["registries"],
        "freeze_request.registries",
    )
    reference_hashes = require_mapping(
        data["reference_hashes"],
        "freeze_request.reference_hashes",
    )
    return build_freeze_manifest(
        dataset_id=require_str(data["dataset_id"], "dataset_id"),
        dataset_version=require_str(
            data["dataset_version"],
            "dataset_version",
        ),
        base_dataset=FactoryArtifactReference.from_dict(
            data["base_dataset"],
            path="freeze_request.base_dataset",
        ),
        factory_protocol=FactoryArtifactReference.from_dict(
            data["factory_protocol"],
            path="freeze_request.factory_protocol",
        ),
        screening_rule_set=FactoryArtifactReference.from_dict(
            data["screening_rule_set"],
            path="freeze_request.screening_rule_set",
        ),
        exclusion_index=FactoryArtifactReference.from_dict(
            data["exclusion_index"],
            path="freeze_request.exclusion_index",
        ),
        registries={
            key: require_str(value, f"registries.{key}")
            for key, value in registries.items()
        },
        entries=entries,
        admissions={item.admission_id: item for item in admissions},
        reference_hashes={
            key: require_str(value, f"reference_hashes.{key}")
            for key, value in reference_hashes.items()
        },
        created_at=require_str(data["created_at"], "created_at"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = _load_request(args.input)
    except (ValueError, ContractError) as exc:
        print(f"[contract_invalid] {exc}", file=sys.stderr)
        return 2
    try:
        freeze = _build(data)
    except ContractError as exc:
        print(f"[freeze_not_rebuildable] {exc}", file=sys.stderr)
        return 1

    try:
        with FactoryArtifactStore(args.output_dir) as store:
            store.write_contract("freeze_manifest.json", freeze)
            store.write_json(
                "dataset.json",
                rebuild_dataset_manifest(freeze),
                artifact_type="dataset_manifest",
                artifact_id=f"dataset:{freeze.dataset_id}",
            )
    except (ContractError, OSError) as exc:
        print(f"[artifact_unsafe] {exc}", file=sys.stderr)
        return 1

    print(
        canonical_json(
            {
                "dataset_hash": freeze.generated_dataset_hash,
                "freeze_hash": freeze.content_hash,
                "status": "frozen",
                "task_count": len(freeze.entries),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
