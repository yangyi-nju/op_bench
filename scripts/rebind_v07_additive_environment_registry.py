#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import load_canonical_json_artifact
from op_bench.factory.quality_admission import (
    QualityAdmissionResultIndex,
    load_quality_accepted_task_index,
    load_quality_admission_result_index,
)
from op_bench.factory.quality_release import quality_bytes_hash
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebind verified v0.7 Admission results after a provably additive-only "
            "environment registry expansion."
        )
    )
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--added-environment-id",
        action="append",
        required=True,
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Permit atomically replacing --results when --output names the same file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = _rooted(args.results).resolve()
    output_path = _rooted(args.output).resolve()
    for selected in (results_path, output_path):
        try:
            selected.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("results and output must remain inside the repository") from exc
    if results_path == output_path and not args.replace:
        raise SystemExit("--replace is required when output overwrites results")
    if output_path != results_path and output_path.exists():
        raise SystemExit("refusing to overwrite an existing output")

    added_ids = tuple(dict.fromkeys(args.added_environment_id))
    if len(added_ids) != len(args.added_environment_id):
        raise SystemExit("duplicate --added-environment-id")

    result = QualityAdmissionResultIndex.from_dict(
        load_canonical_json_artifact(results_path)
    )
    if not all(item.verified for item in result.results):
        raise ContractError("results must contain only verified outcomes")

    registry_path = (ROOT / result.environment_registry_path).resolve()
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    environments = registry.get("environments")
    if not isinstance(environments, list):
        raise ContractError("environment registry must contain environments")
    by_id = {
        item.get("id"): item
        for item in environments
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    missing = sorted(set(added_ids) - set(by_id))
    if missing:
        raise ContractError(f"added environments are absent: {missing}")

    previous = dict(registry)
    previous["environments"] = [
        item for item in environments if item.get("id") not in set(added_ids)
    ]
    previous_bytes = (json.dumps(previous, indent=2) + "\n").encode("utf-8")
    previous_hash = quality_bytes_hash(previous_bytes)
    if previous_hash != result.environment_registry_hash:
        raise ContractError(
            "registry expansion is not byte-for-byte additive relative to results"
        )

    accepted_path = (ROOT / result.accepted_index_path).resolve()
    accepted = load_quality_accepted_task_index(
        ROOT,
        accepted_path,
        require_complete=False,
    )
    for record in accepted.tasks:
        task = load_canonical_json_artifact(ROOT / record.task_path / "task.json")
        if task.get("environment_ref") in added_ids:
            raise ContractError(
                f"{record.task_id}: existing accepted Task references an added environment"
            )

    rebound = replace(
        result,
        environment_registry_hash=quality_bytes_hash(registry_bytes),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(rebound.to_dict()).encode("utf-8")
    if output_path == results_path:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=output_path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(payload)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, output_path)
    else:
        output_path.write_bytes(payload)

    loaded = load_quality_admission_result_index(
        ROOT,
        output_path,
        accepted_path,
        require_private_bundles=False,
    )
    print(
        canonical_json(
            {
                "content_hash": loaded.content_hash,
                "environment_registry_hash": loaded.environment_registry_hash,
                "task_count": loaded.task_count,
                "verified_count": loaded.verified_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
