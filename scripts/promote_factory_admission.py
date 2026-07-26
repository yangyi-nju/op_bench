#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from op_bench.factory.artifacts import (
    FactoryArtifactStore,
    load_factory_contract,
)
from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    FactoryAdmissionRecord,
)
from op_bench.factory.lifecycle import validate_admission_chain
from op_bench.factory.promotion import build_verified_admission_chain
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote validated Factory inputs to a verified chain.",
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--admission-evidence", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _json_mapping(path: Path, label: str) -> dict[str, object]:
    if path.is_symlink():
        raise ContractError(f"{label}: symlink is denied")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: cannot read JSON") from exc
    if not isinstance(value, Mapping):
        raise ContractError(f"{label}: expected JSON object")
    return {str(key): item for key, item in value.items()}


def _load_inputs(
    *,
    candidate_path: Path,
    decision_path: Path,
    task_path: Path,
    admission_path: Path,
    review_path: Path,
) -> tuple[
    CandidateRecord,
    DecisionRecord,
    TaskManifest,
    dict[str, object],
    dict[str, object],
]:
    candidate = load_factory_contract(candidate_path)
    if not isinstance(candidate, CandidateRecord):
        raise ContractError("candidate: expected factory_candidate contract")
    decision = load_factory_contract(decision_path)
    if not isinstance(decision, DecisionRecord):
        raise ContractError("decision: expected factory_decision contract")
    if task_path.is_symlink():
        raise ContractError("task: symlink is denied")
    try:
        task = TaskManifest.load(task_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("task: cannot read JSON") from exc
    admission = _json_mapping(admission_path, "admission evidence")
    review = _json_mapping(review_path, "review")
    return candidate, decision, task, admission, review


def _write_atomic_chain(
    output: Path,
    records: tuple[FactoryAdmissionRecord, ...],
) -> None:
    if os.path.lexists(output):
        raise ContractError("output directory: must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
        )
    )
    chain_paths: list[Path] = []
    try:
        with FactoryArtifactStore(temporary) as store:
            for index, record in enumerate(records, start=1):
                filename = (
                    f"chain/{index:02d}-{record.state.replace('_', '-')}.json"
                )
                store.write_contract(filename, record)
                chain_paths.append(temporary / filename)
            store.write_contract("admission.json", records[-1])

        parsed = tuple(load_factory_contract(path) for path in chain_paths)
        if not all(
            isinstance(record, FactoryAdmissionRecord) for record in parsed
        ):
            raise ContractError("output verification: unexpected contract type")
        validate_admission_chain(parsed)
        final = load_factory_contract(temporary / "admission.json")
        if final != parsed[-1]:
            raise ContractError("output verification: final Admission mismatch")
        if os.path.lexists(output):
            raise ContractError("output directory: appeared during promotion")
        os.rename(temporary, output)
        directory = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if os.path.lexists(args.output_dir):
            raise ContractError("output directory: must not already exist")
        candidate, decision, task, admission, review = _load_inputs(
            candidate_path=args.candidate,
            decision_path=args.decision,
            task_path=args.task,
            admission_path=args.admission_evidence,
            review_path=args.review,
        )
        records = build_verified_admission_chain(
            candidate=candidate,
            decision=decision,
            task=task,
            admission=admission,
            review=review,
            created_at=args.created_at,
        )
        _write_atomic_chain(args.output_dir, records)
    except (ContractError, OSError) as exc:
        print(f"[contract_invalid] {exc}", file=sys.stderr)
        return 2

    print(
        canonical_json(
            {
                "admission_id": records[-1].admission_id,
                "records": len(records),
                "state": records[-1].state,
                "status": "promoted",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
