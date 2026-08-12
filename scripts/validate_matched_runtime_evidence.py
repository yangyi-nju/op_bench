#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.matched_runtime.contracts import (  # noqa: E402
    CHECK_STATUSES,
    COMPATIBILITY_STATUSES,
    MATCH_STRATEGIES,
    CompatibilityEvidence,
)
from op_bench.runtime.validation import ContractError  # noqa: E402


DEFAULT_SCHEMA = ROOT / "schemas" / "matched_runtime_compatibility.schema.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate canonical OpBench matched-runtime evidence.",
    )
    parser.add_argument("evidence", help="Path to compatibility evidence JSON.")
    parser.add_argument(
        "--schema",
        default=str(DEFAULT_SCHEMA),
        help="Schema whose public shape must match the Python contract.",
    )
    return parser


def _validate_schema_shape(schema: object) -> None:
    if not isinstance(schema, dict):
        raise ContractError("schema: expected object")
    required = schema.get("required")
    properties = schema.get("properties")
    if not isinstance(required, list) or not isinstance(properties, dict):
        raise ContractError("schema: required and properties are mandatory")
    expected = set(CompatibilityEvidence.wire_fields())
    if set(required) != expected or set(properties) != expected:
        raise ContractError("schema: top-level fields do not match contract")
    if schema.get("additionalProperties") is not False:
        raise ContractError("schema: additionalProperties must be false")
    if properties.get("contract_type", {}).get("const") != CompatibilityEvidence.contract_type:
        raise ContractError("schema: contract_type drift")
    if properties.get("schema_version", {}).get("const") != CompatibilityEvidence.schema_version:
        raise ContractError("schema: schema_version drift")
    if properties.get("strategy", {}).get("enum") != list(MATCH_STRATEGIES):
        raise ContractError("schema: strategy enum drift")
    if properties.get("status", {}).get("enum") != list(COMPATIBILITY_STATUSES):
        raise ContractError("schema: status enum drift")
    try:
        check_statuses = schema["$defs"]["check"]["properties"]["status"]["enum"]
    except (KeyError, TypeError) as exc:
        raise ContractError("schema: check status enum is missing") from exc
    if check_statuses != list(CHECK_STATUSES):
        raise ContractError("schema: check status enum drift")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evidence_path = Path(args.evidence)
    schema_path = Path(args.schema)
    try:
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise ContractError("evidence: expected a regular non-symlink file")
        payload = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence = CompatibilityEvidence.from_dict(payload)
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        _validate_schema_shape(schema)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"{evidence_path}: invalid matched-runtime evidence: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "content_hash": evidence.content_hash,
                "evidence_id": evidence.evidence_id,
                "status": evidence.status,
                "task_id": evidence.task_id,
            },
            sort_keys=True,
        )
    )
    return 0 if evidence.status == "compatible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
