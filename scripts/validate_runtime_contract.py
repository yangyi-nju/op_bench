#!/usr/bin/env python

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

from op_bench.runtime.schema import load_runtime_schema, parse_runtime_contract
from op_bench.runtime.validation import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an OpBench v0.6 runtime contract locally. "
            "This command performs no Agent launch or remote validation."
        )
    )
    parser.add_argument("contract", help="Path to a runtime contract JSON file.")
    parser.add_argument("--schema", help="Optional path to runtime_contracts.schema.json.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.contract)
    if not path.is_file():
        print(f"contract not found: {path}", file=sys.stderr)
        return 2

    try:
        with path.open("r", encoding="utf-8") as handle:
            instance = json.load(handle)
        schema = load_runtime_schema(args.schema)
        contract = parse_runtime_contract(instance, schema)
        contract_type = contract.contract_type
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(f"invalid runtime contract: {exc}", file=sys.stderr)
        return 1

    print(f"{path}: valid {contract_type}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
