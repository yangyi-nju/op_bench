#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from op_bench.factory.artifacts import load_factory_contract
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one canonical OpBench Factory artifact.",
    )
    parser.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = load_factory_contract(args.path)
    except (ContractError, OSError) as exc:
        print(f"[contract_invalid] {exc}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "content_hash": contract.content_hash,
                "contract_type": contract.contract_type,
                "status": "valid",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
