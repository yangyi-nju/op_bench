#!/usr/bin/env python3
"""Audit every pre-quality v0.7 Task into a deterministic disposition index."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.quality_release import write_historical_dispositions
from op_bench.runtime.validation import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args(argv)
    try:
        records = write_historical_dispositions(
            ROOT,
            _resolve(args.dataset),
            _resolve(args.review_root),
            _resolve(args.output),
            args.created_at,
        )
    except (ContractError, OSError) as exc:
        print(f"historical audit failed: {exc}", file=sys.stderr)
        return 1
    retained = sum(item.disposition == "retained" for item in records)
    deferred = sum(item.disposition == "deferred" for item in records)
    retired = sum(item.disposition == "retired" for item in records)
    print(
        f"{args.output}: wrote {len(records)} dispositions "
        f"(retained={retained}, deferred={deferred}, retired={retired})"
    )
    return 0


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
