#!/usr/bin/env python3
"""Build the canonical v0.7 quality candidate screening funnel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.quality_release import write_quality_candidate_funnel
from op_bench.runtime.validation import ContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen private merged-PR captures without assigning final "
            "taxonomy, complexity, or Admission truth."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index = write_quality_candidate_funnel(
            ROOT,
            _resolve(args.input),
            _resolve(args.receipts),
            _resolve(args.historical_index),
            _resolve(args.output_dir),
            args.created_at,
        )
    except (ContractError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"candidate screening failed: {exc}", file=sys.stderr)
        return 1
    print(
        "screened "
        f"{index['candidate_count']} real candidates "
        f"({index['eligible_candidate_count']} eligible; "
        f"required={index['required_candidate_count']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
