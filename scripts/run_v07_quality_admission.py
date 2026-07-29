#!/usr/bin/env python3
"""Run v0.7 quality admission for the exact accepted-index Task subset."""

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

from op_bench.factory.quality_admission import (  # noqa: E402
    run_quality_admission,
)
from op_bench.runtime.validation import ContractError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-index",
        required=True,
        help="Canonical accepted_tasks.json; it is the only Task source.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Canonical admission result index to write.",
    )
    parser.add_argument(
        "--environment-registry",
        default="environments/registry.json",
        help="Environment registry used to resolve every accepted Task.",
    )
    parser.add_argument(
        "--source-registry",
        default="sources/registry.json",
        help="Source registry used to resolve every accepted Task.",
    )
    parser.add_argument(
        "--created-at",
        required=True,
        help="Canonical UTC RFC3339-seconds timestamp for the result index.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reserve terminal output for the final JSON summary.",
    )
    return parser


def _rooted(value: str) -> Path:
    selected = Path(value)
    return (
        selected.absolute()
        if selected.is_absolute()
        else (ROOT / selected).absolute()
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    accepted_index = _rooted(args.accepted_index)
    output = _rooted(args.output)
    environment_registry = _rooted(args.environment_registry)
    source_registry = _rooted(args.source_registry)
    try:
        results = run_quality_admission(
            root=ROOT,
            accepted_index_path=accepted_index,
            output_path=output,
            environment_registry_path=environment_registry,
            source_registry_path=source_registry,
            created_at=args.created_at,
        )
    except (ContractError, OSError, UnicodeDecodeError, ValueError) as exc:
        print(f"quality admission failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "accepted_index": str(accepted_index),
                "output": str(output),
                "task_count": results.task_count,
                "verified_count": results.verified_count,
                "verified": (
                    results.verified_count == results.task_count
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if results.verified_count == results.task_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
