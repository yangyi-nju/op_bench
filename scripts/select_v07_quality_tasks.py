#!/usr/bin/env python3
"""Select an ordered v0.7 accepted-Task subset by exact Task id."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.quality_admission import (  # noqa: E402
    load_quality_accepted_task_index,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--task-id",
        action="append",
        required=True,
        help="Exact private Task id; repeat for every selected Task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = _rooted(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output}")
    requested = tuple(args.task_id)
    if len(requested) != len(set(requested)):
        raise ContractError("task selection contains duplicates")
    source = load_quality_accepted_task_index(
        ROOT,
        _rooted(args.accepted_index),
        require_complete=False,
    )
    requested_set = set(requested)
    selected = tuple(
        record for record in source.tasks if record.task_id in requested_set
    )
    missing = sorted(requested_set - {record.task_id for record in selected})
    if missing:
        raise ContractError(f"selected Task ids are unavailable: {missing}")
    subset = replace(
        source,
        status="building",
        task_count=len(selected),
        tasks=selected,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(subset.to_dict()).encode("utf-8"))
    loaded = load_quality_accepted_task_index(
        ROOT, output, require_complete=False
    )
    print(canonical_json({
        "content_hash": loaded.content_hash,
        "task_count": loaded.task_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
