#!/usr/bin/env python3
"""Create a strict Task/result shard from an existing v0.7 Admission index."""

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
    load_quality_admission_result_index,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"output is outside repository: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-index", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output-accepted", required=True)
    parser.add_argument("--output-results", required=True)
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="Select every and only fully verified outcome.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verified_only:
        raise SystemExit("a selection policy such as --verified-only is required")
    accepted_path = _rooted(args.accepted_index)
    result_path = _rooted(args.results)
    output_accepted = _rooted(args.output_accepted)
    output_results = _rooted(args.output_results)
    for output in (output_accepted, output_results):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")

    accepted = load_quality_accepted_task_index(
        ROOT, accepted_path, require_complete=False
    )
    results = load_quality_admission_result_index(
        ROOT,
        result_path,
        accepted_path,
        require_private_bundles=True,
    )
    selected_results = tuple(
        result for result in results.results if result.verified
    )
    if not selected_results:
        raise SystemExit("selection is empty")
    tasks_by_id = {task.task_id: task for task in accepted.tasks}
    selected_tasks = tuple(
        tasks_by_id[result.task_id] for result in selected_results
    )
    shard_accepted = replace(
        accepted,
        status="building",
        task_count=len(selected_tasks),
        tasks=selected_tasks,
    )
    shard_results = replace(
        results,
        accepted_index_path=_relative(output_accepted),
        accepted_index_hash=shard_accepted.content_hash,
        task_count=len(selected_results),
        verified_count=len(selected_results),
        results=selected_results,
    )
    try:
        output_accepted.parent.mkdir(parents=True, exist_ok=True)
        output_results.parent.mkdir(parents=True, exist_ok=True)
        output_accepted.write_bytes(
            canonical_json(shard_accepted.to_dict()).encode("utf-8")
        )
        output_results.write_bytes(
            canonical_json(shard_results.to_dict()).encode("utf-8")
        )
        loaded = load_quality_admission_result_index(
            ROOT,
            output_results,
            output_accepted,
            require_private_bundles=True,
        )
    except BaseException:
        output_results.unlink(missing_ok=True)
        output_accepted.unlink(missing_ok=True)
        raise
    print(canonical_json({
        "accepted_index_hash": shard_accepted.content_hash,
        "result_index_hash": loaded.content_hash,
        "task_count": loaded.task_count,
        "verified_count": loaded.verified_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
