#!/usr/bin/env python3
"""Recover and revalidate v0.7 quality gates without rerunning Tasks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.quality_admission import (  # noqa: E402
    load_quality_admission_result_index,
    revalidate_quality_admission_result_index,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--accepted-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--replace-input",
        action="store_true",
        help="Atomically replace --input with the rebound result index.",
    )
    parser.add_argument(
        "--confirm-reuse-runtime-evidence",
        action="store_true",
        help=(
            "Assert that exact bound Runtime bundles may be reused to finish "
            "an interrupted lifecycle."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_reuse_runtime_evidence:
        raise SystemExit(
            "refusing to revalidate without "
            "--confirm-reuse-runtime-evidence"
        )
    input_path = _rooted(args.input)
    accepted_path = _rooted(args.accepted_index)
    output_path = _rooted(args.output)
    if args.replace_input and output_path.resolve() != input_path.resolve():
        raise SystemExit("--replace-input requires --output to equal --input")
    if output_path.exists() and not args.replace_input:
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    result = revalidate_quality_admission_result_index(
        ROOT,
        input_path,
        accepted_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(result.to_dict()).encode("utf-8")
    if args.replace_input:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.name}.",
            dir=output_path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output_path)
        finally:
            if temporary.exists():
                temporary.unlink()
    else:
        output_path.write_bytes(content)
    loaded = load_quality_admission_result_index(
        ROOT,
        output_path,
        accepted_path,
        require_private_bundles=True,
    )
    print(canonical_json({
        "content_hash": loaded.content_hash,
        "task_count": loaded.task_count,
        "verified_count": loaded.verified_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
