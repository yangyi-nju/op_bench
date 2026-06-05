#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.curation import curate_dataset, summarize_dataset
from scripts.validate_dataset import validate_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an auditable OpBench dataset slice and summary.")
    parser.add_argument("--dataset", required=True, help="Source dataset.json path.")
    parser.add_argument("--output-dataset", required=True, help="Output path for the curated dataset.")
    parser.add_argument("--output-summary", required=True, help="Output path for the curated dataset summary.")
    parser.add_argument("--verified-only", action="store_true", help="Include only tasks with admission_status=verified.")
    parser.add_argument("--dataset-id", help="Override the output dataset_id.")
    parser.add_argument("--version", help="Override the output dataset version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = Path(args.dataset).resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_errors = validate_dataset(source, source_path.parent)
    if source_errors:
        _print_errors("source dataset is invalid", source_errors)
        return 1

    curated = curate_dataset(
        source,
        verified_only=args.verified_only,
        dataset_id=args.dataset_id,
        version=args.version,
    )
    output_dataset = Path(args.output_dataset).resolve()
    output_dataset.parent.mkdir(parents=True, exist_ok=True)
    curated_errors = validate_dataset(
        curated,
        output_dataset.parent,
        require_verified=curated.get("status") == "verified",
    )
    if curated_errors:
        _print_errors("curated dataset is invalid", curated_errors)
        return 1

    output_dataset.write_text(json.dumps(curated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_summary = Path(args.output_summary).resolve()
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summarize_dataset(curated), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(output_dataset), "summary": str(output_summary)}, sort_keys=True))
    return 0


def _print_errors(title: str, errors: list[str]) -> None:
    print(title, file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
