#!/usr/bin/env python3
"""Validate formal quality evidence for a v0.7 Dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.dataset import DatasetManifest
from op_bench.factory.quality_release import (
    validate_candidate_index,
    validate_historical_index,
    validate_quality_task,
)
from op_bench.runtime.validation import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--historical-index", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--require-verified", action="store_true")
    args = parser.parse_args(argv)
    selected = sum(
        item is not None
        for item in (
            args.dataset,
            args.historical_index,
            args.candidate_index,
        )
    )
    if selected != 1:
        parser.error(
            "provide exactly one Dataset, --historical-index, "
            "or --candidate-index"
        )
    if args.candidate_index is not None:
        index_path = (
            args.candidate_index
            if args.candidate_index.is_absolute()
            else ROOT / args.candidate_index
        )
        errors = validate_candidate_index(ROOT, index_path)
        for error in errors:
            print(error, file=sys.stderr)
        if errors:
            print(
                f"{index_path}: candidate quality validation failed",
                file=sys.stderr,
            )
            return 1
        print(f"{index_path}: candidate funnel passed quality validation")
        return 0
    if args.historical_index is not None:
        index_path = (
            args.historical_index
            if args.historical_index.is_absolute()
            else ROOT / args.historical_index
        )
        errors = validate_historical_index(ROOT, index_path)
        for error in errors:
            print(error, file=sys.stderr)
        if errors:
            print(
                f"{index_path}: historical quality validation failed",
                file=sys.stderr,
            )
            return 1
        print(
            f"{index_path}: 25 Tasks passed historical quality validation "
            "(retained=14, deferred=1, retired=10)"
        )
        return 0
    assert args.dataset is not None
    dataset_path = (
        args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    )
    try:
        tasks = DatasetManifest.load(dataset_path).load_tasks()
    except (ContractError, OSError, ValueError) as exc:
        print(f"{dataset_path}: cannot load Dataset: {exc}", file=sys.stderr)
        return 1
    failures = 0
    for task in tasks:
        errors = validate_quality_task(
            ROOT,
            task,
            require_verified=args.require_verified,
        )
        for error in errors:
            print(f"{task.task_id}: {error}", file=sys.stderr)
        failures += bool(errors)
    if failures:
        print(
            f"{dataset_path}: {failures}/{len(tasks)} Tasks failed quality validation",
            file=sys.stderr,
        )
        return 1
    print(f"{dataset_path}: {len(tasks)} Tasks passed quality validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
