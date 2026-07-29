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
from op_bench.factory.quality_release import validate_quality_task
from op_bench.runtime.validation import ContractError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--require-verified", action="store_true")
    args = parser.parse_args(argv)
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
