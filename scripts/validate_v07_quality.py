#!/usr/bin/env python3
"""Validate formal quality evidence for a v0.7 Dataset."""

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

from op_bench.dataset import DatasetManifest
from op_bench.factory.quality_release import (
    validate_candidate_index,
    validate_historical_index,
    validate_quality_task,
)
from op_bench.runtime.validation import ContractError
from op_bench.factory.artifacts import load_regular_file_bytes
from scripts.build_v07_quality_release import build_release_outputs
from scripts.run_v07_quality_replay import validate_quality_replay_index
from scripts.run_v07_quality_validation import validate_quality_validation_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?")
    parser.add_argument("--historical-index", type=Path)
    parser.add_argument("--candidate-index", type=Path)
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--replay-index", type=Path)
    parser.add_argument("--validation-contract", type=Path)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args(argv)
    selected = sum(
        item is not None
        for item in (
            args.dataset,
            args.historical_index,
            args.candidate_index,
            args.release,
        )
    )
    if selected != 1:
        parser.error(
            "provide exactly one Dataset, --historical-index, "
            "--candidate-index, or --release with --replay-index"
        )
    if args.release is not None:
        if args.replay_index is None:
            parser.error("--release requires --replay-index")
        if (args.validation_contract is None) != (args.run_root is None):
            parser.error(
                "--validation-contract and --run-root must be provided together"
            )
        release_path = args.release if args.release.is_absolute() else ROOT / args.release
        replay_path = (
            args.replay_index
            if args.replay_index.is_absolute()
            else ROOT / args.replay_index
        )
        try:
            release = json.loads(
                load_regular_file_bytes(release_path).decode("utf-8")
            )
            created_at = release["created_at"]
            outputs = build_release_outputs(
                root=ROOT,
                historical_index_path=ROOT / "factory/v0.7/p7/historical_readmission.json",
                accepted_index_path=ROOT / "factory/v0.7/p8/accepted_tasks.json",
                admission_results_path=ROOT / "factory/v0.7/p8/admission_results.json",
                request_path=ROOT / "factory/v0.7/p9/release_request.json",
                created_at=created_at,
            )
            mismatches = [
                relative
                for relative, content in outputs.items()
                if not (ROOT / relative).is_file()
                or load_regular_file_bytes(ROOT / relative) != content
            ]
            replay_errors = validate_quality_replay_index(
                ROOT,
                replay_path,
                release_manifest_path=release_path,
            )
            validation_errors: list[str] = []
            if args.validation_contract is not None and args.run_root is not None:
                contract_path = (
                    args.validation_contract
                    if args.validation_contract.is_absolute()
                    else ROOT / args.validation_contract
                )
                run_root = (
                    args.run_root if args.run_root.is_absolute() else ROOT / args.run_root
                )
                validation_errors = validate_quality_validation_index(
                    ROOT,
                    run_root / "index.json",
                    release_path=release_path,
                    contract_path=contract_path,
                )
        except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
            print(f"final quality release validation failed: {exc}", file=sys.stderr)
            return 1
        if mismatches or replay_errors or validation_errors:
            if mismatches:
                print(
                    f"final quality release validation failed: {len(mismatches)} artifact mismatch(es)",
                    file=sys.stderr,
                )
            for error in replay_errors:
                print(f"final quality replay validation failed: {error}", file=sys.stderr)
            for error in validation_errors:
                print(
                    f"final quality Agent validation failed: {error}",
                    file=sys.stderr,
                )
            return 1
        suffix = (
            " and 122/122 Agent Attempts"
            if args.validation_contract is not None
            else ""
        )
        print(
            "final v0.7 quality release, 50/50 replay"
            f"{suffix} passed validation"
        )
        return 0
    if args.replay_index is not None:
        parser.error("--replay-index requires --release")
    if args.validation_contract is not None or args.run_root is not None:
        parser.error("Agent validation options require --release and --replay-index")
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
