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

from op_bench.admission import AdmissionRunner
from op_bench.environment import EnvironmentManager
from op_bench.evaluator import Evaluator
from op_bench.progress import ProgressLogger
from op_bench.task import TaskManifest
from scripts.validate_task import validate_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline/gold admission replay for one OpBench task.")
    parser.add_argument("--task", required=True, help="Task directory or task.json path.")
    parser.add_argument("--output-dir", help="Directory for full admission evidence and logs.")
    parser.add_argument(
        "--write-task-evidence",
        action="store_true",
        help="Write a stable evidence summary to <task>/admission/evidence.json.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal progress logs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_path = Path(args.task).resolve()
    if task_path.is_dir():
        task_path = task_path / "task.json"
    task = TaskManifest.load(task_path)
    validation_errors = validate_manifest(task.data)
    if validation_errors:
        print(f"{task_path}: invalid task", file=sys.stderr)
        for error in validation_errors:
            print(f"  - {error}", file=sys.stderr)
        return 2
    progress = ProgressLogger(enabled=not args.quiet)
    environment_manager = EnvironmentManager(progress=progress)
    evaluator = Evaluator(environment_manager=environment_manager, progress=progress)
    runner = AdmissionRunner(evaluator=evaluator, progress=progress)

    evidence = runner.run(task)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(evidence.task_id, evidence.created_at)
    runner.write_bundle(evidence, output_dir)
    stable_evidence = runner.write_task_evidence(task, evidence) if args.write_task_evidence else None
    summary = {
        "task_id": evidence.task_id,
        "decision": evidence.decision,
        "verified": evidence.verified,
        "baseline": evidence.baseline["status"],
        "gold": evidence.gold["status"] if evidence.gold else None,
        "output_dir": str(output_dir),
        "task_evidence": str(stable_evidence) if stable_evidence else None,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if evidence.verified else 1


def _default_output_dir(task_id: str, created_at: str) -> Path:
    timestamp = created_at.replace(":", "").replace("-", "")
    return ROOT / "runs" / "admission" / task_id / timestamp


if __name__ == "__main__":
    raise SystemExit(main())
