#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.evaluator import Evaluator
from op_bench.task import TaskManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify baseline and gold replay for one op_bench task.")
    parser.add_argument("task", help="Task directory containing task.json")
    parser.add_argument("--output", help="Optional JSON file for replay evidence")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = TaskManifest.load(Path(args.task) / "task.json")
    evaluator = Evaluator()
    baseline = evaluator.evaluate_baseline(task)
    gold = evaluator.evaluate_gold(task)
    evidence = {
        "task_id": task.task_id,
        "verified": baseline.status == "baseline_reproduced" and gold.status == "resolved",
        "baseline": baseline.to_dict(),
        "gold": gold.to_dict(),
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"task_id": task.task_id, "verified": evidence["verified"], "baseline": baseline.status, "gold": gold.status}))
    return 0 if evidence["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
