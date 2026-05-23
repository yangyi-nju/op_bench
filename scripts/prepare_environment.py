#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.environment import EnvironmentManager
from op_bench.task import TaskManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and preflight the executable environment declared by one or more op_bench tasks. "
            "This checks Docker availability, builds a missing image when a Dockerfile is declared, "
            "starts an isolated container, runs preflight commands, then removes the container."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Task directory containing task.json. May be provided multiple times.",
    )
    parser.add_argument("--output", help="Optional JSON file for environment preparation evidence.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = EnvironmentManager()
    records: list[dict[str, object]] = []

    for task_dir in args.task:
        task = TaskManifest.load(Path(task_dir) / "task.json")
        with tempfile.TemporaryDirectory(prefix=f"op-bench-env-{task.task_id}-") as tmp:
            workspace = Path(tmp) / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            preparation = manager.prepare(task, workspace)
            cleanup_result = manager.cleanup(preparation)
            commands = preparation.commands_as_dicts()
            if cleanup_result is not None:
                commands.append(cleanup_result.to_dict())
            record: dict[str, object] = {
                "task_id": task.task_id,
                "status": preparation.status,
                "available": preparation.available,
                "environment": preparation.evidence,
                "commands": commands,
            }
            if preparation.error:
                record["error"] = preparation.error
            records.append(record)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(records, sort_keys=True))
    return 0 if all(record["available"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
