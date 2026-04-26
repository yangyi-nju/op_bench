#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.agents import agent_by_name
from op_bench.evaluator import Evaluator
from op_bench.reporter import summarize_results, write_json, write_jsonl
from op_bench.task import TaskManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an op_bench experiment.")
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Task directory containing task.json. May be provided multiple times.",
    )
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        choices=["noop", "gold", "codex"],
        help="Agent adapter to run. May be provided multiple times.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for result artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    patches_dir = output_dir / "patches"
    evaluator = Evaluator()
    records: list[dict[str, object]] = []

    for task_dir in args.task:
        task = TaskManifest.load(Path(task_dir) / "task.json")
        baseline = evaluator.evaluate_baseline(task)
        baseline_record = _record(agent="baseline", result=baseline)
        records.append(baseline_record)

        for agent_name in args.agent:
            agent = agent_by_name(agent_name)
            workspace = None
            if agent_name == "codex":
                workspace = output_dir / "workspaces" / task.task_id / agent_name
                if workspace.exists():
                    shutil.rmtree(workspace)
                workspace.parent.mkdir(parents=True, exist_ok=True)
                prepare_error = evaluator.prepare_workspace(task, workspace)
                if prepare_error is not None:
                    records.append(
                        {
                            "agent": agent_name,
                            "task_id": task.task_id,
                            "mode": f"agent:{agent_name}",
                            "status": "runner_error",
                            "error": prepare_error,
                            "fail_to_pass_total": len(task.fail_to_pass_tests),
                            "fail_to_pass_passed": 0,
                            "pass_to_pass_total": len(task.pass_to_pass_tests),
                            "pass_to_pass_passed": 0,
                            "duration_sec": 0.0,
                            "environment": {},
                            "commands": [],
                        }
                    )
                    continue
            agent_output = agent.produce_patch(task, patches_dir / agent_name, workspace=workspace)
            result = evaluator.evaluate_patch(task, agent_output.patch_path, agent_name)
            record = _record(agent=agent_name, result=result)
            record["agent_metadata"] = agent_output.metadata
            record["patch_path"] = str(agent_output.patch_path)
            records.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "results.jsonl", records)
    summary = summarize_results([record for record in records if record["agent"] != "baseline"])
    summary["baselines"] = [record for record in records if record["agent"] == "baseline"]
    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")
    return 0


def _record(agent: str, result: object) -> dict[str, object]:
    data = result.to_dict()
    return {
        "agent": agent,
        "task_id": data["task_id"],
        "mode": data["mode"],
        "status": data["status"],
        "fail_to_pass_total": data["fail_to_pass_total"],
        "fail_to_pass_passed": data["fail_to_pass_passed"],
        "pass_to_pass_total": data["pass_to_pass_total"],
        "pass_to_pass_passed": data["pass_to_pass_passed"],
        "duration_sec": data["duration_sec"],
        "environment": data["environment"],
        "commands": data["commands"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
