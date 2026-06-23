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

from op_bench.agents import AgentRuntimeUnsupported, agent_by_name
from op_bench.actions import WorkspaceActions
from op_bench.dataset import DatasetManifest
from op_bench.environment import EnvironmentManager
from op_bench.evaluator import Evaluator
from op_bench.progress import ProgressLogger, format_duration
from op_bench.registry import load_resolved_task
from op_bench.reporter import summarize_results, write_json, write_jsonl
from op_bench.task import TaskManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an op_bench experiment.")
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Task directory containing task.json. May be provided multiple times.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset manifest containing task entries. May be provided multiple times.",
    )
    parser.add_argument(
        "--verified-only",
        action="store_true",
        help="When using --dataset, run only task entries with admission_status='verified'.",
    )
    parser.add_argument(
        "--filter-tasks",
        nargs="+",
        default=None,
        metavar="PATTERN",
        help=(
            "Only run tasks whose task_id contains one of the given substrings. "
            "Applied after --verified-only filtering. "
            "Example: --filter-tasks lazylinear autograd"
        ),
    )
    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help=(
            "Agent adapter to run, e.g. gold or codex_action_bridge. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--agent-repeat",
        type=int,
        default=1,
        help="Number of independent attempts to run for each agent on each reproduced task.",
    )
    parser.add_argument(
        "--no-public-tests",
        action="store_true",
        help=(
            "Hide public tests from the agent (skip applying public_test.patch in the agent workspace "
            "and omit public_tests from the agent prompt). Used for ablation experiments."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for result artifacts.")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal progress logs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.task and not args.dataset:
        print("at least one --task or --dataset is required", file=sys.stderr)
        return 2
    if args.agent_repeat < 1:
        print("--agent-repeat must be >= 1", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir).resolve()
    patches_dir = output_dir / "patches"
    progress = ProgressLogger(enabled=not args.quiet)
    environment_manager = EnvironmentManager()
    environment_manager.progress = progress
    evaluator = Evaluator(environment_manager=environment_manager)
    evaluator.progress = progress
    records: list[dict[str, object]] = []
    tasks = _load_tasks(args.task, args.dataset, verified_only=args.verified_only)
    if args.filter_tasks:
        tasks = [t for t in tasks if any(pat in t.task_id for pat in args.filter_tasks)]
        if not tasks:
            print(f"no tasks matched --filter-tasks {args.filter_tasks}", file=sys.stderr)
            return 2
    progress(
        f"experiment start: tasks={len(tasks)}, agents={args.agent}, "
        f"repeat={args.agent_repeat}, output={output_dir}"
    )

    for task_index, task in enumerate(tasks, start=1):
        progress(f"task {task_index}/{len(tasks)} start: {task.task_id}")
        baseline = evaluator.evaluate_baseline(task)
        baseline_record = _record(agent="baseline", result=baseline)
        records.append(baseline_record)
        progress(
            f"task {task.task_id} baseline: status={baseline.status}, "
            f"fail_to_pass={baseline_record['fail_to_pass_passed']}/{baseline_record['fail_to_pass_total']}, "
            f"pass_to_pass={baseline_record['pass_to_pass_passed']}/{baseline_record['pass_to_pass_total']}"
        )
        if baseline.status == "environment_unavailable":
            progress(f"task {task.task_id} skipped: environment_unavailable")
            records.extend(_skipped_agent_records(args.agent, task, "environment_unavailable", baseline, args.agent_repeat))
            continue
        if baseline.status != "baseline_reproduced":
            progress(f"task {task.task_id} skipped: baseline status={baseline.status}")
            records.extend(_skipped_agent_records(args.agent, task, "task_not_reproduced", baseline, args.agent_repeat))
            continue

        for agent_name in args.agent:
            for attempt in range(1, args.agent_repeat + 1):
                agent = agent_by_name(agent_name, progress=progress, hide_public_tests=args.no_public_tests)
                agent_label = str(getattr(agent, "agent_id", getattr(agent, "name", agent_name)))
                attempt_label = f"attempt_{attempt:03d}"
                progress(f"agent start: task={task.task_id}, agent={agent_label}, attempt={attempt}/{args.agent_repeat}")
                workspace = None
                actions = None
                environment_preparation = None
                if getattr(agent, "requires_workspace", False) or getattr(agent, "requires_actions", False):
                    workspace = output_dir / "workspaces" / task.task_id / _safe_name(agent_label) / attempt_label
                    if workspace.exists():
                        shutil.rmtree(workspace)
                    workspace.parent.mkdir(parents=True, exist_ok=True)
                    progress(f"agent workspace prepare: {workspace}")
                    prepare_error = evaluator.prepare_workspace(task, workspace)
                    if prepare_error is not None:
                        progress(f"agent workspace failed: task={task.task_id}, agent={agent_label}, error={prepare_error}")
                        records.append(
                            {
                                "agent": agent_label,
                                "attempt": attempt,
                                "task_id": task.task_id,
                                "mode": f"agent:{agent_label}",
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
                    if task.public_test_patch_path is not None and task.public_test_patch_path.exists() and not args.no_public_tests:
                        import subprocess
                        patch_result = subprocess.run(
                            ["git", "apply", str(task.public_test_patch_path)],
                            cwd=str(workspace), capture_output=True, text=True, timeout=30,
                        )
                        if patch_result.returncode != 0:
                            progress(f"agent public_test_patch apply failed: {patch_result.stderr}")
                    progress(f"agent environment prepare: task={task.task_id}, agent={agent_label}")
                    environment_preparation = environment_manager.prepare(task, workspace)
                    if not environment_preparation.available:
                        progress(
                            f"agent environment unavailable: task={task.task_id}, "
                            f"agent={agent_label}, error={environment_preparation.error}"
                        )
                        records.append(
                            {
                                "agent": agent_label,
                                "attempt": attempt,
                                "task_id": task.task_id,
                                "mode": f"agent:{agent_label}",
                                "status": "environment_unavailable",
                                "error": environment_preparation.error,
                                "fail_to_pass_total": len(task.fail_to_pass_tests),
                                "fail_to_pass_passed": 0,
                                "pass_to_pass_total": len(task.pass_to_pass_tests),
                                "pass_to_pass_passed": 0,
                                "duration_sec": 0.0,
                                "environment": environment_preparation.evidence,
                                "commands": environment_preparation.commands_as_dicts(),
                            }
                        )
                        continue
                    if getattr(agent, "requires_actions", False):
                        actions = WorkspaceActions(task=task, workspace=workspace, command_executor=environment_preparation.executor)
                try:
                    progress(f"agent patch generation start: task={task.task_id}, agent={agent_label}")
                    agent_output = agent.produce_patch(
                        task,
                        patches_dir / _safe_name(agent_label) / attempt_label,
                        workspace=workspace,
                        actions=actions,
                    )
                    progress(
                        f"agent patch generation done: task={task.task_id}, agent={agent_label}, "
                        f"patch={agent_output.patch_path}"
                    )
                except AgentRuntimeUnsupported as exc:
                    environment = environment_preparation.evidence if environment_preparation is not None else {}
                    commands = environment_preparation.commands_as_dicts() if environment_preparation is not None else []
                    if environment_preparation is not None:
                        cleanup_result = environment_manager.cleanup(environment_preparation)
                        environment_preparation = None
                        if cleanup_result is not None:
                            commands.append(cleanup_result.to_dict())
                    progress(f"agent unsupported: task={task.task_id}, agent={agent_label}, error={exc}")
                    records.append(
                        {
                            "agent": agent_label,
                            "attempt": attempt,
                            "task_id": task.task_id,
                            "mode": f"agent:{agent_label}",
                            "status": "agent_runtime_unsupported",
                            "error": str(exc),
                            "fail_to_pass_total": len(task.fail_to_pass_tests),
                            "fail_to_pass_passed": 0,
                            "pass_to_pass_total": len(task.pass_to_pass_tests),
                            "pass_to_pass_passed": 0,
                            "duration_sec": 0.0,
                            "environment": environment,
                            "commands": commands,
                        }
                    )
                    continue
                finally:
                    if environment_preparation is not None:
                        progress(f"agent environment cleanup: task={task.task_id}, agent={agent_label}")
                        environment_manager.cleanup(environment_preparation)
                progress(f"agent patch evaluation start: task={task.task_id}, agent={agent_label}")
                result = evaluator.evaluate_patch(task, agent_output.patch_path, agent_label)
                record = _record(agent=agent_label, result=result)
                record["attempt"] = attempt
                record["agent_metadata"] = agent_output.metadata
                record["patch_path"] = str(agent_output.patch_path)
                records.append(record)
                progress(
                    f"agent result: task={task.task_id}, agent={agent_label}, status={result.status}, "
                    f"fail_to_pass={result.fail_to_pass_passed}/{result.fail_to_pass_total}, "
                    f"pass_to_pass={result.pass_to_pass_passed}/{result.pass_to_pass_total}, "
                    f"duration={format_duration(result.duration_sec)}"
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress(f"write results: {output_dir / 'results.jsonl'}")
    write_jsonl(output_dir / "results.jsonl", records)
    summary = summarize_results([record for record in records if record["agent"] != "baseline"])
    summary["baselines"] = [record for record in records if record["agent"] == "baseline"]
    progress(f"write summary: {output_dir / 'summary.json'}")
    write_json(output_dir / "summary.json", summary)
    print(output_dir / "summary.json")
    return 0


def _load_tasks(task_dirs: list[str], dataset_paths: list[str], verified_only: bool = False) -> list[TaskManifest]:
    tasks = [
        load_resolved_task(
            Path(task_dir) / "task.json",
            environment_registry_path=ROOT / "environments/registry.json",
            source_registry_path=ROOT / "sources/registry.json",
        )
        for task_dir in task_dirs
    ]
    for dataset_path in dataset_paths:
        dataset = DatasetManifest.load(dataset_path)
        tasks.extend(dataset.load_tasks(verified_only=verified_only))
    return tasks


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)


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


def _skipped_agent_records(
    agent_names: list[str],
    task: TaskManifest,
    status: str,
    baseline: object,
    repeat: int = 1,
) -> list[dict[str, object]]:
    baseline_data = baseline.to_dict()
    return [
        {
            "agent": agent_name,
            "attempt": attempt,
            "task_id": task.task_id,
            "mode": f"agent:{agent_name}",
            "status": status,
            "baseline_status": baseline_data["status"],
            "fail_to_pass_total": len(task.fail_to_pass_tests),
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": len(task.pass_to_pass_tests),
            "pass_to_pass_passed": 0,
            "duration_sec": 0.0,
            "environment": baseline_data["environment"],
            "commands": baseline_data["commands"],
        }
        for agent_name in agent_names
        for attempt in range(1, repeat + 1)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
