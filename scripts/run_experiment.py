#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
import sys
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
from op_bench.resume import BASELINE_CACHE_DIRNAME, BaselineCache, ResultsStore, RunState, RUN_STATE_FILE
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
        "--only-tasks",
        nargs="+",
        default=None,
        metavar="TASK_ID",
        help=(
            "Only run these exact task_ids (post-resume filter). "
            "Combines with --filter-tasks. Useful for precise replay."
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
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Maximum number of (task × agent × attempt) triples to run concurrently. "
            "Default 1 (serial). Parallelism is at the attempt level; each attempt gets "
            "its own workspace and container. GPU tiers should keep N=1 to avoid "
            "contending for the single --gpus all allocation."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for result artifacts.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete --output-dir contents before starting. Without --fresh, existing "
            "results.jsonl / baselines.jsonl are honored and only missing attempts are run."
        ),
    )
    parser.add_argument(
        "--no-baseline-cache",
        action="store_true",
        help="Disable the cross-run baseline cache (runs/_baseline_cache/).",
    )
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

    tasks = _load_tasks(args.task, args.dataset, verified_only=args.verified_only)
    if args.filter_tasks:
        tasks = [t for t in tasks if any(pat in t.task_id for pat in args.filter_tasks)]
    if args.only_tasks:
        wanted = set(args.only_tasks)
        tasks = [t for t in tasks if t.task_id in wanted]
    if not tasks:
        print("no tasks matched filters", file=sys.stderr)
        return 2

    # --- Resume housekeeping --------------------------------------------------

    if args.fresh and output_dir.exists():
        progress(f"--fresh: removing existing output dir {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    current_state = RunState.build(
        task_ids=[t.task_id for t in tasks],
        agents=args.agent,
        agent_repeat=args.agent_repeat,
        only_tasks=args.only_tasks or (),
    )
    prior_state = RunState.load(output_dir / RUN_STATE_FILE)
    resuming = prior_state is not None
    if resuming:
        ok, reason = current_state.is_compatible(prior_state)
        if not ok:
            print(f"cannot resume: {reason}", file=sys.stderr)
            return 3
        progress(f"resuming from {output_dir} (previous run state matched)")
    current_state.save(output_dir / RUN_STATE_FILE)

    store = ResultsStore(output_dir)
    completed_keys = store.completed_agent_keys() if resuming else set()
    completed_baselines = store.completed_baseline_task_ids() if resuming else {}

    cache_dir = None if args.no_baseline_cache else (ROOT / "runs" / BASELINE_CACHE_DIRNAME)
    baseline_cache = BaselineCache(cache_dir)

    if resuming and (completed_keys or completed_baselines):
        progress(
            f"resume: {len(completed_baselines)} baseline(s) + {len(completed_keys)} agent attempt(s) already recorded"
        )

    # --- Runtime setup --------------------------------------------------------

    environment_manager = EnvironmentManager()
    environment_manager.progress = progress
    evaluator = Evaluator(environment_manager=environment_manager)
    evaluator.progress = progress

    progress(
        f"experiment start: tasks={len(tasks)}, agents={args.agent}, "
        f"repeat={args.agent_repeat}, output={output_dir}"
    )

    total_attempts = len(tasks) * len(args.agent) * args.agent_repeat
    remaining_attempts = sum(
        1
        for t in tasks
        for a in args.agent
        for i in range(1, args.agent_repeat + 1)
        if (t.task_id, a, i) not in completed_keys
    )
    progress(f"attempts: {remaining_attempts} pending / {total_attempts} total")

    # --- Main loop ------------------------------------------------------------

    for task_index, task in enumerate(tasks, start=1):
        progress(f"task {task_index}/{len(tasks)} start: {task.task_id}")

        # -- Baseline (with resume + cross-run cache) --------------------------

        cached_baseline_record = completed_baselines.get(task.task_id)
        baseline_record: dict[str, object] | None = None
        baseline_status: str | None = None
        if cached_baseline_record is not None:
            baseline_record = cached_baseline_record
            baseline_status = str(cached_baseline_record.get("status", ""))
            progress(f"task {task.task_id} baseline: reused from prior run (status={baseline_status})")

        if baseline_record is None:
            cache_key = baseline_cache.key_for(
                task.task_id, task.source_snapshot_hash, task.hidden_test_patch_path
            )
            cross_cache = baseline_cache.get(cache_key)
            if cross_cache is not None:
                baseline_record = dict(cross_cache)
                baseline_record["agent"] = "baseline"
                baseline_record["task_id"] = task.task_id
                baseline_status = str(baseline_record.get("status", ""))
                progress(f"task {task.task_id} baseline: cache hit (status={baseline_status})")
                store.append_baseline(baseline_record)

        if baseline_record is None:
            baseline = evaluator.evaluate_baseline(task)
            baseline_record = _record(agent="baseline", result=baseline)
            baseline_status = baseline.status
            progress(
                f"task {task.task_id} baseline: status={baseline.status}, "
                f"fail_to_pass={baseline_record['fail_to_pass_passed']}/{baseline_record['fail_to_pass_total']}, "
                f"pass_to_pass={baseline_record['pass_to_pass_passed']}/{baseline_record['pass_to_pass_total']}"
            )
            store.append_baseline(baseline_record)
            cache_key = baseline_cache.key_for(
                task.task_id, task.source_snapshot_hash, task.hidden_test_patch_path
            )
            if cache_key is not None and baseline_status == "baseline_reproduced":
                # only cache stable, successful baselines — a failed baseline may be
                # a transient environment issue and shouldn't stick across runs.
                baseline_cache.put(cache_key, baseline_record)

        if baseline_status == "environment_unavailable":
            progress(f"task {task.task_id} skipped: environment_unavailable")
            _append_skipped_agents(
                store, completed_keys, args.agent, task,
                "environment_unavailable", baseline_record, args.agent_repeat,
            )
            continue
        if baseline_status != "baseline_reproduced":
            progress(f"task {task.task_id} skipped: baseline status={baseline_status}")
            _append_skipped_agents(
                store, completed_keys, args.agent, task,
                "task_not_reproduced", baseline_record, args.agent_repeat,
            )
            continue

        # -- Agent attempts (with resume) --------------------------------------

        # -- Agent attempts (with resume + optional parallelism) ---------------

        for agent_name in args.agent:
            for attempt in range(1, args.agent_repeat + 1):
                if (task.task_id, agent_name, attempt) in completed_keys:
                    progress(f"skip (already done): task={task.task_id}, agent={agent_name}, attempt={attempt}")

        pending_to_run = [
            (agent_name, attempt)
            for agent_name in args.agent
            for attempt in range(1, args.agent_repeat + 1)
            if (task.task_id, agent_name, attempt) not in completed_keys
        ]

        if args.max_parallel > 1 and len(pending_to_run) > 1:
            import concurrent.futures
            import threading
            store_lock = threading.Lock()

            def run_and_store(agent_name: str, attempt: int) -> None:
                record = _run_single_attempt(
                    task=task,
                    agent_name=agent_name,
                    attempt=attempt,
                    agent_repeat=args.agent_repeat,
                    output_dir=output_dir,
                    patches_dir=patches_dir,
                    hide_public_tests=args.no_public_tests,
                    environment_manager=environment_manager,
                    evaluator=evaluator,
                    progress=progress,
                )
                with store_lock:
                    store.append_result(record)
                    completed_keys.add((task.task_id, agent_name, attempt))

            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
                futures = [pool.submit(run_and_store, a, i) for a, i in pending_to_run]
                for f in concurrent.futures.as_completed(futures):
                    f.result()  # propagate exceptions
        else:
            for agent_name, attempt in pending_to_run:
                record = _run_single_attempt(
                    task=task,
                    agent_name=agent_name,
                    attempt=attempt,
                    agent_repeat=args.agent_repeat,
                    output_dir=output_dir,
                    patches_dir=patches_dir,
                    hide_public_tests=args.no_public_tests,
                    environment_manager=environment_manager,
                    evaluator=evaluator,
                    progress=progress,
                )
                store.append_result(record)
                completed_keys.add((task.task_id, agent_name, attempt))

    # --- Post-run: write summary from stored records --------------------------

    _write_summary(store, output_dir, progress)
    print(output_dir / "summary.json")
    return 0


def _run_single_attempt(
    *,
    task: TaskManifest,
    agent_name: str,
    attempt: int,
    agent_repeat: int,
    output_dir: Path,
    patches_dir: Path,
    hide_public_tests: bool,
    environment_manager: EnvironmentManager,
    evaluator: Evaluator,
    progress,
) -> dict[str, object]:
    agent = agent_by_name(agent_name, progress=progress, hide_public_tests=hide_public_tests)
    agent_label = str(getattr(agent, "agent_id", getattr(agent, "name", agent_name)))
    attempt_label = f"attempt_{attempt:03d}"
    progress(f"agent start: task={task.task_id}, agent={agent_label}, attempt={attempt}/{agent_repeat}")

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
            return _error_record(
                agent_label, attempt, task, "runner_error", prepare_error, {}, [],
            )
        if task.public_test_patch_path is not None and task.public_test_patch_path.exists() and not hide_public_tests:
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
            return _error_record(
                agent_label, attempt, task, "environment_unavailable",
                environment_preparation.error,
                environment_preparation.evidence,
                environment_preparation.commands_as_dicts(),
            )
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
        return _error_record(
            agent_label, attempt, task, "agent_runtime_unsupported", str(exc), environment, commands,
        )
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
    progress(
        f"agent result: task={task.task_id}, agent={agent_label}, status={result.status}, "
        f"fail_to_pass={result.fail_to_pass_passed}/{result.fail_to_pass_total}, "
        f"pass_to_pass={result.pass_to_pass_passed}/{result.pass_to_pass_total}, "
        f"duration={format_duration(result.duration_sec)}"
    )
    return record


def _write_summary(store: ResultsStore, output_dir: Path, progress) -> None:
    results = store.load_all_results()
    baselines = store.load_all_baselines()
    progress(f"write summary: {output_dir / 'summary.json'}")
    # Preserve the legacy layout: agents-summary from result records,
    # baselines listed separately, and a top-level baseline_count.
    summary = summarize_results(results)
    summary["baselines"] = baselines
    summary["baseline_count"] = len(baselines)
    write_json(output_dir / "summary.json", summary)
    # Also refresh results.jsonl in canonical (sorted) form for downstream tooling —
    # the append-only file is authoritative, but a re-sorted mirror is nicer to diff.
    # (Skipped to avoid touching evidence; consumers should read results.jsonl directly.)


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


def _error_record(
    agent_label: str,
    attempt: int,
    task: TaskManifest,
    status: str,
    error: str | None,
    environment: dict[str, object],
    commands: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "agent": agent_label,
        "attempt": attempt,
        "task_id": task.task_id,
        "mode": f"agent:{agent_label}",
        "status": status,
        "error": error,
        "fail_to_pass_total": len(task.fail_to_pass_tests),
        "fail_to_pass_passed": 0,
        "pass_to_pass_total": len(task.pass_to_pass_tests),
        "pass_to_pass_passed": 0,
        "duration_sec": 0.0,
        "environment": environment,
        "commands": commands,
    }


def _append_skipped_agents(
    store: ResultsStore,
    completed_keys: set[tuple[str, str, int]],
    agent_names: list[str],
    task: TaskManifest,
    status: str,
    baseline_record: dict[str, object],
    repeat: int,
) -> None:
    for agent_name in agent_names:
        for attempt in range(1, repeat + 1):
            key = (task.task_id, agent_name, attempt)
            if key in completed_keys:
                continue
            record = {
                "agent": agent_name,
                "attempt": attempt,
                "task_id": task.task_id,
                "mode": f"agent:{agent_name}",
                "status": status,
                "baseline_status": baseline_record.get("status"),
                "fail_to_pass_total": len(task.fail_to_pass_tests),
                "fail_to_pass_passed": 0,
                "pass_to_pass_total": len(task.pass_to_pass_tests),
                "pass_to_pass_passed": 0,
                "duration_sec": 0.0,
                "environment": baseline_record.get("environment", {}),
                "commands": baseline_record.get("commands", []),
            }
            store.append_result(record)
            completed_keys.add(key)


if __name__ == "__main__":
    raise SystemExit(main())
