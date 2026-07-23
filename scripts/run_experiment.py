#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
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
from op_bench.integrity import replay_spec_hash
from op_bench.progress import ProgressLogger, format_duration
from op_bench.registry import load_resolved_task
from op_bench.reporter import compute_extended_metrics, summarize_results, write_json, write_jsonl
from op_bench.resume import BASELINE_CACHE_DIRNAME, BaselineCache, ResultsStore, RunState, RUN_STATE_FILE
from op_bench.task import TaskManifest


_PRIVATE_PROCESS_GROUP_RECOVERY = "private_process_group_recovery.json"


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
            "results.jsonl is honored and only missing attempts are run."
        ),
    )
    parser.add_argument(
        "--no-baseline-cache",
        action="store_true",
        help="Disable the cross-run baseline cache (runs/_baseline_cache/).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal progress logs.")
    parser.add_argument(
        "--runtime-protocol",
        choices=("legacy", "v1"),
        default="legacy",
        help="Runtime protocol. The default preserves the legacy runner.",
    )
    parser.add_argument(
        "--runtime-profile",
        default=None,
        metavar="PROFILE_ID",
        help="Exact Runtime Profile ID; required with --runtime-protocol v1.",
    )
    parser.add_argument(
        "--runtime-profile-registry",
        default=str(ROOT / "configs" / "runtime_profiles.v1.json"),
        metavar="PATH",
        help="Versioned Runtime Profile registry used only by the v1 runner.",
    )
    parser.add_argument(
        "--target-config",
        default=None,
        metavar="PATH",
        help="Private exact runtime target config. No target discovery is performed.",
    )
    parser.add_argument(
        "--codex-model",
        default=None,
        metavar="MODEL_ID",
        help="Exact model ID; required only for the v1 codex_mcp_canonical adapter.",
    )
    parser.add_argument(
        "--enable-external-canary",
        action="store_true",
        help="Explicitly permit a real v1 Codex adapter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runtime_protocol == "v1":
        return _main_v1(args)
    default_registry = str(ROOT / "configs" / "runtime_profiles.v1.json")
    if (
        args.runtime_profile is not None
        or args.target_config is not None
        or args.enable_external_canary
        or args.codex_model is not None
        or args.runtime_profile_registry != default_registry
    ):
        print("v1 runtime controls require --runtime-protocol v1", file=sys.stderr)
        return 2
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
        task_signatures=[replay_spec_hash(t) for t in tasks],
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

        _write_summary(store, output_dir, progress, tasks)

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
                    environment_manager=environment_manager,
                    evaluator=evaluator,
                    progress=progress,
                )
                with store_lock:
                    store.append_result(record)
                    completed_keys.add((task.task_id, agent_name, attempt))
                    _write_summary(store, output_dir, progress, tasks)

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
                    environment_manager=environment_manager,
                    evaluator=evaluator,
                    progress=progress,
                )
                store.append_result(record)
                completed_keys.add((task.task_id, agent_name, attempt))
                _write_summary(store, output_dir, progress, tasks)

    # --- Post-run: write summary from stored records --------------------------

    _write_summary(store, output_dir, progress, tasks)
    print(output_dir / "summary.json")
    return 0


def detect_codex_cli_version(codex_binary: str = "codex") -> str:
    """Read one exact local Codex CLI identity without network or discovery."""

    completed = subprocess.run(
        (codex_binary, "--version"),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = completed.stdout.strip() if isinstance(completed.stdout, str) else ""
    if (
        completed.returncode != 0
        or re.fullmatch(r"codex-cli [A-Za-z0-9][A-Za-z0-9.+-]*", output) is None
    ):
        from op_bench.runtime.validation import ContractError

        raise ContractError("cannot determine exact Codex CLI version")
    return output


def _recovery_payload(process_group_id: int) -> bytes:
    from op_bench.runtime.canonical import canonical_json
    from op_bench.runtime.validation import require_int

    selected = require_int(process_group_id, "process_group_id", minimum=1)
    return (
        canonical_json(
            {
                "record_type": "exact_process_group_cleanup_recovery",
                "schema_version": "v1",
                "process_group_id": selected,
            }
        )
        + "\n"
    ).encode("utf-8")


def _open_real_output_root(output_root: Path) -> int:
    descriptor = os.open(
        output_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise OSError("output root is not a directory")
    return descriptor


def _write_process_group_recovery(output_root: Path, process_group_id: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    encoded = _recovery_payload(process_group_id)
    root_descriptor = _open_real_output_root(output_root)
    try:
        try:
            descriptor = os.open(
                _PRIVATE_PROCESS_GROUP_RECOVERY,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_descriptor,
            )
        except FileExistsError:
            descriptor = os.open(
                _PRIVATE_PROCESS_GROUP_RECOVERY,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_descriptor,
            )
            try:
                metadata = os.fstat(descriptor)
                existing = os.read(descriptor, len(encoded) + 1)
            finally:
                os.close(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or existing != encoded:
                raise OSError("process group recovery marker conflict")
            return
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(root_descriptor)
    finally:
        os.close(root_descriptor)


def _resolve_process_group_recovery(output_root: Path) -> bool:
    from op_bench.runtime.process_group import exact_process_group_is_absent
    from op_bench.runtime.validation import require_exact_fields, require_int, require_str

    if not output_root.exists():
        return True
    root_descriptor = _open_real_output_root(output_root)
    try:
        try:
            descriptor = os.open(
                _PRIVATE_PROCESS_GROUP_RECOVERY,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=root_descriptor,
            )
        except FileNotFoundError:
            return True
        try:
            metadata = os.fstat(descriptor)
            encoded = os.read(descriptor, 1_025)
        finally:
            os.close(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or len(encoded) > 1_024:
            raise OSError("process group recovery marker is invalid")
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise OSError("process group recovery marker is invalid") from None
        data = require_exact_fields(
            value,
            "process_group_recovery",
            ("record_type", "schema_version", "process_group_id"),
        )
        if require_str(data["record_type"], "record_type") != (
            "exact_process_group_cleanup_recovery"
        ) or require_str(data["schema_version"], "schema_version") != "v1":
            raise OSError("process group recovery marker is invalid")
        process_group_id = require_int(
            data["process_group_id"],
            "process_group_id",
            minimum=1,
        )
        if encoded != _recovery_payload(process_group_id):
            raise OSError("process group recovery marker is not canonical")
        if not exact_process_group_is_absent(process_group_id):
            return False
        os.unlink(_PRIVATE_PROCESS_GROUP_RECOVERY, dir_fd=root_descriptor)
        os.fsync(root_descriptor)
        return True
    finally:
        os.close(root_descriptor)


def _main_v1(args: argparse.Namespace) -> int:
    """Validate explicit v1 selection before creating output or runtime resources."""

    from op_bench.runtime.backends import load_runtime_target_binding
    from op_bench.runtime.profiles import load_runtime_profile_registry
    from op_bench.runtime.validation import ContractError

    if args.runtime_profile is None:
        print("--runtime-profile is required with --runtime-protocol v1", file=sys.stderr)
        return 2
    if args.task:
        print("--task is a legacy-only input", file=sys.stderr)
        return 2
    if len(args.dataset) != 1:
        print("v1 requires exactly one --dataset", file=sys.stderr)
        return 2
    if args.fresh:
        print("--fresh is not supported by the v1 resume protocol", file=sys.stderr)
        return 2
    if args.no_baseline_cache:
        print("--no-baseline-cache is a legacy-only input", file=sys.stderr)
        return 2
    if args.max_parallel != 1:
        print("--max-parallel is not supported by the v1 runner", file=sys.stderr)
        return 2
    if args.agent_repeat < 1:
        print("--agent-repeat must be >= 1", file=sys.stderr)
        return 2
    if len(args.agent) != 1 or args.agent[0] not in {
        "scripted_canonical",
        "codex_canonical",
        "codex_mcp_canonical",
    }:
        print(
            "v1 --agent must explicitly select scripted_canonical, "
            "codex_canonical, or codex_mcp_canonical",
            file=sys.stderr,
        )
        return 2
    adapter_id = args.agent[0]
    if adapter_id == "codex_mcp_canonical" and not args.codex_model:
        print("--codex-model is required for codex_mcp_canonical", file=sys.stderr)
        return 2
    if adapter_id != "codex_mcp_canonical" and args.codex_model is not None:
        print(
            "--codex-model is only supported for codex_mcp_canonical",
            file=sys.stderr,
        )
        return 2
    if adapter_id in {
        "codex_canonical",
        "codex_mcp_canonical",
    } and not args.enable_external_canary:
        print(
            f"--enable-external-canary is required for {adapter_id}",
            file=sys.stderr,
        )
        return 2
    codex_cli_version = None
    if adapter_id == "codex_mcp_canonical":
        try:
            codex_cli_version = detect_codex_cli_version()
        except (ContractError, OSError):
            print("cannot determine exact Codex CLI version", file=sys.stderr)
            return 2
    try:
        registry = load_runtime_profile_registry(
            Path(args.runtime_profile_registry).resolve()
        )
        try:
            profile = registry.get(args.runtime_profile)
        except ContractError as exc:
            raise ContractError("unknown Runtime Profile") from exc
    except (ContractError, OSError, ValueError):
        print("cannot load Runtime Profile registry or unknown Runtime Profile", file=sys.stderr)
        return 2
    if profile.backend == "remote_docker" and args.target_config is None:
        print("--target-config is required for a Remote Runtime Profile", file=sys.stderr)
        return 2
    if args.target_config is None:
        target = None
    else:
        try:
            target = load_runtime_target_binding(
                Path(args.target_config).resolve(strict=True),
                local_workspace_parent=Path(args.output_dir).resolve().parent,
            )
        except (ContractError, OSError, ValueError):
            print("invalid private target config", file=sys.stderr)
            return 2
        if target.backend != profile.backend:
            print("private target backend does not match Runtime Profile", file=sys.stderr)
            return 2

    return _execute_v1(
        args,
        registry,
        profile,
        target,
        codex_cli_version=codex_cli_version,
    )


def _execute_v1(
    args,
    registry,
    profile,
    target,
    *,
    codex_cli_version,
) -> int:
    """Build one frozen v1 request and dispatch it without legacy fallback."""

    from op_bench.runtime.adapters import ScriptedCanonicalAdapter
    from op_bench.runtime.backends import (
        DockerRuntimeBackend,
        LocalProcessBackend,
        RemoteDockerRuntimeBackend,
        RuntimeTargetBinding,
    )
    from op_bench.runtime.codex_adapter import (
        CodexCanonicalAdapter,
        subprocess_command_runner,
    )
    from op_bench.runtime.codex_mcp_adapter import CodexMcpCanonicalAdapter
    from op_bench.runtime.process_group import (
        ProcessGroupCleanupError,
        run_process_group,
    )
    from op_bench.runtime.legacy import (
        LegacyV05Defaults,
        agent_spec_for_v1_adapter,
        runtime_bundle_from_v05_dataset,
    )
    from op_bench.runtime.orchestrator import V06Orchestrator, V06RunRequest
    from op_bench.runtime.validation import ContractError

    dataset_path = Path(args.dataset[0]).resolve()
    try:
        dataset = DatasetManifest.load(dataset_path)
        task_ids = [task.task_id for task in dataset.load_tasks(verified_only=True)]
        if args.filter_tasks:
            task_ids = [
                task_id
                for task_id in task_ids
                if any(pattern in task_id for pattern in args.filter_tasks)
            ]
        if args.only_tasks:
            exact = set(args.only_tasks)
            task_ids = [task_id for task_id in task_ids if task_id in exact]
        if not task_ids:
            print("no verified v1 tasks matched filters", file=sys.stderr)
            return 2
        agent = agent_spec_for_v1_adapter(
            args.agent[0],
            model_id=args.codex_model,
            codex_cli_version=codex_cli_version,
        )
        defaults = LegacyV05Defaults.standard()
        defaults = replace(
            defaults,
            budget_policy=replace(
                defaults.budget_policy,
                wall_clock_ms=profile.timeout_ms,
            ),
        )
        bundle = runtime_bundle_from_v05_dataset(
            dataset_path,
            agents=(agent,),
            repeat=args.agent_repeat,
            created_at="1970-01-01T00:00:00Z",
            defaults=defaults,
            selected_task_ids=tuple(task_ids),
        )
    except (ContractError, OSError, ValueError, KeyError):
        print("cannot construct frozen v1 runtime inputs", file=sys.stderr)
        return 2

    if any(task.runtime != profile for task in bundle.manifest.tasks):
        print("selected Task Runtime Profile does not match --runtime-profile", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir).resolve()
    if output_root.is_symlink() or (output_root.exists() and not output_root.is_dir()):
        print("v1 --output-dir must be a real directory or an absent path", file=sys.stderr)
        return 2
    try:
        recovery_resolved = _resolve_process_group_recovery(output_root)
    except (ContractError, OSError, ProcessGroupCleanupError, ValueError):
        print("exact process group recovery evidence is invalid", file=sys.stderr)
        return 1
    if not recovery_resolved:
        print(
            "previous exact process group cleanup is still unproven; run is blocked",
            file=sys.stderr,
        )
        return 1
    if target is None:
        if profile.backend != "local":
            print("explicit target is required for this Runtime Profile", file=sys.stderr)
            return 2
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            target = RuntimeTargetBinding(
                backend="local",
                local_workspace_parent=output_root,
            )
        except (ContractError, OSError):
            print("cannot prepare local v1 output binding", file=sys.stderr)
            return 1

    backend_types = {
        "local": LocalProcessBackend,
        "docker": DockerRuntimeBackend,
        "remote_docker": RemoteDockerRuntimeBackend,
    }

    def backend_factory(selected_profile, target_binding, phase):
        del target_binding, phase
        try:
            backend_type = backend_types[selected_profile.backend]
        except KeyError as exc:
            raise ContractError("Runtime Profile backend is not executable") from exc
        return backend_type()

    def adapter_factory(agent_spec, adapter_id):
        del agent_spec
        if adapter_id == "scripted_canonical":
            return ScriptedCanonicalAdapter()
        if adapter_id == "codex_canonical":
            return CodexCanonicalAdapter(subprocess_command_runner)
        if adapter_id == "codex_mcp_canonical":
            return CodexMcpCanonicalAdapter(
                run_process_group,
                model_id=args.codex_model,
                codex_cli_version=codex_cli_version,
            )
        raise ContractError("unsupported v1 Adapter")

    origin_ns = time.monotonic_ns()

    def clock_ms() -> int:
        return max(1, (time.monotonic_ns() - origin_ns) // 1_000_000 + 1)

    orchestrator = V06Orchestrator(
        source_resolver=bundle.source_for,
        hidden_asset_resolver=bundle.hidden_asset_for,
        source_overlay_resolver=bundle.source_overlay_paths_for,
        backend_factory=backend_factory,
        adapter_factory=adapter_factory,
        python_executable=(sys.executable if profile.backend == "local" else "python"),
    )
    try:
        result = orchestrator.run(
            V06RunRequest(
                manifest=bundle.manifest,
                selected_attempt_ids=tuple(
                    attempt.attempt_id
                    for attempt in bundle.manifest.expected_attempts
                ),
                runtime_profile_registry=registry,
                runtime_profile_id=profile.profile_id,
                target_binding=target,
                output_root=output_root,
                resume_policy="retry_infrastructure",
                adapter_id=args.agent[0],
                enable_external_canary=args.enable_external_canary,
                clock_ms=clock_ms,
            )
        )
    except ProcessGroupCleanupError as exc:
        process_group_id = exc.process_group_id
        if process_group_id is None:
            print("exact process group cleanup failed without recovery identity", file=sys.stderr)
            return 1
        try:
            _write_process_group_recovery(output_root, process_group_id)
        except (ContractError, OSError, ValueError):
            print("exact process group cleanup recovery could not be persisted", file=sys.stderr)
            return 1
        print(
            "exact process group cleanup is unproven; recovery evidence persisted",
            file=sys.stderr,
        )
        return 1
    except (ContractError, OSError, ValueError):
        print("v1 orchestration failed before a valid run result", file=sys.stderr)
        return 1

    if result.integrity.status != "passed" or result.blocked_attempt_ids:
        print("v1 run did not pass Integrity", file=sys.stderr)
        return 1
    try:
        with (output_root / "summary.json").open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        totals = summary["totals"]
        infrastructure_invalid = int(totals["infrastructure_invalid"])
        valid = int(totals["valid"])
        expected = int(totals["expected"])
    except (OSError, ValueError, KeyError, TypeError):
        print("v1 summary is unavailable or invalid", file=sys.stderr)
        return 1
    if infrastructure_invalid or valid != expected:
        print("v1 run completed with infrastructure-invalid Attempts", file=sys.stderr)
        return 1
    if not args.quiet:
        print(
            f"v1 run complete: ran={len(result.ran_attempt_ids)}, "
            f"skipped={len(result.skipped_attempt_ids)}, output={output_root}"
        )
    print(output_root / "summary.json")
    return 0


def _run_single_attempt(
    *,
    task: TaskManifest,
    agent_name: str,
    attempt: int,
    agent_repeat: int,
    output_dir: Path,
    patches_dir: Path,
    environment_manager: EnvironmentManager,
    evaluator: Evaluator,
    progress,
) -> dict[str, object]:
    agent = agent_by_name(agent_name, progress=progress)
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


def _write_summary(store: ResultsStore, output_dir: Path, progress, tasks: list["TaskManifest"] | None = None) -> None:
    results = store.load_all_results()
    baselines = store.load_all_baselines()
    progress(f"write summary: {output_dir / 'summary.json'}")
    # Preserve the legacy layout: agents-summary from result records,
    # baselines listed separately, and a top-level baseline_count.
    summary = summarize_results(results)
    summary["baselines"] = baselines
    summary["baseline_count"] = len(baselines)
    # v0.5 extended metrics (patch conciseness, tier-weighted score, per-dimension).
    if tasks is not None:
        task_meta = {
            t.task_id: {
                "gold_patch_lines": _count_lines(t.gold_patch_path),
                "runtime_tier": t.runtime_tier,
                "problem_type": t.problem_type,
                "problem_dimension": t.problem_dimension,
                "problem_subclass": t.problem_subclass,
            }
            for t in tasks
        }
    else:
        task_meta = {}
    summary["extended"] = compute_extended_metrics(results, task_meta)
    write_json(output_dir / "summary.json", summary)


def _count_lines(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeDecodeError):
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
