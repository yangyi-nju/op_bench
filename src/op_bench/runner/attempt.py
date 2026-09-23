"""One model attempt: prepare, solve, freeze its patch, then request grading.

The model loop receives public inputs and workspace tools only. This orchestration
owns the lifetime of the attempt's workspaces and records each terminal phase.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, replace
import math
from pathlib import Path
import shutil
import tempfile
import time
import uuid

from op_bench.data.task import TaskSpec
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.io import read_json, write_json
from op_bench.provenance import software_identity as observed_software_identity
from op_bench.results.records import load_attempt
from op_bench.runner.loop import HARNESS_VERSION, run_harness
from op_bench.runner.mcp import MCPToolClient
from op_bench.runner.model_client import ModelSpec, create_client
from op_bench.runner.tools import TOOL_VERSION, ToolExecutor
from op_bench.runtime.baseline import BaselineError, build_baseline, fork_baseline
from op_bench.runtime.execution import create_session
from op_bench.runtime.submission import freeze_patch, initialize_workspace
from op_bench.runtime.workspace import age_cached_workspace


TIMING_PROTOCOL_REVISION = "3"


def _initial_upload_duration(info: dict) -> float | None:
    observation = info.get("initial_workspace_upload")
    if not isinstance(observation, dict) or observation.get("status") not in {"completed", "not_applicable"}:
        return None
    duration = observation.get("duration_sec")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
        return None
    return float(duration)


@dataclass(frozen=True)
class HarnessSpec:
    budget_sec: float = 900
    max_turns: int = 60
    max_context_chars: int = 200000
    per_tool_timeout_sec: float = 120

    def __post_init__(self):
        for field in ("budget_sec", "per_tool_timeout_sec"):
            x = getattr(self, field)
            try:
                valid = not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and x > 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"harness.{field} must be positive and finite")
        for field in ("max_turns", "max_context_chars"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"harness.{field} must be a positive integer")

    @classmethod
    def from_dict(cls, value: dict) -> "HarnessSpec":
        if not isinstance(value, dict) or set(value) - cls.__dataclass_fields__.keys():
            raise ValueError("unsupported harness configuration")
        return cls(**value)

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def validate_conditions(task: TaskSpec, profile: str) -> None:
    if profile not in {"controlled", "development"}:
        raise ValueError("information_profile must be controlled or development")
    if profile == "controlled" and task.environment.backend != "docker":
        raise ValueError("Controlled tools require Docker; local runs must explicitly use development")


def _boundary_record(session, initialized: bool) -> dict:
    info = session.info if session is not None else {}
    enforced = (initialized and info.get("backend") == "docker" and
                info.get("observed_network_mode") == "none" and not info.get("grader_access"))
    return {"status": "enforced" if enforced else "unverified",
            "source_history": "fresh_baseline" if initialized else "unavailable",
            "solver_network": info.get("observed_network_mode", info.get("network")),
            "model_access": "trusted_controller", "tools": TOOL_VERSION,
            "scope": "Workspace execution boundary; model transport and backend limits are recorded separately."}


def _build_attempt_baseline(task: TaskSpec, artifact: Path, evidence: Path) -> dict:
    """Keep build evidence, while the per-attempt workspace lives temporarily."""
    try:
        return build_baseline(task, artifact)
    finally:
        pending = [(artifact, evidence)]
        while pending:
            original, saved = pending.pop()
            record_file = original / "baseline.json"
            if not record_file.is_file():
                continue
            record = read_json(record_file)
            write_json(saved / "baseline.json", record)
            if (original / "logs").is_dir():
                shutil.copytree(original / "logs", saved / "logs", dirs_exist_ok=True)
            if record.get("kind") == "trusted_baseline_variants":
                pending.extend((original / relative, saved / relative) for relative in record["variants"].values())


def run_attempt(task: TaskSpec, model: ModelSpec, output_dir: str | Path, *,
                harness: HarnessSpec | None = None,
                attempt_id: str | None = None,
                information_profile: str | None = None,
                dataset_identity: dict | None = None, software_identity: dict | None = None,
                baseline_artifact: Path | None = None,
                solver_baseline_artifact: Path | None = None) -> dict:
    """Own one workspace lifetime and persist its submission and grading records.

    Trusted preparation precedes the solving budget. The loop receives only
    public inputs; after execution stops, the frozen patch enters independent
    grading. Return the loaded attempt record, including available score evidence.
    """
    harness = harness or HarnessSpec()
    profile = information_profile or ("development" if task.environment.backend == "local" else "controlled")
    validate_conditions(task, profile)
    actual_software = observed_software_identity()
    if software_identity is not None and software_identity != actual_software:
        raise ValueError("Installed software differs from the experiment plan; do not resume new executions under the old version")
    output = task.validate_output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "attempt_id": attempt_id or str(uuid.uuid4()),
              "task_id": task.task_id, "model_id": model.model_id,
              "terminal_status": "running", "evaluation_path": "evaluation/result.json",
              "harness_path": "harness/harness-result.json", "harness_version": HARNESS_VERSION,
              "submission": {"status": "pending", "path": "patch.diff"},
              "information_profile": profile,
              "task_identity": task.identity_dict(), "dataset_identity": dataset_identity,
              "software_identity": actual_software,
              "timing_protocol_revision": TIMING_PROTOCOL_REVISION,
              "solver_wall_duration_sec": None, "solve_duration_sec": None,
              "solver_startup_duration_sec": None, "budgeted_solver_startup_duration_sec": None,
              "initial_workspace_upload": {"status": "not_started", "duration_sec": None},
              "budget": {"solve_wall_time_sec": harness.budget_sec,
                         "timing_protocol_revision": TIMING_PROTOCOL_REVISION,
                         "solve_includes": ["solver_container_startup", "model_calls", "controlled_tools", "incremental_builds", "public_tests"],
                         "solve_excludes": ["trusted_source_preparation", "baseline_build", "initial_workspace_upload", "final_grading"],
                         "timing_explanation": "Only completed initial workspace upload/ownership initialization is deducted from solver wall time. Its network-none helper runs no model or candidate command. Container startup and fixed harness remain budgeted; missing measurements remain unknown."},
              "model": model.to_dict(), "harness": harness.to_dict(), "tool_version": TOOL_VERSION,
              "network": "none" if task.environment.backend == "docker" else "host",
              "generation_assurance": "development_only" if task.environment.backend == "local" else "controlled_tools_container"}
    write_json(output / "attempt.json", record)
    (output / "patch.diff").write_text("", encoding="utf-8")
    started = time.monotonic()
    interrupted: KeyboardInterrupt | None = None
    patch = None
    session = None
    initialized = False
    reserved, generated = (".opbench",), ()
    with tempfile.TemporaryDirectory(prefix="opbench-agent-") as temporary:
        root = Path(temporary).resolve()
        workspace, database = root / "workspace", root / "capture.git"
        try:
            preparation_started = time.monotonic()
            if baseline_artifact is None and solver_baseline_artifact is None:
                local_baselines = root / "trusted-baselines"
                solver_baseline_artifact = local_baselines / "solver"
                baseline_artifact = local_baselines / "grading" if task.variants else solver_baseline_artifact
                record["baseline_preparation"] = {}
                for role, baseline_task, destination in (
                    [("solver", replace(task, variants=()), solver_baseline_artifact)] +
                    ([("grading", task, baseline_artifact)] if task.variants else [])
                ):
                    record["baseline_preparation"][role] = {
                        "record": f"preparation/{role}/baseline.json",
                        "workspace_retention": "attempt_lifetime_only"}
                    built = _build_attempt_baseline(baseline_task, destination, output / "preparation" / role)
                    record["baseline_preparation"][role].update(status=built["status"], duration_sec=built["duration_sec"])
                    if role == "solver":
                        record["preparation_environment"] = built.get("session")
                        record["baseline_build_duration_sec"] = built["duration_sec"]
                    if built["status"] != "ready":
                        record["terminal_status"] = "environment_error"
                        raise BaselineError(f"Trusted {role} baseline failed; see preparation/{role}/baseline.json")
            selected_baseline = solver_baseline_artifact or baseline_artifact
            if task.variants and (solver_baseline_artifact is None or baseline_artifact is None):
                raise BaselineError("Separate solver and grading baselines are required for tasks with grading variants")
            cached = fork_baseline(replace(task, variants=()), selected_baseline, workspace)
            age_cached_workspace(workspace)
            record["source"] = cached["compatibility"]["source"]
            original_paths = cached["source_files"]
            contract = cached["artifact_contract"]
            runtime = contract["runtime_directory"]
            reserved = (".opbench", *contract.get("runtime_paths", [runtime]))
            sources = set(original_paths)
            generated = tuple(str(path.relative_to(workspace)) for path in workspace.rglob("*")
                if (path.is_file() or path.is_symlink()) and str(path.relative_to(workspace)) not in sources
                and not any(str(path.relative_to(workspace)) == prefix or
                            str(path.relative_to(workspace)).startswith(prefix + "/") for prefix in reserved))
            record["baseline_reuse"] = {
                "scope": "attempt_local" if "baseline_preparation" in record else "experiment_shared",
                "solver_artifact": str(Path(selected_baseline).resolve()),
                "grading_artifact": str(Path(baseline_artifact).resolve()) if baseline_artifact else None,
                "compatibility": cached["compatibility"], "assurance": cached["assurance"],
                "fork_and_preparation_duration_sec": time.monotonic() - preparation_started,
                "build_cost_allocation": "Baseline construction is recorded in attempt preparation or once in the shared experiment plan.",
            }
            public = task.visible_dict()
            public["workspace_policy"] = {
                "editable": "Task source, public tests, self-authored tests, scripts and build files.",
                "reserved": "Paths listed in reserved_paths contain public inputs or runtime dependencies, HOME, caches and logs. They are excluded from Git history and submission, including force-added files.",
                "submission": "All changes, deletions and mode changes to initial tracked source are captured. New files are captured using the Git ignore rules frozen before solving; changing or deleting .gitignore does not change that boundary. The write_file and replace_text tools explicitly register new source files, including ignored paths. Reserved runtime paths and initial baseline build artifacts are never submitted, even if staged. The finish tool ends solving; the controller also freezes at the time limit.",
                "submission_policy_file": ".opbench/submission-policy.json",
                "time_budget_sec": harness.budget_sec,
                "build_artifacts": "Prepared dependencies are in the image; prepare/build outputs needed later must be inside the workspace. Temporary or container-root changes are not transferred into grading.",
                "final_feedback": "One final patch is captured after stopping execution. Private grading results are not available during this attempt.",
            }
            public["workspace_policy"].update(
                baseline_artifacts="Each attempt starts from an independent trusted unpatched build. Initial build-generated files are excluded from submission; source and build recipe changes are captured. Final grading always builds the frozen patch independently.",
                reserved_paths=list(reserved))
            input_dir = workspace / ".opbench"
            input_dir.mkdir()
            write_json(input_dir / "submission-policy.json", {
                "schema_version": 1, "baseline_source_files": original_paths,
                "reserved_paths": list(reserved), "baseline_generated_files": list(generated),
                "automatic_new_files": "Git ignore rules frozen before solving; Agent ignore changes do not alter this policy.",
                "explicit_new_files": "Explicitly written new files are registered by the tools and included unless reserved or baseline generated.",
                "existing_source": "Edits, deletions and executable mode changes are always captured independently of the Agent HEAD, index and ignore rules.",
            })
            write_json(output / "task_input.json", public)
            variables = dict(cached["execution_environment"]["environment"])
            initialize_workspace(workspace, database, original_paths, reserved=reserved)
            initialized = True
            record["preparation_duration_sec"] = time.monotonic() - preparation_started
            with ExitStack() as stack:
                environment = replace(task.environment, image=cached["execution_environment"]["image"],
                                      environment=variables)
                agent_task = replace(task, environment=environment)
                # Container startup consumes solving time. Only the measured,
                # completed initial upload is deducted; solving commands stay budgeted.
                solve_started = time.monotonic()
                failed_start_info = None
                try:
                    session = stack.enter_context(create_session(
                        agent_task, workspace, None))
                    record["solver_startup_duration_sec"] = time.monotonic() - solve_started
                    upload_duration = _initial_upload_duration(session.info)
                    if upload_duration is None:
                        raise RuntimeError("Initial workspace upload timing is unavailable; no Agent command was started")
                    record["budgeted_solver_startup_duration_sec"] = max(
                        0.0, record["solver_startup_duration_sec"] - upload_duration)
                    command_started = time.monotonic()
                    remaining = harness.budget_sec - (command_started - solve_started - upload_duration)
                    if remaining <= 0:
                        record["terminal_status"] = "timed_out"
                    else:
                        executor = ToolExecutor(session, output / "tool-logs",
                            build_commands=task.environment.build, public_commands=task.public_commands,
                            reserved=reserved, per_tool_timeout_sec=harness.per_tool_timeout_sec)
                        tools = MCPToolClient(executor)
                        client = create_client(model, output / "model")
                        outcome = run_harness(client, public, tools, output / "harness",
                            budget_sec=max(.001, harness.budget_sec - (time.monotonic() - solve_started - upload_duration)),
                            max_turns=harness.max_turns, max_context_chars=harness.max_context_chars)
                        record["harness_duration_sec"] = time.monotonic() - command_started
                        record["terminal_status"] = {"completed": "finished", "timeout": "timed_out",
                            "model_error": "model_service_error", "protocol_error": "model_protocol_error",
                            "tool_error": "tool_error", "turn_limit": "budget_exhausted",
                            "context_limit": "budget_exhausted"}[outcome["state"]]
                except BaseException as exc:
                    failed_start_info = getattr(exc, "execution_session_info", None)
                    raise
                finally:
                    # Measure before ExitStack stops/captures the sessions. A
                    # failed upload has partial elapsed evidence, not a completed
                    # duration that can be silently deducted from the budget.
                    record["solver_wall_duration_sec"] = time.monotonic() - solve_started
                    info = session.info if session is not None else failed_start_info or {}
                    record["initial_workspace_upload"] = info.get("initial_workspace_upload", {
                        "status": "unavailable", "duration_sec": None})
                    upload_duration = _initial_upload_duration(info)
                    if upload_duration is not None:
                        record["solve_duration_sec"] = max(0.0, record["solver_wall_duration_sec"] - upload_duration)
        except KeyboardInterrupt as exc:
            record["terminal_status"] = "cancelled"
            interrupted = exc
        except Exception as exc:
            if record["terminal_status"] == "running":
                record["terminal_status"] = "runner_error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            if getattr(exc, "execution_session_info", None) is not None:
                record["failed_session"] = exc.execution_session_info
        finally:
            # Session exit stops execution before capturing the current files.
            if session is not None:
                record["environment"] = session.info
                record["cleanup_errors"] = list(session.cleanup_errors)
            record["information_boundary"] = _boundary_record(session, initialized)
            capture_ready = session is not None and session.workspace_capture_ready
            if database.exists() and not capture_ready:
                record["capture_error"] = "Final workspace was not successfully stopped and retrieved; no patch was scored."
                record["submission"]["status"] = "capture_failed"
            if database.exists() and capture_ready:
                try:
                    patch = freeze_patch(workspace, database, reserved=reserved, generated=generated)
                    (output / "patch.diff").write_text(patch, encoding="utf-8")
                    record["submission"]["status"] = "frozen"
                except Exception as exc:
                    record["capture_error"] = f"{type(exc).__name__}: {exc}"
                    record["submission"]["status"] = "capture_failed"
            record["duration_sec"] = time.monotonic() - started
            write_json(output / "attempt.json", record)
        # Grade a captured patch even after a tool/service fault for diagnosis.
        # The reporter separately decides whether the attempt is a valid model outcome.
        if patch is not None and record["terminal_status"] in {"finished", "timed_out", "budget_exhausted", "model_service_error", "model_protocol_error", "tool_error"}:
            try:
                PatchEvaluator().evaluate(task, patch, output / "evaluation",
                    **({"baseline_artifact": baseline_artifact} if baseline_artifact is not None else {}))
            except KeyboardInterrupt as exc:
                record["evaluation_interrupted"] = True
                interrupted = exc
            except Exception as exc:
                record["evaluation_error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["duration_sec"] = time.monotonic() - started
    write_json(output / "attempt.json", record)
    if interrupted:
        raise interrupted
    return load_attempt(output / "attempt.json")
