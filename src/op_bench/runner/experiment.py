"""Plan and resume one fixed model-by-task experiment.

Single-attempt execution belongs to attempt.py; recoverable input preparation
belongs to preparation.py. Reports are derived from the saved attempts.
"""
from __future__ import annotations

from pathlib import Path
import random
import shutil
import tempfile

from op_bench.data.dataset import Dataset
from op_bench.data.task import TaskSpec
from op_bench.io import read_json, write_json
from op_bench.provenance import software_identity as observed_software_identity
from op_bench.results.records import load_attempt
from op_bench.results.report import summarize_run
from op_bench.runner.attempt import HarnessSpec, TIMING_PROTOCOL_REVISION, run_attempt, validate_conditions
from op_bench.runner.loop import HARNESS_VERSION
from op_bench.runner.model_client import ModelSpec
from op_bench.runner.preparation import prepare_experiment
from op_bench.runner.tools import TOOL_VERSION
from op_bench.runtime.provisioning import publish_directory


def make_schedule(task_ids: list[str], model_ids: list[str], seed: int = 0, *,
                  information_profile: str = "controlled") -> list[dict]:
    if not task_ids or not model_ids or len(set(task_ids)) != len(task_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("task/model selections must be nonempty and unique")
    cells = [(task, model) for task in task_ids for model in model_ids]
    random.Random(seed).shuffle(cells)
    return [{"attempt_id": f"attempt-{index:06d}", "task_id": task, "model_id": model,
             "information_profile": information_profile}
            for index, (task, model) in enumerate(cells, 1)]


def run_experiment(path: str | Path, output_dir: str | Path, *, resume: bool = False) -> dict:
    """Persist a fixed model/task plan, run unstarted cells, and return its report.

    Preparation may recover independently. Once an attempt record exists,
    resume preserves that model opportunity rather than silently trying again.
    The plan remains the report denominator even when preparation or solving fails.
    """
    path, output = Path(path).resolve(), Path(output_dir).resolve()
    specification = read_json(path)
    if not isinstance(specification, dict) or specification.get("schema_version") != 2:
        raise ValueError("experiment.schema_version must be 2 (fixed model harness)")
    if set(specification) - {"schema_version", "dataset", "task_ids", "models", "harness", "information_profile", "reuse_baseline", "seed"}:
        raise ValueError("unsupported experiment field; native agents and aggregated repeats are retired")
    harness = HarnessSpec.from_dict(specification.get("harness", {}))
    reuse_baseline = specification.get("reuse_baseline", False)
    if type(reuse_baseline) is not bool:
        raise ValueError("experiment.reuse_baseline must be a boolean")
    if output.exists():
        if not resume or not (output / "plan.json").is_file():
            raise ValueError("output exists; use --resume only for a recorded experiment")
        plan = read_json(output / "plan.json")
        if plan.get("schema_version") != 2 or plan.get("mode") != "fixed_models":
            raise ValueError("cannot resume an obsolete native experiment")
        if plan["specification"] != specification:
            raise ValueError("resume configuration differs from the planned experiment; use a new output directory")
    else:
        dataset = Dataset.load(path.parent / specification["dataset"])
        selected = dataset.select(specification.get("task_ids"))
        models = [ModelSpec.from_dict(item) for item in specification["models"]]
        profile = specification.get("information_profile", "controlled")
        for task in selected:
            output = task.validate_output_path(output)
            validate_conditions(task, profile)
        seed = specification.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("experiment.seed must be an integer")
        schedule = make_schedule([task.task_id for task in selected], [model.model_id for model in models],
                                 seed, information_profile=profile)
        identities = {model.model_id: model.to_dict() for model in models}
        task_identities = {task.task_id: task.identity_dict() for task in selected}
        dataset_identity = dataset.identity_dict()
        software_identity = observed_software_identity()
        for cell in schedule:
            cell.update(model=identities[cell["model_id"]], harness=harness.to_dict(),
                        harness_version=HARNESS_VERSION, tool_version=TOOL_VERSION, task_identity=task_identities[cell["task_id"]],
                        dataset_identity=dataset_identity, software_identity=software_identity,
                        timing_protocol_revision=TIMING_PROTOCOL_REVISION)
        task_preparation = [{"task_id": task.task_id, "input": task.to_dict(),
                             "destination": f"inputs/task-{index:06d}", "status": "pending", "attempts": []}
                            for index, task in enumerate(selected, 1)]
        plan = {"schema_version": 2, "mode": "fixed_models", "dataset_id": dataset.dataset_id, "dataset_version": dataset.version,
                "dataset_identity": dataset_identity, "software_identity": software_identity,
                "timing_protocol_revision": TIMING_PROTOCOL_REVISION,
                "information_profile": profile, "specification": specification, "schedule": schedule,
                "task_files": [entry["destination"] + "/task.json" for entry in task_preparation],
                "models": [model.to_dict() for model in models], "harness": harness.to_dict(),
                "harness_version": HARNESS_VERSION, "tool_version": TOOL_VERSION,
                "preparation": {"status": "pending", "tasks": task_preparation}}
        if reuse_baseline:
            baseline_entries, baseline_tasks = [], {}
            for index, task in enumerate(selected, 1):
                task_file = task_preparation[index - 1]["destination"] + "/task.json"
                roles = ("solver", "grading") if task.variants else ("shared",)
                paths = {}
                for role in roles:
                    destination = f"inputs/baseline-{index:06d}-{role}"
                    baseline_entries.append({"task_id": task.task_id, "task_file": task_file,
                        "role": role, "destination": destination, "status": "pending", "attempts": []})
                    paths[role] = destination
                baseline_tasks[task.task_id] = {"solver": paths.get("solver", paths.get("shared")),
                                                 "grading": paths.get("grading", paths.get("shared"))}
            plan["preparation"]["baselines"] = baseline_entries
            plan["baseline_artifacts"] = baseline_tasks
            plan["baseline_policy"] = {
                "reuse_baseline": True, "source": "trusted_unpatched_task_snapshot",
                "solve_budget": "Baseline builds and per-attempt forks are outside solving; candidate local builds remain inside solving and independent final grading always builds.",
                "cost": "Baseline build duration is recorded once per prepared artifact; monetary cost is unknown.",
                "scope": "Metadata compatibility is not tamper verification. Incremental dependency correctness, image compatibility and large-framework resource limits require task-specific validation.",
            }
        # Publish the first plan before any source copy or image inspection. Even an interruption during initial plan writing leaves no
        # unusable output directory claiming to be a recorded experiment.
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".opbench-plan-", dir=output.parent))
        try:
            write_json(staging / "plan.json", plan)
            publish_directory(staging, output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    schedule = plan["schedule"]
    profile = plan.get("information_profile", "legacy_unverified")
    try:
        pending = [cell for cell in schedule
                   if not (output / "attempts" / cell["attempt_id"] / "attempt.json").exists()]
        if pending and plan.get("timing_protocol_revision") != TIMING_PROTOCOL_REVISION:
            raise ValueError("Budget timing protocol differs from the experiment plan; use a new plan for new Agent executions")
        if any(cell.get("tool_version") != TOOL_VERSION for cell in pending):
            raise ValueError("Tool protocol differs from the experiment plan; use a new plan for new model executions")
        if any(cell.get("harness_version") != HARNESS_VERSION for cell in pending):
            raise ValueError("Harness protocol differs from the experiment plan; use a new plan for new model executions")
        prepare_experiment(plan, output)
        tasks = [TaskSpec.load(output / relative) for relative in plan["task_files"]]
        models = [ModelSpec.from_dict(item) for item in plan["models"]]
        by_task, by_model = {t.task_id: t for t in tasks}, {m.model_id: m for m in models}
        for cell in schedule:
            attempt_dir = output / "attempts" / cell["attempt_id"]
            existing = attempt_dir / "attempt.json"
            if existing.exists():
                record = load_attempt(existing)
                if any(record.get(key) != cell[key] for key in cell):
                    raise ValueError(f"existing attempt identity differs: {cell['attempt_id']}")
                if record.get("terminal_status") == "running":
                    # A stale running record cannot tell us how much model work
                    # completed. Retain it as interrupted instead of granting a retry.
                    original = read_json(existing)
                    original["terminal_status"] = "interrupted"
                    original["error"] = "Previous process ended without a terminal record; not automatically rerun."
                    write_json(existing, original)
            else:
                run_attempt(by_task[cell["task_id"]], by_model[cell["model_id"]], attempt_dir, harness=harness,
                                     attempt_id=cell["attempt_id"],
                                     information_profile=profile,
                                     dataset_identity=plan.get("dataset_identity"),
                                     software_identity=plan.get("software_identity"),
                                     **({"solver_baseline_artifact": output / plan["baseline_artifacts"][cell["task_id"]]["solver"],
                                         "baseline_artifact": output / plan["baseline_artifacts"][cell["task_id"]]["grading"]}
                                        if plan.get("baseline_artifacts") else {}))
    finally:
        # Preparation failures leave all cells missing; interrupted executions
        # retain their actual terminal record. Neither is removed from the plan.
        # Summarize once: verifying all prior scores after every cell would
        # repeatedly read numeric evidence and make aggregation quadratic.
        write_json(output / "report.json", summarize_run(output, plan))
    return read_json(output / "report.json")
