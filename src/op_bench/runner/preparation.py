"""Recoverable preparation transactions for trusted task and Agent inputs.

Callers persist their complete schedule before entering preparation. Publication
markers establish ownership and completion; they do not verify file integrity.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import uuid

from op_bench.io import read_json, write_json
from op_bench.provenance import software_identity as observed_software_identity
from op_bench.runtime.baseline import BaselineError, build_baseline
from op_bench.runtime.provisioning import publish_directory
from op_bench.runtime.snapshot import snapshot_task
from op_bench.data.task import TaskSpec


def prepare_input(output: Path, entry: dict, kind: str, prepare, persist) -> None:
    """Publish only completed inputs; retain each failed staging directory.

    The marker records ownership and completion, not file integrity. It also
    closes the crash window between directory publication and the plan update.
    """
    destination = output / entry["destination"]
    if entry["status"] == "ready":
        if not destination.is_dir():
            raise ValueError(f"Recorded prepared input is missing: {destination}")
        return
    previous = entry["attempts"][-1] if entry["attempts"] else None
    if previous is not None:
        staged = output / previous["staging"] / "artifact"
        completed = destination if destination.exists() else staged
        marker_path = completed / ".preparation.json"
        if marker_path.is_file():
            marker = read_json(marker_path)
            if marker.get("attempt_id") != previous["attempt_id"] or marker.get("kind") != kind:
                raise ValueError(f"Preparation artifact ownership differs: {completed}")
            if completed != destination:
                destination.parent.mkdir(parents=True, exist_ok=True)
                publish_directory(completed, destination)
            previous["recovered_from"] = {key: previous.get(key) for key in ("status", "stage", "duration_sec", "error")}
            previous.update(status="ready", stage="published", duration_sec=marker["duration_sec"],
                            error=None, recovered_after_interruption=True)
            entry.update(status="ready", result=marker["result"])
            persist()
            return
        if previous["status"] == "running":
            previous.update(status="interrupted", duration_sec=None,
                            error={"type": "ProcessInterrupted", "stage": previous["stage"],
                                   "message": "Previous process ended before preparation completed; elapsed work is unknown."})
    token = uuid.uuid4().hex
    staging = Path("preparation") / kind / destination.name / token
    trial = {"attempt_id": token, "staging": str(staging), "status": "running", "stage": "staging",
             "started_at_unix": time.time(), "duration_sec": None, "error": None}
    entry["attempts"].append(trial)
    entry["status"] = "running"
    started = time.monotonic()
    try:
        persist()
        if destination.exists():
            raise ValueError(f"Refusing to replace an unrecognized prepared input: {destination}")
        directory = output / staging
        directory.mkdir(parents=True, exist_ok=False)

        def stage(name: str) -> None:
            trial["stage"] = name
            persist()

        result = prepare(directory, stage)
        artifact = directory / "artifact"
        write_json(artifact / ".preparation.json", {"attempt_id": token, "kind": kind,
                   "result": result, "duration_sec": time.monotonic() - started})
        stage("publish")
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_directory(artifact, destination)
        trial.update(status="ready", stage="published", duration_sec=time.monotonic() - started)
        entry.update(status="ready", result=result)
    except BaseException as exc:
        status = "failed" if isinstance(exc, Exception) else "interrupted"
        trial.update(status=status, duration_sec=time.monotonic() - started,
                     error={"stage": trial["stage"], "type": type(exc).__name__, "message": str(exc)})
        entry["status"] = status
        raise
    finally:
        persist()


def prepare_task_input(output: Path, entry: dict, persist) -> None:
    """Freeze entry['input'] into entry['destination'] under output.

    A new entry has status='pending' and attempts=[]. The input is a TaskSpec
    to_dict() declaration with absolute original source/grader paths. Callers
    retain the entry in their plan and supply an atomic plan-writing callback.
    """
    def prepare(directory, stage):
        stage("task_declaration")
        declaration = directory / "declared-task.json"
        write_json(declaration, entry["input"])
        task = TaskSpec.load(declaration)
        stage("task_snapshot")
        snapshot_task(task, directory / "artifact")
        return {"task_id": task.task_id}

    prepare_input(output, entry, "tasks", prepare, persist)


def _prepare_baseline(plan: dict, output: Path, entry: dict) -> None:
    def prepare(directory, stage):
        task = TaskSpec.load(output / entry["task_file"])
        if entry["role"] == "solver":
            task = replace(task, variants=())
        stage("baseline_build")
        artifact = directory / "artifact"
        result = build_baseline(task, artifact)
        entry["build_evidence"] = {"record": str((artifact / "baseline.json").relative_to(output)),
            "status": result["status"], "duration_sec": result["duration_sec"]}
        if result["status"] != "ready":
            raise BaselineError(f"Trusted baseline did not complete; see {entry['build_evidence']['record']}")
        entry["build_evidence"]["record"] = entry["destination"] + "/baseline.json"
        return {"task_id": task.task_id, "role": entry["role"],
                "baseline_record": entry["destination"] + "/baseline.json",
                "duration_sec": result["duration_sec"]}

    prepare_input(output, entry, "baselines", prepare, lambda: write_json(output / "plan.json", plan))


def prepare_experiment(plan: dict, output: Path) -> None:
    preparation = plan.get("preparation")
    if preparation is None:
        return  # Earlier completed plans already contain their frozen inputs.
    entries = preparation["tasks"] + preparation.get("baselines", [])
    if any(entry["status"] != "ready" for entry in entries):
        expected = plan.get("software_identity")
        if expected is not None and expected != observed_software_identity():
            raise ValueError("Installed software differs from the experiment plan; do not resume preparation under the old version")
    preparation["status"] = "running"
    write_json(output / "plan.json", plan)
    failures = []
    try:
        for entry in preparation["tasks"]:
            try:
                prepare_task_input(output, entry, lambda: write_json(output / "plan.json", plan))
            except Exception as exc:
                failures.append(exc)
        ready_tasks = {entry["task_id"] for entry in preparation["tasks"] if entry["status"] == "ready"}
        for entry in preparation.get("baselines", []):
            if entry["task_id"] not in ready_tasks:
                continue
            try:
                _prepare_baseline(plan, output, entry)
            except Exception as exc:
                failures.append(exc)
        preparation["status"] = "failed" if failures else "ready"
        if failures:
            raise failures[0]
    except BaseException as exc:
        preparation["status"] = "failed" if isinstance(exc, Exception) else "interrupted"
        raise
    finally:
        write_json(output / "plan.json", plan)
