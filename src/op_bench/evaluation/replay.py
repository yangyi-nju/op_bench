"""Append-only physical reevaluation of an already frozen Agent submission."""
from dataclasses import replace
from pathlib import Path
import time
import uuid

from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.io import read_json, write_json
from op_bench.provenance import software_identity
from op_bench.runtime.snapshot import snapshot_task
from op_bench.data.task import TaskSpec


def replay_attempt(attempt: Path, task_file: Path, output: Path, *, reason: str,
                   baseline_artifact: Path | None = None) -> dict:
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Replay requires a reason for this additional physical evaluation")
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    if baseline_artifact is not None:
        baseline_artifact = Path(baseline_artifact).expanduser().resolve()
    if attempt.is_file():
        attempt = attempt.parent
    source = read_json(attempt / "attempt.json")
    if not isinstance(source, dict):
        raise ValueError("Original attempt record must be an object")
    task = TaskSpec.load(task_file)
    output = task.validate_output_path(output)
    if source.get("task_id") != task.task_id:
        raise ValueError("Replay task_id must match the original logical task")
    patch = (attempt / "patch.diff").read_text(encoding="utf-8")
    if source.get("capture_error"):
        raise ValueError("The original attempt did not capture a reliable submission")
    submission = source.get("submission")
    frozen = isinstance(submission, dict) and submission.get("status") == "frozen"
    if not frozen:
        raise ValueError("The original attempt did not freeze a submission")
    previous_patch = attempt / "evaluation/patch.diff"
    if previous_patch.is_file() and previous_patch.read_text(encoding="utf-8") != patch:
        raise ValueError("The saved evaluation patch differs from the original frozen submission")
    output.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": 1, "physical_execution_id": str(uuid.uuid4()),
              "kind": "patch_replay", "reason": reason, "status": "running",
              "task_id": task.task_id, "task_identity": task.identity_dict(),
              "software_identity": software_identity(), "evaluation_path": "evaluation/result.json",
              "baseline_artifact": None if baseline_artifact is None else {
                  "path": str(baseline_artifact), "selection": "explicit"},
              "generation_performed": False, "new_independent_repeat": False,
              "generation_comparability": "requires_review",
              "origin": {"attempt_path": str(attempt), **{key: source.get(key) for key in (
                  "attempt_id", "model_id", "model", "harness", "tool_version", "repeat", "task_identity", "dataset_identity", "software_identity",
                  "information_profile", "terminal_status", "timing_protocol_revision", "budget")}},
              "duration_sec": 0.0}
    if "model_id" not in source and source.get("agent_id"):
        record["origin"]["legacy_agent_id"] = source["agent_id"]
    (output / "patch.diff").write_text(patch, encoding="utf-8")
    write_json(output / "replay.json", record)
    started = time.monotonic()
    try:
        options = {}
        if baseline_artifact is not None:
            from op_bench.runtime.execution import source_identity
            original_source = source_identity(task)
            record["baseline_artifact"]["source_provenance"] = original_source
            write_json(output / "replay.json", record)
            # Preserve the source identity across the history-free snapshot.
            # Pin a moving Git ref before exporting, so the snapshot cannot
            # accidentally use a different commit from the selected cache.
            if original_source["kind"] == "git":
                task = replace(task, source=replace(task.source, revision=original_source["revision"]))
            options.update(baseline_artifact=baseline_artifact, baseline_source=original_source)
        frozen_task = snapshot_task(task, output / "inputs/task")
        evaluation = PatchEvaluator().evaluate(frozen_task, patch, output / "evaluation", **options)
        record["status"] = "completed"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        # The evaluator retains partial evidence at evaluation_path before
        # propagating an interruption; this record only tracks the replay.
        raise
    finally:
        record["duration_sec"] = time.monotonic() - started
        write_json(output / "replay.json", record)
    return {**record, "evaluation": evaluation}
