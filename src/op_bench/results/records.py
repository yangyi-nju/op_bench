"""Read saved attempts and independently verified outcomes without modifying them."""
from __future__ import annotations

from pathlib import Path

from op_bench.io import read_json
from op_bench.evaluation.verification import verify_evaluation


def normalize_record(record: dict) -> dict:
    """Translate historical labels once, without changing saved evidence."""
    if not isinstance(record, dict):
        raise ValueError("attempt must be an object")
    record = dict(record)
    for current, previous in (("model_id", "agent_id"), ("model", "agent")):
        if previous in record:
            if current in record and record[current] != record[previous]:
                raise ValueError(f"conflicting {current} and legacy {previous}")
            record[current] = record.pop(previous)
    terminal = record.get("terminal_status")
    if isinstance(terminal, str):
        record["terminal_status"] = {
            "completed": "finished", "external_submission": "finished",
            "deadline": "timed_out", "deadline_exceeded": "timed_out",
            "output_limit": "budget_exhausted",
        }.get(terminal, terminal)
    return record


def load_attempt(path: str | Path) -> dict:
    """Load attempt.json using its independently saved evaluation as authority.

    The independent evaluation file is the only scoring source. Legacy inline
    values are ignored, and invalid or absent evidence remains unknown. No
    files are changed, including attempt.json.
    A declared submission must be frozen successfully and match the saved patch.
    This helper never launches an Agent or reruns evaluation, and preserves the
    generation terminal status.
    """
    path = Path(path).resolve()
    if path.is_dir():
        path /= "attempt.json"
    record = normalize_record(read_json(path))
    for key in ("attempt_id", "task_id", "model_id"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Attempt {key} must be a nonempty string: {path}")
    if type(record.get("repeat", 1)) is not int or record.get("repeat", 1) < 1:
        raise ValueError(f"Attempt repeat must be a positive integer: {path}")
    record.pop("unverified_evaluation", None)
    evaluation_dir = path.parent / "evaluation"
    verified = None
    if not (evaluation_dir / "result.json").is_file():
        recovery = {"status": "unavailable", "findings": ["No saved evaluation result; outcome remains unknown."]}
    else:
        validation = verify_evaluation(evaluation_dir, expected_task_id=record["task_id"])
        findings = list(validation["findings"])
        if validation["valid"]:
            candidate = read_json(evaluation_dir / "result.json")
            if (record.get("task_identity") is not None
                    and record["task_identity"] != candidate.get("task_identity")):
                findings.append("evaluation task_identity does not match the Attempt")
            submission = record.get("submission")
            frozen = isinstance(submission, dict) and submission.get("status") == "frozen"
            if record.get("capture_error"):
                findings.append("the Attempt records a capture_error; no reliable submission can be recovered")
            if not frozen:
                findings.append("the Attempt's submission is not frozen")
            else:
                try:
                    frozen_patch = (path.parent / "patch.diff").read_text(encoding="utf-8")
                    evaluated_patch = (evaluation_dir / "patch.diff").read_text(encoding="utf-8")
                    if frozen_patch != evaluated_patch:
                        findings.append("evaluation patch does not match the frozen submission")
                except (OSError, UnicodeError) as exc:
                    findings.append(f"cannot read frozen submission and evaluation patches: {exc}")
            if not findings:
                verified = candidate
        if verified is not None:
            recovery = {"status": "verified", "source": "evaluation/result.json"}
        else:
            recovery = {"status": "rejected", "findings": findings}
    record["evaluation"] = verified
    record["evaluation_recovery"] = recovery
    return record
