"""One resolved rate over a fixed task plan, without modifying run evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from op_bench.io import read_json
from op_bench.provenance import software_identity as observed_software_identity
from op_bench.results.records import load_attempt, normalize_record
from op_bench.evaluation.judgment import RESULT_STATUSES
from op_bench.evaluation.verification import evaluation_outcome_issue

TASK_SCOPES = {"operator", "operator_integration", "framework_support", "uncertain"}
REPORT_PROTOCOL_REVISION = "5"
# Budget exhaustion and invalid model tool decisions end the opportunity to
# solve; infrastructure failures do not become model outcomes merely because
# a partially edited patch was gradeable.
SCORED_TERMINALS = {"finished", "timed_out", "budget_exhausted", "model_protocol_error"}


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _timing_revision(rows: list[dict]) -> str | None:
    values = [row.get("timing_protocol_revision") for row in rows]
    if any(value is not None and not _nonempty(value) for value in values):
        raise ValueError("timing_protocol_revision must be a nonempty string or null")
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError("cannot combine different timing_protocol_revision values")
    return values[0] if values else None


def _task_identity(row: dict) -> dict | None:
    value = row.get("task_identity")
    if value is None:
        return None
    if not isinstance(value, dict) or value.get("task_id") != row["task_id"]:
        raise ValueError("task_identity must name the row's logical task_id")
    if not isinstance(value.get("status"), str) or value["status"] not in {"declared", "legacy_unidentified"}:
        raise ValueError("unsupported task_identity status")
    for field in ("task_revision", "defect_group", "scoring_revision"):
        item = value.get(field)
        if item is not None and not _nonempty(item):
            raise ValueError(f"task_identity.{field}: expected a nonempty string")
    if value.get("scope") is not None and (
        not isinstance(value["scope"], str) or value["scope"] not in TASK_SCOPES
    ):
        raise ValueError("unsupported task_identity scope")
    if value["status"] == "declared":
        if any(not _nonempty(value.get(field)) for field in
               ("task_revision", "scope", "defect_group", "scoring_revision")):
            raise ValueError("declared task_identity is missing revision, scope or grouping")
        environments = [value.get("solver_environment")]
        variants = value.get("required_variants")
        if not isinstance(variants, list) or not variants:
            raise ValueError("declared task_identity requires grading variants")
        ids = set()
        for variant in variants:
            if not isinstance(variant, dict) or not _nonempty(variant.get("variant_id")):
                raise ValueError("invalid required variant identity")
            if variant["variant_id"] in ids:
                raise ValueError("duplicate required variant identity")
            ids.add(variant["variant_id"])
            environments.append(variant.get("environment"))
        if any(not isinstance(env, dict) or not _nonempty(env.get("environment_id"))
               or not _nonempty(env.get("revision")) for env in environments):
            raise ValueError("declared task_identity requires environment identities")
    return value


def _provenance(rows: list[dict], field: str) -> dict | None:
    values = [row.get(field) for row in rows]
    for value in values:
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError(f"{field} must be an object")
            required = ("dataset_id", "dataset_version") if field == "dataset_identity" else ("version",)
            if any(not _nonempty(value.get(key)) for key in required):
                raise ValueError(f"incomplete {field}")
    if values and any(value != values[0] for value in values[1:]):
        raise ValueError(f"one report cannot aggregate different {field} values")
    return values[0] if values else None


def _index(rows: list[dict]) -> dict[str, dict]:
    result, cells = {}, set()
    for row in rows:
        row = normalize_record(row)
        for field in ("attempt_id", "task_id", "model_id"):
            if not _nonempty(row.get(field)):
                raise ValueError(f"attempt.{field}: expected a nonempty string")
        if row.get("terminal_status") is not None and not _nonempty(row["terminal_status"]):
            raise ValueError("attempt.terminal_status: expected a nonempty string")
        if row.get("model") is not None and not isinstance(row["model"], dict):
            raise ValueError("model configuration must be an object")
        if row.get("variant_id") is not None:
            raise ValueError("grading variants belong inside one task result")
        if type(row.get("repeat", 1)) is not int or row.get("repeat", 1) != 1:
            raise ValueError("one attempt per model and task is required; repeats need a separate experiment")
        cell = (row["model_id"], row["task_id"])
        if row["attempt_id"] in result:
            raise ValueError(f"duplicate attempt_id: {row['attempt_id']}")
        if cell in cells:
            raise ValueError(f"duplicate model/task: {cell}; retries must not become additional samples")
        cells.add(cell)
        result[row["attempt_id"]] = row
    return result


def _outcome_issue(row: dict, evaluation: dict | None, identity: dict | None) -> str | None:
    if row.get("terminal_status") not in SCORED_TERMINALS:
        return row.get("terminal_status") or "terminal_status_unavailable"
    if row.get("capture_error"):
        return "capture_failed"
    submission = row.get("submission")
    if not isinstance(submission, dict) or submission.get("status") != "frozen":
        return "submission_not_frozen"
    if not evaluation:
        return "evaluation_unavailable"
    recovery = row.get("evaluation_recovery")
    if not isinstance(recovery, dict) or recovery.get("status") != "verified":
        return "evaluation_unverified"
    return evaluation_outcome_issue(evaluation, identity)


def build_report(records: list[dict], *, planned: list[dict] | None = None) -> dict:
    """Consume loaded attempts and a fixed plan; never rerun or repair anything.

    ``records`` should come from ``records.load_attempt``, which verifies saved
    grading evidence and its binding to the frozen submission. Without a saved
    plan, the observed ratio is available but explicitly incomplete.
    """
    started = _index(records)
    schedule = _index(records if planned is None else planned)
    all_rows = list(schedule.values()) + list(started.values())
    timing_revision = _timing_revision(all_rows)
    dataset = _provenance(all_rows, "dataset_identity")
    software = _provenance(all_rows, "software_identity")
    fixed_conditions = {}
    for field in ("harness", "harness_version", "tool_version", "information_profile"):
        values = [cell.get(field) for cell in schedule.values()]
        if values and any(value != values[0] for value in values[1:]):
            raise ValueError(f"models must use the same planned {field}")
        fixed_conditions[field] = values[0] if values else None
    tasks, configurations, groups = {}, {}, defaultdict(list)
    for cell in schedule.values():
        task_id, model_id = cell["task_id"], cell["model_id"]
        identity = _task_identity(cell)
        if task_id in tasks and tasks[task_id] != identity:
            raise ValueError(f"different task revisions or conditions share task_id: {task_id}")
        tasks[task_id] = identity
        configuration = cell.get("model")
        if model_id in configurations and configurations[model_id] != configuration:
            raise ValueError(f"different model configurations share model_id: {model_id}")
        configurations[model_id] = configuration
        groups[model_id].append(cell)
    task_sets = [{cell["task_id"] for cell in rows} for rows in groups.values()]
    if task_sets and any(selected != task_sets[0] for selected in task_sets[1:]):
        raise ValueError("models must use the same planned task set")
    for attempt_id, row in started.items():
        cell = schedule.get(attempt_id)
        if cell is None or (row["task_id"], row["model_id"]) != (cell["task_id"], cell["model_id"]):
            raise ValueError(f"attempt does not match planned identity: {attempt_id}")
        if _task_identity(row) != _task_identity(cell):
            raise ValueError(f"task_identity differs from planned identity: {attempt_id}")
        for field in ("harness", "harness_version", "tool_version", "information_profile"):
            if row.get(field) != cell.get(field):
                raise ValueError(f"{field} differs from planned configuration: {attempt_id}")
        if row.get("model") != cell.get("model"):
            raise ValueError(f"model configuration differs from planned configuration: {attempt_id}")
        evaluation = row.get("evaluation")
        if evaluation is not None:
            if (not isinstance(evaluation, dict) or not isinstance(evaluation.get("status"), str)
                    or evaluation["status"] not in RESULT_STATUSES):
                raise ValueError(f"unknown evaluation status in {attempt_id}")
            if type(evaluation.get("resolved")) is not bool or evaluation["resolved"] != (evaluation["status"] == "resolved"):
                raise ValueError(f"inconsistent resolved flag in {attempt_id}")
            if evaluation.get("task_id", row["task_id"]) != row["task_id"]:
                raise ValueError(f"evaluation task_id differs from attempt: {attempt_id}")
    models = {}
    for model_id, cells in sorted(groups.items()):
        statuses, terminals = Counter(), Counter()
        outcomes = {}
        resolved = unresolved = missing = errors = 0
        for cell in cells:
            row = started.get(cell["attempt_id"])
            evaluation = row.get("evaluation") if row else None
            issue = _outcome_issue(row, evaluation, tasks[cell["task_id"]]) if row else "missing"
            if row:
                terminals[str(row.get("terminal_status", "unknown"))] += 1
            if row is None:
                state = "missing"
                missing += 1
            elif issue:
                state = "error"
                errors += 1
            elif evaluation["resolved"]:
                state = "resolved"
                resolved += 1
            else:
                state = "unresolved"
                unresolved += 1
            status = issue or evaluation["status"]
            statuses[status] += 1
            outcomes[cell["task_id"]] = {
                "attempt_id": cell["attempt_id"], "state": state, "status": status,
                "resolved": state == "resolved", "task_identity": tasks[cell["task_id"]],
            }
        complete = planned is not None and not missing and not errors
        models[model_id] = {
            "model_configuration": configurations[model_id],
            "planned": len(cells), "started": len(cells) - missing,
            "resolved": resolved, "unresolved": unresolved, "missing": missing, "errors": errors,
            "resolved_rate": resolved / len(cells),
            "complete": complete, "status": "complete" if complete else "incomplete",
            "statuses": dict(statuses), "terminal_statuses": dict(terminals),
            "tasks": dict(sorted(outcomes.items())),
        }
    complete = bool(schedule) and all(result["complete"] for result in models.values())
    return {
        "schema_version": 2, "planned_schedule_supplied": planned is not None,
        "report_generation": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "software_identity": observed_software_identity(),
            "report_protocol_revision": REPORT_PROTOCOL_REVISION,
        },
        "dataset_identity": dataset, "software_identity": software,
        **fixed_conditions,
        "timing_protocol_revision": timing_revision,
        "complete": complete, "status": "complete" if complete else "incomplete", "models": models,
        "interpretation": {
            "resolved_rate": "Independently verified resolved tasks / all tasks in the fixed plan. Missing tasks and infrastructure failures remain in the denominator. An incomplete report is progress, not a final model comparison.",
            "errors": "Unavailable or untrusted generation, submission or evaluation evidence; separate from valid unresolved model attempts.",
            "report_generation": "Current reporter provenance is separate from original execution provenance. Recomputing never edits the original attempts or results.",
        },
    }


def summarize_run(output: Path, plan: dict | None = None) -> dict:
    """Read one run's plan and verified attempts without changing its files."""
    output = Path(output)
    if plan is None:
        plan = read_json(output / "plan.json")
    records = [load_attempt(path) for path in sorted((output / "attempts").glob("*/attempt.json"))]
    report = build_report(records, planned=plan["schedule"])
    for field in ("harness", "harness_version", "information_profile", "tool_version", "dataset_identity", "software_identity", "timing_protocol_revision"):
        if field in plan and plan[field] != report.get(field):
            raise ValueError(f"plan {field} differs from its scheduled configuration")
    report["mode"] = plan.get("mode", "external")
    return report
