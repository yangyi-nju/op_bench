"""Executable quality controls, separate from human task admission."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.io import read_json, write_json
from op_bench.runtime.snapshot import snapshot_task
from op_bench.data.task import TaskSpec


def _not_run(name: str) -> dict:
    return {"evaluation_path": f"{name}/result.json", "status": "not_run", "resolved": False, "error": None}


def _record(control: dict, evaluation: dict) -> None:
    control.update(status=evaluation["status"], resolved=evaluation["resolved"], error=evaluation.get("error"))


def _cases(output: Path, control: dict):
    """Read the sole case evidence from each independent evaluation."""
    if control["status"] == "not_run":
        return
    path = output / control["evaluation_path"]
    directory = path.parent
    saved = read_json(path)
    if "required_variants" in saved:
        for name, child in saved["variants"].items():
            for case in _cases(directory, child):
                yield {**case, "id": f"{name}/{case['id']}"}
    else:
        yield from saved["cases"]


def _controls(result: dict) -> list[dict]:
    return [result["baseline"], result["reference"], *result["alternatives"], *result["mutations"]]


def _summarize(result: dict, output: Path) -> None:
    baseline, reference = result["baseline"], result["reference"]
    cases = list(_cases(output, baseline))
    mutations = [list(_cases(output, control)) for control in result["mutations"]]
    targets = [case for case in cases if case["group"] == "fail_to_pass"]
    checks = {
        "baseline_fail_to_pass_fail": baseline["status"] == "test_failed" and bool(targets)
            and all(case["status"] == "failed" for case in targets),
        "baseline_pass_to_pass_pass": baseline["status"] in {"test_failed", "resolved"}
            and all(case["status"] == "passed" for case in cases if case["group"] == "pass_to_pass"),
        "reference_passes": reference["resolved"],
        "provided_alternatives_pass": all(control["resolved"] for control in result["alternatives"]),
        # An unavailable environment or malformed patch does not demonstrate
        # detection of an incorrect implementation.
        "provided_mutations_rejected": all(
            control["status"] == "test_failed" and any(case["status"] == "failed" for case in cases)
            for control, cases in zip(result["mutations"], mutations)),
    }
    result.update(checks=checks, execution_checks_passed=result["error"] is None and all(checks.values()))
    result["mutation_review"] = [
        {"index": index, "status": control["status"],
         "failed_cases": [case["id"] for case in cases if case["status"] == "failed"],
         "review_required": "Confirm the declared behavior was exercised and its assertion detected the intended defect. Import, syntax and setup failures do not demonstrate semantic mutation coverage."}
        for index, (control, cases) in enumerate(zip(result["mutations"], mutations), 1)]


def _block_pending(result: dict, error: dict) -> None:
    result["error"] = error
    for control in _controls(result):
        if control["status"] == "not_run":
            control["error"] = error


def check_task(task: TaskSpec, reference: str, output_dir: str | Path, *,
               alternatives: list[str] | None = None, mutations: list[str] | None = None,
               reuse_baseline: bool = False, baseline_artifact: Path | None = None) -> dict:
    if reuse_baseline and baseline_artifact is not None:
        raise ValueError("Choose a new baseline build or an existing artifact, not both")
    alternatives, mutations = list(alternatives or []), list(mutations or [])
    output = task.validate_output_path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema_version": 2, "task_id": task.task_id,
              "task_identity": task.identity_dict(), "baseline_build": None,
              "reused_baseline_artifact": str(Path(baseline_artifact).resolve()) if baseline_artifact is not None else None,
              "task_file": "inputs/task.json", "phase": "declare_controls", "error": None,
              "execution_checks_passed": False, "checks": {}, "duration_sec": 0.0,
              "limitations": "Execution checks do not establish prompt sufficiency, independent oracle correctness, alternative-solution coverage or mutation coverage. No verified status is assigned.",
              "baseline": _not_run("baseline"), "reference": _not_run("reference"),
              "alternatives": [_not_run(f"alternative-{i}") for i in range(1, len(alternatives) + 1)],
              "mutations": [_not_run(f"mutation-{i}") for i in range(1, len(mutations) + 1)], "control_plan": []}
    write_json(output / "controls.json", result)
    active_control = None

    def phase(name: str) -> None:
        result["phase"] = name
        write_json(output / "controls.json", result)

    try:
        declared = [("baseline", ""), ("reference", reference)]
        declared += [(f"alternative-{index}", patch) for index, patch in enumerate(alternatives, 1)]
        declared += [(f"mutation-{index}", patch) for index, patch in enumerate(mutations, 1)]
        for name, patch in declared:
            relative = f"declared-controls/{name}.patch"
            path = output / relative
            path.parent.mkdir(exist_ok=True)
            path.write_text(patch, encoding="utf-8")
            result["control_plan"].append({"control": name, "patch_file": relative})
        phase("source_identity")
        original_source = None
        if baseline_artifact is not None:
            from op_bench.runtime.execution import source_identity
            original_source = source_identity(task)
            # The cache identity and history-free snapshot must refer to the
            # same commit, even if HEAD or a declared branch moves meanwhile.
            if original_source["kind"] == "git":
                task = replace(task, source=replace(task.source, revision=original_source["revision"]))
        # Freeze all source, grader and installed-image inputs once before the
        # potentially long controls, including when reusing a trusted build.
        phase("snapshot")
        task = snapshot_task(task, output / "inputs")
        options = {}
        if baseline_artifact is not None:
            options.update(baseline_artifact=baseline_artifact, baseline_source=original_source)
        if reuse_baseline:
            from op_bench.runtime.baseline import build_baseline
            phase("baseline_build")
            built = build_baseline(task, output / "baseline-build")
            result["baseline_build"] = {"path": "baseline-build", "status": built["status"],
                "duration_sec": built["duration_sec"], "error": built.get("error"),
                "variants": {name: None if child is None else {"status": child["status"], "error": child.get("error")}
                             for name, child in built.get("variant_results", {}).items()}}
            if built["status"] != "ready":
                _block_pending(result, {"stage": "baseline_build", "type": "baseline_unavailable",
                    "message": "Trusted baseline construction did not complete; no controls were evaluated",
                    "baseline_record": "baseline-build/baseline.json"})
                return result
            options["baseline_artifact"] = output / "baseline-build"
        evaluator = PatchEvaluator()
        for control, (name, patch) in zip(_controls(result), declared):
            phase(f"{name}_evaluation" if name in {"baseline", "reference"} else name)
            active_control = control
            _record(control, evaluator.evaluate(task, patch, output / name, **options))
            active_control = None
            write_json(output / "controls.json", result)
            if name == "baseline" and control["status"] not in {"test_failed", "resolved"}:
                _block_pending(result, {"stage": "baseline_evaluation", "type": "baseline_unavailable",
                    "message": "Baseline evaluation was unavailable; remaining controls were not evaluated",
                    "cause": control.get("error")})
                return result
        result["phase"] = "complete"
    except BaseException as exc:
        # Evaluators persist partial evidence in their finally blocks before
        # propagating an interruption. Retain it instead of relabeling already
        # executed tests as not_run in the control matrix.
        if active_control is not None:
            try:
                partial = read_json(output / active_control["evaluation_path"])
                if isinstance(partial, dict) and partial.get("task_id") == task.task_id:
                    _record(active_control, partial)
            except (OSError, ValueError, KeyError):
                pass
        _block_pending(result, {"stage": result["phase"], "type": type(exc).__name__, "message": str(exc)})
        if not isinstance(exc, Exception):
            raise
    finally:
        try:
            _summarize(result, output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(execution_checks_passed=False, checks={}, mutation_review=[])
            result["error"] = result["error"] or {"stage": "control_summary", "type": type(exc).__name__, "message": str(exc)}
        result["duration_sec"] = time.monotonic() - started
        write_json(output / "controls.json", result)
    return result
