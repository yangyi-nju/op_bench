"""One frozen patch, independently saved variants, one logical task outcome."""
from __future__ import annotations

from pathlib import Path
import time

from op_bench.io import read_json, write_json
from op_bench.provenance import software_identity
from op_bench.data.task import TaskInputError
from op_bench.evaluation.judgment import variant_status


def evaluate_variants(evaluator, task, patch: str, output_dir: Path,
                      *, baseline_artifact: Path | None = None,
                      baseline_source: dict | None = None) -> dict:
    root = Path(output_dir).expanduser().resolve()
    if (root / "result.json").exists():
        raise TaskInputError(f"Refusing to overwrite an existing evaluation: {root}")
    if not isinstance(patch, str):
        raise TaskInputError("patch must be a string")
    root.mkdir(parents=True, exist_ok=True)
    (root / "patch.diff").write_text(patch, encoding="utf-8")
    started = time.monotonic()
    variants = task.required_variants()
    required = [variant.variant_id for variant in variants]
    result = {"schema_version": 2, "task_id": task.task_id,
              "software_identity": software_identity(),
              "task_identity": task.identity_dict(), "status": "evaluation_error",
              "resolved": False, "required_variants": required, "variants": {},
              "duration_sec": 0.0, "error": None,
              "aggregation": "all_required_variants_one_frozen_patch_one_logical_task"}
    phase = "baseline_artifact"
    active_variant = None

    def record(variant_id, relative, child):
        result["variants"][variant_id] = {"evaluation_path": f"{relative}/result.json",
                                          "status": child["status"], "resolved": child["resolved"]}

    try:
        baseline_paths = {}
        if baseline_artifact is not None:
            artifact = Path(baseline_artifact).expanduser().resolve()
            index = read_json(artifact / "baseline.json")
            if (index.get("kind") != "trusted_baseline_variants" or index.get("status") != "ready"
                    or set(index.get("variants", {})) != set(required)):
                raise TaskInputError("A compatible ready baseline is required for every grading variant")
            for variant_id, relative in index["variants"].items():
                if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                        or not (artifact / relative).resolve().is_relative_to(artifact)):
                    raise TaskInputError("Variant baseline paths must stay inside their artifact")
                baseline_paths[variant_id] = artifact / relative
        for index, variant in enumerate(variants, 1):
            phase = f"variant:{variant.variant_id}"
            relative = f"variants/{index:06d}"
            active_variant = (variant.variant_id, relative)
            child = evaluator.evaluate(task.as_variant(variant.variant_id), patch, root / relative,
                **({"baseline_artifact": baseline_paths[variant.variant_id],
                    "baseline_source": baseline_source} if baseline_paths else {}))
            record(variant.variant_id, relative, child)
            active_variant = None
    except BaseException as exc:
        if active_variant is not None:
            # The child owns partial execution evidence even when interrupted.
            name, relative = active_variant
            try:
                partial = read_json(root / relative / "result.json")
                if isinstance(partial, dict) and partial.get("task_id") == task.task_id:
                    record(name, relative, partial)
            except (OSError, ValueError, KeyError):
                pass
        result["error"] = {"stage": phase, "type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, Exception):
            raise
    finally:
        result["status"] = variant_status(result["variants"], required)
        result["resolved"] = result["status"] == "resolved"
        result["duration_sec"] = time.monotonic() - started
        write_json(root / "result.json", result)
    return result
