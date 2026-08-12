#!/usr/bin/env python3
"""Bind selected v0.7 source-build Tasks to their compatible build image."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


ENVIRONMENT_REF_OVERRIDES = {
    "181469_dimension_annotation_guards": (
        "pytorch-boundary-cpu-source-build-cmake3-py311"
    ),
    "182083_autograd_worker_teardown": (
        "pytorch-boundary-cpu-source-build-cmake3-py311"
    ),
}


def repair() -> tuple[str, ...]:
    repaired = []
    for directory, environment_ref in ENVIRONMENT_REF_OVERRIDES.items():
        path = ROOT / "tasks/pytorch" / directory / "task.json"
        manifest = json.loads(path.read_bytes())
        if not isinstance(manifest, dict):
            raise ContractError(f"{path}: expected object")
        environment = manifest.get("environment")
        if not isinstance(environment, dict):
            raise ContractError(f"{path}: environment must be an object")
        source_loading = environment.get("source_loading")
        if not isinstance(source_loading, dict):
            raise ContractError(
                f"{path}: environment.source_loading must be an object"
            )
        if source_loading.get("mode") != "inplace_build":
            raise ContractError(f"{path}: expected inplace_build")
        manifest["environment_ref"] = environment_ref
        path.write_text(canonical_json(manifest), encoding="utf-8")
        repaired.append(str(manifest.get("task_id")))
    return tuple(repaired)


def main() -> int:
    repaired = repair()
    print(
        canonical_json(
            {"repaired_task_count": len(repaired), "tasks": list(repaired)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
