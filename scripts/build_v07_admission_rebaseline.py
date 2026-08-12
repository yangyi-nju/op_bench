#!/usr/bin/env python3
"""Rebuild the frozen t0007 legacy Admission contract migration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.admission_rebaseline import (  # noqa: E402
    build_admission_contract_rebaseline,
    migrated_admission_payload,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


BASELINE_COMMIT = "4f5addc"
TASK_RELATIVE = Path("tasks/pytorch/132616_cuda_mem_get_info")
REBASELINE_RELATIVE = Path(
    "factory/v0.7/p6/cuda_mem_get_info_contract_rebaseline.json"
)


def _git_bytes(root: Path, relative_path: Path) -> bytes:
    completed = subprocess.run(
        (
            "git",
            "show",
            f"{BASELINE_COMMIT}:{relative_path.as_posix()}",
        ),
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ContractError(
            f"baseline {relative_path}: git show failed at {BASELINE_COMMIT}"
        )
    return completed.stdout


def build(root: Path) -> None:
    selected_root = root.resolve()
    task_path = selected_root / TASK_RELATIVE / "task.json"
    task = TaskManifest.load(task_path)
    baseline_manifest = _git_bytes(selected_root, TASK_RELATIVE / "task.json")
    baseline_admission = _git_bytes(
        selected_root,
        TASK_RELATIVE / "admission/evidence.json",
    )
    baseline_artifacts = {
        "gold_patch": _git_bytes(
            selected_root,
            TASK_RELATIVE / "artifacts/gold.patch",
        ),
        "hidden_test_patch": _git_bytes(
            selected_root,
            TASK_RELATIVE / "artifacts/hidden_test.patch",
        ),
    }
    rebaseline = build_admission_contract_rebaseline(
        task=task,
        baseline_commit=BASELINE_COMMIT,
        baseline_manifest_bytes=baseline_manifest,
        baseline_admission_bytes=baseline_admission,
        baseline_artifact_bytes=baseline_artifacts,
    )
    migrated = migrated_admission_payload(
        task=task,
        baseline_admission_bytes=baseline_admission,
    )
    rebaseline_path = selected_root / REBASELINE_RELATIVE
    rebaseline_path.parent.mkdir(parents=True, exist_ok=True)
    rebaseline_path.write_text(
        canonical_json(rebaseline) + "\n",
        encoding="utf-8",
    )
    admission_path = selected_root / TASK_RELATIVE / "admission/evidence.json"
    admission_path.write_text(
        json.dumps(
            migrated,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    try:
        build(args.repo_root)
    except (ContractError, OSError) as exc:
        print(f"admission rebaseline failed: {exc}", file=sys.stderr)
        return 1
    print(REBASELINE_RELATIVE.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
