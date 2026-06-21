#!/usr/bin/env python3
"""Batch admission runner for v0.3 tasks.

Usage:
    PYTHONPATH=src python3 scripts/run_admission_batch.py [--stop-on-failure]

Requires Docker with op-bench/pytorch-cpu:torch2.6.0-py311 image available.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    "tasks/pytorch/168295_autograd_create_graph",
    "tasks/pytorch/124385_load_state_dict_prefix",
    "tasks/pytorch/161488_lbfgs_wolfe",
    "tasks/pytorch/168159_embeddingbag_2d_offset",
    "tasks/pytorch/162340_nn_arg_length",
    "tasks/pytorch/150975_autograd_backward_inputs",
    "tasks/pytorch/149312_lr_scheduler_last_epoch",
    "tasks/pytorch/163961_dataloader_subset",
    "tasks/pytorch/127190_lr_scheduler_deepcopy",
    "tasks/pytorch/143455_set_submodule",
]


def main() -> int:
    stop_on_failure = "--stop-on-failure" in sys.argv
    results: list[dict] = []

    for task_rel in TASKS:
        task_dir = ROOT / task_rel
        task_data = json.loads((task_dir / "task.json").read_text())
        task_id = task_data["task_id"]

        if task_data.get("admission", {}).get("status") == "verified":
            print(f"SKIP {task_id} (already verified)")
            results.append({"task_id": task_id, "decision": "skip"})
            continue

        print(f"\n{'='*60}")
        print(f"RUNNING: {task_id}")
        print(f"{'='*60}")

        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_admission.py"),
            "--task", str(task_dir),
            "--write-task-evidence",
            "--environment-registry", str(ROOT / "environments" / "registry.json"),
            "--source-registry", str(ROOT / "sources" / "registry.json"),
        ]
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, env={
            **__import__("os").environ,
            "PYTHONPATH": str(ROOT / "src"),
        })

        print(r.stdout)
        if r.stderr:
            print(r.stderr, file=sys.stderr)

        # Parse result from stdout (last JSON line)
        decision = "unknown"
        for line in reversed(r.stdout.strip().splitlines()):
            try:
                parsed = json.loads(line)
                decision = parsed.get("decision", "unknown")
                break
            except (json.JSONDecodeError, ValueError):
                continue

        status = "PASS" if decision == "verified" else "FAIL"
        print(f"\n{status}: {task_id} -> {decision}")
        results.append({"task_id": task_id, "decision": decision, "exit_code": r.returncode})

        if status == "FAIL" and stop_on_failure:
            print("\nStopping on first failure (--stop-on-failure)")
            break

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    verified = sum(1 for r in results if r["decision"] == "verified")
    skipped = sum(1 for r in results if r["decision"] == "skip")
    failed = len(results) - verified - skipped
    for r in results:
        print(f"  {r['decision']:20s} | {r['task_id']}")
    print(f"\nVerified: {verified} | Failed: {failed} | Skipped: {skipped} | Total: {len(results)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
