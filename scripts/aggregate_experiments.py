#!/usr/bin/env python
"""Aggregate one or more experiment output directories into a combined report.

Usage:
    PYTHONPATH=src python3 scripts/aggregate_experiments.py \\
        runs/v0.5_codex_cpu runs/v0.5_codex_gpu \\
        --dataset datasets/pytorch_v0.5/dataset.json \\
        --output runs/v0.5_codex_all/summary.json

Combines the `results.jsonl` from each input directory, then computes extended
v0.5 metrics (patch conciseness, tier-weighted score, per-dimension rate).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.dataset import DatasetManifest
from op_bench.reporter import compute_extended_metrics, summarize_results, write_json
from op_bench.resume import ResultsStore, TRANSIENT_STATUSES


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _task_metadata_from_dataset(dataset_paths: list[Path]) -> dict[str, dict]:
    """Build a task_id -> metadata map from dataset manifests. Metadata carries
    the fields needed by `compute_extended_metrics` (gold_patch_lines,
    runtime_tier, problem_type, problem_dimension)."""
    metadata: dict[str, dict] = {}
    for dataset_path in dataset_paths:
        dataset = DatasetManifest.load(dataset_path)
        for task in dataset.load_tasks(verified_only=False):
            gold_path = task.gold_patch_path
            gold_lines = 0
            if gold_path.exists():
                try:
                    gold_lines = len(gold_path.read_text(encoding="utf-8").splitlines())
                except (OSError, UnicodeDecodeError):
                    pass
            metadata[task.task_id] = {
                "gold_patch_lines": gold_lines,
                "runtime_tier": task.runtime_tier,
                "problem_type": task.problem_type,
                "problem_dimension": task.problem_dimension,
                "problem_subclass": task.problem_subclass,
            }
    return metadata


def _experiment_integrity(
    records: list[dict], expected_task_ids: set[str], expected_repeat: int
) -> dict:
    agents = sorted(
        {str(r["agent"]) for r in records if r.get("agent") not in (None, "baseline")}
    )
    agent_records = {
        (str(r.get("task_id")), str(r.get("agent")), int(r.get("attempt"))): r
        for r in records
        if r.get("agent") != "baseline" and r.get("attempt") is not None
    }
    expected_keys = {
        (task_id, agent, attempt)
        for task_id in expected_task_ids
        for agent in agents
        for attempt in range(1, expected_repeat + 1)
    }
    missing = sorted(expected_keys - set(agent_records))
    transient = sorted(
        key
        for key in expected_keys & set(agent_records)
        if str(agent_records[key].get("status", "")) in TRANSIENT_STATUSES
    )
    baselines = {
        str(r.get("task_id")): str(r.get("status", ""))
        for r in records
        if r.get("agent") == "baseline" and r.get("task_id")
    }
    missing_baselines = sorted(expected_task_ids - set(baselines))
    invalid_baselines = sorted(
        (task_id, status)
        for task_id, status in baselines.items()
        if task_id in expected_task_ids and status != "baseline_reproduced"
    )
    unexpected_attempts = sorted(set(agent_records) - expected_keys)
    complete = bool(agents) and not any(
        (missing, transient, missing_baselines, invalid_baselines, unexpected_attempts)
    )
    return {
        "complete": complete,
        "expected_task_count": len(expected_task_ids),
        "agents": agents,
        "expected_repeat": expected_repeat,
        "expected_attempt_count": len(expected_keys),
        "observed_attempt_count": len(expected_keys & set(agent_records)),
        "missing_attempts": [list(key) for key in missing],
        "transient_attempts": [list(key) for key in transient],
        "missing_baselines": missing_baselines,
        "invalid_baselines": [list(item) for item in invalid_baselines],
        "unexpected_attempts": [list(key) for key in unexpected_attempts],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", nargs="+", help="Directories containing results.jsonl")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset manifest(s) for task metadata")
    parser.add_argument("--output", required=True, help="Output summary.json path")
    parser.add_argument("--expected-repeat", type=int, help="Expected repeats per task and agent")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless dataset, baselines, and expected attempts are complete",
    )
    args = parser.parse_args(argv)

    if args.require_complete and (not args.dataset or args.expected_repeat is None):
        parser.error("--require-complete requires --dataset and --expected-repeat")

    all_records: list[dict] = []
    for d in args.input_dirs:
        input_dir = Path(d)
        results_path = input_dir / "results.jsonl"
        raw_count = len(_load_jsonl(results_path))
        loaded = ResultsStore(input_dir).load_all_records()
        print(
            f"[aggregate] {results_path}: {len(loaded)} logical records "
            f"({raw_count} raw)",
            file=sys.stderr,
        )
        all_records.extend(loaded)

    task_meta = _task_metadata_from_dataset([Path(p) for p in args.dataset]) if args.dataset else {}
    agent_records = [r for r in all_records if r.get("agent") != "baseline"]
    baseline_records = [r for r in all_records if r.get("agent") == "baseline"]

    legacy = summarize_results(agent_records)
    extended = compute_extended_metrics(agent_records, task_meta)

    combined = {
        "input_dirs": [str(d) for d in args.input_dirs],
        "datasets": [str(p) for p in args.dataset],
        "total_records": len(all_records),
        "baseline_count": len(baseline_records),
        "legacy": legacy,
        "extended": extended,
    }
    if args.expected_repeat is not None:
        combined["integrity"] = _experiment_integrity(
            all_records, set(task_meta), args.expected_repeat
        )
    output = Path(args.output).resolve()
    write_json(output, combined)
    print(output)
    if args.require_complete and not combined["integrity"]["complete"]:
        print("[aggregate] experiment is incomplete; see summary.integrity", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
