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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", nargs="+", help="Directories containing results.jsonl")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset manifest(s) for task metadata")
    parser.add_argument("--output", required=True, help="Output summary.json path")
    args = parser.parse_args(argv)

    all_records: list[dict] = []
    for d in args.input_dirs:
        results_path = Path(d) / "results.jsonl"
        loaded = _load_jsonl(results_path)
        print(f"[aggregate] {results_path}: {len(loaded)} records", file=sys.stderr)
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
    output = Path(args.output).resolve()
    write_json(output, combined)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
