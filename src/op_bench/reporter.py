from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# Weight per tier for `tier_weighted_score`. Kernel builds are far harder than
# python overlays; the weights reflect that observed cost/difficulty ratio.
TIER_WEIGHTS = {
    "cpu_python_overlay": 1.0,
    "cuda_python_overlay": 2.0,
    "cuda_kernel_build": 3.0,
}

def normalized_result_status(record: dict[str, Any]) -> str:
    """Normalize legacy test-outcome labels from their actual counters."""
    status = str(record.get("status", "unknown"))
    if status != "pass_to_pass_regressed":
        return status
    if "fail_to_pass_total" not in record or "pass_to_pass_total" not in record:
        return status
    f2p_total = int(record.get("fail_to_pass_total", 0) or 0)
    f2p_passed = int(record.get("fail_to_pass_passed", 0) or 0)
    if f2p_total > 0 and f2p_passed < f2p_total:
        return "fail_to_pass_failed"
    return status


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy summary shape (resolved_rate + failure_reasons + median_runtime).
    Kept for backward compatibility with v0.3/v0.4 downstream tooling."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        agent = str(record.get("agent", "unknown"))
        grouped[agent].append(record)

    agents: dict[str, dict[str, Any]] = {}
    for agent, agent_records in sorted(grouped.items()):
        total = len(agent_records)
        resolved = sum(1 for record in agent_records if normalized_result_status(record) == "resolved")
        durations = [float(record.get("duration_sec", 0.0)) for record in agent_records]
        failure_reasons = Counter(normalized_result_status(record) for record in agent_records)
        agents[agent] = {
            "total": total,
            "resolved": resolved,
            "resolved_rate": resolved / total if total else 0.0,
            "median_runtime_sec": _median(durations),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        }

    return {
        "total_records": len(records),
        "agents": agents,
    }


def compute_extended_metrics(
    records: list[dict[str, Any]],
    task_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """v0.5 multi-dimensional scoring. Callers pass agent records (baseline rows
    should be excluded before invoking) and, optionally, a `task_metadata` map
    keyed by task_id with values `{gold_patch_lines, runtime_tier,
    problem_dimension, problem_subclass}` for tier weighting and per-group breakdowns.

    Returned shape:

        {
          "<agent>": {
            "resolved_rate": ...,
            "patch_conciseness": ...,      # median gold_lines / agent_lines, in [0,1]
            "pass_to_pass_kept_rate": ...,
            "regression_rate": ...,        # fail_to_pass passed but pass_to_pass broken
            "tier_weighted_score": ...,
            "per_tier": {tier: rate, ...},
            "per_problem_type": {problem_type: rate, ...},
            "per_problem_dimension": {dim: rate, ...},
          },
          ...
        }
    """
    task_metadata = task_metadata or {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        agent = str(record.get("agent", "unknown"))
        if agent == "baseline":
            continue
        grouped[agent].append(record)

    out: dict[str, Any] = {}
    for agent, agent_records in sorted(grouped.items()):
        out[agent] = _metrics_for_agent(agent_records, task_metadata)
    return out


def _metrics_for_agent(
    records: list[dict[str, Any]],
    task_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {
            "resolved_rate": 0.0,
            "patch_conciseness": 0.0,
            "pass_to_pass_kept_rate": 0.0,
            "fail_to_pass_only_rate": 0.0,
            "regression_rate": 0.0,
            "tier_weighted_score": 0.0,
            "per_tier": {},
            "per_problem_type": {},
            "per_problem_dimension": {},
            "per_problem_subclass": {},
        }
    resolved = sum(1 for r in records if normalized_result_status(r) == "resolved")

    # patch_conciseness: median(gold_lines / agent_lines), only for resolved attempts
    # with a captured patch path. Clamped to [0, 1] (agents with tighter patch than
    # gold score 1.0, not >1).
    conciseness_ratios: list[float] = []
    for r in records:
        if normalized_result_status(r) != "resolved":
            continue
        meta = task_metadata.get(str(r.get("task_id", "")), {})
        gold_lines = int(meta.get("gold_patch_lines", 0)) if meta else 0
        agent_lines = _count_patch_lines(r.get("patch_path"))
        if gold_lines > 0 and agent_lines > 0:
            ratio = min(1.0, gold_lines / agent_lines)
            conciseness_ratios.append(ratio)
    patch_conciseness = _median(conciseness_ratios)

    # pass_to_pass_kept_rate: mean of (pass_to_pass_passed / total) across records
    # with a nonzero pass_to_pass_total. Records without pass_to_pass are skipped.
    p2p_ratios: list[float] = []
    for r in records:
        p2p_total = int(r.get("pass_to_pass_total", 0) or 0)
        if p2p_total <= 0:
            continue
        p2p_passed = int(r.get("pass_to_pass_passed", 0) or 0)
        p2p_ratios.append(p2p_passed / p2p_total)
    pass_to_pass_kept_rate = sum(p2p_ratios) / len(p2p_ratios) if p2p_ratios else 0.0

    strict_resolved_count = 0
    for r in records:
        f2p_pass = int(r.get("fail_to_pass_passed", 0) or 0)
        f2p_total = int(r.get("fail_to_pass_total", 0) or 0)
        p2p_pass = int(r.get("pass_to_pass_passed", 0) or 0)
        p2p_total = int(r.get("pass_to_pass_total", 0) or 0)
        if f2p_total > 0 and f2p_pass == f2p_total and p2p_pass == p2p_total:
            strict_resolved_count += 1
    fail_to_pass_only_rate = strict_resolved_count / total

    # regression_rate: fraction of attempts where fail_to_pass_passed == total
    # but pass_to_pass_passed < pass_to_pass_total. Isolates "fixed the bug but
    # broke something else" from "didn't fix the bug".
    regression_count = 0
    for r in records:
        f2p_pass = int(r.get("fail_to_pass_passed", 0) or 0)
        f2p_total = int(r.get("fail_to_pass_total", 0) or 0)
        p2p_pass = int(r.get("pass_to_pass_passed", 0) or 0)
        p2p_total = int(r.get("pass_to_pass_total", 0) or 0)
        if f2p_total > 0 and f2p_pass == f2p_total and p2p_total > 0 and p2p_pass < p2p_total:
            regression_count += 1
    regression_rate = regression_count / total

    # tier_weighted_score: sum(weight * resolved) / sum(weight * total) per tier.
    # Unknown tiers weight 1.0.
    weight_sum = 0.0
    weighted_resolved = 0.0
    for r in records:
        meta = task_metadata.get(str(r.get("task_id", "")), {})
        tier = str(meta.get("runtime_tier", "cpu_python_overlay"))
        weight = TIER_WEIGHTS.get(tier, 1.0)
        weight_sum += weight
        if normalized_result_status(r) == "resolved":
            weighted_resolved += weight
    tier_weighted_score = weighted_resolved / weight_sum if weight_sum else 0.0

    per_tier = _group_rate(records, lambda r: task_metadata.get(str(r.get("task_id", "")), {}).get("runtime_tier"))
    per_problem_type = _group_rate(records, lambda r: task_metadata.get(str(r.get("task_id", "")), {}).get("problem_type"))
    per_problem_dimension = _group_rate(records, lambda r: task_metadata.get(str(r.get("task_id", "")), {}).get("problem_dimension"))
    per_problem_subclass = _group_rate(records, lambda r: task_metadata.get(str(r.get("task_id", "")), {}).get("problem_subclass"))

    return {
        "total": total,
        "resolved": resolved,
        "resolved_rate": resolved / total,
        "patch_conciseness": patch_conciseness,
        "pass_to_pass_kept_rate": pass_to_pass_kept_rate,
        "fail_to_pass_only_rate": fail_to_pass_only_rate,
        "regression_rate": regression_rate,
        "tier_weighted_score": tier_weighted_score,
        "per_tier": per_tier,
        "per_problem_type": per_problem_type,
        "per_problem_dimension": per_problem_dimension,
        "per_problem_subclass": per_problem_subclass,
    }


def _group_rate(records: list[dict[str, Any]], key_fn) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = key_fn(r)
        groups[str(key) if key else "unclassified"].append(r)
    return {
        group: {
            "total": len(items),
            "resolved": sum(1 for r in items if normalized_result_status(r) == "resolved"),
            "resolved_rate": (
                sum(1 for r in items if normalized_result_status(r) == "resolved") / len(items)
                if items else 0.0
            ),
        }
        for group, items in sorted(groups.items())
    }


def _count_patch_lines(patch_path: object) -> int:
    if not patch_path:
        return 0
    try:
        return sum(1 for _ in Path(str(patch_path)).read_text(encoding="utf-8").splitlines())
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return 0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
