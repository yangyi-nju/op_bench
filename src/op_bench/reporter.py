from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def summarize_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        agent = str(record.get("agent", "unknown"))
        grouped[agent].append(record)

    agents: dict[str, dict[str, Any]] = {}
    for agent, agent_records in sorted(grouped.items()):
        total = len(agent_records)
        resolved = sum(1 for record in agent_records if record.get("status") == "resolved")
        durations = [float(record.get("duration_sec", 0.0)) for record in agent_records]
        failure_reasons = Counter(str(record.get("status", "unknown")) for record in agent_records)
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


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
