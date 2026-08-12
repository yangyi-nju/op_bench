#!/usr/bin/env python3
"""Rebuild only the two repaired Batch-7 candidates without touching verified tasks."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for selected in (SRC, SCRIPTS):
    if str(selected) not in sys.path:
        sys.path.insert(0, str(selected))

import build_v07_overlay_replacements_batch7 as batch7  # noqa: E402
import build_v07_replacement_tasks as replacement_builder  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402


REPAIRED_TASK_IDS = {
    "pytorch__179028__integer_range_lowering_registry",
}
TASKS = tuple(
    spec for spec in batch7.TASKS if spec["task_id"] in REPAIRED_TASK_IDS
)


def main() -> int:
    for spec in TASKS:
        batch7._materialize(spec)
    replacement_builder.TASKS = TASKS
    replacement_builder.BASE_SOURCE_REGISTRY = ROOT / "sources/staging_v07_replacements.json"
    for spec in TASKS:
        replacement_builder._build_task(spec)
    replacement_builder._register_sources()
    replacement_builder._update_review_packets()
    print(canonical_json({"rebuilt": [spec["task_id"] for spec in TASKS]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
