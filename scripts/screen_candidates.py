#!/usr/bin/env python3
"""Screen v0.5 precision-dimension PR candidates against hard filter rules.

Input: JSON array of candidate PRs, e.g. from `gh pr list --json ...` output.
Output: JSON with two arrays — passed candidates and rejected candidates with
reason strings. Rules match docs/v0.5/candidate_search.md §"硬性过滤条件".

Usage:
    gh pr list --repo pytorch/pytorch --state merged --limit 200 \\
        --search '(autocast OR "mixed precision") is:merged merged:2024-01-01..2025-04-30' \\
        --json number,title,url,mergedAt,body,files > candidates.json

    PYTHONPATH=src python3 scripts/screen_candidates.py \\
        --input candidates.json \\
        --subclass P3 \\
        --output candidates_screened.json

Rules applied:
    1. Title does not contain "revert" or "reland"
    2. Merged in 2024-01-01 to 2025-04-30
    3. File count ≤ 3
    4. Total additions + deletions in 20..200
    5. At least one test file modified
    6. Title/body does not read as "Add support for X" (feature add, not fix)

Rule 4 counts changed lines from gh's files.additions + files.deletions.

`--subclass` is echoed onto every output entry (does not affect filtering).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path


MERGE_WINDOW_START = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
MERGE_WINDOW_END = dt.datetime(2025, 4, 30, 23, 59, 59, tzinfo=dt.timezone.utc)

MAX_FILES = 3
MIN_LINES = 20
MAX_LINES = 200

REVERT_RE = re.compile(r"\b(revert|reland)\b", re.IGNORECASE)
FEATURE_ADD_RE = re.compile(
    r"\b(add\s+support|enable\s+.*\s+for|introduce\s+.*\s+support|support\s+for)\b",
    re.IGNORECASE,
)
TEST_PATH_RE = re.compile(r"(^|/)test[/_]|/tests?/", re.IGNORECASE)


def screen(candidate: dict, subclass: str | None) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    title = str(candidate.get("title", ""))
    body = str(candidate.get("body", ""))
    merged_at = candidate.get("mergedAt")
    files = candidate.get("files") or []

    # Rule 1
    if REVERT_RE.search(title):
        reasons.append("rule1: title looks like revert/reland")

    # Rule 2
    if merged_at:
        try:
            when = dt.datetime.fromisoformat(str(merged_at).replace("Z", "+00:00"))
            if when < MERGE_WINDOW_START or when > MERGE_WINDOW_END:
                reasons.append(f"rule2: merged {when.date()} outside 2024-01-01..2025-04-30 window")
        except ValueError:
            reasons.append(f"rule2: unparseable mergedAt={merged_at!r}")
    else:
        reasons.append("rule2: missing mergedAt")

    # Rule 3
    if len(files) > MAX_FILES:
        reasons.append(f"rule3: {len(files)} files changed (>{MAX_FILES})")

    # Rule 4
    total_lines = sum(int(f.get("additions", 0)) + int(f.get("deletions", 0)) for f in files)
    if total_lines < MIN_LINES or total_lines > MAX_LINES:
        reasons.append(f"rule4: {total_lines} lines changed (need {MIN_LINES}..{MAX_LINES})")

    # Rule 5
    test_files = [f for f in files if TEST_PATH_RE.search(str(f.get("path", "")))]
    if not test_files:
        reasons.append("rule5: no test file modified")

    # Rule 6 — feature-add PR pattern
    # Consider both title and first ~500 chars of body.
    body_snippet = body[:500]
    if FEATURE_ADD_RE.search(title) or FEATURE_ADD_RE.search(body_snippet):
        # Heuristic exception: PR title starts with "Fix" outweighs the feature-add pattern
        if not re.match(r"^\s*(fix|correct|handle)\b", title, re.IGNORECASE):
            reasons.append("rule6: reads as feature-add rather than correctness fix")

    return (len(reasons) == 0, reasons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to JSON array from gh pr list --json")
    parser.add_argument("--subclass", default=None, help="Echo this subclass tag (P1..P5) onto outputs")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    candidates = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(candidates, list):
        print(f"expected JSON array, got {type(candidates).__name__}", file=sys.stderr)
        return 2

    passed: list[dict] = []
    rejected: list[dict] = []
    for c in candidates:
        ok, reasons = screen(c, args.subclass)
        entry = {
            "number": c.get("number"),
            "url": c.get("url"),
            "title": c.get("title"),
            "mergedAt": c.get("mergedAt"),
            "subclass": args.subclass,
            "files": [f.get("path") for f in (c.get("files") or [])],
            "lines_changed": sum(int(f.get("additions", 0)) + int(f.get("deletions", 0)) for f in (c.get("files") or [])),
        }
        if ok:
            passed.append(entry)
        else:
            entry["reject_reasons"] = reasons
            rejected.append(entry)

    result = {
        "input": str(args.input),
        "subclass": args.subclass,
        "total": len(candidates),
        "passed_count": len(passed),
        "rejected_count": len(rejected),
        "passed": passed,
        "rejected": rejected,
    }
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[screen] {args.input}: {len(passed)} passed / {len(rejected)} rejected → {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
