#!/usr/bin/env python3

"""Minimal validator for op_bench task manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REQUIRED_PATHS = [
    ("task_id",),
    ("version",),
    ("source", "repo"),
    ("source", "pr_url"),
    ("source", "issue_url"),
    ("source", "issue_number"),
    ("source", "pr_number"),
    ("source", "base_commit"),
    ("source", "merge_commit"),
    ("statement", "title"),
    ("statement", "body"),
    ("operator", "framework"),
    ("operator", "operator_name"),
    ("environment", "tier"),
    ("environment", "build_mode"),
    ("environment", "hardware", "device"),
    ("agent_visible", "repo_setup_commands"),
    ("evaluation", "fail_to_pass"),
    ("evaluation", "pass_to_pass"),
    ("evaluation", "test_command"),
    ("artifacts", "gold_patch"),
    ("artifacts", "test_patch"),
    ("metadata", "difficulty"),
    ("metadata", "curation_status"),
    ("metadata", "deterministic"),
]

ALLOWED_TIERS = {"cpu-deterministic", "single-gpu", "kernel-build"}
ALLOWED_BUILD_MODES = {"editable-python", "source-build", "prebuilt-wheel"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_CURATION_STATUSES = {"draft", "verified"}
ALLOWED_CHECKOUT_MODES = {"git", "local-copy"}
ALLOW_EMPTY_REQUIRED_PATHS = {
    ("agent_visible", "repo_setup_commands"),
}


def lookup(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for path in REQUIRED_PATHS:
        try:
            value = lookup(data, path)
        except KeyError:
            errors.append(f"missing required field: {'.'.join(path)}")
            continue

        if path not in ALLOW_EMPTY_REQUIRED_PATHS and value in ("", [], {}):
            errors.append(f"empty required field: {'.'.join(path)}")

    source = data.get("source", {})
    checkout_mode = source.get("checkout_mode", "git")
    if checkout_mode not in ALLOWED_CHECKOUT_MODES:
        errors.append(
            f"invalid source.checkout_mode: {checkout_mode!r}; expected one of {sorted(ALLOWED_CHECKOUT_MODES)}"
        )
    if checkout_mode == "local-copy" and not source.get("local_path"):
        errors.append("source.local_path is required when source.checkout_mode is 'local-copy'")

    try:
        tier = lookup(data, ("environment", "tier"))
        if tier not in ALLOWED_TIERS:
            errors.append(
                f"invalid environment.tier: {tier!r}; expected one of {sorted(ALLOWED_TIERS)}"
            )
    except KeyError:
        pass

    try:
        build_mode = lookup(data, ("environment", "build_mode"))
        if build_mode not in ALLOWED_BUILD_MODES:
            errors.append(
                "invalid environment.build_mode: "
                f"{build_mode!r}; expected one of {sorted(ALLOWED_BUILD_MODES)}"
            )
    except KeyError:
        pass

    try:
        difficulty = lookup(data, ("metadata", "difficulty"))
        if difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(
                f"invalid metadata.difficulty: {difficulty!r}; expected one of {sorted(ALLOWED_DIFFICULTIES)}"
            )
    except KeyError:
        pass

    try:
        curation_status = lookup(data, ("metadata", "curation_status"))
        if curation_status not in ALLOWED_CURATION_STATUSES:
            errors.append(
                "invalid metadata.curation_status: "
                f"{curation_status!r}; expected one of {sorted(ALLOWED_CURATION_STATUSES)}"
            )
    except KeyError:
        pass

    try:
        fail_to_pass = lookup(data, ("evaluation", "fail_to_pass"))
        if not isinstance(fail_to_pass, list) or not fail_to_pass:
            errors.append("evaluation.fail_to_pass must be a non-empty list")
        elif contains_draft_test_entry(fail_to_pass):
            errors.append("evaluation.fail_to_pass contains unresolved draft test entries")
    except KeyError:
        pass

    try:
        pass_to_pass = lookup(data, ("evaluation", "pass_to_pass"))
        if not isinstance(pass_to_pass, list) or not pass_to_pass:
            errors.append("evaluation.pass_to_pass must be a non-empty list")
        elif contains_draft_test_entry(pass_to_pass):
            errors.append("evaluation.pass_to_pass contains unresolved draft test entries")
    except KeyError:
        pass

    return errors


def contains_draft_test_entry(tests: list[Any]) -> bool:
    draft_prefix = "TO" + "DO:"
    return any(str(test).startswith(draft_prefix) for test in tests)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_task.py <task_manifest.json>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    errors = validate_manifest(data)
    if errors:
        print(f"{path}: invalid manifest")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"{path}: manifest looks valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
