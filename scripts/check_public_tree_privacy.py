#!/usr/bin/env python3
"""Reject private connection details in tracked public artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SRC = REPOSITORY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.public_tree_privacy import scan_public_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    tracked_paths = _git_tracked_paths(root)
    if tracked_paths is None:
        print(
            json.dumps(
                {"error": "tracked_paths_unavailable"},
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    findings = scan_public_tree(root, tracked_paths=tracked_paths)
    payload = {
        "finding_count": len(findings),
        "findings": [
            {
                "code": finding.code,
                "line": finding.line,
                "path": finding.path,
            }
            for finding in findings
        ],
    }
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    return 1 if findings else 0


def _git_tracked_paths(root: Path) -> tuple[Path, ...] | None:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return tuple(Path(value) for value in decoded.split("\0") if value)


if __name__ == "__main__":
    raise SystemExit(main())
