#!/usr/bin/env python3
"""Rebuild source snapshots from sources/registry.json.

Run this after cloning the repo on a new machine:

    PYTHONPATH=src python3 scripts/setup_sources.py

Requires git and network access to github.com.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "sources" / "registry.json"


def setup_source(entry: dict) -> bool:
    commit = entry["commit"]
    local_path = ROOT / entry["local_path"]

    if local_path.exists() and (local_path / ".git").exists():
        print(f"  skip (exists): {commit[:7]}")
        return True

    repo_url = entry["repo_url"]
    print(f"  fetch: {commit[:7]} from {repo_url}")

    local_path.mkdir(parents=True, exist_ok=True)

    commands = [
        ["git", "init"],
        ["git", "remote", "add", "origin", repo_url],
        ["git", "config", "core.sparseCheckout", "true"],
    ]
    for cmd in commands:
        r = subprocess.run(cmd, cwd=str(local_path), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    FAIL: {' '.join(cmd)}: {r.stderr.strip()}")
            return False

    sparse_checkout = local_path / ".git" / "info" / "sparse-checkout"
    sparse_checkout.parent.mkdir(parents=True, exist_ok=True)
    sparse_checkout.write_text("torch/\ntest/\n")

    r = subprocess.run(
        ["git", "fetch", "--depth=1", "origin", commit],
        cwd=str(local_path), capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        print(f"    FAIL: git fetch: {r.stderr.strip()[:200]}")
        return False

    r = subprocess.run(
        ["git", "checkout", "FETCH_HEAD"],
        cwd=str(local_path), capture_output=True, text=True,
    )
    if r.returncode != 0:
        print(f"    FAIL: git checkout: {r.stderr.strip()}")
        return False

    subprocess.run(["git", "add", "-A"], cwd=str(local_path), capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", f"source snapshot at {commit}", "--allow-empty"],
        cwd=str(local_path), capture_output=True,
    )

    print(f"    done: {commit[:7]}")
    return True


def main() -> int:
    if not REGISTRY_PATH.exists():
        print(f"registry not found: {REGISTRY_PATH}", file=sys.stderr)
        return 1

    registry = json.loads(REGISTRY_PATH.read_text())
    sources = registry.get("sources", [])
    print(f"sources/registry.json: {len(sources)} entries\n")

    ok = 0
    fail = 0
    for entry in sources:
        print(f"[{entry['id']}]")
        if setup_source(entry):
            ok += 1
        else:
            fail += 1

    print(f"\nDone: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
