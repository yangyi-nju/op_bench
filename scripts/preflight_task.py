#!/usr/bin/env python3
"""Pre-admission preflight: verify a task is admission-ready without spending
30-60 minutes on a remote build.

Checks performed (all local, no docker, no GPU needed):

1. Task manifest validates.
2. Source snapshot directory exists and is non-empty.
3. hidden_test.patch applies cleanly to the snapshot.
4. gold.patch applies cleanly after hidden_test.patch.
5. For each fail_to_pass / pass_to_pass test name, the unittest test loader
   can resolve the name in a fresh workspace (after hidden_test.patch). This
   catches the `instantiate_device_type_tests` rename trap.
6. Resolved task has environment.host set if backend is remote_docker.

Usage:
    PYTHONPATH=src python3 scripts/preflight_task.py tasks/pytorch/<task_dir>
    PYTHONPATH=src python3 scripts/preflight_task.py --all  # all tasks in dataset

Exits 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.registry import load_resolved_task  # noqa: E402


def _apply_patch(workspace: Path, patch_path: Path) -> tuple[bool, str]:
    r = subprocess.run(
        ["git", "apply", "--check", str(patch_path)],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if r.returncode == 0:
        subprocess.run(["git", "apply", str(patch_path)],
                       cwd=str(workspace), capture_output=True)
        return True, "git apply"
    r2 = subprocess.run(
        ["bash", "-c", f"patch -p1 -F 3 -i {patch_path}"],
        cwd=str(workspace), capture_output=True, text=True,
    )
    if r2.returncode == 0:
        return True, "patch -F 3"
    return False, (r.stderr or r2.stderr or "patch failed").strip()


def _resolve_test_names(workspace: Path, test_file: str, test_names: list[str]) -> list[tuple[str, str]]:
    """Try to resolve each `Class.method` via unittest loader inside the workspace.

    Returns list of (name, status). status is 'OK' or an error string.
    """
    # Build a tiny Python probe that imports the test module and tries to load
    # each test by name. Uses unittest's defaultTestLoader.loadTestsFromName
    # to follow the same resolution PyTorch test runners use.
    test_dir = (workspace / test_file).parent
    module_name = Path(test_file).stem

    probe = f"""
import sys, importlib, traceback, unittest
sys.path.insert(0, {str(test_dir)!r})

# Common PyTorch test setup helpers may rely on these env vars.
import os
os.environ.setdefault('PYTORCH_TEST_WITH_SLOW', '0')

results = []
try:
    mod = importlib.import_module({module_name!r})
except Exception as exc:
    print('MODULE_IMPORT_FAIL:' + str(exc))
    sys.exit(0)

names = {test_names!r}
for full in names:
    cls_name, _, method = full.partition('.')
    cls = getattr(mod, cls_name, None)
    if cls is None:
        # Names may match dynamically generated subclasses (instantiate_device_type_tests)
        candidates = [a for a in dir(mod) if a.startswith(cls_name.split('CUDA')[0].split('CPU')[0])]
        results.append((full, 'MISSING_CLASS', candidates[:6]))
        continue
    if method and not hasattr(cls, method):
        # Try fuzzy match for parametrized variants
        methods = [m for m in dir(cls) if m.startswith(method.rsplit('_cuda', 1)[0].rsplit('_cpu', 1)[0])]
        results.append((full, 'MISSING_METHOD', methods[:8]))
        continue
    results.append((full, 'OK', None))

for r in results:
    print('RESULT:' + repr(r))
"""

    # Run probe with bare Python (no torch needed for symbol resolution if mod imports cleanly;
    # if mod requires torch, we still need it installed locally).
    r = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(workspace), capture_output=True, text=True, timeout=60,
    )
    out = r.stdout

    if "MODULE_IMPORT_FAIL:" in out:
        msg = out.split("MODULE_IMPORT_FAIL:", 1)[1].strip().split("\n", 1)[0]
        return [(name, f"MODULE_IMPORT_FAIL: {msg[:200]}") for name in test_names]

    results: list[tuple[str, str]] = []
    for line in out.splitlines():
        if line.startswith("RESULT:"):
            try:
                tup = eval(line[len("RESULT:"):])
                name, status, hint = tup
                if status == "OK":
                    results.append((name, "OK"))
                else:
                    h = f" (similar: {hint})" if hint else ""
                    results.append((name, f"{status}{h}"))
            except Exception:
                continue
    # Fallback if probe crashed
    if not results:
        results = [(name, f"PROBE_ERROR: stdout={out[:200]} stderr={r.stderr[:200]}") for name in test_names]
    return results


def preflight_task(task_dir: Path) -> tuple[bool, list[str]]:
    """Run all preflight checks on a task. Returns (ok, messages)."""
    messages: list[str] = []

    # 1. Resolve task manifest with registries
    try:
        task = load_resolved_task(
            task_dir / "task.json",
            environment_registry_path=ROOT / "environments" / "registry.json",
            source_registry_path=ROOT / "sources" / "registry.json",
        )
    except Exception as exc:
        return False, [f"FAIL: cannot load task: {exc}"]

    messages.append(f"task_id: {task.task_id}")
    messages.append(f"backend: {task.environment_backend}")
    messages.append(f"runtime_tier: {task.runtime_tier}")

    # 2. Snapshot exists
    snapshot = task.source_snapshot_path
    if not snapshot or not snapshot.exists():
        return False, messages + [f"FAIL: snapshot missing at {snapshot}"]
    messages.append(f"snapshot: {snapshot} ✓")

    # 3. Host for remote_docker
    if task.environment_backend == "remote_docker":
        if not task.environment_host:
            return False, messages + ["FAIL: backend is remote_docker but no environment.host (check registry default)"]
        messages.append(f"host: {task.environment_host} ✓")

    # 4. Patches apply in clean workspace
    with tempfile.TemporaryDirectory(prefix=f"preflight-{task.task_id}-") as tmp:
        ws = Path(tmp) / "ws"
        shutil.copytree(snapshot, ws, symlinks=True)

        hidden = task.hidden_test_patch_path
        if hidden.exists():
            ok, msg = _apply_patch(ws, hidden)
            if not ok:
                return False, messages + [f"FAIL: hidden_test.patch does not apply: {msg[:200]}"]
            messages.append(f"hidden_test.patch: {msg} ✓")

        # Resolve test names AFTER hidden patch (so newly-added tests are visible)
        test_command = task.data["evaluation"]["test_command"]
        # Extract test file: "{python} test/foo.py {test}" -> "test/foo.py"
        parts = test_command.split()
        test_file = None
        for p in parts:
            if p.endswith(".py"):
                test_file = p
                break
        if test_file:
            messages.append(f"test_file: {test_file}")
            all_names = list(task.fail_to_pass_tests) + list(task.pass_to_pass_tests)
            try:
                results = _resolve_test_names(ws, test_file, all_names)
                # MODULE_IMPORT_FAIL is a warning (likely missing torch locally), not a blocker.
                # MISSING_CLASS / MISSING_METHOD are real bugs we want to catch.
                hard_failures = [(n, s) for n, s in results if s != "OK" and not s.startswith("MODULE_IMPORT_FAIL")]
                soft_warnings = [(n, s) for n, s in results if s.startswith("MODULE_IMPORT_FAIL")]
                if hard_failures:
                    for n, s in hard_failures:
                        messages.append(f"  {n}: {s}")
                    return False, messages + ["FAIL: some test names did not resolve"]
                if soft_warnings:
                    messages.append(f"test names: {len(all_names)} declared, module not importable locally (skipping resolver) ⚠")
                else:
                    messages.append(f"test names ({len(all_names)}): all resolved ✓")
            except Exception as exc:
                messages.append(f"WARN: test name resolution probe failed: {exc} (continuing)")

        # Apply gold patch on top
        gold = task.gold_patch_path
        if gold.exists() and gold.read_text().strip() and not gold.read_text().startswith("# TODO"):
            ok, msg = _apply_patch(ws, gold)
            if not ok:
                return False, messages + [f"FAIL: gold.patch does not apply: {msg[:200]}"]
            messages.append(f"gold.patch: {msg} ✓")

    return True, messages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_dir", nargs="?", help="Path to a task directory")
    parser.add_argument("--all", action="store_true", help="Run on every task in pytorch_v0.4 dataset")
    args = parser.parse_args()

    if args.all:
        dataset = json.loads((ROOT / "datasets" / "pytorch_v0.4" / "dataset.json").read_text())
        task_dirs = [
            (ROOT / "datasets" / "pytorch_v0.4" / t["task_path"]).resolve()
            for t in dataset["tasks"]
        ]
    elif args.task_dir:
        task_dirs = [Path(args.task_dir).resolve()]
    else:
        parser.print_help()
        return 2

    total = len(task_dirs)
    failures = 0
    for i, td in enumerate(task_dirs, 1):
        print(f"\n[{i}/{total}] {td.name}")
        print("-" * 60)
        ok, msgs = preflight_task(td)
        for m in msgs:
            print(f"  {m}")
        if ok:
            print("  PREFLIGHT OK")
        else:
            failures += 1
            print("  PREFLIGHT FAILED")

    print(f"\n{'='*60}")
    print(f"Total: {total} | OK: {total - failures} | Failed: {failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
