#!/usr/bin/env python3
"""Rebuild source snapshots from sources/registry.json.

Run this after cloning the repo on a new machine:

    PYTHONPATH=src python3 scripts/setup_sources.py

Registry entry `snapshot_mode` controls what gets fetched:

- "overlay" (default): sparse checkout of `torch/` + `test/` only.
  Cheap (~500 MB), sufficient for python_overlay tasks.

- "kernel_full": full clone (no sparse) + required submodules initialized
  at depth 1. About 4 GB. Required for cuda_kernel_build tasks that must
  actually recompile PyTorch source (setup.py develop). Sparse-checkout
  is unreliable for kernel builds because PyTorch's build system reads
  many files (root-level .bzl, functorch/csrc/, tools/, third_party/*)
  that are hard to enumerate ahead of time.

Requires git and network access to github.com.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "sources" / "registry.json"
# Registry local_path values are relative to the registry file (sources/), matching
# how EnvironmentAsset in src/op_bench/registry.py resolves them.
REGISTRY_DIR = REGISTRY_PATH.parent

# Submodules needed for a kernel_build PyTorch source compile.
# Kept explicit (not `--init --recursive`) to keep the checkout small.
KERNEL_BUILD_SUBMODULES = [
    "third_party/cutlass",
    "third_party/cudnn_frontend",
    "third_party/eigen",
    "third_party/fbgemm",
    "third_party/flatbuffers",
    "third_party/fmt",
    "third_party/gloo",
    "third_party/googletest",
    "third_party/ideep",
    "third_party/kineto",
    "third_party/nlohmann",
    "third_party/onnx",
    "third_party/opentelemetry-cpp",
    "third_party/pocketfft",
    "third_party/protobuf",
    "third_party/pthreadpool",
    "third_party/pybind11",
    "third_party/sleef",
    "third_party/tensorpipe",
    "third_party/XNNPACK",
    "third_party/FP16",
    "third_party/FXdiv",
    "third_party/psimd",
    "third_party/cpuinfo",
]


def _run(cmd: list[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def _setup_overlay(local_path: Path, repo_url: str, commit: str) -> tuple[bool, str]:
    """Sparse checkout of torch/ + test/ only. Cheap, python_overlay use."""
    local_path.mkdir(parents=True, exist_ok=True)
    for cmd in [
        ["git", "init"],
        ["git", "remote", "add", "origin", repo_url],
        ["git", "config", "core.sparseCheckout", "true"],
    ]:
        r = _run(cmd, local_path)
        if r.returncode != 0:
            return False, f"{' '.join(cmd)}: {r.stderr.strip()}"

    sparse = local_path / ".git" / "info" / "sparse-checkout"
    sparse.parent.mkdir(parents=True, exist_ok=True)
    sparse.write_text("torch/\ntest/\n")

    r = _run(["git", "fetch", "--depth=1", "origin", commit], local_path, timeout=600)
    if r.returncode != 0:
        return False, f"git fetch: {r.stderr.strip()[:200]}"

    r = _run(["git", "checkout", "FETCH_HEAD"], local_path)
    if r.returncode != 0:
        return False, f"git checkout: {r.stderr.strip()}"

    _run(["git", "add", "-A"], local_path)
    _run(["git", "commit", "-m", f"overlay snapshot at {commit}", "--allow-empty"], local_path)
    return True, "overlay OK"


def _setup_kernel_full(local_path: Path, repo_url: str, commit: str) -> tuple[bool, str]:
    """Full clone (no sparse) + init submodules at depth 1. cuda_kernel_build use."""
    local_path.mkdir(parents=True, exist_ok=True)
    for cmd in [
        ["git", "init"],
        ["git", "remote", "add", "origin", repo_url],
    ]:
        r = _run(cmd, local_path)
        if r.returncode != 0:
            return False, f"{' '.join(cmd)}: {r.stderr.strip()}"

    # Full fetch (no sparse) — the whole point of kernel_full mode.
    r = _run(["git", "fetch", "--depth=1", "origin", commit], local_path, timeout=1200)
    if r.returncode != 0:
        return False, f"git fetch: {r.stderr.strip()[:200]}"

    r = _run(["git", "checkout", "FETCH_HEAD"], local_path, timeout=300)
    if r.returncode != 0:
        return False, f"git checkout: {r.stderr.strip()[:200]}"

    # Init the submodules PyTorch build needs.
    # Use `git submodule update --init` per-path so a missing/optional
    # submodule doesn't abort the whole init.
    ok_subs, failed_subs = 0, []
    for sub in KERNEL_BUILD_SUBMODULES:
        gitmod_check = local_path / sub / ".git"
        # Skip if already initialized
        if gitmod_check.exists():
            ok_subs += 1
            continue
        r = _run(
            ["git", "submodule", "update", "--init", "--depth=1", "--", sub],
            local_path,
            timeout=600,
        )
        if r.returncode != 0:
            # Some submodules may not exist on old base commits — record but continue
            failed_subs.append(f"{sub}: {r.stderr.strip()[:80]}")
        else:
            ok_subs += 1

    _run(["git", "add", "-A"], local_path)
    _run(["git", "commit", "-m", f"kernel_full snapshot at {commit}", "--allow-empty"], local_path)

    msg = f"kernel_full OK ({ok_subs}/{len(KERNEL_BUILD_SUBMODULES)} submodules)"
    if failed_subs:
        msg += f", {len(failed_subs)} skipped: {failed_subs[:3]}"
    return True, msg


def setup_source(entry: dict) -> bool:
    commit = entry["commit"]
    # Resolve local_path against the registry directory (sources/), not ROOT.
    # This matches how src/op_bench/registry.py::SourceAsset.local_path resolves paths.
    local_path = (REGISTRY_DIR / entry["local_path"]).resolve()
    repo_url = entry["repo_url"]
    mode = entry.get("snapshot_mode", "overlay")

    if local_path.exists() and (local_path / ".git").exists():
        # Verify the mode is consistent — for kernel_full we also want setup.py present.
        if mode == "kernel_full" and not (local_path / "setup.py").exists():
            print(f"  {commit[:7]}: exists but no setup.py (mode changed to kernel_full?), will re-create")
            import shutil
            shutil.rmtree(local_path)
        else:
            print(f"  skip (exists): {commit[:7]} [{mode}]")
            return True

    print(f"  fetch: {commit[:7]} [{mode}] from {repo_url}")

    if mode == "overlay":
        ok, msg = _setup_overlay(local_path, repo_url, commit)
    elif mode == "kernel_full":
        ok, msg = _setup_kernel_full(local_path, repo_url, commit)
    else:
        print(f"    FAIL: unknown snapshot_mode: {mode}")
        return False

    print(f"    {msg}")
    return ok


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
