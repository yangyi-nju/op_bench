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

# Submodule paths to always skip for kernel_full snapshots. These are large
# and unused by CPU/CUDA python-level builds. If future tasks need them,
# either add a per-task override or drop from this exclude list.
KERNEL_BUILD_SUBMODULE_EXCLUDES = frozenset({
    # Android/mobile — PyTorch check_submodules validates their presence but
    # setup.py only requires the *directory* to be non-empty, which submodule
    # init accomplishes. Keep them in for safety.
})


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


def _parse_gitmodules_paths(gitmodules_path: Path) -> list[str]:
    """Read paths from a .gitmodules file. Return list of `path = ...` values."""
    if not gitmodules_path.exists():
        return []
    paths: list[str] = []
    for line in gitmodules_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("path"):
            _, _, val = line.partition("=")
            val = val.strip()
            if val:
                paths.append(val)
    return paths


def _find_uninitialized_submodules(local_path: Path) -> list[str]:
    """Return top-level submodule paths that need (re-)initialization.

    A submodule needs init if either:
    - its own directory is empty (not initialized at all), OR
    - it has a .gitmodules of its own and any of its nested submodule
      directories is empty (nested submodules not initialized).

    We only return the *top-level* path — `git submodule update --init
    --recursive -- <path>` fixes both cases.
    """
    submodule_paths = _parse_gitmodules_paths(local_path / ".gitmodules")
    missing = []
    for sub in submodule_paths:
        if sub in KERNEL_BUILD_SUBMODULE_EXCLUDES:
            continue
        sub_dir = local_path / sub
        if not sub_dir.exists() or not any(sub_dir.iterdir()):
            missing.append(sub)
            continue
        # Check nested submodules if this one has its own .gitmodules
        nested_gitmodules = sub_dir / ".gitmodules"
        if nested_gitmodules.exists():
            for nested_sub in _parse_gitmodules_paths(nested_gitmodules):
                nested_dir = sub_dir / nested_sub
                if not nested_dir.exists() or not any(nested_dir.iterdir()):
                    missing.append(sub)  # top-level parent
                    break
    return missing


def _init_missing_submodules(local_path: Path, submodule_paths: list[str]) -> tuple[bool, str]:
    """Initialize the listed submodules recursively at depth 1. Used as a repair path."""
    ok_subs, failed_subs = 0, []
    for sub in submodule_paths:
        err = _init_submodule_recursive(local_path, sub)
        if err is None:
            ok_subs += 1
        else:
            failed_subs.append(f"{sub}: {err[:80]}")

    if failed_subs:
        return False, f"repair: {ok_subs}/{len(submodule_paths)} initialized, {len(failed_subs)} failed: {failed_subs[:3]}"
    return True, f"repair: {ok_subs}/{len(submodule_paths)} submodules initialized ✓"


def _setup_kernel_full(local_path: Path, repo_url: str, commit: str) -> tuple[bool, str]:
    """Full clone (no sparse) + init ALL submodules declared in .gitmodules at depth 1.

    PyTorch's setup.py runs check_submodules() which walks .gitmodules and errors
    on any submodule whose directory is empty. So we can't cherry-pick — we must
    init every path .gitmodules mentions (minus any explicit excludes).

    depth=1 keeps each submodule small (no history).
    """
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

    # Now read the committed .gitmodules and init every declared submodule.
    submodule_paths = _parse_gitmodules_paths(local_path / ".gitmodules")
    if not submodule_paths:
        return False, ".gitmodules missing or empty after checkout"

    ok_subs, failed_subs = 0, []
    for sub in submodule_paths:
        if sub in KERNEL_BUILD_SUBMODULE_EXCLUDES:
            continue
        # `git submodule update --init --depth=1 --recursive` for a single path.
        # --recursive is needed because PyTorch's setup.py check_submodules() also
        # verifies nested submodule paths like fbgemm/third_party/asmjit.
        # Retry once — submodule fetches sometimes fail transiently on slow networks.
        r = _init_submodule_recursive(local_path, sub)
        if r is None:
            ok_subs += 1
        else:
            failed_subs.append(f"{sub}: {r[:80]}")

    _run(["git", "add", "-A"], local_path)
    _run(["git", "commit", "-m", f"kernel_full snapshot at {commit}", "--allow-empty"], local_path)

    total = len(submodule_paths) - len(KERNEL_BUILD_SUBMODULE_EXCLUDES)
    msg = f"kernel_full OK ({ok_subs}/{total} submodules from .gitmodules, recursive)"
    if failed_subs:
        msg += f", {len(failed_subs)} failed: {failed_subs[:3]}"
        return False, msg
    return True, msg


def _init_submodule_recursive(local_path: Path, sub: str) -> str | None:
    """Init a submodule and its nested submodules at depth=1. Returns None on success,
    or the last stderr on failure. Retries once."""
    for _ in range(2):
        r = _run(
            ["git", "submodule", "update", "--init", "--depth=1", "--recursive", "--", sub],
            local_path,
            timeout=900,
        )
        if r.returncode == 0:
            return None
    return r.stderr.strip()


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
        elif mode == "kernel_full":
            # kernel_full exists — verify all submodules from .gitmodules are initialized.
            # This is cheap (just directory checks) and fixes the case where a previous
            # setup ran with a narrower whitelist.
            missing = _find_uninitialized_submodules(local_path)
            if missing:
                print(f"  {commit[:7]}: kernel_full needs {len(missing)} submodule(s) initialized")
                ok, msg = _init_missing_submodules(local_path, missing)
                print(f"    {msg}")
                return ok
            print(f"  skip (exists): {commit[:7]} [{mode}]")
            return True
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
