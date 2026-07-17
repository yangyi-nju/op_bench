from __future__ import annotations

import os
from pathlib import Path
import subprocess


_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "OpBench Test",
    "GIT_AUTHOR_EMAIL": "opbench@example.invalid",
    "GIT_AUTHOR_DATE": "2026-07-17T00:00:00Z",
    "GIT_COMMITTER_NAME": "OpBench Test",
    "GIT_COMMITTER_EMAIL": "opbench@example.invalid",
    "GIT_COMMITTER_DATE": "2026-07-17T00:00:00Z",
}


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ("git", "-c", "core.autocrlf=false", "-C", str(root), *args),
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_GIT_ENV,
    )


def initialize_git_repo(root: Path) -> str:
    root.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "operator.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "helper.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (root / "tests" / "test_operator.py").write_text(
        "def test_operator():\n    assert True\n", encoding="utf-8"
    )
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "-m", "base")
    return git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()
