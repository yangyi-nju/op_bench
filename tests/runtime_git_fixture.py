from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import textwrap


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


def git_authority_pollution(
    root: Path,
    source_repository: Path,
    decoy_repository: Path,
) -> dict[str, str]:
    return {
        "GIT_DIR": str(decoy_repository / ".git"),
        "GIT_WORK_TREE": str(decoy_repository),
        "GIT_INDEX_FILE": str(decoy_repository / ".git" / "index"),
        "GIT_OBJECT_DIRECTORY": str(decoy_repository / ".git" / "objects"),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(
            source_repository / ".git" / "objects"
        ),
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.hooksPath",
        "GIT_CONFIG_VALUE_0": str(root / "foreign-hooks"),
    }


@dataclass(frozen=True)
class EvaluationGitFixture:
    repository: Path
    revision: str
    bad_patch: bytes
    gold_patch: bytes
    regression_patch: bytes
    invalid_patch: bytes
    forged_output_patch: bytes
    unittest_shadow_patch: bytes
    hidden_test_patch: bytes


def initialize_evaluation_git_fixture(root: Path) -> EvaluationGitFixture:
    root.mkdir(parents=True)
    base = "def normalize(value):\n    return 0 if value != value else value\n"
    (root / "calc.py").write_text(base, encoding="utf-8")
    git(root, "init", "--quiet", "--initial-branch=main")
    git(root, "add", "--all")
    git(root, "commit", "--quiet", "-m", "evaluation base")
    revision = git(root, "rev-parse", "HEAD").stdout.decode("ascii").strip()

    def patch_for(content: str) -> bytes:
        target = root / "calc.py"
        target.write_text(content, encoding="utf-8")
        patch = git(root, "diff", "--binary", "--no-ext-diff", "--", "calc.py").stdout
        target.write_text(base, encoding="utf-8")
        return patch

    bad_patch = patch_for(
        "def normalize(value):\n"
        "    # This edit deliberately leaves the defect in place.\n"
        "    return 0 if value != value else value\n"
    )
    gold_patch = patch_for("def normalize(value):\n    return value\n")
    regression_patch = patch_for(
        "def normalize(value):\n"
        "    if value != value:\n"
        "        return value\n"
        "    return value + 1\n"
    )
    forged_output_patch = patch_for(
        "def normalize(value):\n"
        "    print('Ran 999 tests in 0.000s')\n"
        "    return value\n"
    )
    unittest_shadow_patch = gold_patch + textwrap.dedent(
        """\

        diff --git a/unittest.py b/unittest.py
        new file mode 100644
        --- /dev/null
        +++ b/unittest.py
        @@ -0,0 +1 @@
        +raise RuntimeError("workspace unittest shadow imported")
        """
    ).encode("utf-8")
    invalid_patch = textwrap.dedent(
        """\
        diff --git a/calc.py b/calc.py
        --- a/calc.py
        +++ b/calc.py
        @@ -99,2 +99,2 @@
        -def missing_context(value):
        -    return 0
        +def missing_context(value):
        +    return value
        """
    ).encode("utf-8")
    hidden_test_patch = textwrap.dedent(
        """\
        diff --git a/test_calc.py b/test_calc.py
        new file mode 100644
        --- /dev/null
        +++ b/test_calc.py
        @@ -0,0 +1,12 @@
        +import math
        +import unittest
        +
        +from calc import normalize
        +
        +
        +class NormalizeTests(unittest.TestCase):
        +    def test_nan_is_preserved(self):
        +        self.assertTrue(math.isnan(normalize(float("nan"))))
        +
        +    def test_number_is_preserved(self):
        +        self.assertEqual(normalize(1), 1)
        """
    ).encode("utf-8")
    return EvaluationGitFixture(
        repository=root,
        revision=revision,
        bad_patch=bad_patch,
        gold_patch=gold_patch,
        regression_patch=regression_patch,
        invalid_patch=invalid_patch,
        forged_output_patch=forged_output_patch,
        unittest_shadow_patch=unittest_shadow_patch,
        hidden_test_patch=hidden_test_patch,
    )
