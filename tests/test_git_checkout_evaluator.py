from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from op_bench.evaluator import Evaluator
from op_bench.task import TaskManifest


class GitCheckoutEvaluatorTests(unittest.TestCase):
    def test_evaluator_replays_git_checkout_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            repo = root / "repo"
            repo.mkdir()
            self._run(["git", "init"], repo)
            self._run(["git", "config", "user.name", "Test"], repo)
            self._run(["git", "config", "user.email", "test@example.com"], repo)
            (repo / "calc.py").write_text(
                "def normalize(value):\n    return 0 if value != value else value\n",
                encoding="utf-8",
            )
            self._run(["git", "add", "calc.py"], repo)
            self._run(["git", "commit", "-m", "base"], repo)
            base_commit = self._run(["git", "rev-parse", "HEAD"], repo).stdout.strip()

            task_dir = root / "task"
            artifacts = task_dir / "artifacts"
            artifacts.mkdir(parents=True)
            (artifacts / "test.patch").write_text(
                textwrap.dedent(
                    """\
                    diff --git a/test_calc.py b/test_calc.py
                    new file mode 100644
                    index 0000000..bbed1c3
                    --- /dev/null
                    +++ b/test_calc.py
                    @@ -0,0 +1,12 @@
                    +import math
                    +import unittest
                    +
                    +from calc import normalize
                    +
                    +
                    +class TestNormalize(unittest.TestCase):
                    +    def test_nan_is_preserved(self):
                    +        self.assertTrue(math.isnan(normalize(float("nan"))))
                    +
                    +    def test_number_is_preserved(self):
                    +        self.assertEqual(normalize(1), 1)
                    """
                ),
                encoding="utf-8",
            )
            (artifacts / "gold.patch").write_text(
                textwrap.dedent(
                    """\
                    diff --git a/calc.py b/calc.py
                    index 9185a80..83f0271 100644
                    --- a/calc.py
                    +++ b/calc.py
                    @@ -1,2 +1,2 @@
                     def normalize(value):
                    -    return 0 if value != value else value
                    +    return value
                    """
                ),
                encoding="utf-8",
            )
            manifest = {
                "task_id": "git__normalize_nan",
                "version": "v1",
                "source": {
                    "pr_url": "https://github.com/local/repo/pull/1",
                    "issue_url": "https://github.com/local/repo/issues/1",
                    "repo": "local/repo",
                    "repo_url": str(repo),
                    "issue_number": 1,
                    "pr_number": 1,
                    "base_commit": base_commit,
                    "merge_commit": base_commit,
                    "checkout_mode": "git",
                },
                "statement": {"title": "preserve nan", "body": "body", "labels": []},
                "operator": {
                    "framework": "pytorch",
                    "component": "test",
                    "operator_name": "normalize",
                    "problem_type": "numerical-semantics",
                    "tags": [],
                },
                "environment": {
                    "tier": "cpu-deterministic",
                    "image": "local",
                    "python_version": "3",
                    "os": "local",
                    "build_mode": "editable-python",
                    "hardware": {"device": "cpu", "min_memory_gb": 1},
                    "dependencies": [],
                },
                "agent_visible": {
                    "repo_setup_commands": [],
                    "known_constraints": [],
                    "allowed_test_commands": ["{python} -m unittest {test}"],
                },
                "evaluation": {
                    "setup_commands": [],
                    "fail_to_pass": ["test_calc.TestNormalize.test_nan_is_preserved"],
                    "pass_to_pass": ["test_calc.TestNormalize.test_number_is_preserved"],
                    "test_command": "{python} -m unittest {test}",
                    "timeout_sec": 30,
                },
                "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
                "metadata": {
                    "difficulty": "easy",
                    "curation_status": "verified",
                    "deterministic": True,
                    "estimated_runtime_min": 1,
                },
            }
            (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")

            task = TaskManifest.load(task_dir / "task.json")
            baseline = Evaluator().evaluate_baseline(task)
            gold = Evaluator().evaluate_gold(task)

            self.assertEqual(baseline.status, "baseline_reproduced")
            self.assertEqual(gold.status, "resolved")

    def _run(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
