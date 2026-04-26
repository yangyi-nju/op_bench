from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from op_bench.task import TaskManifest


class TaskManifestTests(unittest.TestCase):
    def test_load_resolves_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "artifacts").mkdir()
            data = {
                "task_id": "smoke",
                "version": "v1",
                "source": {
                    "pr_url": "https://github.com/local/op-smoke/pull/1",
                    "issue_url": "https://github.com/local/op-smoke/issues/1",
                    "repo": "local/op-smoke",
                    "issue_number": 1,
                    "pr_number": 1,
                    "base_commit": "localbase",
                    "merge_commit": "localmerge",
                    "checkout_mode": "local-copy",
                    "local_path": "../../fixtures/smoke_repo",
                },
                "statement": {"title": "bug", "body": "body", "labels": []},
                "operator": {
                    "framework": "pytorch",
                    "component": "torch.special",
                    "operator_name": "torch.special.expit",
                    "problem_type": "numerical-semantics",
                    "tags": ["cpu"],
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
                    "fail_to_pass": ["tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
                    "pass_to_pass": ["tests.test_special.TestSpecialExpit.test_regular_value"],
                    "test_command": "{python} -m unittest {test}",
                    "timeout_sec": 30,
                },
                "artifacts": {
                    "gold_patch": "artifacts/gold.patch",
                    "test_patch": "artifacts/test.patch",
                },
                "metadata": {
                    "difficulty": "easy",
                    "curation_status": "verified",
                    "deterministic": True,
                    "estimated_runtime_min": 1,
                },
            }
            (root / "task.json").write_text(json.dumps(data), encoding="utf-8")

            task = TaskManifest.load(root / "task.json")

            self.assertEqual(task.task_id, "smoke")
            self.assertEqual(task.task_dir, root)
            self.assertEqual(task.gold_patch_path, root / "artifacts/gold.patch")
            self.assertEqual(
                task.command_for_test("tests.test_special.TestSpecialExpit.test_nan_is_preserved"),
                [sys.executable, "-m", "unittest", "tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
            )


if __name__ == "__main__":
    unittest.main()
