from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.source_loading import build_source_loading_command
from op_bench.task import TaskManifest


class SourceLoadingTests(unittest.TestCase):
    def test_builds_python_overlay_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text(
                json.dumps(
                    {
                        "task_id": "overlay_task",
                        "version": "v1",
                        "source": {
                            "repo": "local/repo",
                            "local_path": str(root),
                            "base_commit": "localbase",
                            "merge_commit": "localmerge",
                            "checkout_mode": "local-copy",
                        },
                        "statement": {"title": "overlay", "body": "body", "labels": []},
                        "operator": {
                            "framework": "pytorch",
                            "component": "torch.nn",
                            "operator_name": "torch.nn.Linear",
                            "problem_type": "python-overlay",
                            "tags": [],
                        },
                        "environment": {
                            "backend": "docker",
                            "tier": "cpu-deterministic",
                            "image": "example/image:tag",
                            "workspace_dir": "/workspace",
                            "python_executable": "python",
                            "python_version": "3.11",
                            "os": "linux",
                            "build_mode": "prebuilt-wheel",
                            "hardware": {"device": "cpu", "min_memory_gb": 1},
                            "dependencies": [],
                            "preflight_commands": ["{python} --version"],
                            "source_loading": {
                                "mode": "python_overlay",
                                "installed_package": "torch",
                                "overlay_paths": ["torch/nn/modules/linear.py"],
                                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                                "sync_before_tests": True,
                            },
                        },
                        "agent_visible": {
                            "repo_setup_commands": [],
                            "known_constraints": [],
                            "allowed_test_commands": ["{python} -m unittest {test}"],
                        },
                        "evaluation": {
                            "setup_commands": [],
                            "fail_to_pass": ["test_case"],
                            "pass_to_pass": ["test_case"],
                            "test_command": "{python} -m unittest {test}",
                            "timeout_sec": 30,
                        },
                        "artifacts": {"gold_patch": "gold.patch", "test_patch": "test.patch"},
                        "metadata": {
                            "difficulty": "easy",
                            "curation_status": "draft",
                            "deterministic": True,
                            "estimated_runtime_min": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )

            command = build_source_loading_command(TaskManifest.load(task_path))

            self.assertIsNotNone(command)
            assert command is not None
            self.assertEqual(command[0:2], ["python", "-c"])
            config = json.loads(command[3])
            self.assertEqual(config["workspace_dir"], "/workspace")
            self.assertEqual(config["installed_package"], "torch")
            self.assertEqual(config["overlay_paths"], ["torch/nn/modules/linear.py"])


if __name__ == "__main__":
    unittest.main()
