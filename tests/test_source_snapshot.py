from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.task import TaskManifest
from scripts.prepare_source_snapshot import record


class SourceSnapshotRecordTests(unittest.TestCase):
    def test_record_contains_registry_ready_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "file.py").write_text("VALUE = 1\n", encoding="utf-8")
            task = self._task(root)

            result = record(task, snapshot, "ready", [])

            self.assertEqual(result["source_id"], "pytorch-base")
            self.assertEqual(result["repo_url"], "https://github.com/pytorch/pytorch.git")
            self.assertEqual(result["commit"], "abcdef1")
            self.assertEqual(result["submodules"], {"policy": "none_required", "status": "not_initialized"})
            self.assertEqual(result["source_loading_modes"], ["python_overlay"])
            self.assertEqual(result["related_tasks"], ["fixture"])
            self.assertEqual(result["file_count"], 1)

    def _task(self, root: Path) -> TaskManifest:
        task_dir = root / "task"
        task_dir.mkdir()
        (task_dir / "task.json").write_text(
            json.dumps(
                {
                    "task_id": "fixture",
                    "source_ref": "pytorch-base",
                    "source": {
                        "repo": "pytorch/pytorch",
                        "repo_url": "https://github.com/pytorch/pytorch.git",
                        "base_commit": "abcdef1",
                    },
                    "environment": {
                        "source_loading": {
                            "mode": "python_overlay",
                            "installed_package": "torch",
                            "overlay_paths": ["torch/nn/modules/linear.py"],
                            "runtime_site_packages": "/tmp/site-packages",
                            "sync_before_tests": True,
                        }
                    },
                    "artifacts": {"gold_patch": "gold.patch", "test_patch": "test.patch"},
                    "evaluation": {
                        "fail_to_pass": ["fail"],
                        "pass_to_pass": ["pass"],
                        "test_command": "{python} -m unittest {test}",
                        "timeout_sec": 30,
                    },
                    "metadata": {},
                }
            ),
            encoding="utf-8",
        )
        return TaskManifest.load(task_dir / "task.json")


if __name__ == "__main__":
    unittest.main()
