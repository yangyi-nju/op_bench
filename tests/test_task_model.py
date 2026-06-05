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
                "task_id": "fixture",
                "version": "v1",
                "environment_ref": "pytorch-cpu",
                "runtime_tier": "cpu_python_overlay",
                "source_ref": "pytorch-fixture",
                "admission": {
                    "status": "verified",
                    "evidence": "admission/evidence.json",
                    "verified_at": "2026-06-04T00:00:00Z",
                },
                "source": {
                    "pr_url": "https://github.com/local/op-fixture/pull/1",
                    "issue_url": "https://github.com/local/op-fixture/issues/1",
                    "repo": "local/op-fixture",
                    "issue_number": 1,
                    "pr_number": 1,
                    "base_commit": "localbase",
                    "merge_commit": "localmerge",
                    "checkout_mode": "local-copy",
                    "local_path": "../../fixtures/source_repo",
                    "snapshot_path": "snapshot/source",
                    "snapshot_hash": "sha256:abc123",
                    "snapshot_method": "from_local_repo",
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
                    "image_digest": "sha256:image123",
                    "digest_kind": "local_image_id",
                    "platform": "linux/amd64",
                    "python_version": "3",
                    "os": "local",
                    "build_mode": "editable-python",
                    "hardware": {"device": "cpu", "min_memory_gb": 1},
                    "dependencies": [],
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
                    "layer": "A",
                    "admission_status": "verified",
                    "source_loading_verified": True,
                },
            }
            (root / "task.json").write_text(json.dumps(data), encoding="utf-8")

            task = TaskManifest.load(root / "task.json")

            self.assertEqual(task.task_id, "fixture")
            self.assertEqual(task.task_dir, root)
            self.assertEqual(task.gold_patch_path, root / "artifacts/gold.patch")
            self.assertEqual(task.source_snapshot_path, root / "snapshot" / "source")
            self.assertEqual(task.source_snapshot_hash, "sha256:abc123")
            self.assertEqual(task.source_snapshot_method, "from_local_repo")
            self.assertEqual(task.source_ref, "pytorch-fixture")
            self.assertEqual(task.environment_image_digest, "sha256:image123")
            self.assertEqual(task.environment_digest_kind, "local_image_id")
            self.assertEqual(task.environment_platform, "linux/amd64")
            self.assertEqual(task.environment_ref, "pytorch-cpu")
            self.assertEqual(task.runtime_tier, "cpu_python_overlay")
            self.assertEqual(task.environment_preflight_workdir, "/tmp")
            self.assertEqual(task.source_loading_mode, "python_overlay")
            self.assertEqual(task.source_loading_overlay_paths, ["torch/nn/modules/linear.py"])
            self.assertEqual(task.metadata_layer, "A")
            self.assertEqual(task.metadata_admission_status, "verified")
            self.assertIs(task.metadata_source_loading_verified, True)
            self.assertEqual(task.admission_status, "verified")
            self.assertEqual(task.admission_evidence_path, root / "admission/evidence.json")
            self.assertEqual(task.admission_verified_at, "2026-06-04T00:00:00Z")
            self.assertEqual(
                task.command_for_test("tests.test_special.TestSpecialExpit.test_nan_is_preserved"),
                [sys.executable, "-m", "unittest", "tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
            )


if __name__ == "__main__":
    unittest.main()
