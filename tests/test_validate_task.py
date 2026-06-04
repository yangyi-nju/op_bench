from __future__ import annotations

import unittest

from scripts.validate_task import validate_manifest


class ValidateTaskTests(unittest.TestCase):
    def test_accepts_v02_runtime_tier_and_admission(self) -> None:
        manifest = self._manifest()
        manifest["environment_ref"] = "pytorch-cpu"
        manifest["source_ref"] = "pytorch-base"
        manifest["runtime_tier"] = "cpu_python_overlay"
        manifest["admission"] = {
            "status": "verified",
            "evidence": "admission/evidence.json",
            "verified_at": "2026-06-04T00:00:00Z",
        }
        manifest["metadata"]["admission_status"] = "verified"

        errors = validate_manifest(manifest)

        self.assertEqual(errors, [])

    def test_rejects_verified_admission_without_evidence(self) -> None:
        manifest = self._manifest()
        manifest["admission"] = {"status": "verified"}

        errors = validate_manifest(manifest)

        self.assertIn("admission.evidence is required when admission.status is 'verified'", errors)

    def test_rejects_conflicting_admission_statuses(self) -> None:
        manifest = self._manifest()
        manifest["admission"] = {"status": "blocked_source"}
        manifest["metadata"]["admission_status"] = "verified"

        errors = validate_manifest(manifest)

        self.assertIn("admission.status must match metadata.admission_status when both are provided", errors)

    def _manifest(self) -> dict[str, object]:
        return {
            "task_id": "fixture",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "pr_url": "https://github.com/local/repo/pull/1",
                "issue_url": "https://github.com/local/repo/issues/1",
                "issue_number": 1,
                "pr_number": 1,
                "base_commit": "abcdef1",
                "merge_commit": "abcdef2",
                "checkout_mode": "local-copy",
                "local_path": "/tmp/source",
            },
            "statement": {"title": "bug", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "torch.nn",
                "operator_name": "Fixture",
                "problem_type": "behavior",
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
                "fail_to_pass": ["test_fail"],
                "pass_to_pass": ["test_pass"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {
                "difficulty": "easy",
                "curation_status": "draft",
                "deterministic": True,
            },
        }


if __name__ == "__main__":
    unittest.main()
