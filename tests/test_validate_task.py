from __future__ import annotations

import unittest

from scripts.validate_task import validate_manifest, validate_source_loading


class ValidateTaskTests(unittest.TestCase):
    def test_rejects_invalid_inplace_build_environment(self) -> None:
        errors = validate_source_loading(
            {
                "mode": "inplace_build",
                "build_environment": {"../BAD": "1", "GOOD": ["not-scalar"]},
            }
        )

        self.assertIn(
            "environment.source_loading.build_environment keys must be shell variable names",
            errors,
        )
        self.assertIn(
            "environment.source_loading.build_environment values must be scalar",
            errors,
        )

    def test_rejects_artifact_path_traversal(self) -> None:
        manifest = self._manifest()
        manifest["artifacts"]["gold_patch"] = "../gold.patch"

        errors = validate_manifest(manifest)

        self.assertIn(
            "artifacts.gold_patch must be a task-relative path without '..': '../gold.patch'",
            errors,
        )

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

    def test_environment_ref_allows_registry_driven_environment_fields(self) -> None:
        manifest = self._manifest()
        manifest["environment_ref"] = "pytorch-cpu"
        manifest["runtime_tier"] = "cpu_python_overlay"
        manifest["environment"] = {
            "source_loading": {
                "mode": "python_overlay",
                "installed_package": "torch",
                "overlay_paths": ["torch/nn/modules/linear.py"],
                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                "sync_before_tests": True,
            }
        }

        errors = validate_manifest(manifest)

        self.assertEqual(errors, [])

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
