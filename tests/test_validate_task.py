from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.validate_task import validate_manifest, validate_source_loading


class ValidateTaskTests(unittest.TestCase):
    def test_boundary_requires_b_subclass(self) -> None:
        manifest = self._manifest()
        manifest["operator"].update(
            {
                "problem_dimension": "boundary",
                "problem_subclass": "P3",
                "failure_contract": "wrong-result",
            }
        )

        self.assertIn(
            "operator.problem_subclass: boundary requires B1..B5",
            validate_manifest(manifest),
        )

    def test_historical_task_may_omit_taxonomy(self) -> None:
        self.assertEqual(validate_manifest(self._manifest()), [])

    def test_remote_task_rejects_non_opaque_inline_host_alias(self) -> None:
        manifest = self._manifest()
        manifest["environment"].update(
            {
                "backend": "remote_docker",
                "host": "user@private-host:22",
            }
        )

        self.assertIn(
            "environment.host must be an opaque lowercase host alias",
            validate_manifest(manifest),
        )

    def test_source_can_explicitly_record_no_linked_issue(self) -> None:
        manifest = self._manifest()
        manifest["source"]["issue_url"] = None
        manifest["source"]["issue_number"] = None
        schema = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "schemas/task_manifest.schema.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(validate_manifest(manifest), [])
        source = schema["properties"]["source"]["properties"]
        self.assertEqual(
            source["issue_url"]["type"], ["string", "null"]
        )
        self.assertEqual(
            source["issue_number"]["type"], ["integer", "null"]
        )

    def test_source_linked_issue_fields_must_be_jointly_absent(self) -> None:
        manifest = self._manifest()
        manifest["source"]["issue_url"] = None
        self.assertIn(
            "source.issue_url and source.issue_number must both be null or both be present",
            validate_manifest(manifest),
        )

        manifest = self._manifest()
        manifest["source"]["issue_number"] = None
        self.assertIn(
            "source.issue_url and source.issue_number must both be null or both be present",
            validate_manifest(manifest),
        )

    def test_invalid_taxonomy_v2_returns_contract_error(self) -> None:
        manifest = self._manifest()
        manifest["taxonomy"] = {
            "taxonomy_version": "v2",
            "contract_family": "other",
            "contract_detail_tags": [],
            "trigger_tags": [],
            "execution_context": {
                "devices": ["cpu"],
                "modes": ["eager"],
                "phases": ["forward"],
                "distributed": False,
            },
            "failure_type": "wrong_result",
            "root_cause_tags": [],
            "component_tags": [],
        }

        self.assertIn(
            "taxonomy.contract_family: unsupported value 'other'",
            validate_manifest(manifest),
        )

    def test_historical_precision_pair_may_omit_failure_contract(self) -> None:
        manifest = self._manifest()
        manifest["operator"].update(
            {
                "problem_dimension": "precision",
                "problem_subclass": "P3",
            }
        )

        self.assertEqual(validate_manifest(manifest), [])

    def test_complete_precision_taxonomy_is_valid(self) -> None:
        manifest = self._manifest()
        manifest["operator"].update(
            {
                "problem_dimension": "precision",
                "problem_subclass": "P5",
                "failure_contract": "crash-oob",
            }
        )

        self.assertEqual(validate_manifest(manifest), [])

    def test_partial_taxonomy_is_rejected(self) -> None:
        manifest = self._manifest()
        manifest["operator"]["problem_dimension"] = "boundary"

        self.assertIn(
            "operator taxonomy fields must be provided together",
            validate_manifest(manifest),
        )

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

    def test_accepts_matched_runtime_probe_and_promotion_metadata(self) -> None:
        manifest = self._manifest()
        manifest["environment_ref"] = "pytorch-matched"
        manifest["source_ref"] = "pytorch-source"
        manifest["runtime_tier"] = "cpu_python_overlay"
        manifest["environment"] = {
            "source_loading": {
                "mode": "python_overlay",
                "installed_package": "torch",
                "overlay_paths": ["torch/_refs/__init__.py"],
                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                "sync_before_tests": True,
            }
        }
        manifest["compatibility"] = {
            "target_module": "torch/_refs/__init__.py",
            "target_import": "torch._refs",
            "selector_module": "test/test_decomp.py",
            "minimal_operation": "import torch; torch.ones(1)",
            "evidence": "compatibility/evidence.json",
        }
        manifest["admission"] = {
            "status": "verified",
            "evidence": "admission/evidence.json",
            "compatibility_evidence": "compatibility/evidence.json",
            "verified_at": "2026-07-26T12:00:00Z",
        }
        manifest["metadata"]["curation_status"] = "verified"
        manifest["metadata"]["admission_status"] = "verified"

        self.assertEqual(validate_manifest(manifest), [])

    def test_rejects_invalid_matched_runtime_probe_metadata(self) -> None:
        cases = (
            (
                "target_module",
                "../torch/_refs/__init__.py",
                "compatibility.target_module must be a task-relative path without '..'",
            ),
            (
                "selector_module",
                "/test/test_decomp.py",
                "compatibility.selector_module must be a task-relative path without '..'",
            ),
            (
                "minimal_operation",
                "torch.ones(1)",
                "compatibility.minimal_operation must begin with 'import torch;'",
            ),
            (
                "evidence",
                "../evidence.json",
                "compatibility.evidence must be a task-relative path without '..'",
            ),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                manifest = self._manifest()
                manifest["compatibility"] = {
                    "target_module": "torch/_refs/__init__.py",
                    "target_import": "torch._refs",
                    "selector_module": "test/test_decomp.py",
                    "minimal_operation": "import torch; torch.ones(1)",
                    "evidence": "compatibility/evidence.json",
                }
                manifest["compatibility"][field] = value

                self.assertIn(message, validate_manifest(manifest))

    def test_verified_matched_runtime_admission_binds_the_same_evidence_path(self) -> None:
        manifest = self._manifest()
        manifest["compatibility"] = {
            "target_module": "torch/_refs/__init__.py",
            "target_import": "torch._refs",
            "selector_module": "test/test_decomp.py",
            "minimal_operation": "import torch; torch.ones(1)",
            "evidence": "compatibility/evidence.json",
        }
        manifest["admission"] = {
            "status": "verified",
            "evidence": "admission/evidence.json",
            "compatibility_evidence": "compatibility/stale.json",
            "verified_at": "2026-07-26T12:00:00Z",
        }
        manifest["metadata"]["admission_status"] = "verified"

        self.assertIn(
            "admission.compatibility_evidence must match compatibility.evidence",
            validate_manifest(manifest),
        )

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
