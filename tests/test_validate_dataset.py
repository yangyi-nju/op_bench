from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_dataset import validate_dataset


class ValidateDatasetTests(unittest.TestCase):
    def test_verified_task_requires_admission_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, dataset_dir = self._verified_dataset(Path(tmp))
            del dataset["tasks"][0]["admission_evidence"]

            errors = validate_dataset(dataset, dataset_dir, require_verified=True)

            self.assertIn("fixture: admission_evidence is required for verified tasks", errors)

    def test_verified_evidence_must_match_task_and_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_dir = self._verified_dataset(root)
            evidence_path = root / "tasks/fixture/admission/evidence.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["task_id"] = "different"
            evidence["admission"]["decision"] = "gold_failed"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

            errors = validate_dataset(dataset, dataset_dir, require_verified=True)

            self.assertIn("fixture: admission evidence task_id mismatch: different", errors)
            self.assertIn("fixture: verified admission evidence must have decision='verified'", errors)

    def test_verified_evidence_must_match_current_manifest_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_dir = self._verified_dataset(root)
            task_path = root / "tasks/fixture/task.json"
            manifest = json.loads(task_path.read_text(encoding="utf-8"))
            manifest["statement"]["body"] = "changed after admission"
            task_path.write_text(json.dumps(manifest), encoding="utf-8")

            errors = validate_dataset(dataset, dataset_dir, require_verified=True)

            self.assertIn("fixture: admission evidence task_manifest_hash does not match current task.json", errors)

    def test_verified_task_requires_resolvable_asset_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, dataset_dir = self._verified_dataset(Path(tmp))
            invalid = copy.deepcopy(dataset)
            invalid["registries"]["environments"] = "missing-environments.json"

            errors = validate_dataset(invalid, dataset_dir, require_verified=True)

            self.assertTrue(any("cannot load environment registry" in error for error in errors), errors)

    def test_valid_verified_dataset_passes_evidence_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dataset, dataset_dir = self._verified_dataset(Path(tmp))

            errors = validate_dataset(dataset, dataset_dir, require_verified=True)

            self.assertEqual(errors, [])

    def test_asset_references_must_be_compatible_with_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset, dataset_dir = self._verified_dataset(root)
            source_registry_path = dataset_dir / "source-registry.json"
            source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
            source_registry["sources"][0]["commit"] = "different"
            source_registry["sources"][0]["source_loading_modes"] = ["full_source_build"]
            source_registry_path.write_text(json.dumps(source_registry), encoding="utf-8")
            environment_registry_path = dataset_dir / "environment-registry.json"
            environment_registry = json.loads(environment_registry_path.read_text(encoding="utf-8"))
            environment_registry["environments"][0]["runtime_tier"] = "cuda_declared"
            environment_registry["environments"][0]["source_loading_modes"] = ["full_source_build"]
            environment_registry_path.write_text(json.dumps(environment_registry), encoding="utf-8")

            errors = validate_dataset(dataset, dataset_dir, require_verified=True)

            self.assertIn("fixture: task runtime_tier must match environment asset runtime_tier", errors)
            self.assertIn("fixture: task base_commit must match source asset commit", errors)
            self.assertIn("fixture: task source_loading mode is not supported by environment asset", errors)
            self.assertIn("fixture: task source_loading mode is not supported by source asset", errors)

    def _verified_dataset(self, root: Path) -> tuple[dict[str, object], Path]:
        task_dir = root / "tasks/fixture"
        admission_dir = task_dir / "admission"
        admission_dir.mkdir(parents=True)
        manifest = self._manifest()
        task_path = task_dir / "task.json"
        task_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        task_hash = f"sha256:{hashlib.sha256(task_path.read_bytes()).hexdigest()}"
        (admission_dir / "evidence.json").write_text(
            json.dumps(
                {
                    "schema_version": "v1",
                    "evidence_id": "fixture-evidence",
                    "task_id": "fixture",
                    "task_manifest_hash": task_hash,
                    "created_at": "2026-06-04T00:00:00Z",
                    "source": {"id": "source-fixture"},
                    "environment": {"id": "environment-fixture"},
                    "baseline": {"status": "baseline_reproduced"},
                    "gold": {"status": "resolved"},
                    "admission": {
                        "decision": "verified",
                        "verified": True,
                        "failure_classification": None,
                    },
                }
            ),
            encoding="utf-8",
        )
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        (dataset_dir / "environment-registry.json").write_text(
            json.dumps(
                {
                    "version": "v1",
                    "environments": [
                        {
                            "id": "environment-fixture",
                            "framework": "pytorch",
                            "runtime_tier": "cpu_python_overlay",
                            "docker": {"image": "fixture"},
                            "preflight": {"workdir": "/tmp", "commands": ["python --version"]},
                            "source_loading_modes": ["python_overlay"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (dataset_dir / "source-registry.json").write_text(
            json.dumps(
                {
                    "version": "v1",
                    "sources": [
                        {
                            "id": "source-fixture",
                            "repo_url": "https://github.com/local/repo.git",
                            "commit": "abcdef1",
                            "local_path": "../source",
                            "submodules": {"policy": "none_required", "status": "not_initialized"},
                            "source_loading_modes": ["python_overlay"],
                            "related_tasks": ["fixture"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        dataset = {
            "dataset_id": "fixture",
            "version": "v1",
            "status": "verified",
            "registries": {
                "environments": "environment-registry.json",
                "sources": "source-registry.json",
            },
            "tasks": [
                {
                    "task_id": "fixture",
                    "task_path": str(task_dir),
                    "admission_status": "verified",
                    "admission_evidence": str(admission_dir / "evidence.json"),
                    "environment_status": "ready",
                    "source_status": "ready",
                    "replay_status": "verified",
                }
            ],
        }
        return dataset, dataset_dir

    def _manifest(self) -> dict[str, object]:
        return {
            "task_id": "fixture",
            "version": "v1",
            "environment_ref": "environment-fixture",
            "runtime_tier": "cpu_python_overlay",
            "source_ref": "source-fixture",
            "admission": {
                "status": "verified",
                "evidence": "admission/evidence.json",
                "verified_at": "2026-06-04T00:00:00Z",
            },
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
                "fail_to_pass": ["test_fail"],
                "pass_to_pass": ["test_pass"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {
                "difficulty": "easy",
                "curation_status": "verified",
                "deterministic": True,
                "admission_status": "verified",
            },
        }


if __name__ == "__main__":
    unittest.main()
