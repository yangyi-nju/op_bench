from __future__ import annotations

import copy
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.matched_runtime.contracts import CompatibilityEvidence
from op_bench.matched_runtime.promotion import (
    promote_matched_runtime_task,
    validate_matched_runtime_promotion,
)
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest
from scripts.validate_task import validate_manifest
from scripts.promote_matched_runtime_task import main as promote_main
from tests.test_matched_runtime_contracts import compatible_evidence


BASE_COMMIT = "1" * 40
MERGE_COMMIT = "2" * 40


def task_manifest(root: Path) -> TaskManifest:
    task_dir = root / "task"
    artifacts = task_dir / "artifacts"
    compatibility_dir = task_dir / "compatibility"
    admission_dir = task_dir / "admission"
    artifacts.mkdir(parents=True)
    compatibility_dir.mkdir()
    admission_dir.mkdir()
    (artifacts / "gold.patch").write_text(
        "diff --git a/torch/_refs/__init__.py b/torch/_refs/__init__.py\n",
        encoding="utf-8",
    )
    (artifacts / "hidden.patch").write_text("", encoding="utf-8")
    manifest = {
        "task_id": "pytorch__fixture",
        "version": "v1",
        "environment_ref": "pytorch-matched-fixture",
        "runtime_tier": "cuda_python_overlay",
        "source_ref": "pytorch-source-fixture",
        "admission": {"status": "deprecated"},
        "patch_scope": {
            "allowed_paths": ["torch/_refs/__init__.py"],
            "mode": "enforced",
        },
        "source": {
            "pr_url": "https://github.com/pytorch/pytorch/pull/1",
            "issue_url": "https://github.com/pytorch/pytorch/issues/1",
            "issue_number": 1,
            "repo": "pytorch/pytorch",
            "repo_url": "https://github.com/pytorch/pytorch.git",
            "pr_number": 1,
            "base_commit": BASE_COMMIT,
            "merge_commit": MERGE_COMMIT,
            "checkout_mode": "git",
            "snapshot_path": "snapshot/source",
        },
        "statement": {
            "title": "fixture",
            "body": "fixture body",
            "labels": [],
        },
        "operator": {
            "framework": "pytorch",
            "component": "torch._refs",
            "operator_name": "torch._refs.exponential",
            "problem_type": "numerical-stability",
            "problem_dimension": "precision",
            "problem_subclass": "P4",
            "tags": [],
        },
        "environment": {
            "source_loading": {
                "mode": "python_overlay",
                "installed_package": "torch",
                "overlay_paths": ["torch/_refs/__init__.py"],
                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                "sync_before_tests": True,
            }
        },
        "agent_visible": {
            "repo_setup_commands": [],
            "known_constraints": [],
            "allowed_test_commands": ["{python} test/test_decomp.py {test}"],
        },
        "evaluation": {
            "fail_to_pass": ["DecompTestsCUDA.test_target_cuda"],
            "pass_to_pass": ["DecompTestsCUDA.test_control_cuda"],
            "test_command": "{python} test/test_decomp.py {test}",
            "timeout_sec": 30,
        },
        "artifacts": {
            "gold_patch": "artifacts/gold.patch",
            "hidden_test_patch": "artifacts/hidden.patch",
        },
        "metadata": {
            "difficulty": "easy",
            "curation_status": "draft",
            "deterministic": True,
            "admission_status": "deprecated",
            "source_loading_verified": False,
            "notes": "Deprecated because the historical wheel was incompatible.",
        },
        "compatibility": {
            "target_module": "torch/_refs/__init__.py",
            "target_import": "torch._refs",
            "selector_module": "test/test_decomp.py",
            "minimal_operation": "import torch; torch.ones(1)",
            "evidence": "compatibility/evidence.json",
        },
    }
    (task_dir / "task.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return TaskManifest.load(task_dir / "task.json")


def compatibility(task: TaskManifest) -> CompatibilityEvidence:
    selected = compatible_evidence()
    selected = replace(
        selected,
        task_id=task.task_id,
        source=replace(
            selected.source,
            source_id=task.source_ref,
            commit=task.base_commit,
            target_module_path="torch/_refs/__init__.py",
            runtime_path_suffix="torch/_refs/__init__.py",
        ),
        runtime=replace(
            selected.runtime,
            environment_id=task.environment_ref,
            target_module_path_suffix="torch/_refs/__init__.py",
        ),
    )
    path = task.task_dir / "compatibility/evidence.json"
    path.write_text(canonical_json(selected.to_dict()), encoding="utf-8")
    return selected


def admission(task: TaskManifest) -> dict[str, object]:
    return {
        "schema_version": "v1",
        "evidence_id": "admission-fixture",
        "task_id": task.task_id,
        "task_manifest_hash": replay_spec_hash(task),
        "task_manifest_hash_kind": REPLAY_SPEC_HASH_KIND,
        "created_at": "2026-07-26T12:10:00Z",
        "source": {
            "id": task.source_ref,
            "base_commit": task.base_commit,
        },
        "environment": {
            "id": task.environment_ref,
            "runtime_tier": task.runtime_tier,
        },
        "baseline": {
            "task_id": task.task_id,
            "mode": "baseline",
            "status": "baseline_reproduced",
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 1,
        },
        "gold": {
            "task_id": task.task_id,
            "mode": "gold",
            "status": "resolved",
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 1,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 1,
        },
        "admission": {
            "decision": "verified",
            "verified": True,
            "failure_classification": None,
        },
    }


def write_admission(task: TaskManifest, value: dict[str, object]) -> Path:
    path = task.task_dir / "admission/evidence.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


class MatchedRuntimePromotionTests(unittest.TestCase):
    def test_valid_evidence_promotes_only_curation_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            selected = compatibility(task)
            admission_path = write_admission(task, admission(task))
            before = copy.deepcopy(task.data)

            promoted = promote_matched_runtime_task(
                task.task_json_path,
                task.task_dir / "compatibility/evidence.json",
                admission_path,
                "2026-07-26T12:20:00Z",
            )

            self.assertEqual(promoted["admission"]["status"], "verified")
            self.assertEqual(
                promoted["admission"]["compatibility_evidence"],
                "compatibility/evidence.json",
            )
            self.assertEqual(promoted["metadata"]["curation_status"], "verified")
            self.assertEqual(promoted["metadata"]["admission_status"], "verified")
            self.assertTrue(promoted["metadata"]["source_loading_verified"])
            for field in (
                "task_id",
                "source",
                "operator",
                "evaluation",
                "artifacts",
                "patch_scope",
                "compatibility",
            ):
                self.assertEqual(promoted[field], before[field])
            self.assertIn(selected.content_hash, promoted["metadata"]["notes"])
            self.assertEqual(validate_manifest(promoted), [])

    def test_draft_promotion_uses_verified_not_restored_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            manifest = copy.deepcopy(task.data)
            manifest["admission"] = {"status": "draft"}
            manifest["metadata"]["admission_status"] = "draft"
            manifest["metadata"]["notes"] = "New v0.7 boundary task."
            task.task_json_path.write_text(
                json.dumps(manifest, indent=2) + "\n",
                encoding="utf-8",
            )
            task = TaskManifest.load(task.task_json_path)
            selected = compatibility(task)
            admission_path = write_admission(task, admission(task))

            promoted = promote_matched_runtime_task(
                task.task_json_path,
                task.task_dir / "compatibility/evidence.json",
                admission_path,
                "2026-07-27T12:20:00Z",
            )

            self.assertTrue(
                promoted["metadata"]["notes"].startswith(
                    "Verified by v0.7 matched-runtime compatibility"
                )
            )
            self.assertNotIn("Restored by", promoted["metadata"]["notes"])
            self.assertIn(selected.content_hash, promoted["metadata"]["notes"])

    def test_identity_and_status_mismatches_are_rejected(self) -> None:
        cases = {
            "task_id mismatch": lambda task, comp, adm: (
                replace(comp, task_id="pytorch__other"),
                adm,
            ),
            "source commit mismatch": lambda task, comp, adm: (
                replace(
                    comp,
                    source=replace(comp.source, commit="f" * 40),
                ),
                adm,
            ),
            "environment mismatch": lambda task, comp, adm: (
                replace(
                    comp,
                    runtime=replace(
                        comp.runtime,
                        environment_id="pytorch-other",
                    ),
                ),
                adm,
            ),
            "compatibility status": lambda task, comp, adm: (
                self._incompatible(comp),
                adm,
            ),
            "admission decision": lambda task, comp, adm: (
                comp,
                {
                    **adm,
                    "admission": {
                        "decision": "gold_failed",
                        "verified": False,
                        "failure_classification": "fail_to_pass_failed",
                    },
                },
            ),
            "replay hash mismatch": lambda task, comp, adm: (
                comp,
                {**adm, "task_manifest_hash": "sha256:" + "f" * 64},
            ),
            "test execution": lambda task, comp, adm: (
                comp,
                {
                    **adm,
                    "baseline": {
                        **adm["baseline"],
                        "fail_to_pass_total": 0,
                    },
                },
            ),
        }
        for message, mutate in cases.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                task = task_manifest(Path(tmp))
                comp = compatibility(task)
                adm = admission(task)
                comp, adm = mutate(task, comp, adm)

                with self.assertRaisesRegex(ContractError, message):
                    validate_matched_runtime_promotion(task, comp, adm)

    def test_failed_promotion_leaves_task_bytes_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            compatibility(task)
            failed = admission(task)
            failed["task_manifest_hash"] = "sha256:" + "f" * 64
            admission_path = write_admission(task, failed)
            before = task.task_json_path.read_bytes()

            with self.assertRaisesRegex(ContractError, "replay hash mismatch"):
                promote_matched_runtime_task(
                    task.task_json_path,
                    task.task_dir / "compatibility/evidence.json",
                    admission_path,
                    "2026-07-26T12:20:00Z",
                )

            self.assertEqual(task.task_json_path.read_bytes(), before)

    def test_promotion_cli_emits_verified_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            compatibility(task)
            admission_path = write_admission(task, admission(task))
            stdout = StringIO()

            with redirect_stdout(stdout):
                exit_code = promote_main(
                    [
                        "--task",
                        str(task.task_json_path),
                        "--compatibility-evidence",
                        str(task.task_dir / "compatibility/evidence.json"),
                        "--admission-evidence",
                        str(admission_path),
                        "--verified-at",
                        "2026-07-26T12:20:00Z",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "verified")

    def test_replay_hash_tracks_compatibility_configuration_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            compatibility(task)
            first = replay_spec_hash(task)

            evidence_path = task.task_dir / "compatibility/evidence.json"
            payload = json.loads(evidence_path.read_text(encoding="utf-8"))
            payload["content_hash"] = "sha256:" + "f" * 64
            evidence_path.write_text(json.dumps(payload), encoding="utf-8")
            second = replay_spec_hash(task)

            self.assertNotEqual(first, second)

            manifest = json.loads(task.task_json_path.read_text(encoding="utf-8"))
            manifest["compatibility"]["minimal_operation"] += "; torch.ones(2)"
            task.task_json_path.write_text(json.dumps(manifest), encoding="utf-8")
            third = replay_spec_hash(TaskManifest.load(task.task_json_path))

            self.assertNotEqual(second, third)

    def _incompatible(
        self,
        evidence: CompatibilityEvidence,
    ) -> CompatibilityEvidence:
        checks = list(evidence.checks)
        checks[3] = replace(
            checks[3],
            exit_code=1,
            status="failed",
            summary="target import failed",
        )
        from op_bench.matched_runtime.contracts import CompatibilityFailure

        return replace(
            evidence,
            status="incompatible",
            checks=tuple(checks),
            failure=CompatibilityFailure(
                code="target_import_failed",
                check="target_import",
                summary="target import failed",
            ),
        )


if __name__ == "__main__":
    unittest.main()
