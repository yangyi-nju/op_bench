from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from op_bench.factory.artifacts import load_factory_contract
from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    factory_content_hash,
)
from op_bench.patch_scope import extract_patch_paths
from op_bench.registry import EnvironmentRegistry, SourceRegistry
from op_bench.task import TaskManifest
from scripts.validate_task import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "factory" / "v0.7" / "p3"
CAPTURE_PRS = {
    143792,
    117065,
    126461,
    147352,
    118762,
    139751,
    143461,
    147433,
    127448,
    139502,
}
SELECTED_SUBCLASSES = {
    143792: "B1",
    117065: "B2",
    126461: "B2",
    147352: "B3",
    118762: "B4",
    139751: "B5",
}
SCREEN_CLI = ROOT / "scripts" / "factory_screen_candidates.py"
ENVIRONMENT_REGISTRY = ROOT / "environments" / "registry.json"
SOURCE_REGISTRY = ROOT / "sources" / "registry.json"
BOUNDARY_ENVIRONMENTS = {
    "pytorch-matched-boundary-torch2.2.0-cpu": "2.2.0+cpu",
    "pytorch-matched-boundary-torch2.3.0-cpu": "2.3.0+cpu",
    "pytorch-matched-boundary-torch2.5.1-cu124": "2.5.1+cu124",
}
BOUNDARY_SOURCES = {
    "pytorch-38b3375-boundary-overlay": (
        "38b3375a81dc3c21d7c0420773187c7c2c3d5835",
        "overlay",
        "torch/_decomp/decompositions.py",
        "pytorch__143792__addmv_empty_matrix",
    ),
    "pytorch-b9293e7-boundary-overlay": (
        "b9293e74a2b476534fa2aee5a0708e3fca255d8e",
        "overlay",
        "torch/_decomp/decompositions.py",
        "pytorch__117065__index_copy_zero_dim",
    ),
    "pytorch-15ca562-boundary-overlay": (
        "15ca562f863ffe69d76c0ccaf448b27d18ceb2e8",
        "overlay",
        "torch/_inductor/lowering.py",
        "pytorch__126461__cummin_rank_zero",
    ),
    "pytorch-6ccbff1-boundary-kernel-full": (
        "6ccbff1450bb3936636377d3910906f5666ddcfa",
        "kernel_full",
        "aten/src/ATen/native/Resize.h",
        "pytorch__147352__storage_offset_overflow",
    ),
    "pytorch-e426924-boundary-overlay": (
        "e426924c19500011aa419accc251a923fe857b94",
        "overlay",
        "torch/_decomp/decompositions.py",
        "pytorch__118762__weight_norm_default_dim",
    ),
    "pytorch-cc98a1b-boundary-overlay": (
        "cc98a1b59909e6f3ce929380e02f0345c862e31d",
        "overlay",
        "torch/_inductor/codegen/triton.py",
        "pytorch__139751__triton_ygrid_mask",
    ),
}
PYTHON_BOUNDARY_TASKS = {
    "143792_addmv_empty_matrix": (
        "pytorch__143792__addmv_empty_matrix",
        "B1",
        "torch/_decomp/decompositions.py",
    ),
    "117065_index_copy_zero_dim": (
        "pytorch__117065__index_copy_zero_dim",
        "B2",
        "torch/_decomp/decompositions.py",
    ),
    "126461_cummin_rank_zero": (
        "pytorch__126461__cummin_rank_zero",
        "B2",
        "torch/_inductor/lowering.py",
    ),
    "118762_weight_norm_default_dim": (
        "pytorch__118762__weight_norm_default_dim",
        "B4",
        "torch/_decomp/decompositions.py",
    ),
}


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


class BoundaryArtifactTests(unittest.TestCase):
    def test_real_screening_funnel(self) -> None:
        captures = json.loads(
            (P3 / "captures.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {item["pr_number"] for item in captures},
            CAPTURE_PRS,
        )

        screening = P3 / "screening"
        index = json.loads(
            (screening / "screening_index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            index["counts"],
            {"accepted": 6, "deferred": 2, "rejected": 2},
        )
        self.assertEqual(
            index["content_hash"],
            factory_content_hash(index),
        )

        candidates: dict[int, CandidateRecord] = {}
        decisions: dict[int, DecisionRecord] = {}
        for entry in index["decisions"]:
            candidate = load_factory_contract(
                screening / entry["candidate"]["relative_path"]
            )
            decision = load_factory_contract(
                screening / entry["decision"]["relative_path"]
            )
            self.assertIsInstance(candidate, CandidateRecord)
            self.assertIsInstance(decision, DecisionRecord)
            assert isinstance(candidate, CandidateRecord)
            assert isinstance(decision, DecisionRecord)
            self.assertEqual(
                candidate.content_hash,
                entry["candidate"]["content_hash"],
            )
            self.assertEqual(
                decision.content_hash,
                entry["decision"]["content_hash"],
            )
            self.assertEqual(candidate.pr_number, entry["pr_number"])
            candidates[candidate.pr_number] = candidate
            decisions[candidate.pr_number] = decision

        self.assertEqual(set(candidates), CAPTURE_PRS)
        self.assertEqual(
            {
                pr_number: candidates[pr_number].proposed_subclass
                for pr_number in SELECTED_SUBCLASSES
            },
            SELECTED_SUBCLASSES,
        )
        self.assertEqual(
            tuple(
                finding.code
                for finding in decisions[143461].findings
                if finding.severity == "reject"
            ),
            ("taxonomy.not_boundary",),
        )

        human = load_factory_contract(
            P3 / "human_decisions" / "pr-147433.json"
        )
        self.assertIsInstance(human, DecisionRecord)
        assert isinstance(human, DecisionRecord)
        automated = decisions[147433]
        self.assertEqual(automated.disposition, "deferred")
        self.assertEqual(human.decision_source, "human_review")
        self.assertEqual(human.disposition, "rejected")
        self.assertEqual(
            tuple(finding.code for finding in human.findings),
            ("review.upstream_reverted",),
        )
        self.assertIsNotNone(human.prior_decision)
        assert human.prior_decision is not None
        self.assertEqual(
            human.prior_decision.artifact_id,
            automated.decision_id,
        )
        self.assertEqual(
            human.prior_decision.content_hash,
            automated.content_hash,
        )

        review = json.loads(
            (P3 / "reviews" / "pr-147433.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review["pr_number"], 147433)
        self.assertEqual(
            review["landed_commit"],
            "1d7397a2d04a4d636559f41511a20f7dadbe5777",
        )
        self.assertEqual(
            review["revert_commit"],
            "841451af9f8c4a49d720a0132532c5468963f643",
        )
        self.assertEqual(review["reason_code"], "review.upstream_reverted")
        self.assertEqual(review["content_hash"], factory_content_hash(review))

    def test_real_screening_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            regenerated = Path(raw) / "screening"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCREEN_CLI),
                    "--input",
                    str(P3 / "captures.json"),
                    "--output-dir",
                    str(regenerated),
                    "--created-at",
                    "2026-07-27T03:00:00Z",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                tree_hashes(regenerated),
                tree_hashes(P3 / "screening"),
            )

    def test_runtime_and_source_registry(self) -> None:
        environment_raw = json.loads(
            ENVIRONMENT_REGISTRY.read_text(encoding="utf-8")
        )
        environment_entries = {
            entry["id"]: entry
            for entry in environment_raw["environments"]
        }
        environment_registry = EnvironmentRegistry.load(
            ENVIRONMENT_REGISTRY
        )
        for environment_id, torch_version in BOUNDARY_ENVIRONMENTS.items():
            with self.subTest(environment=environment_id):
                entry = environment_entries[environment_id]
                asset = environment_registry.get(environment_id)
                self.assertRegex(
                    entry["docker"]["digest"],
                    r"^sha256:[0-9a-f]{64}$",
                )
                self.assertEqual(
                    entry["docker"]["digest_kind"],
                    "local_image_id",
                )
                self.assertEqual(
                    entry["runtime_artifact"]["torch_version"],
                    torch_version,
                )
                self.assertEqual(
                    entry["runtime_artifact"]["artifact_kind"],
                    "official_wheel",
                )
                self.assertEqual(
                    entry["runtime_artifact"]["artifact_digest_kind"],
                    "wheel_sha256",
                )
                self.assertRegex(
                    entry["runtime_artifact"]["artifact_digest"],
                    r"^sha256:[0-9a-f]{64}$",
                )
                dockerfile = asset.dockerfile_path
                self.assertIsNotNone(dockerfile)
                assert dockerfile is not None
                contents = dockerfile.read_text(encoding="utf-8")
                self.assertIn(
                    entry["runtime_artifact"]["artifact_digest"][
                        len("sha256:"):
                    ],
                    contents,
                )

        source_raw = json.loads(
            SOURCE_REGISTRY.read_text(encoding="utf-8")
        )
        source_entries = {
            entry["id"]: entry for entry in source_raw["sources"]
        }
        source_registry = SourceRegistry.load(SOURCE_REGISTRY)
        for source_id, expected in BOUNDARY_SOURCES.items():
            commit, mode, target, task_id = expected
            with self.subTest(source=source_id):
                entry = source_entries[source_id]
                asset = source_registry.get(source_id)
                self.assertEqual(asset.commit, commit)
                self.assertEqual(entry["snapshot_mode"], mode)
                self.assertEqual(entry["related_tasks"], [task_id])
                self.assertIsNotNone(asset.local_path)
                assert asset.local_path is not None
                self.assertTrue((asset.local_path / target).is_file())
                self.assertRegex(
                    entry["snapshot_git_commit"],
                    r"^[0-9a-f]{40}$",
                )
                self.assertGreater(entry["tracked_file_count"], 0)
                snapshot_head = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=asset.local_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                snapshot_parent = subprocess.run(
                    ["git", "rev-parse", "HEAD^"],
                    cwd=asset.local_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
                tracked_count = subprocess.run(
                    ["git", "ls-files"],
                    cwd=asset.local_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.count("\n")
                dirty = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=asset.local_path,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout
                self.assertEqual(snapshot_head, entry["snapshot_git_commit"])
                self.assertEqual(snapshot_parent, commit)
                self.assertEqual(tracked_count, entry["tracked_file_count"])
                self.assertEqual(dirty, "")
                if mode == "kernel_full":
                    self.assertEqual(asset.submodule_status, "initialized")

    def test_python_task_bundles(self) -> None:
        for directory, expected in PYTHON_BOUNDARY_TASKS.items():
            task_id, subclass, allowed_source = expected
            with self.subTest(task=task_id):
                task_dir = ROOT / "tasks" / "pytorch" / directory
                task = TaskManifest.load(task_dir / "task.json")
                self.assertEqual(validate_manifest(task.data), [])
                self.assertEqual(task.task_id, task_id)
                self.assertEqual(task.admission_status, "draft")
                self.assertEqual(
                    task.data["operator"]["problem_dimension"],
                    "boundary",
                )
                self.assertEqual(
                    task.data["operator"]["problem_subclass"],
                    subclass,
                )
                self.assertEqual(task.patch_scope_paths, [allowed_source])
                self.assertEqual(
                    extract_patch_paths(
                        task.gold_patch_path.read_text(encoding="utf-8")
                    ),
                    [allowed_source],
                )
                hidden_paths = extract_patch_paths(
                    task.hidden_test_patch_path.read_text(encoding="utf-8")
                )
                self.assertEqual(len(hidden_paths), 1)
                self.assertRegex(
                    hidden_paths[0],
                    r"^test/op_bench_boundary/test_[a-z0-9_]+\.py$",
                )
                fail_to_pass = task.fail_to_pass_tests
                pass_to_pass = task.pass_to_pass_tests
                self.assertTrue(fail_to_pass)
                self.assertTrue(pass_to_pass)
                self.assertTrue(
                    set(fail_to_pass).isdisjoint(pass_to_pass)
                )

    def test_b3_storage_overflow_bundle(self) -> None:
        task_dir = (
            ROOT
            / "tasks"
            / "pytorch"
            / "147352_storage_offset_overflow"
        )
        task = TaskManifest.load(task_dir / "task.json")
        self.assertEqual(validate_manifest(task.data), [])
        self.assertEqual(task.admission_status, "draft")
        self.assertEqual(
            task.data["operator"]["problem_dimension"],
            "boundary",
        )
        self.assertEqual(
            task.data["operator"]["problem_subclass"],
            "B3",
        )
        self.assertEqual(
            task.data["operator"]["problem_type"],
            "checked-overflow",
        )
        self.assertEqual(
            task.patch_scope_paths,
            ["aten/src/ATen/native/Resize.h"],
        )
        self.assertEqual(
            extract_patch_paths(
                task.gold_patch_path.read_text(encoding="utf-8")
            ),
            ["aten/src/ATen/native/Resize.h"],
        )
        self.assertEqual(
            task.source_ref,
            "pytorch-6ccbff1-boundary-kernel-full",
        )
        source_loading = task.data["environment"]["source_loading"]
        self.assertEqual(source_loading["mode"], "inplace_build")
        build_environment = source_loading["build_environment"]
        self.assertEqual(build_environment["BUILD_TEST"], "0")
        self.assertEqual(build_environment["USE_CUDA"], "0")
        self.assertEqual(build_environment["MAX_JOBS"], "8")
        hidden = task.hidden_test_patch_path.read_text(encoding="utf-8")
        self.assertIn("torch.empty((2, 3)", hidden)
        self.assertNotRegex(
            hidden,
            r"torch\.(?:empty|randn|zeros|ones)\([^\n]*(?:17|2\*\*[0-9])",
        )


if __name__ == "__main__":
    unittest.main()
