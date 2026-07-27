from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from op_bench.factory.contracts import FactoryArtifactReference
from op_bench.factory.release import (
    DatasetReleaseInput,
    VerifiedReleaseEntry,
)
from op_bench.runtime.canonical import canonical_json
from scripts.compose_dataset_release import main as compose_main


ROOT = Path(__file__).resolve().parents[1]
TASKS = {
    "base": "149693_lazylinear_init",
    "precision": "140557_layer_norm_decomp_precision",
    "restored": "129154_exp_decomp_numerics",
    "boundary": "117065_index_copy_zero_dim",
}


def file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(
    artifact_type: str,
    artifact_id: str,
    path: str,
    repo_root: Path,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash=file_hash(repo_root / path),
        relative_path=path,
    )


class DatasetReleaseCliTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        (root / "tasks" / "pytorch").mkdir(parents=True)
        for directory in TASKS.values():
            shutil.copytree(
                ROOT / "tasks" / "pytorch" / directory,
                root / "tasks" / "pytorch" / directory,
            )
        (root / "environments").mkdir()
        (root / "sources").mkdir()
        shutil.copy2(
            ROOT / "environments" / "registry.json",
            root / "environments" / "registry.json",
        )
        shutil.copy2(
            ROOT / "sources" / "registry.json",
            root / "sources" / "registry.json",
        )

        datasets = root / "datasets"
        cumulative = self._write_dataset(
            datasets / "base" / "dataset.json",
            "fixture_base",
            (TASKS["base"], TASKS["precision"]),
        )
        precision = self._write_dataset(
            datasets / "precision" / "dataset.json",
            "fixture_precision",
            (TASKS["precision"],),
        )
        boundary = self._write_dataset(
            datasets / "boundary" / "dataset.json",
            "fixture_boundary",
            (TASKS["boundary"],),
        )
        freeze_path = (
            root
            / "factory"
            / "boundary_freeze"
            / "freeze_manifest.json"
        )
        freeze_path.parent.mkdir(parents=True)
        freeze_path.write_text(
            canonical_json(
                {
                    "fixture": "boundary-freeze",
                    "dataset_hash": file_hash(boundary),
                }
            ),
            encoding="utf-8",
        )

        inputs = (
            DatasetReleaseInput(
                role="cumulative_base",
                dataset=artifact(
                    "dataset_manifest",
                    "dataset:fixture_base",
                    "datasets/base/dataset.json",
                    root,
                ),
                provenance=None,
            ),
            DatasetReleaseInput(
                role="precision_base",
                dataset=artifact(
                    "dataset_manifest",
                    "dataset:fixture_precision",
                    "datasets/precision/dataset.json",
                    root,
                ),
                provenance=None,
            ),
            DatasetReleaseInput(
                role="boundary_freeze",
                dataset=artifact(
                    "dataset_manifest",
                    "dataset:fixture_boundary",
                    "datasets/boundary/dataset.json",
                    root,
                ),
                provenance=artifact(
                    "dataset_freeze",
                    "freeze:v1:" + "f" * 64,
                    "factory/boundary_freeze/freeze_manifest.json",
                    root,
                ),
            ),
        )
        entries = (
            self._entry(
                root,
                TASKS["base"],
                origin="inherited_cumulative",
                slices=("cumulative",),
                failure_contract="unclassified",
            ),
            self._entry(
                root,
                TASKS["precision"],
                origin="inherited_precision",
                slices=("cumulative", "precision"),
                failure_contract="unclassified",
            ),
            self._entry(
                root,
                TASKS["restored"],
                origin="restored_precision",
                slices=("cumulative", "precision"),
                failure_contract="unclassified",
            ),
            self._entry(
                root,
                TASKS["boundary"],
                origin="factory_boundary",
                slices=("cumulative", "boundary"),
                failure_contract="exception",
            ),
        )
        request = {
            "schema_version": "v1",
            "release_version": "v0.7-fixture",
            "created_at": "2026-07-27T05:30:00Z",
            "inputs": [item.to_dict() for item in inputs],
            "registries": {
                "environments": artifact(
                    "environment_registry",
                    "registry:environments:v1",
                    "environments/registry.json",
                    root,
                ).to_dict(),
                "sources": artifact(
                    "source_registry",
                    "registry:sources:v1",
                    "sources/registry.json",
                    root,
                ).to_dict(),
            },
            "entries": [item.to_dict() for item in entries],
            "dataset_ids": {
                "cumulative": "fixture_v0.7",
                "boundary": "fixture_v0.7_boundary",
                "precision": "fixture_v0.7_precision",
            },
            "output_paths": {
                "release_manifest": "factory/release_manifest.json",
                "cumulative": {
                    "dataset": "release/cumulative/dataset.json",
                    "summary": "release/cumulative/summary.json",
                },
                "boundary": {
                    "dataset": "release/boundary/dataset.json",
                    "summary": "release/boundary/summary.json",
                },
                "precision": {
                    "dataset": "release/precision/dataset.json",
                    "summary": "release/precision/summary.json",
                },
            },
        }
        request_path = root / "release_request.json"
        request_path.write_text(
            canonical_json(request) + "\n",
            encoding="utf-8",
        )
        return request_path

    def _write_dataset(
        self,
        path: Path,
        dataset_id: str,
        directories: tuple[str, ...],
    ) -> Path:
        path.parent.mkdir(parents=True)
        tasks = []
        for directory in directories:
            task = json.loads(
                (
                    path.parents[2]
                    / "tasks"
                    / "pytorch"
                    / directory
                    / "task.json"
                ).read_text(encoding="utf-8")
            )
            tasks.append(
                {
                    "task_id": task["task_id"],
                    "task_path": (
                        f"../../tasks/pytorch/{directory}"
                    ),
                    "admission_status": "verified",
                    "environment_status": "ready",
                    "source_status": "ready",
                    "replay_status": "verified",
                    "runtime_tier": task["runtime_tier"],
                    "admission_evidence": (
                        "../../tasks/pytorch/"
                        f"{directory}/admission/evidence.json"
                    ),
                }
            )
        payload = {
            "dataset_id": dataset_id,
            "version": "fixture",
            "status": "verified",
            "registries": {
                "environments": "../../environments/registry.json",
                "sources": "../../sources/registry.json",
            },
            "tasks": tasks,
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def _entry(
        self,
        root: Path,
        directory: str,
        *,
        origin: str,
        slices: tuple[str, ...],
        failure_contract: str,
    ) -> VerifiedReleaseEntry:
        task_path = root / "tasks" / "pytorch" / directory / "task.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        evidence_path = (
            root
            / "tasks"
            / "pytorch"
            / directory
            / "admission"
            / "evidence.json"
        )
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        relative_task_dir = f"tasks/pytorch/{directory}"
        return VerifiedReleaseEntry(
            task=artifact(
                "task_bundle",
                f"task:{task['task_id']}",
                f"{relative_task_dir}/task.json",
                root,
            ),
            admission_evidence=artifact(
                "admission_evidence",
                evidence["evidence_id"],
                f"{relative_task_dir}/admission/evidence.json",
                root,
            ),
            task_id=task["task_id"],
            task_path=relative_task_dir,
            admission_evidence_path=(
                f"{relative_task_dir}/admission/evidence.json"
            ),
            runtime_tier=task["runtime_tier"],
            problem_dimension=task["operator"].get(
                "problem_dimension",
                "unclassified",
            ),
            problem_subclass=task["operator"].get(
                "problem_subclass",
                "unclassified",
            ),
            failure_contract=failure_contract,
            origin=origin,
            slices=slices,
            admission_state="verified",
        )

    def test_cli_composes_and_verifies_existing_release(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request = self._fixture(root)

            first = compose_main(
                ["--input", str(request), "--repo-root", str(root)]
            )
            second = compose_main(
                [
                    "--input",
                    str(request),
                    "--repo-root",
                    str(root),
                    "--verify-existing",
                ]
            )

            self.assertEqual(first, 0)
            self.assertEqual(second, 0)
            manifest = json.loads(
                (root / "factory" / "release_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {
                    item["role"]: len(item["task_ids"])
                    for item in manifest["outputs"]
                },
                {"cumulative": 4, "boundary": 1, "precision": 2},
            )
            boundary_summary = json.loads(
                (root / "release" / "boundary" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                boundary_summary["failure_contract"],
                {"exception": 1},
            )

    def test_cli_rejects_reference_hash_drift_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request_path = self._fixture(root)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["inputs"][0]["dataset"]["content_hash"] = (
                "sha256:" + "0" * 64
            )
            request_path.write_text(
                canonical_json(request) + "\n",
                encoding="utf-8",
            )

            result = compose_main(
                [
                    "--input",
                    str(request_path),
                    "--repo-root",
                    str(root),
                ]
            )

            self.assertEqual(result, 2)
            self.assertFalse(
                (root / "factory" / "release_manifest.json").exists()
            )
            self.assertFalse((root / "release").exists())

    def test_cli_rejects_noncanonical_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request_path = self._fixture(root)
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request_path.write_text(
                json.dumps(request, indent=2) + "\n",
                encoding="utf-8",
            )

            result = compose_main(
                [
                    "--input",
                    str(request_path),
                    "--repo-root",
                    str(root),
                ]
            )

            self.assertEqual(result, 2)
            self.assertFalse(
                (root / "factory" / "release_manifest.json").exists()
            )

    def test_cli_rejects_symlink_request(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            request_path = self._fixture(root)
            linked = root / "request-link.json"
            linked.symlink_to(request_path)

            result = compose_main(
                [
                    "--input",
                    str(linked),
                    "--repo-root",
                    str(root),
                ]
            )

            self.assertEqual(result, 2)
            self.assertFalse(
                (root / "factory" / "release_manifest.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
