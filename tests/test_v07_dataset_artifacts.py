from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from op_bench.curation import summarize_verified_dataset
from op_bench.factory.contracts import (
    DatasetFreezeManifest,
    FactoryAdmissionRecord,
)
from op_bench.factory.freeze import rebuild_dataset_manifest
from op_bench.factory.release import (
    DatasetReleaseManifest,
    rebuild_release_datasets,
)
from op_bench.integrity import replay_spec_hash
from op_bench.runtime.canonical import canonical_json
from op_bench.task import TaskManifest
from scripts.build_v07_dataset_requests import (
    build_boundary_freeze_request,
    build_release_request,
)
from scripts.validate_dataset import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
BOUNDARY_DIRECTORIES = (
    "117065_index_copy_zero_dim",
    "118762_weight_norm_default_dim",
    "126461_cummin_rank_zero",
    "139751_triton_ygrid_mask",
    "143792_addmv_empty_matrix",
    "147352_storage_offset_overflow",
)
BOUNDARY_IDS = frozenset(
    {
        "pytorch__117065__index_copy_zero_dim",
        "pytorch__118762__weight_norm_default_dim",
        "pytorch__126461__cummin_rank_zero",
        "pytorch__139751__triton_ygrid_mask",
        "pytorch__143792__addmv_empty_matrix",
        "pytorch__147352__storage_offset_overflow",
    }
)
RESTORED_IDS = frozenset(
    {
        "pytorch__129154__exp_decomp_numerics",
        "pytorch__144073__vector_norm_scalar_overflow",
    }
)
DATASET_PATHS = {
    "cumulative": ROOT / "archives/v0.7-pre-quality/datasets/pytorch_v0.7/dataset.json",
    "boundary": ROOT / "archives/v0.7-pre-quality/datasets/pytorch_v0.7_boundary/dataset.json",
    "precision": ROOT / "archives/v0.7-pre-quality/datasets/pytorch_v0.7_precision/dataset.json",
}


class V07DatasetRequestBuilderTests(unittest.TestCase):
    def test_release_request_resolves_dataset_relative_task_paths(self) -> None:
        request = build_release_request(ROOT)

        self.assertEqual(len(request["entries"]), 25)
        self.assertEqual(
            {
                item["task_path"]
                for item in request["entries"]
                if item["origin"] == "inherited_cumulative"
            }
            | {
                item["task_path"]
                for item in request["entries"]
                if item["origin"] == "inherited_precision"
            },
            {
                f"tasks/pytorch/{Path(item['task_path']).name}"
                for item in json.loads(
                    (
                        ROOT / "datasets/pytorch_v0.5/dataset.json"
                    ).read_text(encoding="utf-8")
                )["tasks"]
            },
        )


def load_canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    encoded = canonical_json(value).encode("utf-8")
    if raw not in (encoded, encoded + b"\n"):
        raise AssertionError(f"{path} is not canonical JSON")
    return value


def task_ids(dataset: dict[str, object]) -> frozenset[str]:
    return frozenset(
        str(item["task_id"])
        for item in dataset["tasks"]
        if isinstance(item, dict)
    )


class V07DatasetArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freeze_request = load_canonical(
            ROOT / "factory" / "v0.7" / "p4" / "boundary_freeze_request.json"
        )
        cls.freeze = DatasetFreezeManifest.from_dict(
            load_canonical(
                ROOT
                / "factory"
                / "v0.7"
                / "p4"
                / "boundary_freeze"
                / "freeze_manifest.json"
            )
        )
        cls.freeze_dataset = load_canonical(
            ROOT
            / "factory"
            / "v0.7"
            / "p4"
            / "boundary_freeze"
            / "dataset.json"
        )
        cls.release_request = load_canonical(
            ROOT
            / "archives/v0.7-pre-quality/factory/v0.7/p4/release_request.json"
        )
        cls.release = DatasetReleaseManifest.from_dict(
            load_canonical(
                ROOT
                / "archives/v0.7-pre-quality/factory/v0.7/p4/release_manifest.json"
            )
        )
        cls.datasets = {
            role: load_canonical(path)
            for role, path in DATASET_PATHS.items()
        }

    def test_exact_25_6_8_membership_and_set_algebra(self) -> None:
        cumulative_base = task_ids(
            json.loads(
                (
                    ROOT / "datasets" / "pytorch_v0.5" / "dataset.json"
                ).read_text(encoding="utf-8")
            )
        )
        precision_base = task_ids(
            json.loads(
                (
                    ROOT
                    / "datasets"
                    / "pytorch_v0.5_precision"
                    / "dataset.json"
                ).read_text(encoding="utf-8")
            )
        )
        observed = {
            role: task_ids(dataset)
            for role, dataset in self.datasets.items()
        }

        self.assertEqual(len(cumulative_base), 17)
        self.assertEqual(len(precision_base), 6)
        self.assertEqual(
            observed["cumulative"],
            cumulative_base | RESTORED_IDS | BOUNDARY_IDS,
        )
        self.assertEqual(observed["boundary"], BOUNDARY_IDS)
        self.assertEqual(
            observed["precision"],
            precision_base | RESTORED_IDS,
        )
        self.assertEqual(
            {role: len(ids) for role, ids in observed.items()},
            {"cumulative": 25, "boundary": 6, "precision": 8},
        )
        for role, dataset in self.datasets.items():
            with self.subTest(role=role):
                identifiers = [
                    item["task_id"]
                    for item in dataset["tasks"]
                    if isinstance(item, dict)
                ]
                self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_boundary_freeze_binds_six_final_factory_admissions(self) -> None:
        self.assertEqual(
            {entry.task_id for entry in self.freeze.entries},
            BOUNDARY_IDS,
        )
        by_task = {entry.task_id: entry for entry in self.freeze.entries}
        request_admissions = {
            item["admission_id"]: FactoryAdmissionRecord.from_dict(item)
            for item in self.freeze_request["admissions"]
        }

        for directory in BOUNDARY_DIRECTORIES:
            admission = FactoryAdmissionRecord.from_dict(
                load_canonical(
                    ROOT
                    / "tasks"
                    / "pytorch"
                    / directory
                    / "factory"
                    / "admission.json"
                )
            )
            entry = by_task[admission.task.artifact_id.removeprefix("task:")]
            with self.subTest(task=entry.task_id):
                self.assertEqual(admission.state, "verified")
                self.assertEqual(entry.admission.artifact_id, admission.admission_id)
                self.assertEqual(entry.admission.content_hash, admission.content_hash)
                self.assertEqual(entry.candidate, admission.candidate)
                self.assertEqual(entry.decision, admission.decision)
                self.assertEqual(entry.task, admission.task)
                self.assertEqual(
                    request_admissions[admission.admission_id],
                    admission,
                )

        self.assertEqual(
            rebuild_dataset_manifest(self.freeze),
            self.freeze_dataset,
        )

    def test_release_rebuilds_all_datasets_and_summaries(self) -> None:
        rebuilt = rebuild_release_datasets(self.release)
        self.assertEqual(rebuilt, self.datasets)
        by_task = {entry.task_id: entry for entry in self.release.entries}
        for output in self.release.outputs:
            metadata = {
                task_id: {
                    "problem_dimension": by_task[task_id].problem_dimension,
                    "problem_subclass": by_task[task_id].problem_subclass,
                    "failure_contract": by_task[task_id].failure_contract,
                }
                for task_id in output.task_ids
            }
            expected = summarize_verified_dataset(
                self.datasets[output.role],
                dataset_hash=output.generated_dataset_hash,
                task_metadata=metadata,
            )
            actual = load_canonical(
                DATASET_PATHS[output.role].with_name("summary.json")
            )
            with self.subTest(role=output.role):
                self.assertEqual(actual, expected)

    def test_historical_requests_remain_archived_after_quality_release(self) -> None:
        self.assertEqual(self.release_request["release_version"], "v0.7")
        self.assertEqual(len(self.release_request["entries"]), 25)
        self.assertEqual(
            self.release_request["output_paths"]["release_manifest"],
            "factory/v0.7/p4/release_manifest.json",
        )

    def test_restored_precision_entries_bind_current_task_and_evidence(self) -> None:
        entries = {
            entry.task_id: entry
            for entry in self.release.entries
            if entry.origin == "restored_precision"
        }
        self.assertEqual(set(entries), RESTORED_IDS)
        for task_id, entry in entries.items():
            task_path = ROOT / entry.task.relative_path
            evidence_path = ROOT / entry.admission_evidence.relative_path
            task = TaskManifest.load(task_path)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            with self.subTest(task=task_id):
                self.assertEqual(
                    entry.task.content_hash,
                    "sha256:" + hashlib.sha256(task_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    entry.admission_evidence.content_hash,
                    "sha256:"
                    + hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    evidence["task_manifest_hash"],
                    replay_spec_hash(task),
                )
                self.assertTrue(evidence["admission"]["verified"])

    def test_all_generated_datasets_are_formally_verified(self) -> None:
        datasets = {
            **self.datasets,
            "boundary_freeze": self.freeze_dataset,
        }
        paths = {
            **DATASET_PATHS,
            "boundary_freeze": (
                ROOT
                / "factory"
                / "v0.7"
                / "p4"
                / "boundary_freeze"
                / "dataset.json"
            ),
        }
        for role, dataset in datasets.items():
            with self.subTest(role=role):
                self.assertEqual(
                    validate_dataset(
                        dataset,
                        paths[role].parent,
                        require_verified=True,
                    ),
                    [],
                )


if __name__ == "__main__":
    unittest.main()
