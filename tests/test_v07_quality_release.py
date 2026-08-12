from __future__ import annotations

import json
from pathlib import Path
import unittest

from op_bench.factory.artifacts import load_regular_file_bytes
from scripts.build_v07_quality_release import build_release_outputs


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "factory/v0.7/p9/release_manifest.json"


class V07QualityReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.outputs = build_release_outputs(
            root=ROOT,
            historical_index_path=ROOT / "factory/v0.7/p7/historical_readmission.json",
            accepted_index_path=ROOT / "factory/v0.7/p8/accepted_tasks.json",
            admission_results_path=ROOT / "factory/v0.7/p8/admission_results.json",
            request_path=ROOT / "factory/v0.7/p9/release_request.json",
            created_at=cls.manifest["created_at"],
        )

    def test_final_release_rebuilds_every_artifact_byte_for_byte(self) -> None:
        self.assertEqual(len(self.outputs), 11)
        for relative, expected in self.outputs.items():
            self.assertEqual(load_regular_file_bytes(ROOT / relative), expected, relative)

    def test_final_composition_and_derived_slices_are_exact(self) -> None:
        self.assertEqual(
            self.manifest["composition"],
            {
                "new_or_replacement": 36,
                "retained_historical": 14,
                "total": 50,
            },
        )
        self.assertEqual(
            {
                role: value["task_count"]
                for role, value in self.manifest["datasets"].items()
            },
            {"boundary": 31, "cumulative": 50, "device": 15, "precision": 5},
        )

    def test_pre_quality_dataset_is_preserved_as_immutable_history(self) -> None:
        archived = ROOT / "archives/v0.7-pre-quality/datasets/pytorch_v0.7/dataset.json"
        payload = json.loads(archived.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["tasks"]), 25)
        for relative in (
            "factory/v0.7/p4/release_request.json",
            "factory/v0.7/p4/release_manifest.json",
            "factory/v0.7/p4/validation_contract.json",
            "factory/v0.7/p4/boundary_freeze_request.json",
            "factory/v0.7/p4/boundary_freeze/freeze_manifest.json",
            "factory/v0.7/p4/boundary_freeze/dataset.json",
        ):
            self.assertEqual(
                load_regular_file_bytes(ROOT / "archives/v0.7-pre-quality" / relative),
                load_regular_file_bytes(ROOT / relative),
                relative,
            )


if __name__ == "__main__":
    unittest.main()
