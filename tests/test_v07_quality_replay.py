from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json
from scripts.run_v07_quality_replay import (
    _index_payload,
    validate_quality_replay_index,
)


ROOT = Path(__file__).resolve().parents[1]


class V07QualityReplayTests(unittest.TestCase):
    def test_incomplete_replay_index_is_rejected_by_final_gate(self) -> None:
        payload = _index_payload(
            dataset_path=ROOT / "datasets/pytorch_v0.7/dataset.json",
            environment_registry_path=ROOT / "environments/registry.json",
            source_registry_path=ROOT / "sources/registry.json",
            created_at="1970-01-01T00:00:00Z",
            expected_task_count=50,
            records={},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            errors = validate_quality_replay_index(
                ROOT,
                path,
                release_manifest_path=ROOT / "factory/v0.7/p9/release_manifest.json",
            )
        self.assertEqual(len(errors), 1)
        self.assertIn("final counters", errors[0])

    def test_index_payload_has_content_addressed_frozen_inputs(self) -> None:
        payload = _index_payload(
            dataset_path=ROOT / "datasets/pytorch_v0.7/dataset.json",
            environment_registry_path=ROOT / "environments/registry.json",
            source_registry_path=ROOT / "sources/registry.json",
            created_at="1970-01-01T00:00:00Z",
            expected_task_count=50,
            records={},
        )
        self.assertEqual(payload["task_count"], 50)
        self.assertEqual(payload["completed_count"], 0)
        self.assertEqual(payload["verified_count"], 0)
        for field in (
            "dataset_hash",
            "environment_registry_hash",
            "source_registry_hash",
            "content_hash",
        ):
            self.assertRegex(str(payload[field]), r"^sha256:[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
