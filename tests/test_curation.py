from __future__ import annotations

import unittest

from op_bench.curation import curate_dataset, summarize_dataset


class DatasetCurationTests(unittest.TestCase):
    def test_verified_only_slice_excludes_drafts_and_is_marked_verified(self) -> None:
        source = self._dataset()

        curated = curate_dataset(source, verified_only=True, dataset_id="pytorch_v02_verified", version="v0.2")

        self.assertEqual(curated["dataset_id"], "pytorch_v02_verified")
        self.assertEqual(curated["version"], "v0.2")
        self.assertEqual(curated["status"], "verified")
        self.assertEqual([entry["task_id"] for entry in curated["tasks"]], ["verified"])
        self.assertEqual(len(source["tasks"]), 2)

    def test_summary_groups_admission_runtime_and_environment_status(self) -> None:
        summary = summarize_dataset(self._dataset())

        self.assertEqual(summary["task_count"], 2)
        self.assertEqual(summary["admission_status"], {"draft": 1, "verified": 1})
        self.assertEqual(summary["environment_status"], {"pending": 1, "ready": 1})
        self.assertEqual(summary["runtime_tier"], {"cpu_python_overlay": 1, "unknown": 1})

    def _dataset(self) -> dict[str, object]:
        return {
            "dataset_id": "source",
            "version": "v1",
            "status": "draft",
            "registries": {"environments": "environments.json", "sources": "sources.json"},
            "tasks": [
                {
                    "task_id": "verified",
                    "task_path": "tasks/verified",
                    "admission_status": "verified",
                    "environment_status": "ready",
                    "source_status": "ready",
                    "replay_status": "verified",
                    "runtime_tier": "cpu_python_overlay",
                },
                {
                    "task_id": "draft",
                    "task_path": "tasks/draft",
                    "admission_status": "draft",
                    "environment_status": "pending",
                    "source_status": "pending",
                    "replay_status": "pending",
                },
            ],
        }


if __name__ == "__main__":
    unittest.main()
