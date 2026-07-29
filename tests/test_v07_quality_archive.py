from __future__ import annotations

from pathlib import Path
import unittest

from op_bench.factory.archive import load_pre_quality_archive


ROOT = Path(__file__).resolve().parents[1]


class V07PreQualityArchiveTests(unittest.TestCase):
    def test_archive_binds_the_closed_release(self) -> None:
        archive = load_pre_quality_archive(
            ROOT / "archives" / "v0.7-pre-quality.json"
        )

        self.assertEqual(archive.baseline_commit, "4f5addc")
        self.assertEqual(
            dict(archive.dataset_hashes),
            {
                "boundary": "sha256:810a9cc85c576f44edd2672197ab83b7dfee7f674e597c76c78050bd119d606a",
                "cumulative": "sha256:4d7bde25e747bcc041aa5105ce5ce881a3f1e9fe2a7545667cdbc2c14d85064a",
                "precision": "sha256:65818466a02e99466386cb8e038dc4da59d91dcb3bea7b83c8901d31a96aa8eb",
            },
        )
        self.assertEqual(archive.task_count, 25)
        self.assertEqual(archive.validation_attempts, 18)
        self.assertEqual(len(archive.cohort_ids), 5)


if __name__ == "__main__":
    unittest.main()
