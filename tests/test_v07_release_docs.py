from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = REPO_ROOT / "docs" / "v0.7" / "dataset_card.md"
SUMMARY_PATHS = {
    "cumulative": REPO_ROOT / "datasets" / "pytorch_v0.7" / "summary.json",
    "boundary": (
        REPO_ROOT / "datasets" / "pytorch_v0.7_boundary" / "summary.json"
    ),
    "precision": (
        REPO_ROOT / "datasets" / "pytorch_v0.7_precision" / "summary.json"
    ),
}


class V07ReleaseDocsTests(unittest.TestCase):
    def test_dataset_card_matches_frozen_release(self) -> None:
        self.assertTrue(
            DATASET_CARD.is_file(),
            f"missing v0.7 Dataset Card: {DATASET_CARD}",
        )
        card = DATASET_CARD.read_text(encoding="utf-8")
        for required in (
            "opbench-v0.7.0",
            "25 verified",
            "6 verified Boundary",
            "8 verified Precision",
            "B1",
            "B5",
            "P1",
            "P5",
            "10",
            "6 accepted",
            "2 deferred",
            "non-leaderboard",
        ):
            with self.subTest(required=required):
                self.assertIn(required, card)

        summaries = {
            role: json.loads(path.read_text(encoding="utf-8"))
            for role, path in SUMMARY_PATHS.items()
        }
        self.assertEqual(
            {role: summary["task_count"] for role, summary in summaries.items()},
            {"cumulative": 25, "boundary": 6, "precision": 8},
        )
        for role, summary in summaries.items():
            with self.subTest(role=role):
                self.assertEqual(summary["status"], "verified")
                self.assertEqual(
                    summary["admission_status"],
                    {"verified": summary["task_count"]},
                )
                self.assertEqual(
                    summary["verified_admission_evidence"],
                    summary["task_count"],
                )

        self.assertEqual(
            summaries["boundary"]["problem_subclass"],
            {"B1": 1, "B2": 2, "B3": 1, "B4": 1, "B5": 1},
        )
        self.assertEqual(
            summaries["precision"]["problem_subclass"],
            {"P1": 1, "P2": 1, "P3": 2, "P4": 2, "P5": 2},
        )
        self.assertEqual(
            {
                role: summary["dataset_hash"]
                for role, summary in summaries.items()
            },
            {
                "cumulative": (
                    "sha256:"
                    "4d7bde25e747bcc041aa5105ce5ce881a3f1e9fe2a7545667cdbc2c14d85064a"
                ),
                "boundary": (
                    "sha256:"
                    "810a9cc85c576f44edd2672197ab83b7dfee7f674e597c76c78050bd119d606a"
                ),
                "precision": (
                    "sha256:"
                    "65818466a02e99466386cb8e038dc4da59d91dcb3bea7b83c8901d31a96aa8eb"
                ),
            },
        )

        screening = json.loads(
            (
                REPO_ROOT
                / "factory"
                / "v0.7"
                / "p3"
                / "screening"
                / "screening_index.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            screening["counts"],
            {"accepted": 6, "deferred": 2, "rejected": 2},
        )
        self.assertEqual(len(screening["decisions"]), 10)

        validation = json.loads(
            (
                REPO_ROOT
                / "runs"
                / "v0.7_validation_report"
                / "experiment_summary.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(validation["totals"]["attempts"], 18)
        self.assertEqual(validation["totals"]["retries"], 0)
        self.assertEqual(
            validation["evaluation_outcomes"],
            {"f2p_failed": 3, "no_patch": 1, "resolved": 14},
        )


if __name__ == "__main__":
    unittest.main()
