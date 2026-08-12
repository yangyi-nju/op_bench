from __future__ import annotations

import json
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_CARD = REPO_ROOT / "docs" / "v0.7" / "dataset_card.md"
QUALITY_EXPANSION = REPO_ROOT / "docs" / "v0.7" / "quality_expansion.md"
SUMMARY_PATHS = {
    "cumulative": REPO_ROOT / "datasets" / "pytorch_v0.7" / "summary.json",
    "boundary": (
        REPO_ROOT / "datasets" / "pytorch_v0.7_boundary" / "summary.json"
    ),
    "precision": (
        REPO_ROOT / "datasets" / "pytorch_v0.7_precision" / "summary.json"
    ),
    "device": REPO_ROOT / "datasets" / "pytorch_v0.7_device" / "summary.json",
}
ARCHIVE_SUMMARY_PATHS = {
    role: (
        REPO_ROOT
        / "archives"
        / "v0.7-pre-quality"
        / "datasets"
        / dataset_id
        / "summary.json"
    )
    for role, dataset_id in {
        "cumulative": "pytorch_v0.7",
        "boundary": "pytorch_v0.7_boundary",
        "precision": "pytorch_v0.7_precision",
    }.items()
}
ENTRYPOINTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "README.zh-CN.md",
    REPO_ROOT / "docs" / "README.md",
    REPO_ROOT / "docs" / "README.zh-CN.md",
)
COMPLETION_RECORDS = (
    REPO_ROOT / "docs" / "v0.7" / "design.md",
    REPO_ROOT / "docs" / "project_state.md",
    REPO_ROOT / "docs" / "project_plan.md",
    REPO_ROOT / "CHANGELOG.md",
)


class V07ReleaseDocsTests(unittest.TestCase):
    def test_dataset_card_matches_quality_release_candidate(self) -> None:
        self.assertTrue(DATASET_CARD.is_file())
        card = DATASET_CARD.read_text(encoding="utf-8")
        for required in (
            "opbench-v0.7.0",
            "50 verified Tasks",
            "14 retained historical",
            "21 new",
            "15 replacement",
            "31",
            "5",
            "15",
            "46",
            "hard",
            "medium",
            "AgentTaskView",
            "122 fresh logical Attempts",
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
            {"cumulative": 50, "boundary": 31, "precision": 5, "device": 15},
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

        cumulative = summaries["cumulative"]
        self.assertEqual(
            cumulative["origin"],
            {"new": 21, "replacement": 15, "retained_historical": 14},
        )
        self.assertEqual(cumulative["difficulty"], {"hard": 46, "medium": 4})
        self.assertEqual(cumulative["devices"], {"cpu": 35, "cuda": 15})
        self.assertEqual(summaries["device"]["devices"], {"cuda": 15})
        self.assertEqual(
            {role: summary["dataset_hash"] for role, summary in summaries.items()},
            {
                "cumulative": (
                    "sha256:"
                    "3695622dd2619a760d510ef49e0a9dbff637c98790ad3263c521bae8e99c9518"
                ),
                "boundary": (
                    "sha256:"
                    "2890f5937a5b2c7f5a12c870fc9cc550f0f16ff065467245ecf65223b5976a01"
                ),
                "precision": (
                    "sha256:"
                    "508ec6928d94c159499ae84bf4f37e594b2bdafdef89b04369f481deeddb2c8d"
                ),
                "device": (
                    "sha256:"
                    "b598fdfe94af9921132b147ab693477de8fb360dabe7e5f611792e5f38c0f138"
                ),
            },
        )

    def test_historical_freeze_is_preserved_outside_current_paths(self) -> None:
        summaries = {
            role: json.loads(path.read_text(encoding="utf-8"))
            for role, path in ARCHIVE_SUMMARY_PATHS.items()
        }
        self.assertEqual(
            {role: summary["task_count"] for role, summary in summaries.items()},
            {"cumulative": 25, "boundary": 6, "precision": 8},
        )
        self.assertEqual(
            {role: summary["dataset_hash"] for role, summary in summaries.items()},
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

    def test_bilingual_entrypoints_publish_the_same_quality_release(self) -> None:
        for path in ENTRYPOINTS:
            text = path.read_text(encoding="utf-8")
            for required in (
                "opbench-v0.7.0",
                "50-task cumulative",
                "31-task Boundary",
                "5-task Precision",
                "15-task Device",
                "datasets/pytorch_v0.7/dataset.json",
                "datasets/pytorch_v0.7_boundary/dataset.json",
                "datasets/pytorch_v0.7_precision/dataset.json",
                "datasets/pytorch_v0.7_device/dataset.json",
                "v0.7/dataset_card.md",
                "v0.7/validation_report.md",
                "v0.6/experiment_report.md",
                "non-leaderboard",
            ):
                with self.subTest(path=path.name, required=required):
                    self.assertIn(required, text)
            for forbidden in (
                "v0.7 is a formal multi-Agent ranking",
                "v0.7 是正式多 Agent 排名",
                "v0.7 cohort establishes a leaderboard",
            ):
                with self.subTest(path=path.name, forbidden=forbidden):
                    self.assertNotIn(forbidden, text)

    def test_v07_records_match_the_current_execution_stage(self) -> None:
        records = {
            path: path.read_text(encoding="utf-8") for path in COMPLETION_RECORDS
        }
        expansion = QUALITY_EXPANSION.read_text(encoding="utf-8")
        self.assertIn(
            "50-task 质量版及最终验证均已完成",
            records[COMPLETION_RECORDS[0]],
        )
        self.assertIn(
            "v0.7 50-task 质量扩展（Completed）",
            records[COMPLETION_RECORDS[1]],
        )
        self.assertIn("| v0.7 | 已完成 |", records[COMPLETION_RECORDS[2]])
        self.assertIn(
            "## v0.7 - Completed 2026-08-11 (50-task quality release)",
            records[COMPLETION_RECORDS[3]],
        )
        for required in (
            "14 retained historical tasks",
            "36 new or replacement tasks",
            "36/36 verified",
            "122",
            "AgentTaskView",
            "medium",
            "hard",
            "Prompt",
        ):
            with self.subTest(expansion_required=required):
                self.assertIn(required, expansion)

        for path, text in records.items():
            for required in (
                "opbench-v0.7.0",
                "50-task",
                "122",
                "non-leaderboard",
            ):
                with self.subTest(path=path.name, required=required):
                    self.assertIn(required, text)
        for required in (
            "50/50",
            "122/122",
            "42 resolved",
            "52 F2P failed",
            "28 invalid patch",
        ):
            with self.subTest(completion_required=required):
                self.assertTrue(
                    any(required in text for text in records.values()),
                    required,
                )


if __name__ == "__main__":
    unittest.main()
