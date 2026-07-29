from __future__ import annotations

import json
from pathlib import Path
import unittest

from op_bench.factory.quality_release import validate_quality_task
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/pytorch_v0.7/dataset.json"
PUBLIC_IDS = ROOT / "factory/v0.7/p6/public_task_ids.json"
HISTORICAL_INDEX = ROOT / "factory/v0.7/p7/historical_readmission.json"
PILOT_FACTS = ROOT / "factory/v0.7/p7/pilot_factual_evidence.json"
SECOND_REVIEW = ROOT / "factory/v0.7/p7/second_complexity_review.json"


def load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path}: expected JSON object")
    return value


def task_ids(path: Path) -> frozenset[str]:
    dataset = load_json(path)
    tasks = dataset["tasks"]
    if not isinstance(tasks, list):
        raise AssertionError(f"{path}: tasks must be an array")
    return frozenset(
        str(item["task_id"])
        for item in tasks
        if isinstance(item, dict)
    )


class V07HistoricalQualityArtifactTests(unittest.TestCase):
    EXPECTED_RETAINED = {
        "opbench-v07-t0003",
        "opbench-v07-t0005",
        "opbench-v07-t0006",
        "opbench-v07-t0008",
        "opbench-v07-t0009",
        "opbench-v07-t0012",
        "opbench-v07-t0013",
        "opbench-v07-t0015",
        "opbench-v07-t0016",
        "opbench-v07-t0017",
        "opbench-v07-t0021",
        "opbench-v07-t0022",
        "opbench-v07-t0023",
        "opbench-v07-t0024",
    }
    EXPECTED_DEFERRED = {"opbench-v07-t0019"}
    EXPECTED_RETIRED = {
        "opbench-v07-t0001",
        "opbench-v07-t0002",
        "opbench-v07-t0004",
        "opbench-v07-t0007",
        "opbench-v07-t0010",
        "opbench-v07-t0011",
        "opbench-v07-t0014",
        "opbench-v07-t0018",
        "opbench-v07-t0020",
        "opbench-v07-t0025",
    }

    def test_public_task_ids_are_frozen_in_lexical_canonical_order(self) -> None:
        archive_ids = sorted(task_ids(DATASET))
        self.assertTrue(PUBLIC_IDS.is_file(), "public Task ID mapping is missing")
        mapping = load_json(PUBLIC_IDS)
        tasks = mapping["tasks"]
        self.assertIsInstance(tasks, list)
        self.assertEqual(
            [item["task_id"] for item in tasks],
            archive_ids,
        )
        self.assertEqual(
            [item["public_task_id"] for item in tasks],
            [
                f"opbench-v07-t{index:04d}"
                for index in range(1, len(archive_ids) + 1)
            ],
        )

    def test_all_pre_quality_tasks_have_dispositions(self) -> None:
        archive_ids = task_ids(DATASET)
        self.assertTrue(
            HISTORICAL_INDEX.is_file(),
            "historical readmission index is missing",
        )
        index = load_json(HISTORICAL_INDEX)
        records = index["records"]
        self.assertIsInstance(records, list)
        self.assertEqual(
            {
                item["task_id"]
                for item in records
                if isinstance(item, dict)
            },
            archive_ids,
        )
        self.assertEqual(index["task_count"], 25)
        self.assertEqual(
            index["required_candidate_count"],
            3 * (50 - index["k"]),
        )
        by_disposition = {
            disposition: {
                item["public_task_id"]
                for item in records
                if isinstance(item, dict)
                and item["disposition"] == disposition
            }
            for disposition in ("retained", "deferred", "retired")
        }
        self.assertEqual(by_disposition["retained"], self.EXPECTED_RETAINED)
        self.assertEqual(by_disposition["deferred"], self.EXPECTED_DEFERRED)
        self.assertEqual(by_disposition["retired"], self.EXPECTED_RETIRED)
        self.assertEqual(index["k"], 14)
        self.assertEqual(index["required_candidate_count"], 108)

        for record in records:
            if not isinstance(record, dict):
                self.fail("historical index record must be an object")
            if record["disposition"] != "retained":
                continue
            task = TaskManifest.load(ROOT / str(record["task_path"]) / "task.json")
            with self.subTest(task=task.task_id):
                self.assertEqual(
                    validate_quality_task(
                        ROOT,
                        task,
                        require_verified=True,
                    ),
                    (),
                )
                self.assertIn(
                    task.data["metadata"]["difficulty"],
                    ("medium", "hard"),
                )

    def test_review_and_per_task_quality_artifacts_are_complete(self) -> None:
        reviews = ROOT / "factory/v0.7/p7/reviews"
        review_files = sorted(reviews.glob("*.json"))
        self.assertEqual(len(review_files), 25)
        self.assertFalse(
            any(path.is_dir() for path in reviews.iterdir()),
            "reviews must be exactly 25 canonical Task JSON files",
        )
        mapping = load_json(PUBLIC_IDS)
        expected_names = {
            f"{item['task_id']}.json"
            for item in mapping["tasks"]
            if isinstance(item, dict)
        }
        self.assertEqual({path.name for path in review_files}, expected_names)

        index = load_json(HISTORICAL_INDEX)
        for record in index["records"]:
            if not isinstance(record, dict):
                self.fail("historical index record must be an object")
            task_root = ROOT / str(record["task_path"])
            with self.subTest(task=record["task_id"]):
                self.assertTrue((task_root / "quality/prompt.json").is_file())
                self.assertTrue((task_root / "quality/complexity.json").is_file())
                self.assertTrue((task_root / "quality/readmission.json").is_file())

    def test_reviews_bind_real_reviewer_decisions_and_sanitized_pilot_facts(
        self,
    ) -> None:
        pilot = load_json(PILOT_FACTS)
        self.assertFalse(pilot["counts_toward_final"])
        self.assertTrue(pilot["outcomes_are_factual_only"])
        self.assertEqual(pilot["expected_attempt_count"], 3)
        self.assertEqual(pilot["observed_attempt_count"], 3)
        self.assertEqual(pilot["missing_attempt_count"], 0)
        self.assertEqual(
            [item["evaluation_outcome"] for item in pilot["tasks"]],
            ["resolved", "f2p_failed", "f2p_failed"],
        )

        second = load_json(SECOND_REVIEW)
        self.assertEqual(
            second["reviewer"],
            "complexity-second-reviewer-v07-independent-01",
        )
        self.assertEqual(
            [item["public_task_id"] for item in second["records"]],
            [
                "opbench-v07-t0003",
                "opbench-v07-t0022",
                "opbench-v07-t0024",
            ],
        )
        for item in second["records"]:
            self.assertEqual(item["pilot_decision"], "accepted")
            self.assertIs(item["second_review"], True)
            self.assertEqual(
                item["complexity_evidence_decision"],
                "accepted",
            )
            self.assertEqual(item["complexity_evidence_severity"], "none")

        reviews = {
            load_json(path)["public_task_id"]: load_json(path)
            for path in (ROOT / "factory/v0.7/p7/reviews").glob("*.json")
        }
        for public_task_id in self.EXPECTED_RETAINED:
            review = reviews[public_task_id]
            complexity = review["complexity"]
            self.assertEqual(complexity["reviewer"], "semantic-reviewer-v07-independent-01")
            if complexity["localization"] + complexity["diagnosis"] + complexity[
                "repair_regression"
            ] == 4:
                self.assertIs(complexity["second_review"], True)
                self.assertEqual(
                    complexity["blind_pilot"]["decision"],
                    "accepted",
                )
                self.assertIs(
                    complexity["blind_pilot"]["counts_toward_final"],
                    False,
                )

        self.assertEqual(
            reviews["opbench-v07-t0010"]["complexity"]["hard_rejections"],
            ["no_public_contract_impact"],
        )
        self.assertEqual(
            reviews["opbench-v07-t0018"]["complexity"]["hard_rejections"],
            ["standard_admission_failure"],
        )
        self.assertEqual(
            reviews["opbench-v07-t0019"]["complexity"]["hard_rejections"],
            ["standard_admission_failure"],
        )
        self.assertEqual(
            reviews["opbench-v07-t0019"]["disposition"],
            "deferred",
        )

    def test_tracked_quality_inputs_do_not_depend_on_temporary_run_paths(
        self,
    ) -> None:
        for path in (
            ROOT / "factory/v0.7/p7/pilot_factual_evidence.json",
            ROOT / "factory/v0.7/p7/second_complexity_review.json",
            *(ROOT / "factory/v0.7/p7/reviews").glob("*.json"),
        ):
            encoded = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(".superpowers/", encoded)
                self.assertNotIn(
                    "runs/v0.7_historical_blind_pilots_score4_cpu",
                    encoded,
                )
                self.assertNotIn("/Users/", encoded)


if __name__ == "__main__":
    unittest.main()
