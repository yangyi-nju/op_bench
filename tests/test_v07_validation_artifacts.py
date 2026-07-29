from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "factory" / "v0.7" / "p4" / "validation_contract.json"
REPORT_ROOT = ROOT / "runs" / "v0.7_validation_report"
INDEX = REPORT_ROOT / "experiment_index.json"
SUMMARY = REPORT_ROOT / "experiment_summary.json"
REPORT = REPORT_ROOT / "experiment_report.md"
HUMAN_REPORT = ROOT / "docs" / "v0.7" / "validation_report.md"

EXPECTED_PROFILES = {
    "remote-cpu-boundary-torch2.2-py311-v1",
    "remote-cpu-boundary-torch2.3-py311-v1",
    "remote-cpu-boundary-torch2.4-py311-v1",
    "remote-cpu-source-boundary-py311-v1",
    "remote-cuda-boundary-torch2.6-cu124-v1",
}
EXPECTED_TASKS = {
    "opbench-v07-t0001",
    "opbench-v07-t0002",
    "opbench-v07-t0004",
    "opbench-v07-t0010",
    "opbench-v07-t0014",
    "opbench-v07-t0017",
}


class V07ValidationArtifactTests(unittest.TestCase):
    def test_public_report_matches_frozen_validation_contract(self) -> None:
        for path in (INDEX, SUMMARY):
            self.assertTrue(path.is_file(), f"missing public report artifact: {path}")
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        index = json.loads(INDEX.read_text(encoding="utf-8"))
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

        self.assertEqual(summary["dataset_identifier"], "pytorch_v0.7_boundary")
        self.assertEqual(
            summary["dataset_digest"],
            "sha256:eaaa93301975ebcf3507c1efe18b600c729ae1e978696bb331546ca59013f0cf",
        )
        self.assertEqual(summary["platform_version"], "opbench-v0.6.0")
        self.assertEqual(summary["totals"]["cohorts"], 5)
        self.assertEqual(summary["totals"]["attempts"], 18)
        self.assertEqual(summary["totals"]["trace_complete"], 18)
        self.assertEqual(summary["totals"]["retries"], 0)
        self.assertEqual(
            summary["evaluation_outcomes"],
            {"f2p_failed": 3, "no_patch": 1, "resolved": 14},
        )
        self.assertEqual(summary["agent_terminals"], {"finished": 17, "timeout": 1})
        self.assertEqual(summary["model_id"], "gpt-5.6-sol")
        self.assertEqual(
            summary["codex_cli_version"], "codex-cli 0.146.0-alpha.3.1"
        )
        self.assertEqual(summary["adapter_id"], "codex_mcp_canonical")
        self.assertEqual(summary["mcp"]["protocol_error_count"], 0)

        self.assertEqual(index["dataset_identifier"], contract["dataset_identifier"])
        self.assertEqual(index["dataset_digest"], contract["dataset_digest"])
        self.assertEqual(index["platform_version"], contract["platform_version"])
        self.assertEqual(len(index["cohorts"]), 5)
        self.assertEqual(len(index["attempts"]), 18)
        self.assertEqual(
            {
                profile_id
                for cohort in index["cohorts"]
                for profile_id in cohort["runtime_profile_ids"]
            },
            EXPECTED_PROFILES,
        )

        expected_partition = {
            (
                cohort["profile_id"],
                task["task_id"],
                repeat,
            )
            for cohort in contract["cohorts"]
            for task in cohort["task_repeats"]
            for repeat in task["repeats"]
        }
        observed_partition = {
            (row["runtime_profile_id"], row["task_id"], row["repeat"])
            for row in index["attempts"]
        }
        self.assertEqual(observed_partition, expected_partition)

        repeats_by_task: dict[str, set[int]] = defaultdict(set)
        for row in index["attempts"]:
            repeats_by_task[row["task_id"]].add(row["repeat"])
        self.assertEqual(set(repeats_by_task), EXPECTED_TASKS)
        self.assertTrue(
            all(repeats == {1, 2, 3} for repeats in repeats_by_task.values())
        )

    def test_public_report_is_trace_complete_and_redacted(self) -> None:
        for path in (INDEX, SUMMARY, REPORT, HUMAN_REPORT):
            self.assertTrue(path.is_file(), f"missing public report artifact: {path}")
        index = json.loads(INDEX.read_text(encoding="utf-8"))
        attempts = index["attempts"]
        self.assertEqual(len({row["attempt_id"] for row in attempts}), 18)
        self.assertTrue(all(row["trace_complete"] for row in attempts))
        self.assertTrue(all(row["retry_index"] == 1 for row in attempts))

        public_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (INDEX, SUMMARY, REPORT, HUMAN_REPORT)
        )
        for forbidden in (
            "private_evaluation",
            "private_runtime_resources",
            "/Users/",
            "/home/",
            "identity_file",
            "remote_user",
            "hostname",
            "BEGIN PRIVATE KEY",
            "Authorization: Bearer",
            "github_pat_",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, public_text)

    def test_human_report_documents_scope_and_anomaly_attribution(self) -> None:
        self.assertTrue(
            HUMAN_REPORT.is_file(),
            f"missing human validation report: {HUMAN_REPORT}",
        )
        text = HUMAN_REPORT.read_text(encoding="utf-8")
        for required in (
            "gpt-5.6-sol",
            "codex-cli 0.146.0-alpha.3.1",
            "codex_mcp_canonical",
            "18",
            "14 resolved",
            "3 f2p_failed",
            "1 no_patch",
            "source loading",
            "transport",
            "retry",
            "floor",
            "ceiling",
            "non-leaderboard",
        ):
            with self.subTest(required=required):
                self.assertIn(required, text)
        for profile_id in EXPECTED_PROFILES:
            with self.subTest(profile_id=profile_id):
                self.assertIn(profile_id, text)


if __name__ == "__main__":
    unittest.main()
