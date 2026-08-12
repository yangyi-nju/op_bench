from __future__ import annotations

import unittest

from op_bench.runtime.experiment_report import add_quality_experiment_metadata


class V07QualityValidationReportTests(unittest.TestCase):
    def test_quality_aggregation_is_taxonomy_aware_and_thresholded(self) -> None:
        attempts = [
            {
                "task_id": task_id,
                "evaluation_outcome": outcome,
            }
            for task_id, outcomes in (
                ("opbench-v07-t0001", ("resolved",)),
                ("opbench-v07-t0002", ("f2p_failed",)),
                ("opbench-v07-t0003", ("resolved", "resolved", "no_patch")),
            )
            for outcome in outcomes
        ]
        index = {"attempts": attempts}
        summary = {
            "totals": {"attempts": 5},
            "attribution": {"provider": 1, "mcp": 0, "runtime": 1},
        }
        metadata = {
            "opbench-v07-t0001": {
                "origin": "retained_historical",
                "difficulty": "medium",
                "slices": ["cumulative", "boundary"],
                "taxonomy": {
                    "contract_family": "api_behavior",
                    "failure_type": "missing_error",
                    "devices": ["cpu"],
                    "modes": ["eager"],
                    "phases": ["forward"],
                },
            },
            "opbench-v07-t0002": {
                "origin": "new",
                "difficulty": "hard",
                "slices": ["cumulative", "device"],
                "taxonomy": {
                    "contract_family": "tensor_metadata",
                    "failure_type": "wrong_result",
                    "devices": ["cuda"],
                    "modes": ["compile"],
                    "phases": ["forward"],
                },
            },
            "opbench-v07-t0003": {
                "origin": "replacement",
                "difficulty": "hard",
                "slices": ["cumulative", "precision"],
                "taxonomy": {
                    "contract_family": "result",
                    "failure_type": "wrong_result",
                    "devices": ["cpu"],
                    "modes": ["compile"],
                    "phases": ["backward"],
                },
            },
        }

        quality_index, quality_summary = add_quality_experiment_metadata(
            index,
            summary,
            metadata,
        )

        self.assertEqual(len(quality_index["quality_tasks"]), 3)
        self.assertEqual(quality_summary["totals"]["accepted_invalid"], 0)
        quality = quality_summary["quality"]
        self.assertEqual(set(quality["derived_slices"]), {"boundary", "device", "precision"})
        self.assertEqual(set(quality_summary["difficulty"]), {"hard", "medium"})
        self.assertEqual(
            set(quality_summary["contract_family"]),
            {"api_behavior", "result", "tensor_metadata"},
        )
        self.assertEqual(
            set(quality_summary["derived_slices"]),
            {"boundary", "device", "precision"},
        )
        self.assertFalse(
            quality["contract_family"]["api_behavior"]["standalone_score_eligible"]
        )
        self.assertTrue(quality["devices"]["cpu"]["standalone_score_eligible"] is False)
        self.assertEqual(quality["failure_attribution"]["agent"], 2)
        self.assertEqual(quality["failure_attribution"]["infrastructure_retries"], 2)


if __name__ == "__main__":
    unittest.main()
