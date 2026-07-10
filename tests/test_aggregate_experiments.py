from __future__ import annotations

import unittest

from scripts.aggregate_experiments import _experiment_integrity


class ExperimentIntegrityTests(unittest.TestCase):
    def test_complete_experiment(self) -> None:
        records = [
            {"task_id": "t1", "agent": "baseline", "status": "baseline_reproduced"},
            {"task_id": "t1", "agent": "codex", "attempt": 1, "status": "resolved"},
            {"task_id": "t1", "agent": "codex", "attempt": 2, "status": "fail_to_pass_failed"},
        ]

        result = _experiment_integrity(records, {"t1"}, 2)

        self.assertTrue(result["complete"])
        self.assertEqual(result["expected_attempt_count"], 2)

    def test_transient_and_missing_attempts_are_incomplete(self) -> None:
        records = [
            {"task_id": "t1", "agent": "baseline", "status": "baseline_reproduced"},
            {"task_id": "t1", "agent": "codex", "attempt": 1, "status": "environment_unavailable"},
        ]

        result = _experiment_integrity(records, {"t1"}, 2)

        self.assertFalse(result["complete"])
        self.assertEqual(result["transient_attempts"], [["t1", "codex", 1]])
        self.assertEqual(result["missing_attempts"], [["t1", "codex", 2]])


if __name__ == "__main__":
    unittest.main()
