from __future__ import annotations

import unittest
from pathlib import Path

from op_bench.evaluator import Evaluator
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks" / "smoke" / "expit_nan_cpu" / "task.json"


class EvaluatorTests(unittest.TestCase):
    def test_baseline_reproduces_failure(self) -> None:
        result = Evaluator().evaluate_baseline(TaskManifest.load(TASK_PATH))
        self.assertEqual(result.status, "baseline_reproduced")
        self.assertEqual(result.fail_to_pass_passed, 0)
        self.assertEqual(result.pass_to_pass_passed, 1)

    def test_gold_resolves_task(self) -> None:
        result = Evaluator().evaluate_gold(TaskManifest.load(TASK_PATH))
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.fail_to_pass_passed, 1)
        self.assertEqual(result.pass_to_pass_passed, 1)


if __name__ == "__main__":
    unittest.main()
