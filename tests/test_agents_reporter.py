from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from op_bench.agents import GoldAgent, NoopAgent
from op_bench.reporter import summarize_results
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
TASK = TaskManifest.load(ROOT / "tasks" / "smoke" / "expit_nan_cpu" / "task.json")


class AgentReporterTests(unittest.TestCase):
    def test_noop_agent_returns_empty_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = NoopAgent().produce_patch(TASK, Path(tmp))
            self.assertEqual(output.agent_name, "noop")
            self.assertEqual(output.patch_path.read_text(encoding="utf-8"), "")

    def test_gold_agent_returns_gold_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = GoldAgent().produce_patch(TASK, Path(tmp))
            self.assertEqual(output.agent_name, "gold")
            self.assertEqual(
                output.patch_path.read_text(encoding="utf-8"),
                TASK.gold_patch_path.read_text(encoding="utf-8"),
            )

    def test_summarize_results_counts_resolved_rate(self) -> None:
        records = [
            {"agent": "gold", "status": "resolved", "duration_sec": 1.0},
            {"agent": "noop", "status": "fail_to_pass_failed", "duration_sec": 1.0},
        ]
        summary = summarize_results(records)
        self.assertEqual(summary["agents"]["gold"]["resolved_rate"], 1.0)
        self.assertEqual(summary["agents"]["noop"]["resolved_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
