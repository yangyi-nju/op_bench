from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.agents import GoldAgent
from op_bench.reporter import normalized_result_status, summarize_results
from op_bench.task import TaskManifest


class AgentReporterTests(unittest.TestCase):
    def test_normalizes_legacy_regression_label_when_fix_also_failed(self) -> None:
        record = {
            "status": "pass_to_pass_regressed",
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 0,
        }
        self.assertEqual(normalized_result_status(record), "fail_to_pass_failed")

    def test_preserves_non_test_outcome_status(self) -> None:
        record = {
            "status": "environment_unavailable",
            "fail_to_pass_total": 1,
            "fail_to_pass_passed": 0,
            "pass_to_pass_total": 1,
            "pass_to_pass_passed": 0,
        }
        self.assertEqual(normalized_result_status(record), "environment_unavailable")

    def test_gold_agent_returns_gold_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = self._task(Path(tmp))
            output = GoldAgent().produce_patch(task, Path(tmp) / "out")
            self.assertEqual(output.agent_name, "gold")
            self.assertEqual(
                output.patch_path.read_text(encoding="utf-8"),
                task.gold_patch_path.read_text(encoding="utf-8"),
            )

    def test_summarize_results_counts_resolved_rate(self) -> None:
        records = [
            {"agent": "gold", "status": "resolved", "duration_sec": 1.0},
            {"agent": "codex_action_bridge", "status": "fail_to_pass_failed", "duration_sec": 1.0},
        ]
        summary = summarize_results(records)
        self.assertEqual(summary["agents"]["gold"]["resolved_rate"], 1.0)
        self.assertEqual(summary["agents"]["codex_action_bridge"]["resolved_rate"], 0.0)

    def _task(self, root: Path) -> TaskManifest:
        source = root / "source"
        source.mkdir()
        task_dir = root / "task"
        artifacts = task_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "gold.patch").write_text("diff --git a/a.py b/a.py\n", encoding="utf-8")
        (artifacts / "test.patch").write_text("", encoding="utf-8")
        manifest = {
            "task_id": "local__agent_fixture",
            "version": "v1",
            "source": {
                "repo": "local/repo",
                "local_path": str(source),
                "base_commit": "local",
                "checkout_mode": "local-copy",
            },
            "statement": {"title": "fixture", "body": "body", "labels": []},
            "operator": {
                "framework": "pytorch",
                "component": "test",
                "operator_name": "op",
                "problem_type": "tooling",
                "tags": [],
            },
            "environment": {
                "tier": "cpu-deterministic",
                "image": "local",
                "python_version": "3",
                "os": "local",
                "build_mode": "editable-python",
                "hardware": {"device": "cpu", "min_memory_gb": 1},
                "dependencies": [],
            },
            "evaluation": {
                "setup_commands": [],
                "fail_to_pass": ["unused"],
                "pass_to_pass": ["unused"],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 30,
            },
            "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
            "metadata": {"curation_status": "draft"},
        }
        (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        return TaskManifest.load(task_dir / "task.json")


if __name__ == "__main__":
    unittest.main()
