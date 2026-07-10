from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from op_bench.reporter import compute_extended_metrics


def _record(**kwargs):
    return {
        "agent": kwargs.get("agent", "codex"),
        "task_id": kwargs.get("task_id", "t1"),
        "status": kwargs.get("status", "resolved"),
        "attempt": kwargs.get("attempt", 1),
        "fail_to_pass_total": kwargs.get("f2p_total", 1),
        "fail_to_pass_passed": kwargs.get("f2p_pass", 1),
        "pass_to_pass_total": kwargs.get("p2p_total", 3),
        "pass_to_pass_passed": kwargs.get("p2p_pass", 3),
        "duration_sec": kwargs.get("duration", 10.0),
        "patch_path": kwargs.get("patch_path"),
    }


class ExtendedMetricsTests(unittest.TestCase):
    def test_empty_records_returns_empty(self) -> None:
        self.assertEqual(compute_extended_metrics([], {}), {})

    def test_basic_resolved_rate(self) -> None:
        records = [
            _record(status="resolved"),
            _record(status="resolved"),
            _record(status="fail_to_pass_failed"),
        ]
        out = compute_extended_metrics(records, {})
        self.assertAlmostEqual(out["codex"]["resolved_rate"], 2 / 3)
        self.assertEqual(out["codex"]["total"], 3)
        self.assertEqual(out["codex"]["resolved"], 2)

    def test_baseline_records_are_excluded(self) -> None:
        records = [
            _record(agent="baseline", status="baseline_reproduced"),
            _record(agent="codex", status="resolved"),
        ]
        out = compute_extended_metrics(records, {})
        self.assertNotIn("baseline", out)
        self.assertEqual(out["codex"]["total"], 1)

    def test_regression_rate(self) -> None:
        # fail_to_pass passes but pass_to_pass broken → regression
        records = [
            _record(f2p_pass=1, f2p_total=1, p2p_pass=2, p2p_total=3, status="resolved"),
            _record(f2p_pass=1, f2p_total=1, p2p_pass=3, p2p_total=3, status="resolved"),
        ]
        out = compute_extended_metrics(records, {})
        self.assertAlmostEqual(out["codex"]["regression_rate"], 0.5)

    def test_fail_to_pass_only_rate_is_explicit(self) -> None:
        records = [
            _record(f2p_pass=1, f2p_total=1, p2p_pass=3, p2p_total=3),
            _record(f2p_pass=1, f2p_total=1, p2p_pass=2, p2p_total=3),
            _record(f2p_pass=0, f2p_total=1, p2p_pass=3, p2p_total=3),
        ]
        out = compute_extended_metrics(records, {})
        self.assertAlmostEqual(out["codex"]["fail_to_pass_only_rate"], 1 / 3)

    def test_pass_to_pass_kept_rate(self) -> None:
        records = [
            _record(p2p_pass=2, p2p_total=4),  # 0.5
            _record(p2p_pass=4, p2p_total=4),  # 1.0
        ]
        out = compute_extended_metrics(records, {})
        self.assertAlmostEqual(out["codex"]["pass_to_pass_kept_rate"], 0.75)

    def test_pass_to_pass_kept_rate_ignores_zero_total(self) -> None:
        records = [
            _record(p2p_pass=0, p2p_total=0),
            _record(p2p_pass=4, p2p_total=4),
        ]
        out = compute_extended_metrics(records, {})
        # Only the second record is counted.
        self.assertAlmostEqual(out["codex"]["pass_to_pass_kept_rate"], 1.0)

    def test_patch_conciseness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Two agent patches: 20 lines and 40 lines. Gold is 20 lines.
            p1 = root / "p1.patch"
            p2 = root / "p2.patch"
            p1.write_text("\n".join(f"line {i}" for i in range(20)))
            p2.write_text("\n".join(f"line {i}" for i in range(40)))
            records = [
                _record(task_id="t1", status="resolved", patch_path=str(p1)),
                _record(task_id="t1", status="resolved", patch_path=str(p2)),
            ]
            meta = {"t1": {"gold_patch_lines": 20}}
            out = compute_extended_metrics(records, meta)
            # p1: 20/20 = 1.0. p2: 20/40 = 0.5. Median = 0.75.
            self.assertAlmostEqual(out["codex"]["patch_conciseness"], 0.75)

    def test_patch_conciseness_clamps_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "small.patch"
            p.write_text("only 3\nlines\nhere\n")
            records = [_record(task_id="t1", status="resolved", patch_path=str(p))]
            meta = {"t1": {"gold_patch_lines": 100}}
            out = compute_extended_metrics(records, meta)
            # 100/3 ≫ 1, must clamp to 1
            self.assertEqual(out["codex"]["patch_conciseness"], 1.0)

    def test_patch_conciseness_skips_unresolved(self) -> None:
        records = [
            _record(task_id="t1", status="fail_to_pass_failed", patch_path="/nonexistent"),
        ]
        meta = {"t1": {"gold_patch_lines": 20}}
        out = compute_extended_metrics(records, meta)
        self.assertEqual(out["codex"]["patch_conciseness"], 0.0)  # median of empty list

    def test_tier_weighted_score(self) -> None:
        records = [
            _record(task_id="cpu1", status="resolved"),
            _record(task_id="cuda_py1", status="resolved"),
            _record(task_id="kernel1", status="fail_to_pass_failed"),
        ]
        meta = {
            "cpu1": {"runtime_tier": "cpu_python_overlay"},
            "cuda_py1": {"runtime_tier": "cuda_python_overlay"},
            "kernel1": {"runtime_tier": "cuda_kernel_build"},
        }
        out = compute_extended_metrics(records, meta)
        # weighted resolved = 1*1 + 2*1 + 3*0 = 3
        # weighted total = 1 + 2 + 3 = 6
        # score = 3/6 = 0.5
        self.assertAlmostEqual(out["codex"]["tier_weighted_score"], 0.5)
        # Plain resolved_rate would be 2/3 ≈ 0.667 — the weighting appropriately
        # discounts the resolved cpu task.

    def test_per_tier_breakdown(self) -> None:
        records = [
            _record(task_id="cpu1", status="resolved"),
            _record(task_id="cpu2", status="fail_to_pass_failed"),
            _record(task_id="cuda1", status="resolved"),
        ]
        meta = {
            "cpu1": {"runtime_tier": "cpu_python_overlay"},
            "cpu2": {"runtime_tier": "cpu_python_overlay"},
            "cuda1": {"runtime_tier": "cuda_python_overlay"},
        }
        out = compute_extended_metrics(records, meta)
        per_tier = out["codex"]["per_tier"]
        self.assertAlmostEqual(per_tier["cpu_python_overlay"]["resolved_rate"], 0.5)
        self.assertAlmostEqual(per_tier["cuda_python_overlay"]["resolved_rate"], 1.0)

    def test_per_problem_dimension_groups_unclassified(self) -> None:
        records = [
            _record(task_id="t1", status="resolved"),
            _record(task_id="t2", status="resolved"),
            _record(task_id="t3", status="fail_to_pass_failed"),
        ]
        meta = {
            "t1": {"problem_dimension": "precision"},
            "t2": {"problem_dimension": "precision"},
            "t3": {"problem_dimension": None},
        }
        out = compute_extended_metrics(records, meta)
        per_dim = out["codex"]["per_problem_dimension"]
        self.assertIn("precision", per_dim)
        self.assertIn("unclassified", per_dim)
        self.assertAlmostEqual(per_dim["precision"]["resolved_rate"], 1.0)
        self.assertAlmostEqual(per_dim["unclassified"]["resolved_rate"], 0.0)

    def test_per_problem_subclass_breakdown(self) -> None:
        records = [
            _record(task_id="p1", status="resolved"),
            _record(task_id="p2", status="fail_to_pass_failed"),
        ]
        metadata = {
            "p1": {"problem_subclass": "P1"},
            "p2": {"problem_subclass": "P2"},
        }
        out = compute_extended_metrics(records, metadata)
        self.assertEqual(out["codex"]["per_problem_subclass"]["P1"]["resolved_rate"], 1.0)
        self.assertEqual(out["codex"]["per_problem_subclass"]["P2"]["resolved_rate"], 0.0)

    def test_multi_agent_groups(self) -> None:
        records = [
            _record(agent="codex", task_id="t1", status="resolved"),
            _record(agent="claude", task_id="t1", status="fail_to_pass_failed"),
        ]
        out = compute_extended_metrics(records, {})
        self.assertIn("codex", out)
        self.assertIn("claude", out)
        self.assertEqual(out["codex"]["resolved_rate"], 1.0)
        self.assertEqual(out["claude"]["resolved_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
