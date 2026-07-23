from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.contracts import EvaluationResultV06, TestExecutionSummary
from op_bench.runtime.run_artifacts import AttemptArtifactStore
from op_bench.runtime.summary import (
    SelectedAttempt,
    rebuild_results,
    rebuild_summary,
    write_rebuilt_outputs,
)
from tests.test_runtime_contracts import (
    SHA_A,
    SHA_B,
    SHA_C,
    agent_spec,
    full_task_spec,
    identity,
)
from tests.test_runtime_manifest import manifest


PASS = TestExecutionSummary(1, 1, 1, 0, 0)
FAIL = TestExecutionSummary(1, 1, 0, 1, 0)
ZERO = TestExecutionSummary(0, 0, 0, 0, 0)


class RuntimeSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        task_a = replace(
            full_task_spec(),
            task=identity("task", "task-a", SHA_A),
        )
        task_b = replace(
            full_task_spec(),
            task=identity("task", "task-b", SHA_B),
        )
        agent_a = replace(
            agent_spec(),
            agent=identity("agent", "agent-a", SHA_A),
        )
        agent_b = replace(
            agent_spec(),
            agent=identity("agent", "agent-b", SHA_B),
        )
        self.manifest = manifest(
            tasks=(task_b, task_a),
            agents=(agent_b, agent_a),
            repeat_count=2,
        )

    def selected(
        self,
        expected,
        outcome: str,
        *,
        retry_index: int = 1,
    ) -> SelectedAttempt:
        infrastructure = outcome == "evaluation_error"
        if outcome == "resolved":
            f2p, p2p = PASS, PASS
        elif outcome == "f2p_failed":
            f2p, p2p = FAIL, PASS
        else:
            f2p, p2p = ZERO, ZERO
        result = EvaluationResultV06(
            session_id=f"session-{expected.attempt_id[-12:]}",
            attempt_id=expected.attempt_id,
            attempt_validity=(
                "infrastructure_invalid" if infrastructure else "valid"
            ),
            agent_terminal=None if infrastructure else "finished",
            evaluation_outcome=outcome,
            invalid_reason="evaluator_unavailable" if infrastructure else None,
            patch=identity("patch", "final.patch", SHA_C),
            fail_to_pass=f2p,
            pass_to_pass=p2p,
            duration_ms=25,
            evaluation=identity(
                "evaluation",
                self.manifest.evaluation_protocol,
                SHA_B,
            ),
            scoring=self.manifest.scoring,
        )
        return SelectedAttempt(
            attempt_id=expected.attempt_id,
            retry_index=retry_index,
            evaluation_spec_hash=SHA_A,
            evaluation_result=result,
        )

    def selected_matrix(self) -> tuple[SelectedAttempt, ...]:
        by_agent: dict[str, list] = {"agent-a": [], "agent-b": []}
        for expected in self.manifest.expected_attempts:
            by_agent[expected.agent.identifier].append(expected)
        selected: list[SelectedAttempt] = []
        for agent_id in ("agent-a", "agent-b"):
            expected = by_agent[agent_id]
            selected.extend(
                (
                    self.selected(expected[0], "resolved"),
                    self.selected(expected[1], "f2p_failed"),
                    self.selected(
                        expected[2],
                        "evaluation_error",
                        retry_index=2,
                    ),
                )
            )
        return tuple(reversed(selected))

    def test_results_follow_frozen_matrix_order_and_summary_excludes_infrastructure(self) -> None:
        selected = self.selected_matrix()

        results_bytes = rebuild_results(self.manifest, selected)
        records = [json.loads(line) for line in results_bytes.splitlines()]
        expected_order = [
            item.attempt_id
            for item in self.manifest.expected_attempts
            if item.attempt_id in {selected_item.attempt_id for selected_item in selected}
        ]
        self.assertEqual([record["attempt_id"] for record in records], expected_order)
        self.assertNotIn("/Users/", results_bytes.decode("utf-8"))

        summary = rebuild_summary(self.manifest, selected)
        expected_agent = {
            "expected": 4,
            "observed": 3,
            "valid": 2,
            "infrastructure_invalid": 1,
            "resolved": 1,
            "resolved_denominator": 2,
            "resolved_rate": {"numerator": 1, "denominator": 2},
            "retries": 1,
            "evaluation_outcomes": {
                "evaluation_error": 1,
                "f2p_failed": 1,
                "resolved": 1,
            },
            "agent_terminals": {"finished": 2, "none": 1},
        }
        self.assertEqual(summary["agents"]["agent-a"], expected_agent)
        self.assertEqual(summary["agents"]["agent-b"], expected_agent)
        self.assertEqual(summary["manifest_hash"], self.manifest.content_hash)
        self.assertEqual(
            summary["evaluation_protocol"], self.manifest.evaluation_protocol
        )
        self.assertEqual(summary["evaluation"], self.manifest.evaluation.to_dict())
        self.assertEqual(summary["scoring"], self.manifest.scoring.to_dict())
        self.assertRegex(summary["results_hash"], r"^sha256:[0-9a-f]{64}$")

    def test_rebuild_is_byte_stable_and_protocol_identity_changes_outputs(self) -> None:
        selected = self.selected_matrix()
        first_results = rebuild_results(self.manifest, selected)
        first_summary = rebuild_summary(self.manifest, selected)

        self.assertEqual(rebuild_results(self.manifest, selected), first_results)
        self.assertEqual(rebuild_summary(self.manifest, selected), first_summary)
        changed = list(selected)
        changed[0] = replace(
            changed[0],
            evaluation_result=replace(
                changed[0].evaluation_result,
                evaluation=identity(
                    "evaluation",
                    self.manifest.evaluation_protocol,
                    SHA_C,
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "evaluation identity mismatch"):
            rebuild_results(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "evaluation identity mismatch"):
            rebuild_summary(self.manifest, changed)

    def test_store_writes_exact_rebuilt_bytes_idempotently(self) -> None:
        selected = self.selected_matrix()
        with tempfile.TemporaryDirectory() as temporary:
            store = AttemptArtifactStore(Path(temporary) / "run", self.manifest)
            try:
                results_bytes, summary_bytes = write_rebuilt_outputs(store, selected)
                self.assertEqual(store.read_results_bytes(), results_bytes)
                self.assertEqual(store.read_summary_bytes(), summary_bytes)
                self.assertEqual(
                    write_rebuilt_outputs(store, selected),
                    (results_bytes, summary_bytes),
                )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
