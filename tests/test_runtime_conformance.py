from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import unittest

from op_bench.runtime.conformance import (
    CONFORMANCE_IGNORED_FIELDS_V1,
    ConformanceSnapshot,
    compare_conformance_snapshots,
    normalize_conformance_snapshot,
)
from op_bench.runtime.validation import ContractError


def snapshot() -> ConformanceSnapshot:
    return ConformanceSnapshot(
        contract_version="v1",
        action_observations=(
            {
                "action_id": "action-1",
                "action_name": "workspace_read",
                "observation": {
                    "ok": True,
                    "error_code": "ok",
                    "started_at_ms": 100,
                    "ended_at_ms": 110,
                    "duration_ms": 10,
                    "data": {
                        "private_target_hash": "sha256:" + "a" * 64,
                        "raw_handle_hash": "sha256:" + "b" * 64,
                        "machine_local_path": "/private/a",
                    },
                },
            },
            {
                "action_id": "action-2",
                "action_name": "session_finish",
                "observation": {"ok": True, "error_code": "ok"},
            },
        ),
        budget_usage={"actions": 2, "tests": 0, "commands": 0, "wall_clock_ms": 10},
        patch_identity={"digest": "sha256:" + "c" * 64, "identifier": "final.patch"},
        session_terminal={
            "status": "terminal",
            "terminal_reason": "agent_finished",
            "started_at_ms": 100,
            "ended_at_ms": 200,
        },
        evaluation_outcome="resolved",
        cleanup_status={
            "all_released": True,
            "entries": [{"resource_id": "resource-1", "status": "released"}],
        },
    )


class ConformanceNormalizationTests(unittest.TestCase):
    def test_only_fixed_nondeterministic_fields_are_ignored(self) -> None:
        baseline = snapshot()
        changed = baseline.to_dict()
        observation = changed["action_observations"][0]["observation"]
        observation["started_at_ms"] = 999
        observation["ended_at_ms"] = 1_000
        observation["duration_ms"] = 1
        observation["data"]["private_target_hash"] = "sha256:" + "d" * 64
        observation["data"]["raw_handle_hash"] = "sha256:" + "e" * 64
        observation["data"]["machine_local_path"] = "/another/private/path"
        changed["budget_usage"]["wall_clock_ms"] = 999
        changed["session_terminal"]["started_at_ms"] = 998
        changed["session_terminal"]["ended_at_ms"] = 999
        variant = ConformanceSnapshot.from_dict(changed)

        comparison = compare_conformance_snapshots(baseline, variant)

        self.assertTrue(comparison.equal)
        self.assertEqual(comparison.differences, ())
        self.assertEqual(
            CONFORMANCE_IGNORED_FIELDS_V1,
            (
                "duration_ms",
                "ended_at_ms",
                "machine_local_path",
                "private_target_hash",
                "raw_handle_hash",
                "started_at_ms",
                "wall_clock_ms",
            ),
        )
        with self.assertRaises(TypeError):
            normalize_conformance_snapshot(baseline, ignored_fields=("ok",))

    def test_semantic_mutations_have_precise_sorted_paths(self) -> None:
        baseline = snapshot()
        cases = (
            (
                "action",
                lambda value: value["action_observations"][0]["observation"].update(ok=False),
                "$.action_observations[0].observation.ok",
            ),
            (
                "budget",
                lambda value: value["budget_usage"].update(actions=3),
                "$.budget_usage.actions",
            ),
            (
                "patch",
                lambda value: value["patch_identity"].update(digest="sha256:" + "f" * 64),
                "$.patch_identity.digest",
            ),
            (
                "terminal",
                lambda value: value["session_terminal"].update(terminal_reason="timeout"),
                "$.session_terminal.terminal_reason",
            ),
            (
                "evaluation",
                lambda value: value.update(evaluation_outcome="f2p_failed"),
                "$.evaluation_outcome",
            ),
            (
                "cleanup",
                lambda value: value["cleanup_status"]["entries"][0].update(status="create_failed"),
                "$.cleanup_status.entries[0].status",
            ),
        )
        for name, mutate, expected_path in cases:
            with self.subTest(name=name):
                changed = deepcopy(baseline.to_dict())
                mutate(changed)
                comparison = compare_conformance_snapshots(
                    baseline,
                    ConformanceSnapshot.from_dict(changed),
                )
                self.assertFalse(comparison.equal)
                self.assertIn(expected_path, comparison.differences)
                self.assertEqual(
                    comparison.differences,
                    tuple(sorted(comparison.differences)),
                )

    def test_invalid_semantic_shapes_fail_closed(self) -> None:
        baseline = snapshot()
        with self.assertRaisesRegex(ContractError, "contract_version"):
            replace(baseline, contract_version="v2")
        with self.assertRaisesRegex(ContractError, "duplicate action"):
            replace(
                baseline,
                action_observations=(
                    baseline.action_observations[0],
                    baseline.action_observations[0],
                    baseline.action_observations[1],
                ),
            )
        with self.assertRaisesRegex(ContractError, "session_finish"):
            replace(baseline, action_observations=baseline.action_observations[:1])
        with self.assertRaisesRegex(ContractError, "terminal"):
            replace(baseline, session_terminal={"status": "running"})
        with self.assertRaisesRegex(ContractError, "active|cleanup"):
            replace(
                baseline,
                cleanup_status={
                    "all_released": False,
                    "entries": [{"resource_id": "resource-1", "status": "created"}],
                },
            )


if __name__ == "__main__":
    unittest.main()
