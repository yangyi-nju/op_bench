from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from op_bench.runtime.replay import build_replay_inventory


ROOT = Path(__file__).resolve().parents[1]


class RuntimeReplayInventoryTests(unittest.TestCase):
    def test_checked_in_inventory_is_exactly_17_baseline_17_gold_51_legacy(self) -> None:
        inventory = build_replay_inventory(ROOT)

        self.assertEqual(len(inventory), 85)
        self.assertEqual(
            Counter(case.case_kind for case in inventory),
            {"baseline": 17, "gold": 17, "legacy": 51},
        )
        self.assertEqual(
            Counter(
                case.provenance_root
                for case in inventory
                if case.case_kind == "legacy"
            ),
            {
                "runs/v0.5_codex_legacy_cpu": 30,
                "runs/v0.5_codex_legacy_cuda": 3,
                "runs/v0.5_precision_codex_cpu": 9,
                "runs/v0.5_precision_codex_gpu": 9,
            },
        )

    def test_inventory_is_unique_content_bound_verified_and_stably_sorted(self) -> None:
        inventory = build_replay_inventory(ROOT)
        replay_ids = [case.replay_id for case in inventory]
        sort_keys = [case.sort_key for case in inventory]

        self.assertEqual(len(replay_ids), len(set(replay_ids)))
        self.assertEqual(sort_keys, sorted(sort_keys))
        self.assertTrue(
            all(case.patch_hash.startswith("sha256:") for case in inventory)
        )
        self.assertTrue(all(case.task_verified for case in inventory))
        self.assertTrue(
            all(
                case.patch_path is None
                or (
                    not Path(case.patch_path).is_absolute()
                    and ".." not in Path(case.patch_path).parts
                )
                for case in inventory
            )
        )
        self.assertTrue(
            all(case.runtime_profile_id.endswith("-v1") for case in inventory)
        )
        self.assertEqual(
            {case.expected_outcome for case in inventory if case.case_kind == "baseline"},
            {"f2p_failed"},
        )
        self.assertEqual(
            {case.expected_outcome for case in inventory if case.case_kind == "gold"},
            {"resolved"},
        )

    def test_every_legacy_patch_matches_task_attempt_and_selected_result_provenance(self) -> None:
        inventory = build_replay_inventory(ROOT)
        legacy = [case for case in inventory if case.case_kind == "legacy"]

        for case in legacy:
            with self.subTest(replay_id=case.replay_id):
                self.assertIsNotNone(case.attempt_number)
                self.assertIn(
                    f"attempt_{case.attempt_number:03d}",
                    case.patch_path,
                )
                self.assertIn(case.task_id, Path(case.patch_path).name)
                self.assertGreater(case.provenance_line, 0)
                self.assertTrue(case.provenance_hash.startswith("sha256:"))


if __name__ == "__main__":
    unittest.main()
