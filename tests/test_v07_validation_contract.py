from __future__ import annotations

from pathlib import Path
import unittest

from op_bench.runtime.experiment_report import (
    McpExperimentContract,
    load_mcp_experiment_contract,
    load_public_task_id_aliases,
)
from op_bench.runtime.task_view import assert_public_artifact_safe
from scripts.build_v07_validation_contract import (
    build_validation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "factory/v0.7/p4/validation_contract.json"
EXPECTED_COHORTS = {
    "remote-cpu-boundary-torch2.2-py311-v1": (
        "opbench-v07-t0001",
    ),
    "remote-cpu-boundary-torch2.3-py311-v1": (
        "opbench-v07-t0002",
    ),
    "remote-cpu-boundary-torch2.4-py311-v1": (
        "opbench-v07-t0004",
    ),
    "remote-cpu-source-boundary-py311-v1": (
        "opbench-v07-t0014",
        "opbench-v07-t0017",
    ),
    "remote-cuda-boundary-torch2.6-cu124-v1": (
        "opbench-v07-t0010",
    ),
}


class V07ValidationContractTests(unittest.TestCase):
    def test_frozen_public_identity_mapping_loads_fail_closed(self) -> None:
        aliases = load_public_task_id_aliases(
            ROOT / "factory/v0.7/p6/public_task_ids.json"
        )

        self.assertEqual(len(aliases), 25)
        self.assertEqual(
            {
                aliases[canonical_id]
                for canonical_id in (
                    "pytorch__117065__index_copy_zero_dim",
                    "pytorch__118762__weight_norm_default_dim",
                    "pytorch__126461__cummin_rank_zero",
                    "pytorch__139751__triton_ygrid_mask",
                    "pytorch__143792__addmv_empty_matrix",
                    "pytorch__147352__storage_offset_overflow",
                )
            },
            {
                task_id
                for task_ids in EXPECTED_COHORTS.values()
                for task_id in task_ids
            },
        )

    def test_contract_rebuilds_exact_five_cohort_partition(self) -> None:
        expected = build_validation_contract(ROOT)
        actual = load_mcp_experiment_contract(CONTRACT_PATH)

        self.assertEqual(actual, expected)
        self.assertEqual(actual.expected_attempt_count, 18)
        self.assertEqual(
            {
                cohort.profile_id: cohort.task_ids
                for cohort in actual.cohorts
            },
            EXPECTED_COHORTS,
        )
        self.assertEqual(
            {
                repeats
                for cohort in actual.cohorts
                for _, repeats in cohort.task_repeats
            },
            {(1, 2, 3)},
        )
        self.assertEqual(
            McpExperimentContract.from_dict(actual.to_dict()),
            actual,
        )
        assert_public_artifact_safe(actual.to_dict())
        flattened = repr(actual.to_dict())
        for private_task_id in (
            "pytorch__117065__index_copy_zero_dim",
            "pytorch__118762__weight_norm_default_dim",
            "pytorch__126461__cummin_rank_zero",
            "pytorch__139751__triton_ygrid_mask",
            "pytorch__143792__addmv_empty_matrix",
            "pytorch__147352__storage_offset_overflow",
        ):
            self.assertNotIn(private_task_id, flattened)


if __name__ == "__main__":
    unittest.main()
