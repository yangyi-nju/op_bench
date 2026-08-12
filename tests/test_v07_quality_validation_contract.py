from __future__ import annotations

import json
from pathlib import Path
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.experiment_report import (
    McpExperimentContract,
    load_mcp_experiment_contract,
)
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError
from scripts.build_v07_quality_validation_contract import (
    CODEX_CLI_VERSION,
    MODEL_ID,
    build_validation_contract,
    final_task_origins_by_public_task_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "factory/v0.7/p9/validation_contract.json"


class V07QualityValidationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = build_validation_contract(ROOT)
        cls.frozen = load_mcp_experiment_contract(CONTRACT_PATH)

    def test_contract_rebuilds_byte_for_byte(self) -> None:
        self.assertEqual(self.frozen, self.contract)
        self.assertEqual(
            CONTRACT_PATH.read_text(encoding="utf-8"),
            canonical_json(self.contract.to_dict()) + "\n",
        )
        self.assertEqual(McpExperimentContract.from_dict(self.frozen.to_dict()), self.frozen)

    def test_validation_budget_matches_task_origin(self) -> None:
        origins = final_task_origins_by_public_task_id(ROOT)
        seen = {
            task_id: repeats
            for cohort in self.contract.cohorts
            for task_id, repeats in cohort.task_repeats
        }
        self.assertEqual(len(origins), 50)
        self.assertEqual(set(seen), set(origins))
        self.assertEqual(self.contract.expected_attempt_count, 122)
        for task_id, origin in origins.items():
            expected = (1,) if origin == "retained_historical" else (1, 2, 3)
            self.assertEqual(seen[task_id], expected)

    def test_every_cohort_binds_exact_comparability_inputs(self) -> None:
        config = self.contract.frozen_config
        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.adapter_id, "codex_mcp_canonical")
        self.assertEqual(config.model_id, MODEL_ID)
        self.assertEqual(config.codex_cli_version, CODEX_CLI_VERSION)
        self.assertEqual(config.action_protocol, "action-v1")
        self.assertEqual(config.evaluation_protocol, "evaluation-v1")
        self.assertEqual(config.scoring_protocol, "scoring-v1")
        for cohort in self.contract.cohorts:
            self.assertIsNotNone(cohort.binding)
            assert cohort.binding is not None
            self.assertEqual(
                {task_id for task_id, _ in cohort.binding.task_view_digests},
                set(cohort.task_ids),
            )

    def test_contract_is_public_and_contains_only_opaque_task_ids(self) -> None:
        payload = self.contract.to_dict()
        assert_public_artifact_safe(payload)
        encoded = canonical_json(payload)
        self.assertNotIn("pytorch__", encoded)
        for cohort in self.contract.cohorts:
            for task_id in cohort.task_ids:
                self.assertRegex(task_id, r"^opbench-v07-t[0-9]{4}$")

    def test_bound_contract_rejects_missing_cohort_binding(self) -> None:
        payload = json.loads(canonical_json(self.contract.to_dict()))
        del payload["cohorts"][0]["binding"]
        with self.assertRaisesRegex(ContractError, "every frozen cohort"):
            McpExperimentContract.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
