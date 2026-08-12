from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from scripts.run_v07_quality_validation import (
    _bounded_process_failure_code,
    _bounded_process_diagnostic,
    _release_inputs,
    cohort_executions,
    recommended_canaries,
    validate_quality_validation_index,
)


ROOT = Path(__file__).resolve().parents[1]


class V07QualityValidationRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.executions = cohort_executions(ROOT)
        _, records = _release_inputs(
            ROOT,
            ROOT / "factory/v0.7/p9/release_manifest.json",
        )
        cls.records = records
        cls.origins = {
            task_id: str(record["origin"])
            for task_id, record in records.items()
        }

    def test_execution_plan_reconstructs_all_frozen_attempts(self) -> None:
        self.assertEqual(len(self.executions), 17)
        self.assertEqual(
            sum(len(execution.manifest.expected_attempts) for execution in self.executions),
            122,
        )
        self.assertEqual(
            {
                task_id
                for execution in self.executions
                for task_id in execution.contract.task_ids
            },
            set(self.origins),
        )
        for execution in self.executions:
            self.assertEqual(
                execution.manifest.capability_policy.policy_id,
                "opbench-v0.7-repository-root-v1",
            )
            self.assertEqual(
                execution.manifest.capability_policy.writable_paths,
                (".",),
            )

    def test_canaries_cover_retained_cpu_new_cpu_and_two_cuda_paths(self) -> None:
        canaries = recommended_canaries(self.executions, self.records)
        self.assertEqual(len(canaries), 4)
        selected_profiles = {
            task_id: execution.contract.profile_id
            for execution in self.executions
            for task_id in execution.contract.task_ids
            if task_id in canaries
        }
        self.assertTrue(
            any(
                self.origins[task_id] == "retained_historical"
                and profile == "remote-cpu-pytorch-2.6-py311-v1"
                for task_id, profile in selected_profiles.items()
            )
        )
        self.assertTrue(
            any(
                self.origins[task_id] != "retained_historical"
                and "compile" in self.records[task_id]["taxonomy"]["modes"]
                and profile.startswith("remote-cpu-")
                for task_id, profile in selected_profiles.items()
            )
        )
        self.assertTrue(any("cuda-overlay" in profile for profile in selected_profiles.values()))
        self.assertTrue(any("cuda-kernel" in profile for profile in selected_profiles.values()))

    def test_incomplete_execution_index_is_rejected(self) -> None:
        payload = {
            "contract_type": "quality_validation_execution_index",
            "schema_version": "v1",
            "release_version": "v0.7",
            "created_at": "1970-01-01T00:00:00Z",
            "contract_path": "factory/v0.7/p9/validation_contract.json",
            "contract_digest": "sha256:" + "0" * 64,
            "cohort_count": 17,
            "expected_attempt_count": 122,
            "completed_cohort_count": 0,
            "valid_attempt_count": 0,
            "records": [],
        }
        payload["content_hash"] = canonical_sha256(payload)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            errors = validate_quality_validation_index(ROOT, path)
        self.assertEqual(len(errors), 1)
        self.assertIn("final counters", errors[0])

    def test_owned_process_errors_are_bounded_without_retaining_output(self) -> None:
        self.assertEqual(
            _bounded_process_failure_code(
                b"prefix: v1 orchestration failed before a valid run result\n"
            ),
            "orchestration_failed",
        )
        self.assertIsNone(_bounded_process_failure_code(b"untrusted arbitrary output"))
        diagnostic = _bounded_process_diagnostic(
            b"v1 orchestration failed [error_type=ContractError, "
            b"error_digest=sha256:" + b"a" * 64 + b"]\n"
        )
        self.assertEqual(
            diagnostic,
            {
                "error_type": "ContractError",
                "error_digest": "sha256:" + "a" * 64,
            },
        )


if __name__ == "__main__":
    unittest.main()
