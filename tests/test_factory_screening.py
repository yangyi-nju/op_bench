from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from op_bench.factory.contracts import (
    CandidateRecord,
    ChangedFile,
    DecisionRecord,
    FactoryArtifactReference,
    ScreeningFinding,
    factory_content_hash,
)
from op_bench.factory.screening import (
    V07_BOUNDARY_SCREENING_V1,
    derive_disposition,
    screen_candidate,
)
from op_bench.runtime.validation import ContractError
from tests.test_factory_contracts import SHA_A, candidate


ROOT = Path(__file__).resolve().parents[1]


def finding(severity: str, code: str = "review.root_cause_required") -> ScreeningFinding:
    return ScreeningFinding(
        code=code,
        severity=severity,
        rule_id="v07-root-cause-review",
        message="Root-cause classification requires human review.",
        observed=False,
        expected={"reviewed": True},
    )


def external_test_reference() -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type="external_test",
        artifact_id="test:pytorch/pytorch#170001",
        content_hash=SHA_A,
        relative_path="raw/external-test.json",
    )


def environment_freeze_reference() -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type="environment_freeze",
        artifact_id="runtime:pytorch-vintage",
        content_hash=SHA_A,
        relative_path="raw/environment-freeze.json",
    )


def changed_candidate(
    *,
    file_count: int,
    changed_lines: int,
    has_test: bool = True,
) -> CandidateRecord:
    files = tuple(
        ChangedFile(
            path=(
                f"test/test_boundary_{index}.py"
                if has_test and index == file_count - 1
                else f"aten/src/ATen/native/Boundary{index}.cpp"
            ),
            additions=changed_lines if index == 0 else 0,
            deletions=0,
            is_test=has_test and index == file_count - 1,
        )
        for index in range(file_count)
    )
    return replace(
        candidate(),
        changed_files=files,
        total_files=file_count,
        total_changed_lines=changed_lines,
    )


class DecisionContractTests(unittest.TestCase):
    def test_disposition_uses_highest_finding_severity(self) -> None:
        self.assertEqual(derive_disposition(()), "accepted")
        self.assertEqual(derive_disposition((finding("warning"),)), "accepted")
        self.assertEqual(derive_disposition((finding("defer"),)), "deferred")
        self.assertEqual(
            derive_disposition((finding("defer"), finding("reject"))),
            "rejected",
        )

    def test_decision_round_trip_is_exact(self) -> None:
        decision = screen_candidate(candidate())

        self.assertEqual(DecisionRecord.from_dict(decision.to_dict()), decision)
        self.assertEqual(
            DecisionRecord.from_dict(decision.to_dict()).to_dict(),
            decision.to_dict(),
        )

    def test_decision_rejects_stored_disposition_drift(self) -> None:
        decision = screen_candidate(
            replace(candidate(), title="Revert empty reduction fix")
        )
        payload = decision.to_dict()
        payload["disposition"] = "accepted"
        payload["content_hash"] = factory_content_hash(payload)

        with self.assertRaisesRegex(ContractError, "disposition"):
            DecisionRecord.from_dict(payload)

    def test_decision_schema_required_fields_match_wire_contract(self) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "factory_decision.schema.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(DecisionRecord.wire_fields()))
        self.assertEqual(set(schema["properties"]), set(DecisionRecord.wire_fields()))


class DeterministicScreeningTests(unittest.TestCase):
    def test_clear_candidate_is_accepted_pending_root_cause_review(self) -> None:
        result = screen_candidate(candidate())

        self.assertEqual(result.disposition, "accepted")
        self.assertEqual(
            tuple((item.code, item.severity) for item in result.findings),
            (("review.root_cause_required", "warning"),),
        )

    def test_revert_or_reland_is_rejected(self) -> None:
        for title in ("Revert empty reduction fix", "Reland zero-size support"):
            with self.subTest(title=title):
                result = screen_candidate(replace(candidate(), title=title))
                self.assertEqual(result.disposition, "rejected")
                self.assertEqual(
                    tuple((item.code, item.severity) for item in result.findings),
                    (
                        ("metadata.revert_or_reland", "reject"),
                        ("review.root_cause_required", "warning"),
                    ),
                )

    def test_missing_dates_are_deferred(self) -> None:
        result = screen_candidate(
            replace(candidate(), author_date=None, merge_date=None)
        )

        self.assertEqual(result.disposition, "deferred")
        self.assertEqual(
            tuple((item.code, item.severity) for item in result.findings),
            (
                ("metadata.missing_date", "defer"),
                ("review.root_cause_required", "warning"),
            ),
        )

    def test_outside_window_requires_environment_freeze(self) -> None:
        outside = replace(candidate(), author_date="2025-05-01T00:00:00Z")

        rejected = screen_candidate(outside)
        accepted = screen_candidate(
            replace(outside, environment_freeze=environment_freeze_reference())
        )

        self.assertEqual(rejected.disposition, "rejected")
        self.assertEqual(
            tuple(item.code for item in rejected.findings),
            ("window.outside_stable", "review.root_cause_required"),
        )
        self.assertEqual(accepted.disposition, "accepted")
        self.assertEqual(
            tuple(item.code for item in accepted.findings),
            (
                "review.root_cause_required",
                "window.environment_freeze_exception",
            ),
        )

    def test_refactor_cleanup_and_feature_only_changes_are_rejected(self) -> None:
        for change_kind in ("refactor", "cleanup", "feature"):
            with self.subTest(change_kind=change_kind):
                result = screen_candidate(
                    replace(candidate(), change_kind=change_kind)
                )
                self.assertEqual(result.disposition, "rejected")
                self.assertEqual(
                    tuple(item.code for item in result.findings),
                    (
                        "change.non_bug_change",
                        "review.root_cause_required",
                    ),
                )

    def test_large_diff_is_deferred_for_manual_exception(self) -> None:
        result = screen_candidate(
            changed_candidate(file_count=4, changed_lines=201)
        )

        self.assertEqual(result.disposition, "deferred")
        self.assertEqual(
            tuple(item.code for item in result.findings),
            (
                "change.large_diff_requires_review",
                "review.root_cause_required",
            ),
        )

    def test_tiny_change_without_test_is_rejected_by_both_hard_rules(self) -> None:
        result = screen_candidate(
            changed_candidate(file_count=1, changed_lines=5, has_test=False)
        )

        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(
            tuple(item.code for item in result.findings),
            (
                "change.below_behavioral_threshold",
                "test.missing_executable_delta",
                "review.root_cause_required",
            ),
        )

    def test_external_test_reference_satisfies_test_delta_rule(self) -> None:
        selected = changed_candidate(
            file_count=1,
            changed_lines=30,
            has_test=False,
        )

        missing = screen_candidate(selected)
        present = screen_candidate(
            replace(selected, external_test=external_test_reference())
        )

        self.assertEqual(missing.disposition, "rejected")
        self.assertIn(
            "test.missing_executable_delta",
            tuple(item.code for item in missing.findings),
        )
        self.assertEqual(present.disposition, "accepted")

    def test_missing_commit_identities_are_deferred(self) -> None:
        selected = candidate()
        missing = replace(
            selected,
            candidate_id=CandidateRecord.candidate_id_for(
                repository=selected.repository,
                pr_number=selected.pr_number,
                base_commit=None,
                merge_commit=None,
            ),
            base_commit=None,
            merge_commit=None,
        )

        result = screen_candidate(missing)

        self.assertEqual(result.disposition, "deferred")
        self.assertEqual(
            tuple(item.code for item in result.findings),
            (
                "source.missing_commit_identity",
                "review.root_cause_required",
            ),
        )

    def test_unavailable_source_and_runtime_are_deferred(self) -> None:
        result = screen_candidate(
            replace(
                candidate(),
                source_available=False,
                runtime_supported=False,
            )
        )

        self.assertEqual(result.disposition, "deferred")
        self.assertEqual(
            tuple(item.code for item in result.findings),
            (
                "runtime.unsupported",
                "source.unavailable",
                "review.root_cause_required",
            ),
        )

    def test_findings_have_deterministic_severity_then_code_order(self) -> None:
        selected = changed_candidate(
            file_count=4,
            changed_lines=5,
            has_test=False,
        )
        selected = replace(
            selected,
            title="Revert boundary feature",
            change_kind="feature",
            source_available=False,
            runtime_supported=False,
        )

        result = screen_candidate(selected)

        self.assertEqual(
            tuple((item.severity, item.code) for item in result.findings),
            (
                ("reject", "change.below_behavioral_threshold"),
                ("reject", "change.non_bug_change"),
                ("reject", "metadata.revert_or_reland"),
                ("reject", "test.missing_executable_delta"),
                ("defer", "change.large_diff_requires_review"),
                ("defer", "runtime.unsupported"),
                ("defer", "source.unavailable"),
                ("warning", "review.root_cause_required"),
            ),
        )

    def test_rule_set_identity_changes_when_a_threshold_changes(self) -> None:
        changed = replace(V07_BOUNDARY_SCREENING_V1, max_changed_lines=201)

        self.assertNotEqual(
            changed.rule_set_hash,
            V07_BOUNDARY_SCREENING_V1.rule_set_hash,
        )
        self.assertNotEqual(
            changed.rule_set_id,
            V07_BOUNDARY_SCREENING_V1.rule_set_id,
        )


if __name__ == "__main__":
    unittest.main()
