from __future__ import annotations

import copy
import unittest

from op_bench.factory.complexity import (
    ComplexityEvidence,
    build_complexity_evidence,
    semantic_duplicate_fingerprint,
)
from op_bench.factory.taxonomy import parse_taxonomy_v2
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError


def accepted_pilot() -> dict[str, object]:
    return {
        "decision": "accepted",
        "counts_toward_final": False,
        "pilot_id": "pilot-v07-empty-addmv",
    }


def build_evidence(
    *,
    scores: tuple[int, int, int] = (2, 2, 1),
    hard_rejections: tuple[str, ...] = (),
    risk_signals: tuple[str, ...] = (),
    duplicate_decision: str = "distinct",
    blind_pilot: dict[str, object] | None = None,
    second_review: bool = False,
) -> ComplexityEvidence:
    return build_complexity_evidence(
        task_id="pytorch__empty_addmv",
        localization=scores[0],
        diagnosis=scores[1],
        repair_regression=scores[2],
        dimension_evidence={
            "localization": "The sanitized prompt requires tracing the public operation.",
            "diagnosis": "The observed difference requires comparing contract behavior.",
            "repair_regression": "Several repairs must preserve neighboring behavior.",
        },
        hard_rejections=hard_rejections,
        risk_signals=risk_signals,
        duplicate_fingerprint="sha256:" + "a" * 64,
        duplicate_decision=duplicate_decision,
        blind_pilot=blind_pilot,
        second_review=second_review,
        reviewer="complexity-reviewer",
        reviewed_at="2026-07-29T00:00:00Z",
    )


class ComplexityEvidenceTests(unittest.TestCase):
    def test_complexity_thresholds_and_labels(self) -> None:
        rejected = build_evidence(scores=(1, 1, 1))
        boundary = build_evidence(
            scores=(1, 1, 2),
            blind_pilot=accepted_pilot(),
            second_review=True,
        )
        hard = build_evidence(scores=(2, 2, 1))

        self.assertEqual((rejected.decision, rejected.difficulty), ("rejected", None))
        self.assertEqual((boundary.decision, boundary.difficulty), ("accepted", "medium"))
        self.assertEqual((hard.decision, hard.difficulty), ("accepted", "hard"))

    def test_hard_rejection_overrides_score_six(self) -> None:
        evidence = build_evidence(
            scores=(2, 2, 2),
            hard_rejections=("mechanical_after_sanitization",),
        )

        self.assertEqual(evidence.decision, "rejected")
        self.assertIsNone(evidence.difficulty)

    def test_score_four_requires_non_final_pilot_and_second_review(self) -> None:
        missing_review = build_evidence(
            scores=(2, 1, 1),
            blind_pilot=accepted_pilot(),
        )
        final_budget_pilot = build_evidence(
            scores=(2, 1, 1),
            blind_pilot={**accepted_pilot(), "counts_toward_final": True},
            second_review=True,
        )

        self.assertEqual(missing_review.decision, "deferred")
        self.assertEqual(final_budget_pilot.decision, "deferred")

    def test_score_four_artifact_cannot_claim_second_review_by_changing_decision(self) -> None:
        deferred = build_evidence(
            scores=(2, 1, 1),
            blind_pilot=accepted_pilot(),
            second_review=False,
        )
        payload = deferred.to_dict()
        payload["decision"] = "accepted"
        payload["difficulty"] = "medium"
        payload["content_hash"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_hash"}
        )

        with self.assertRaisesRegex(ContractError, "decision"):
            ComplexityEvidence.from_dict(payload)

    def test_second_review_is_a_required_boolean_wire_field(self) -> None:
        payload = build_evidence().to_dict()

        self.assertFalse(payload["second_review"])
        missing = copy.deepcopy(payload)
        del missing["second_review"]
        with self.assertRaisesRegex(ContractError, "second_review"):
            ComplexityEvidence.from_dict(missing)

        wrong_type = copy.deepcopy(payload)
        wrong_type["second_review"] = "false"
        wrong_type["content_hash"] = canonical_sha256(
            {key: value for key, value in wrong_type.items() if key != "content_hash"}
        )
        with self.assertRaisesRegex(ContractError, "second_review"):
            ComplexityEvidence.from_dict(wrong_type)

    def test_duplicate_decision_injects_hard_rejection(self) -> None:
        evidence = build_evidence(
            scores=(2, 2, 2),
            duplicate_decision="duplicate",
        )

        self.assertEqual(evidence.hard_rejections, ("semantic_duplicate",))
        self.assertEqual((evidence.decision, evidence.difficulty), ("rejected", None))

    def test_dimension_evidence_is_complete_and_non_empty(self) -> None:
        with self.assertRaisesRegex(ContractError, "dimension_evidence"):
            build_complexity_evidence(
                task_id="pytorch__empty_addmv",
                localization=2,
                diagnosis=2,
                repair_regression=1,
                dimension_evidence={
                    "localization": "written",
                    "diagnosis": "written",
                },
                hard_rejections=(),
                risk_signals=(),
                duplicate_fingerprint="sha256:" + "a" * 64,
                duplicate_decision="distinct",
                blind_pilot=None,
                second_review=False,
                reviewer="complexity-reviewer",
                reviewed_at="2026-07-29T00:00:00Z",
            )

    def test_round_trip_recomputes_derived_admission_decision(self) -> None:
        payload = build_evidence().to_dict()
        restored = ComplexityEvidence.from_dict(payload)
        self.assertEqual(restored, build_evidence())

        tampered = copy.deepcopy(payload)
        tampered["decision"] = "rejected"
        tampered["content_hash"] = canonical_sha256(
            {key: value for key, value in tampered.items() if key != "content_hash"}
        )
        with self.assertRaisesRegex(ContractError, "decision"):
            ComplexityEvidence.from_dict(tampered)

    def test_fingerprint_uses_only_normalized_behavior_side_facts(self) -> None:
        taxonomy = parse_taxonomy_v2(
            {
                "taxonomy_version": "v2",
                "contract_family": "result",
                "execution_context": {
                    "devices": ["cpu"],
                    "modes": ["eager"],
                    "phases": ["forward"],
                    "distributed": False,
                },
                "failure_type": "wrong_result",
                "trigger_tags": ["empty_or_zero"],
            }
        )

        fingerprint = semantic_duplicate_fingerprint(
            taxonomy,
            ("empty matrix", "eager reference", "empty matrix"),
        )

        self.assertEqual(
            fingerprint,
            canonical_sha256(
                {
                    "contract_family": "result",
                    "failure_type": "wrong_result",
                    "triggers": ["empty_or_zero"],
                    "contexts": {
                        "devices": ["cpu"],
                        "modes": ["eager"],
                        "phases": ["forward"],
                        "distributed": False,
                    },
                    "behavior_tokens": ["eager reference", "empty matrix"],
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
