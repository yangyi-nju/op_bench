from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from op_bench.factory.contracts import (
    FactoryAdmissionRecord,
    FactoryArtifactReference,
    FactoryEvidence,
)
from op_bench.factory.lifecycle import (
    FactoryTransitionRequest,
    advance_admission,
    required_evidence,
    validate_admission_chain,
)
from op_bench.factory.screening import screen_candidate
from op_bench.runtime.validation import ContractError
from tests.test_factory_contracts import SHA_A, candidate


ROOT = Path(__file__).resolve().parents[1]
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
VALID_PATH = (
    "discovered",
    "screened",
    "bundled",
    "preflight_passed",
    "baseline_reproduced",
    "gold_resolved",
    "reviewed",
    "verified",
)


def artifact(
    artifact_type: str,
    *,
    content_hash: str = SHA_A,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=f"{artifact_type}:fixture",
        content_hash=content_hash,
        relative_path=f"evidence/{artifact_type}.json",
    )


def evidence(
    evidence_type: str,
    *,
    selector_hash: str = SHA_C,
    status: str | None = None,
    extra_claims: dict[str, object] | None = None,
) -> FactoryEvidence:
    default_statuses = {
        "screening_decision": "accepted",
        "task_bundle": "complete",
        "preflight": "passed",
        "baseline": "failed_as_expected",
        "gold": "resolved",
        "human_review": "approved",
        "integrity": "passed",
    }
    claims: dict[str, object] = {
        "status": status or default_statuses[evidence_type]
    }
    if evidence_type in ("baseline", "gold"):
        claims.update(
            {
                "source_hash": SHA_A,
                "runtime_hash": SHA_B,
                "selector_hash": selector_hash,
            }
        )
    if extra_claims:
        claims.update(extra_claims)
    return FactoryEvidence(
        evidence_type=evidence_type,
        reference=artifact(evidence_type),
        claims=claims,
    )


def task_reference() -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type="task_bundle",
        artifact_id="task:pytorch__fixture",
        content_hash=SHA_A,
        relative_path="tasks/pytorch__fixture/task.json",
    )


def stage_evidence(state: str) -> tuple[FactoryEvidence, ...]:
    mapping = {
        "screened": evidence("screening_decision"),
        "bundled": evidence("task_bundle"),
        "preflight_passed": evidence("preflight"),
        "baseline_reproduced": evidence("baseline"),
        "gold_resolved": evidence("gold"),
        "reviewed": evidence("human_review"),
        "verified": evidence("integrity"),
    }
    selected = mapping.get(state)
    return () if selected is None else (selected,)


def transition(
    state: str,
    *,
    previous: FactoryAdmissionRecord | None,
    selected_candidate=None,
    decision=None,
    supplied_evidence: tuple[FactoryEvidence, ...] | None = None,
    task: FactoryArtifactReference | None = None,
    actor_kind: str | None = None,
) -> FactoryTransitionRequest:
    selected_candidate = selected_candidate or candidate()
    if decision is None and state != "discovered":
        decision = screen_candidate(selected_candidate)
    if task is None and state in VALID_PATH[2:]:
        task = task_reference()
    return FactoryTransitionRequest(
        target_state=state,
        candidate=selected_candidate,
        decision=decision,
        task=task,
        evidence=(
            stage_evidence(state)
            if supplied_evidence is None
            else supplied_evidence
        ),
        expected_previous_hash=(
            None if previous is None else previous.content_hash
        ),
        transition_reason=f"Advance fixture to {state}.",
        actor_kind=(
            actor_kind
            or ("human" if state in ("reviewed", "verified") else "automation")
        ),
        created_at="2026-07-26T00:00:00Z",
    )


def admission_at(
    target_state: str,
    *,
    omit: tuple[str, ...] = (),
) -> FactoryAdmissionRecord:
    previous = None
    for state in VALID_PATH:
        supplied = tuple(
            item
            for item in stage_evidence(state)
            if item.evidence_type not in omit
        )
        previous = advance_admission(
            previous,
            transition(
                state,
                previous=previous,
                supplied_evidence=supplied,
            ),
        )
        if state == target_state:
            return previous
    raise AssertionError(f"unknown target state {target_state!r}")


class FactoryLifecycleTests(unittest.TestCase):
    def test_complete_happy_path_forms_a_valid_immutable_chain(self) -> None:
        records: list[FactoryAdmissionRecord] = []
        previous = None

        for state in VALID_PATH:
            current = advance_admission(
                previous,
                transition(state, previous=previous),
            )
            records.append(current)
            previous = current

        self.assertEqual(tuple(record.state for record in records), VALID_PATH)
        self.assertEqual(
            tuple(item.evidence_type for item in records[-1].evidence),
            (
                "baseline",
                "gold",
                "human_review",
                "integrity",
                "preflight",
                "screening_decision",
                "task_bundle",
            ),
        )
        validate_admission_chain(records)

    def test_required_evidence_is_cumulative(self) -> None:
        self.assertEqual(required_evidence("discovered"), ())
        self.assertEqual(
            required_evidence("verified"),
            (
                "screening_decision",
                "task_bundle",
                "preflight",
                "baseline",
                "gold",
                "human_review",
                "integrity",
            ),
        )

    def test_skipped_state_is_rejected(self) -> None:
        discovered = admission_at("discovered")

        with self.assertRaisesRegex(ContractError, "transition"):
            advance_admission(
                discovered,
                transition("bundled", previous=discovered),
            )

    def test_terminal_rejected_and_deprecated_states_cannot_advance(self) -> None:
        for terminal in ("rejected", "deprecated"):
            with self.subTest(terminal=terminal):
                discovered = admission_at("discovered")
                terminal_record = advance_admission(
                    discovered,
                    transition(
                        terminal,
                        previous=discovered,
                        supplied_evidence=(),
                    ),
                )
                with self.assertRaisesRegex(ContractError, "terminal"):
                    advance_admission(
                        terminal_record,
                        transition("discovered", previous=terminal_record),
                    )

    def test_deferred_reassessment_requires_resolution_and_new_capture(self) -> None:
        discovered = admission_at("discovered")
        deferred = advance_admission(
            discovered,
            transition(
                "deferred",
                previous=discovered,
                supplied_evidence=(),
            ),
        )
        refreshed = replace(
            candidate(),
            created_at="2026-07-26T00:00:01Z",
        )

        with self.assertRaisesRegex(ContractError, "resolution"):
            advance_admission(
                deferred,
                transition(
                    "discovered",
                    previous=deferred,
                    selected_candidate=refreshed,
                    supplied_evidence=(),
                ),
            )

        reassessed = advance_admission(
            deferred,
            transition(
                "discovered",
                previous=deferred,
                selected_candidate=refreshed,
                supplied_evidence=(
                    evidence(
                        "human_review",
                        extra_claims={"resolves_deferral": True},
                    ),
                ),
                actor_kind="human",
            ),
        )
        self.assertEqual(reassessed.candidate.content_hash, refreshed.content_hash)
        self.assertEqual(reassessed.state, "discovered")

    def test_previous_record_hash_mismatch_is_rejected(self) -> None:
        discovered = admission_at("discovered")
        request = replace(
            transition("screened", previous=discovered),
            expected_previous_hash=SHA_A,
        )

        with self.assertRaisesRegex(ContractError, "previous"):
            advance_admission(discovered, request)

    def test_screened_requires_an_accepted_matching_decision(self) -> None:
        rejected_candidate = replace(
            candidate(),
            title="Revert boundary fix",
        )
        rejected_decision = screen_candidate(rejected_candidate)
        discovered = advance_admission(
            None,
            transition(
                "discovered",
                previous=None,
                selected_candidate=rejected_candidate,
            ),
        )

        with self.assertRaisesRegex(ContractError, "accepted"):
            advance_admission(
                discovered,
                transition(
                    "screened",
                    previous=discovered,
                    selected_candidate=rejected_candidate,
                    decision=rejected_decision,
                ),
            )

    def test_gold_requires_matching_source_runtime_and_selectors(self) -> None:
        previous = admission_at("baseline_reproduced")

        with self.assertRaisesRegex(ContractError, "selector"):
            advance_admission(
                previous,
                transition(
                    "gold_resolved",
                    previous=previous,
                    supplied_evidence=(
                        evidence("gold", selector_hash=SHA_A),
                    ),
                ),
            )

    def test_verified_requires_all_stage_evidence_and_integrity(self) -> None:
        previous = admission_at("reviewed")

        with self.assertRaisesRegex(ContractError, "integrity"):
            advance_admission(
                previous,
                transition(
                    "verified",
                    previous=previous,
                    supplied_evidence=(),
                ),
            )

    def test_admission_round_trip_and_schema_field_parity(self) -> None:
        selected = admission_at("verified")

        self.assertEqual(
            FactoryAdmissionRecord.from_dict(selected.to_dict()),
            selected,
        )
        schema = json.loads(
            (ROOT / "schemas" / "factory_admission.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(schema["required"]),
            set(FactoryAdmissionRecord.wire_fields()),
        )
        self.assertEqual(
            set(schema["properties"]),
            set(FactoryAdmissionRecord.wire_fields()),
        )


if __name__ == "__main__":
    unittest.main()
