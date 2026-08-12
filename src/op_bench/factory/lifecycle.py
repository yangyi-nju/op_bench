from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from op_bench.factory.contracts import (
    ADMISSION_STAGE_EVIDENCE,
    FACTORY_ACTOR_KINDS,
    FACTORY_ADMISSION_STATES,
    CandidateRecord,
    DecisionRecord,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
    FactoryEvidence,
)
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_str,
)


_MAIN_NEXT_STATE = {
    "discovered": "screened",
    "screened": "bundled",
    "bundled": "preflight_passed",
    "preflight_passed": "baseline_reproduced",
    "baseline_reproduced": "gold_resolved",
    "gold_resolved": "reviewed",
    "reviewed": "verified",
}
_TERMINAL_STATES = ("rejected", "deprecated")


@dataclass(frozen=True)
class FactoryTransitionRequest:
    target_state: str
    candidate: CandidateRecord
    decision: DecisionRecord | None
    task: FactoryArtifactReference | None
    evidence: tuple[FactoryEvidence, ...]
    expected_previous_hash: str | None
    transition_reason: str
    actor_kind: str
    created_at: str

    def __post_init__(self) -> None:
        require_enum(
            self.target_state,
            "target_state",
            FACTORY_ADMISSION_STATES,
        )
        if not isinstance(self.candidate, CandidateRecord):
            raise ContractError("candidate: expected CandidateRecord")
        if self.decision is not None and not isinstance(
            self.decision,
            DecisionRecord,
        ):
            raise ContractError("decision: expected DecisionRecord or null")
        if self.task is not None and not isinstance(
            self.task,
            FactoryArtifactReference,
        ):
            raise ContractError("task: expected FactoryArtifactReference or null")
        if not isinstance(self.evidence, tuple):
            raise ContractError("evidence: expected tuple")
        for index, item in enumerate(self.evidence):
            if not isinstance(item, FactoryEvidence):
                raise ContractError(
                    f"evidence[{index}]: expected FactoryEvidence"
                )
        if self.expected_previous_hash is not None:
            require_str(self.expected_previous_hash, "expected_previous_hash")
        require_str(self.transition_reason, "transition_reason")
        require_enum(self.actor_kind, "actor_kind", FACTORY_ACTOR_KINDS)
        require_str(self.created_at, "created_at")


def required_evidence(state: str) -> tuple[str, ...]:
    require_enum(state, "state", FACTORY_ADMISSION_STATES)
    return ADMISSION_STAGE_EVIDENCE.get(state, ())


def _candidate_reference(candidate: CandidateRecord) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type="factory_candidate",
        artifact_id=candidate.candidate_id,
        content_hash=candidate.content_hash,
        relative_path=f"candidates/{candidate.candidate_id}.json",
    )


def _decision_reference(decision: DecisionRecord) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type="factory_decision",
        artifact_id=decision.decision_id,
        content_hash=decision.content_hash,
        relative_path=f"decisions/{decision.decision_id}.json",
    )


def _validate_edge(previous_state: str | None, target_state: str) -> None:
    if previous_state is None:
        if target_state != "discovered":
            raise ContractError(
                "transition: initial Admission state must be 'discovered'"
            )
        return
    if previous_state in _TERMINAL_STATES:
        raise ContractError(
            f"transition: terminal state {previous_state!r} cannot advance"
        )
    if previous_state == "verified":
        if target_state != "deprecated":
            raise ContractError(
                "transition: verified may only advance to deprecated"
            )
        return
    if previous_state == "deferred":
        if target_state != "discovered":
            raise ContractError(
                "transition: deferred may only be reassessed as discovered"
            )
        return
    allowed = {_MAIN_NEXT_STATE.get(previous_state), *_TERMINAL_STATES}
    if previous_state == "discovered":
        allowed.add("deferred")
    if target_state not in allowed:
        raise ContractError(
            f"transition: cannot advance from {previous_state!r} to {target_state!r}"
        )


def advance_admission(
    previous: FactoryAdmissionRecord | None,
    request: FactoryTransitionRequest,
) -> FactoryAdmissionRecord:
    if previous is not None and not isinstance(previous, FactoryAdmissionRecord):
        raise ContractError("previous: expected FactoryAdmissionRecord or null")
    if not isinstance(request, FactoryTransitionRequest):
        raise ContractError("request: expected FactoryTransitionRequest")

    previous_hash = None if previous is None else previous.content_hash
    if request.expected_previous_hash != previous_hash:
        raise ContractError("previous record hash does not match transition request")
    previous_state = None if previous is None else previous.state
    _validate_edge(previous_state, request.target_state)

    candidate_ref = _candidate_reference(request.candidate)
    if previous is not None:
        if candidate_ref.artifact_id != previous.candidate.artifact_id:
            raise ContractError("candidate: stable candidate identity changed")
        if (
            previous.state != "deferred"
            and candidate_ref.content_hash != previous.candidate.content_hash
        ):
            raise ContractError("candidate: content changed outside reassessment")

    if request.decision is not None:
        if request.decision.candidate_id != request.candidate.candidate_id:
            raise ContractError("decision: candidate identity does not match")
        if (
            request.decision.candidate_content_hash
            != request.candidate.content_hash
        ):
            raise ContractError("decision: candidate content hash does not match")
        decision_ref = _decision_reference(request.decision)
    else:
        decision_ref = None if previous is None else previous.decision

    if request.target_state not in ("discovered", "deferred"):
        if request.decision is None:
            raise ContractError(
                f"decision: required for state {request.target_state!r}"
            )
        if request.decision.disposition != "accepted":
            raise ContractError("decision: screened state requires accepted Decision")
    if previous is not None and previous.decision is not None and decision_ref is not None:
        if decision_ref.content_hash != previous.decision.content_hash:
            prior = request.decision.prior_decision if request.decision else None
            if (
                request.decision is None
                or request.decision.decision_source != "human_review"
                or prior is None
                or prior.content_hash != previous.decision.content_hash
            ):
                raise ContractError(
                    "decision: replacement must reference the prior Decision"
                )

    task = request.task if request.task is not None else (
        None if previous is None else previous.task
    )
    if previous is not None and previous.task is not None and task is not None:
        if task != previous.task:
            raise ContractError("task: immutable Task reference changed")

    evidence_by_type = (
        {}
        if previous is None
        else {item.evidence_type: item for item in previous.evidence}
    )
    for item in request.evidence:
        evidence_by_type[item.evidence_type] = item
    accumulated_evidence = tuple(
        evidence_by_type[key] for key in sorted(evidence_by_type)
    )

    if previous_state == "deferred" and request.target_state == "discovered":
        resolution = evidence_by_type.get("human_review")
        if (
            request.actor_kind != "human"
            or resolution is None
            or resolution.claims.get("resolves_deferral") is not True
            or candidate_ref.content_hash == previous.candidate.content_hash
        ):
            raise ContractError(
                "resolution: deferred reassessment requires a new capture and human evidence"
            )

    admission_id = FactoryAdmissionRecord.admission_id_for(
        candidate=candidate_ref,
        decision=decision_ref,
        task=task,
        state=request.target_state,
        previous_record_hash=previous_hash,
        evidence=accumulated_evidence,
    )
    return FactoryAdmissionRecord(
        admission_id=admission_id,
        candidate=candidate_ref,
        decision=decision_ref,
        task=task,
        state=request.target_state,
        previous_record_hash=previous_hash,
        evidence=accumulated_evidence,
        transition_reason=request.transition_reason,
        actor_kind=request.actor_kind,
        created_at=request.created_at,
    )


def validate_admission_chain(
    records: Sequence[FactoryAdmissionRecord],
) -> None:
    if not records:
        raise ContractError("admission chain: expected at least one record")
    previous: FactoryAdmissionRecord | None = None
    for index, record in enumerate(records):
        if not isinstance(record, FactoryAdmissionRecord):
            raise ContractError(
                f"admission chain[{index}]: expected FactoryAdmissionRecord"
            )
        expected_hash = None if previous is None else previous.content_hash
        if record.previous_record_hash != expected_hash:
            raise ContractError(
                f"admission chain[{index}]: previous record hash mismatch"
            )
        _validate_edge(None if previous is None else previous.state, record.state)
        if previous is not None:
            if record.candidate.artifact_id != previous.candidate.artifact_id:
                raise ContractError(
                    f"admission chain[{index}]: candidate identity changed"
                )
            previous_types = {item.evidence_type for item in previous.evidence}
            current_types = {item.evidence_type for item in record.evidence}
            if not previous_types.issubset(current_types):
                raise ContractError(
                    f"admission chain[{index}]: accumulated evidence was removed"
                )
        previous = record
