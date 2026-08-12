from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
    FactoryEvidence,
)
from op_bench.factory.lifecycle import (
    FactoryTransitionRequest,
    advance_admission,
    validate_admission_chain,
)
from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.runtime.canonical import JsonValue, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_exact_fields,
    require_str,
)
from op_bench.task import TaskManifest


_STATES = (
    "discovered",
    "screened",
    "bundled",
    "preflight_passed",
    "baseline_reproduced",
    "gold_resolved",
    "reviewed",
    "verified",
)
_ADMISSION_FIELDS = (
    "schema_version",
    "evidence_id",
    "task_id",
    "task_manifest_hash",
    "task_manifest_hash_kind",
    "created_at",
    "source",
    "environment",
    "baseline",
    "gold",
    "admission",
)
_REVIEW_FIELDS = (
    "schema_version",
    "task_id",
    "decision",
    "root_cause_confirmed",
    "scope_confirmed",
    "tests_confirmed",
    "surrogate_confirmed",
    "reviewer",
    "reviewed_at",
)
_SOURCE_FIELDS = (
    "id",
    "repo_url",
    "base_commit",
    "snapshot_hash",
    "snapshot_method",
)
_ENVIRONMENT_FIELDS = (
    "id",
    "runtime_tier",
    "backend",
    "image",
    "image_digest",
    "digest_kind",
    "platform",
)
_EXECUTION_FIELDS = (
    "task_id",
    "mode",
    "status",
    "fail_to_pass_total",
    "fail_to_pass_passed",
    "pass_to_pass_total",
    "pass_to_pass_passed",
)


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    return {str(key): item for key, item in value.items()}


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path}: expected integer")
    return value


def _utc_timestamp(value: object, path: str) -> str:
    text = require_str(value, path)
    if not text.endswith("Z"):
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds") from exc
    if parsed.isoformat(timespec="seconds").replace("+00:00", "Z") != text:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    return text


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path)


def _execution(
    value: object,
    *,
    path: str,
    task: TaskManifest,
    baseline: bool,
) -> dict[str, object]:
    execution = _mapping(value, path)
    required = set(_EXECUTION_FIELDS)
    actual = set(execution)
    missing = sorted(required - actual)
    unknown = sorted(actual - required - {"duration_sec"})
    if missing:
        raise ContractError(f"{path}: missing fields {missing}")
    if unknown:
        raise ContractError(f"{path}: unknown fields {unknown}")
    if execution["task_id"] != task.task_id:
        raise ContractError(f"{path}.task_id: does not match Task")
    expected_mode = "baseline" if baseline else "gold"
    if execution["mode"] != expected_mode:
        raise ContractError(f"{path}.mode: expected {expected_mode!r}")
    expected_status = "baseline_reproduced" if baseline else "resolved"
    if execution["status"] != expected_status:
        raise ContractError(
            f"{path}.status: expected {expected_status!r}"
        )

    fail_total = _integer(
        execution["fail_to_pass_total"],
        f"{path}.fail_to_pass_total",
    )
    fail_passed = _integer(
        execution["fail_to_pass_passed"],
        f"{path}.fail_to_pass_passed",
    )
    pass_total = _integer(
        execution["pass_to_pass_total"],
        f"{path}.pass_to_pass_total",
    )
    pass_passed = _integer(
        execution["pass_to_pass_passed"],
        f"{path}.pass_to_pass_passed",
    )
    if fail_total != len(task.fail_to_pass_tests) or fail_total <= 0:
        raise ContractError(
            f"{path}.fail_to_pass_total: must match declared F2P selectors"
        )
    if pass_total != len(task.pass_to_pass_tests) or pass_total <= 0:
        raise ContractError(
            f"{path}.pass_to_pass_total: must match declared P2P selectors"
        )
    if not (0 <= fail_passed <= fail_total):
        raise ContractError(f"{path}.fail_to_pass_passed: out of range")
    if not (0 <= pass_passed <= pass_total):
        raise ContractError(f"{path}.pass_to_pass_passed: out of range")
    if baseline:
        if fail_passed != 0 or pass_passed != pass_total:
            raise ContractError(
                "test execution: baseline must fail every F2P and preserve P2P"
            )
    elif fail_passed != fail_total or pass_passed != pass_total:
        raise ContractError("test execution: Gold must pass every F2P and P2P")
    if "duration_sec" in execution:
        duration = execution["duration_sec"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
        ):
            raise ContractError(f"{path}.duration_sec: expected non-negative number")
        try:
            decimal_duration = Decimal(str(duration))
        except InvalidOperation as exc:
            raise ContractError(
                f"{path}.duration_sec: expected non-negative number"
            ) from exc
        if not decimal_duration.is_finite():
            raise ContractError(
                f"{path}.duration_sec: expected non-negative number"
            )
        execution.pop("duration_sec")
        execution["duration_ms"] = int(
            (decimal_duration * 1000).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )
    return execution


def _validate_inputs(
    *,
    candidate: CandidateRecord,
    decision: DecisionRecord,
    task: TaskManifest,
    admission: Mapping[str, object],
    review: Mapping[str, object],
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    if decision.disposition != "accepted":
        raise ContractError("decision: expected accepted disposition")
    if (
        decision.candidate_id != candidate.candidate_id
        or decision.candidate_content_hash != candidate.content_hash
    ):
        raise ContractError("decision: candidate identity or content mismatch")
    if (
        decision.target_dimension != candidate.proposed_dimension
        or decision.target_subclass != candidate.proposed_subclass
    ):
        raise ContractError("decision: taxonomy does not match Candidate")
    if task.admission_status != "verified":
        raise ContractError("task admission status: expected verified")
    if (
        task.problem_dimension != "boundary"
        or task.problem_dimension != candidate.proposed_dimension
        or task.problem_subclass != candidate.proposed_subclass
    ):
        raise ContractError("task taxonomy: does not match Candidate")
    operator = _mapping(task.data.get("operator"), "task.operator")
    if operator.get("framework") != candidate.framework:
        raise ContractError("task framework: does not match Candidate")
    if task.base_commit != candidate.base_commit:
        raise ContractError("task source: base commit does not match Candidate")
    task_source = _mapping(task.data.get("source"), "task.source")
    if task_source.get("repo") != candidate.repository:
        raise ContractError("task source: repository does not match Candidate")
    if (
        "pr_number" in task_source
        and task_source["pr_number"] != candidate.pr_number
    ):
        raise ContractError("task source: PR number does not match Candidate")

    admission_data = dict(
        require_exact_fields(
            admission,
            "admission evidence",
            _ADMISSION_FIELDS,
        )
    )
    if admission_data["schema_version"] != "v1":
        raise ContractError("admission evidence.schema_version: expected 'v1'")
    require_str(admission_data["evidence_id"], "admission evidence.evidence_id")
    if admission_data["task_id"] != task.task_id:
        raise ContractError("task_id mismatch: admission evidence")
    _utc_timestamp(
        admission_data["created_at"],
        "admission evidence.created_at",
    )
    if admission_data["task_manifest_hash_kind"] != REPLAY_SPEC_HASH_KIND:
        raise ContractError("replay hash mismatch: unsupported hash kind")
    if admission_data["task_manifest_hash"] != replay_spec_hash(task):
        raise ContractError("replay hash mismatch: admission evidence is stale")

    source = dict(
        require_exact_fields(
            admission_data["source"],
            "admission source",
            _SOURCE_FIELDS,
        )
    )
    for field in ("id", "repo_url", "base_commit"):
        require_str(source[field], f"admission source.{field}")
    _optional_string(source["snapshot_hash"], "admission source.snapshot_hash")
    _optional_string(
        source["snapshot_method"],
        "admission source.snapshot_method",
    )
    if (
        source["id"] != task.source_ref
        or source["repo_url"] != task.repo_url
        or source["base_commit"] != task.base_commit
        or source["snapshot_hash"] != task.source_snapshot_hash
        or source["snapshot_method"] != task.source_snapshot_method
    ):
        raise ContractError("source mismatch: admission evidence and Task")

    environment = dict(
        require_exact_fields(
            admission_data["environment"],
            "admission environment",
            _ENVIRONMENT_FIELDS,
        )
    )
    for field in ("id", "runtime_tier", "backend", "image"):
        require_str(environment[field], f"admission environment.{field}")
    for field in ("image_digest", "digest_kind", "platform"):
        _optional_string(
            environment[field],
            f"admission environment.{field}",
        )
    expected_environment = {
        "id": task.environment_ref,
        "runtime_tier": task.runtime_tier,
        "backend": task.environment_backend,
        "image": task.environment_image,
        "image_digest": task.environment_image_digest,
        "digest_kind": task.environment_digest_kind,
        "platform": task.environment_platform,
    }
    if environment != expected_environment:
        raise ContractError("environment mismatch: admission evidence and Task")

    baseline = _execution(
        admission_data["baseline"],
        path="baseline",
        task=task,
        baseline=True,
    )
    gold = _execution(
        admission_data["gold"],
        path="gold",
        task=task,
        baseline=False,
    )
    admission_data["baseline"] = baseline
    admission_data["gold"] = gold
    admission_decision = require_exact_fields(
        admission_data["admission"],
        "admission decision",
        ("decision", "verified", "failure_classification"),
    )
    if (
        admission_decision["decision"] != "verified"
        or require_bool(
            admission_decision["verified"],
            "admission decision.verified",
        )
        is not True
        or admission_decision["failure_classification"] is not None
    ):
        raise ContractError("admission decision: expected verified")

    review_data = dict(
        require_exact_fields(review, "review", _REVIEW_FIELDS)
    )
    if review_data["schema_version"] != "v1":
        raise ContractError("review.schema_version: expected 'v1'")
    if review_data["task_id"] != task.task_id:
        raise ContractError("review.task_id: does not match Task")
    if review_data["decision"] != "approved":
        raise ContractError("review.decision: expected 'approved'")
    for field in (
        "root_cause_confirmed",
        "scope_confirmed",
        "tests_confirmed",
    ):
        if require_bool(review_data[field], f"review.{field}") is not True:
            raise ContractError(f"review.{field}: expected true")
    surrogate = review_data["surrogate_confirmed"]
    if task.problem_subclass == "B5":
        if require_bool(surrogate, "review.surrogate_confirmed") is not True:
            raise ContractError(
                "review.surrogate_confirmed: B5 requires true"
            )
    elif surrogate is not None:
        raise ContractError(
            "review.surrogate_confirmed: expected null outside B5"
        )
    require_str(review_data["reviewer"], "review.reviewer")
    _utc_timestamp(review_data["reviewed_at"], "review.reviewed_at")
    return (
        admission_data,
        review_data,
        source,
        environment,
        baseline,
        gold,
    )


def _reference(
    *,
    artifact_type: str,
    artifact_id: str,
    relative_path: str,
    value: object,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash=canonical_sha256(value),
        relative_path=relative_path,
    )


def _evidence(
    evidence_type: str,
    reference: FactoryArtifactReference,
    claims: Mapping[str, JsonValue],
) -> FactoryEvidence:
    return FactoryEvidence(
        evidence_type=evidence_type,
        reference=reference,
        claims=claims,
    )


def _execution_claims(
    execution: Mapping[str, object],
    *,
    phase: str,
    source_hash: str,
    runtime_hash: str,
    selector_hash: str,
) -> dict[str, JsonValue]:
    status = execution.get("status")
    if phase == "baseline" and status == "baseline_reproduced":
        status = "failed_as_expected"
    return {
        "status": status,
        "source_hash": source_hash,
        "runtime_hash": runtime_hash,
        "selector_hash": selector_hash,
        "fail_to_pass_total": _integer(
            execution.get("fail_to_pass_total"),
            f"{phase}.fail_to_pass_total",
        ),
        "fail_to_pass_passed": _integer(
            execution.get("fail_to_pass_passed"),
            f"{phase}.fail_to_pass_passed",
        ),
        "pass_to_pass_total": _integer(
            execution.get("pass_to_pass_total"),
            f"{phase}.pass_to_pass_total",
        ),
        "pass_to_pass_passed": _integer(
            execution.get("pass_to_pass_passed"),
            f"{phase}.pass_to_pass_passed",
        ),
    }


def build_verified_admission_chain(
    *,
    candidate: CandidateRecord,
    decision: DecisionRecord,
    task: TaskManifest,
    admission: Mapping[str, object],
    review: Mapping[str, object],
    created_at: str,
) -> tuple[FactoryAdmissionRecord, ...]:
    if not isinstance(candidate, CandidateRecord):
        raise ContractError("candidate: expected CandidateRecord")
    if not isinstance(decision, DecisionRecord):
        raise ContractError("decision: expected DecisionRecord")
    if not isinstance(task, TaskManifest):
        raise ContractError("task: expected TaskManifest")

    (
        admission_data,
        review_data,
        source,
        environment,
        baseline,
        gold,
    ) = _validate_inputs(
        candidate=candidate,
        decision=decision,
        task=task,
        admission=admission,
        review=review,
    )

    source_hash = canonical_sha256(source)
    runtime_hash = canonical_sha256(environment)
    selector_hash = canonical_sha256(task.data["evaluation"])
    task_reference = _reference(
        artifact_type="task_bundle",
        artifact_id=f"task:{task.task_id}",
        relative_path=f"tasks/{task.task_id}/task.json",
        value=task.data,
    )
    decision_reference = FactoryArtifactReference(
        artifact_type="factory_decision",
        artifact_id=decision.decision_id,
        content_hash=decision.content_hash,
        relative_path=f"decisions/{decision.decision_id}.json",
    )
    source_reference = _reference(
        artifact_type="source",
        artifact_id=f"source:{task.task_id}",
        relative_path=f"evidence/{task.task_id}/source.json",
        value=source,
    )
    environment_reference = _reference(
        artifact_type="environment",
        artifact_id=f"environment:{task.task_id}",
        relative_path=f"evidence/{task.task_id}/environment.json",
        value=environment,
    )
    admission_reference = _reference(
        artifact_type="admission_evidence",
        artifact_id=str(
            admission_data.get("evidence_id", f"evidence:{task.task_id}")
        ),
        relative_path=f"evidence/{task.task_id}/admission.json",
        value=admission_data,
    )
    review_reference = _reference(
        artifact_type="human_review",
        artifact_id=f"review:{task.task_id}",
        relative_path=f"reviews/{task.task_id}.json",
        value=review_data,
    )
    integrity_payload = {
        "task_id": task.task_id,
        "task_hash": task_reference.content_hash,
        "admission_hash": admission_reference.content_hash,
        "review_hash": review_reference.content_hash,
    }
    integrity_reference = _reference(
        artifact_type="integrity",
        artifact_id=f"integrity:{task.task_id}",
        relative_path=f"integrity/{task.task_id}.json",
        value=integrity_payload,
    )

    stage_evidence = {
        "screened": _evidence(
            "screening_decision",
            decision_reference,
            {"status": decision.disposition},
        ),
        "bundled": _evidence(
            "task_bundle",
            task_reference,
            {"status": "complete", "task_id": task.task_id},
        ),
        "preflight_passed": _evidence(
            "preflight",
            environment_reference,
            {
                "status": "passed",
                "source_hash": source_hash,
                "runtime_hash": runtime_hash,
            },
        ),
        "baseline_reproduced": _evidence(
            "baseline",
            source_reference,
            _execution_claims(
                baseline,
                phase="baseline",
                source_hash=source_hash,
                runtime_hash=runtime_hash,
                selector_hash=selector_hash,
            ),
        ),
        "gold_resolved": _evidence(
            "gold",
            admission_reference,
            _execution_claims(
                gold,
                phase="gold",
                source_hash=source_hash,
                runtime_hash=runtime_hash,
                selector_hash=selector_hash,
            ),
        ),
        "reviewed": _evidence(
            "human_review",
            review_reference,
            {
                "status": review_data.get("decision"),
                "reviewer": review_data.get("reviewer"),
                "reviewed_at": review_data.get("reviewed_at"),
            },
        ),
        "verified": _evidence(
            "integrity",
            integrity_reference,
            {
                "status": "passed",
                "source_hash": source_hash,
                "runtime_hash": runtime_hash,
                "selector_hash": selector_hash,
            },
        ),
    }

    records: list[FactoryAdmissionRecord] = []
    previous: FactoryAdmissionRecord | None = None
    for state in _STATES:
        previous = advance_admission(
            previous,
            FactoryTransitionRequest(
                target_state=state,
                candidate=candidate,
                decision=None if state == "discovered" else decision,
                task=task_reference if state in _STATES[2:] else None,
                evidence=(
                    ()
                    if state == "discovered"
                    else (stage_evidence[state],)
                ),
                expected_previous_hash=(
                    None if previous is None else previous.content_hash
                ),
                transition_reason=f"Factory promotion advanced to {state}.",
                actor_kind=(
                    "human"
                    if state in ("reviewed", "verified")
                    else "automation"
                ),
                created_at=created_at,
            ),
        )
        records.append(previous)

    result = tuple(records)
    validate_admission_chain(result)
    return result
