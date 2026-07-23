from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Protocol

from op_bench.runtime.contracts import (
    ContentIdentity,
    EvaluationResultV06,
    EvaluationSpec,
    SCHEMA_VERSION,
    SHA256_PATTERN,
    SessionResult,
    TestExecutionSummary,
)
from op_bench.runtime.events import EventJournal
from op_bench.runtime.session import termination_attribution
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
)
from op_bench.runtime.workspace import (
    FrozenPatch,
    PatchArtifact,
    assert_patch_identity_handoff,
)


_SELECTOR_GROUPS = ("fail_to_pass", "pass_to_pass")
_SESSION_INVALID_REASONS = {
    "workspace_error": "session_workspace_error",
    "runtime_error": "session_runtime_error",
    "provider_error": "session_provider_error",
    "platform_error": "session_platform_error",
}


class StrictPatchApplyError(Exception):
    """The immutable Agent patch does not strictly apply to the verified base."""


class EvaluationInfrastructureError(Exception):
    """A stable evaluator-owned infrastructure failure."""

    def __init__(self, invalid_reason: str, private_message: str = "") -> None:
        self.invalid_reason = require_str(invalid_reason, "invalid_reason")
        self.private_message = require_str(
            private_message, "private_message", min_length=0
        )
        super().__init__(self.private_message or self.invalid_reason)


@dataclass(frozen=True)
class SelectorExecution:
    selector_id: str
    group: str
    command_digest: str
    exit_code: int | None
    timed_out: bool
    summary: TestExecutionSummary
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        require_str(self.selector_id, "selector_id")
        if self.group not in _SELECTOR_GROUPS:
            raise ContractError(f"group: unsupported value {self.group!r}")
        require_str(self.command_digest, "command_digest", pattern=SHA256_PATTERN)
        if self.exit_code is not None:
            require_int(self.exit_code, "exit_code")
        require_bool(self.timed_out, "timed_out")
        if not isinstance(self.summary, TestExecutionSummary):
            raise ContractError("summary: expected TestExecutionSummary")
        require_str(self.stdout, "stdout", min_length=0)
        require_str(self.stderr, "stderr", min_length=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": "selector_execution",
            "schema_version": SCHEMA_VERSION,
            "selector_id": self.selector_id,
            "group": self.group,
            "command_digest": self.command_digest,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "summary": self.summary.to_dict(),
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SelectorExecution":
        data = require_exact_fields(
            value,
            "selector_execution",
            (
                "record_type",
                "schema_version",
                "selector_id",
                "group",
                "command_digest",
                "exit_code",
                "timed_out",
                "summary",
                "stdout",
                "stderr",
            ),
        )
        if data["record_type"] != "selector_execution":
            raise ContractError("record_type: expected 'selector_execution'")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
        exit_code = data["exit_code"]
        if exit_code is not None:
            exit_code = require_int(exit_code, "exit_code")
        return cls(
            selector_id=require_str(data["selector_id"], "selector_id"),
            group=require_str(data["group"], "group"),
            command_digest=require_str(
                data["command_digest"], "command_digest", pattern=SHA256_PATTERN
            ),
            exit_code=exit_code,
            timed_out=require_bool(data["timed_out"], "timed_out"),
            summary=TestExecutionSummary.from_dict(
                data["summary"], path="selector_execution.summary"
            ),
            stdout=require_str(data["stdout"], "stdout", min_length=0),
            stderr=require_str(data["stderr"], "stderr", min_length=0),
        )


@dataclass(frozen=True)
class PrivateEvaluationEvidence:
    source: ContentIdentity | None
    patch: ContentIdentity | None
    hidden_test_asset: ContentIdentity | None
    selectors: tuple[SelectorExecution, ...]
    cleanup_completed: bool

    def __post_init__(self) -> None:
        for value, path, identity_type in (
            (self.source, "source", "source"),
            (self.patch, "patch", "patch"),
            (self.hidden_test_asset, "hidden_test_asset", "test"),
        ):
            if value is None:
                continue
            if not isinstance(value, ContentIdentity):
                raise ContractError(f"{path}: expected ContentIdentity")
            if value.identity_type != identity_type:
                raise ContractError(
                    f"{path}: expected identity_type {identity_type!r}"
                )
        if not isinstance(self.selectors, tuple):
            raise ContractError("selectors: expected tuple")
        for index, item in enumerate(self.selectors):
            if not isinstance(item, SelectorExecution):
                raise ContractError(
                    f"selectors[{index}]: expected SelectorExecution"
                )
        require_bool(self.cleanup_completed, "cleanup_completed")

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": "private_evaluation_evidence",
            "schema_version": SCHEMA_VERSION,
            "source": None if self.source is None else self.source.to_dict(),
            "patch": None if self.patch is None else self.patch.to_dict(),
            "hidden_test_asset": (
                None
                if self.hidden_test_asset is None
                else self.hidden_test_asset.to_dict()
            ),
            "selectors": [item.to_dict() for item in self.selectors],
            "cleanup_completed": self.cleanup_completed,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PrivateEvaluationEvidence":
        data = require_exact_fields(
            value,
            "private_evaluation_evidence",
            (
                "record_type",
                "schema_version",
                "source",
                "patch",
                "hidden_test_asset",
                "selectors",
                "cleanup_completed",
            ),
        )
        if data["record_type"] != "private_evaluation_evidence":
            raise ContractError(
                "record_type: expected 'private_evaluation_evidence'"
            )
        if data["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
        return cls(
            source=_optional_identity(data["source"], "source"),
            patch=_optional_identity(data["patch"], "patch"),
            hidden_test_asset=_optional_identity(
                data["hidden_test_asset"], "hidden_test_asset"
            ),
            selectors=tuple(
                SelectorExecution.from_dict(item)
                for item in require_list(data["selectors"], "selectors")
            ),
            cleanup_completed=require_bool(
                data["cleanup_completed"], "cleanup_completed"
            ),
        )


@dataclass(frozen=True)
class CompletedEvaluation:
    result: EvaluationResultV06
    private_evidence: PrivateEvaluationEvidence | None
    evaluation_spec: EvaluationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.result, EvaluationResultV06):
            raise ContractError("result: expected EvaluationResultV06")
        if self.private_evidence is not None and not isinstance(
            self.private_evidence, PrivateEvaluationEvidence
        ):
            raise ContractError(
                "private_evidence: expected PrivateEvaluationEvidence"
            )
        if not isinstance(self.evaluation_spec, EvaluationSpec):
            raise ContractError("evaluation_spec: expected EvaluationSpec")
        if self.result.session_id != self.evaluation_spec.session_id:
            raise ContractError("result: session_id does not match EvaluationSpec")
        if self.result.attempt_id != self.evaluation_spec.attempt_id:
            raise ContractError("result: attempt_id does not match EvaluationSpec")
        if self.result.patch != self.evaluation_spec.frozen_patch:
            raise ContractError("result: patch does not match EvaluationSpec")
        if self.result.evaluation != self.evaluation_spec.evaluation:
            raise ContractError("result: evaluation does not match EvaluationSpec")
        if self.result.scoring != self.evaluation_spec.scoring:
            raise ContractError("result: scoring does not match EvaluationSpec")

    @property
    def evaluation_spec_hash(self) -> str:
        return self.evaluation_spec.content_hash


@dataclass(frozen=True)
class ReplayEvaluationEvidence:
    observed_outcome: str
    invalid_reason: str | None
    private_evidence: PrivateEvaluationEvidence | None

    def __post_init__(self) -> None:
        require_str(self.observed_outcome, "observed_outcome")
        if self.invalid_reason is not None:
            require_str(self.invalid_reason, "invalid_reason")
        if self.private_evidence is not None and not isinstance(
            self.private_evidence,
            PrivateEvaluationEvidence,
        ):
            raise ContractError("private_evidence: expected PrivateEvaluationEvidence")


class FreshEvaluationBackend(Protocol):
    def evaluate(
        self,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch,
    ) -> PrivateEvaluationEvidence: ...


class FreshEvaluator:
    def __init__(
        self,
        backend: FreshEvaluationBackend,
        *,
        clock_ms,
    ) -> None:
        evaluate = getattr(backend, "evaluate", None)
        if not callable(evaluate):
            raise ContractError("backend: expected FreshEvaluationBackend")
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        self._backend = backend
        self._clock_ms = clock_ms

    def evaluate(
        self,
        session_result: SessionResult,
        evaluation_spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
        patch_artifact: PatchArtifact | None,
    ) -> CompletedEvaluation:
        if not isinstance(session_result, SessionResult):
            raise ContractError("session_result: expected SessionResult")
        if not isinstance(evaluation_spec, EvaluationSpec):
            raise ContractError("evaluation_spec: expected EvaluationSpec")
        if session_result.session_id != evaluation_spec.session_id:
            raise ContractError("session_id: SessionResult/EvaluationSpec mismatch")
        if session_result.attempt_id != evaluation_spec.attempt_id:
            raise ContractError("attempt_id: SessionResult/EvaluationSpec mismatch")

        started_at_ms = require_int(self._clock_ms(), "clock_ms", minimum=0)
        attribution = termination_attribution(session_result.terminal_reason)
        if attribution.attempt_validity == "infrastructure_invalid":
            patch = self._validate_optional_patch_handoff(
                session_result,
                evaluation_spec,
                frozen_patch,
                patch_artifact,
            )
            return self._completed(
                evaluation_spec=evaluation_spec,
                started_at_ms=started_at_ms,
                attempt_validity="infrastructure_invalid",
                agent_terminal=attribution.agent_terminal,
                outcome="not_evaluated",
                invalid_reason=_SESSION_INVALID_REASONS[session_result.terminal_reason],
                patch=patch,
                private_evidence=None,
            )

        if frozen_patch is None or patch_artifact is None:
            raise ContractError("patch handoff: complete frozen patch is required")
        assert_patch_identity_handoff(
            frozen=frozen_patch,
            session_result=session_result,
            patch_artifact=patch_artifact,
            evaluation_spec=evaluation_spec,
        )
        if frozen_patch.source != evaluation_spec.source:
            raise ContractError("source identity mismatch across frozen/evaluation")
        if frozen_patch.empty:
            return self._completed(
                evaluation_spec=evaluation_spec,
                started_at_ms=started_at_ms,
                attempt_validity="valid",
                agent_terminal=attribution.agent_terminal,
                outcome="no_patch",
                invalid_reason=None,
                patch=frozen_patch.patch,
                private_evidence=None,
            )

        try:
            evidence = self._backend.evaluate(evaluation_spec, frozen_patch)
            fail_to_pass, pass_to_pass = _validate_evidence(
                evaluation_spec,
                evidence,
            )
        except StrictPatchApplyError:
            return self._completed(
                evaluation_spec=evaluation_spec,
                started_at_ms=started_at_ms,
                attempt_validity="valid",
                agent_terminal=attribution.agent_terminal,
                outcome="invalid_patch",
                invalid_reason=None,
                patch=frozen_patch.patch,
                private_evidence=None,
            )
        except EvaluationInfrastructureError as exc:
            return self._completed(
                evaluation_spec=evaluation_spec,
                started_at_ms=started_at_ms,
                attempt_validity="infrastructure_invalid",
                agent_terminal=attribution.agent_terminal,
                outcome="evaluation_error",
                invalid_reason=exc.invalid_reason,
                patch=frozen_patch.patch,
                private_evidence=None,
            )

        if fail_to_pass.failed:
            outcome = "f2p_failed"
        elif pass_to_pass.failed:
            outcome = "p2p_regression"
        else:
            outcome = "resolved"
        return self._completed(
            evaluation_spec=evaluation_spec,
            started_at_ms=started_at_ms,
            attempt_validity="valid",
            agent_terminal=attribution.agent_terminal,
            outcome=outcome,
            invalid_reason=None,
            patch=frozen_patch.patch,
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
            private_evidence=evidence,
        )

    def evaluate_replay(
        self,
        evaluation_spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
        patch_artifact: PatchArtifact | None,
    ) -> ReplayEvaluationEvidence:
        if not isinstance(evaluation_spec, EvaluationSpec):
            raise ContractError("evaluation_spec: expected EvaluationSpec")
        if frozen_patch is None:
            if patch_artifact is not None:
                raise ContractError("replay patch handoff: unexpected patch artifact")
        else:
            if not isinstance(frozen_patch, FrozenPatch):
                raise ContractError("frozen_patch: expected FrozenPatch")
            if not isinstance(patch_artifact, PatchArtifact):
                raise ContractError("patch_artifact: expected PatchArtifact")
            if frozen_patch.source != evaluation_spec.source:
                raise ContractError("replay source identity mismatch")
            if frozen_patch.patch != evaluation_spec.frozen_patch:
                raise ContractError("replay patch identity mismatch")
            if patch_artifact.patch != frozen_patch.patch:
                raise ContractError("replay patch artifact identity mismatch")
            if (
                patch_artifact.size_bytes != len(frozen_patch.patch_bytes)
                or patch_artifact.changed_paths != frozen_patch.changed_paths
                or patch_artifact.workspace != frozen_patch.workspace
            ):
                raise ContractError("replay patch artifact metadata mismatch")
        try:
            replay_method = getattr(self._backend, "evaluate_replay", None)
            if callable(replay_method):
                evidence = replay_method(evaluation_spec, frozen_patch)
            elif frozen_patch is not None:
                evidence = self._backend.evaluate(evaluation_spec, frozen_patch)
            else:
                raise EvaluationInfrastructureError(
                    "baseline_replay_not_supported"
                )
            fail_to_pass, pass_to_pass = _validate_replay_evidence(
                evaluation_spec,
                evidence,
                expected_patch=(
                    None if frozen_patch is None else frozen_patch.patch
                ),
            )
        except StrictPatchApplyError:
            return ReplayEvaluationEvidence("invalid_patch", None, None)
        except EvaluationInfrastructureError as exc:
            return ReplayEvaluationEvidence(
                "evaluation_error",
                exc.invalid_reason,
                None,
            )
        outcome = (
            "f2p_failed"
            if fail_to_pass.failed
            else "p2p_regression"
            if pass_to_pass.failed
            else "resolved"
        )
        return ReplayEvaluationEvidence(outcome, None, evidence)

    def _validate_optional_patch_handoff(
        self,
        session_result: SessionResult,
        evaluation_spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
        patch_artifact: PatchArtifact | None,
    ) -> ContentIdentity | None:
        values = (
            session_result.final_patch,
            evaluation_spec.frozen_patch,
            frozen_patch,
            patch_artifact,
        )
        if all(value is None for value in values):
            return None
        if frozen_patch is None or patch_artifact is None:
            raise ContractError("patch handoff: partial infrastructure patch bundle")
        assert_patch_identity_handoff(
            frozen=frozen_patch,
            session_result=session_result,
            patch_artifact=patch_artifact,
            evaluation_spec=evaluation_spec,
        )
        if frozen_patch.source != evaluation_spec.source:
            raise ContractError("source identity mismatch across frozen/evaluation")
        return frozen_patch.patch

    def _completed(
        self,
        *,
        evaluation_spec: EvaluationSpec,
        started_at_ms: int,
        attempt_validity: str,
        agent_terminal: str | None,
        outcome: str,
        invalid_reason: str | None,
        patch: ContentIdentity | None,
        private_evidence: PrivateEvaluationEvidence | None,
        fail_to_pass: TestExecutionSummary | None = None,
        pass_to_pass: TestExecutionSummary | None = None,
    ) -> CompletedEvaluation:
        ended_at_ms = require_int(self._clock_ms(), "clock_ms", minimum=0)
        if ended_at_ms < started_at_ms:
            raise ContractError("clock_ms: evaluation clock moved backwards")
        result = EvaluationResultV06(
            session_id=evaluation_spec.session_id,
            attempt_id=evaluation_spec.attempt_id,
            attempt_validity=attempt_validity,
            agent_terminal=agent_terminal,
            evaluation_outcome=outcome,
            invalid_reason=invalid_reason,
            patch=patch,
            fail_to_pass=fail_to_pass or _zero_summary(),
            pass_to_pass=pass_to_pass or _zero_summary(),
            duration_ms=ended_at_ms - started_at_ms,
            evaluation=evaluation_spec.evaluation,
            scoring=evaluation_spec.scoring,
        )
        return CompletedEvaluation(
            result=result,
            private_evidence=private_evidence,
            evaluation_spec=evaluation_spec,
        )


class AttemptEvaluationCoordinator:
    def __init__(
        self,
        evaluator: FreshEvaluator,
        journal: EventJournal,
        artifact_store,
        *,
        retry_index: int = 1,
        clock_ms,
    ) -> None:
        if not isinstance(evaluator, FreshEvaluator):
            raise ContractError("evaluator: expected FreshEvaluator")
        if not isinstance(journal, EventJournal):
            raise ContractError("journal: expected EventJournal")
        from op_bench.runtime.run_artifacts import AttemptArtifactStore

        if not isinstance(artifact_store, AttemptArtifactStore):
            raise ContractError("artifact_store: expected AttemptArtifactStore")
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        require_int(retry_index, "retry_index", minimum=1)
        self._evaluator = evaluator
        self._journal = journal
        self._artifact_store = artifact_store
        self._clock_ms = clock_ms
        self._retry_index = retry_index
        self._lock = threading.RLock()
        self._completed: CompletedEvaluation | None = None
        self._artifact_index = None
        self._failed = False

    @property
    def artifact_index(self):
        with self._lock:
            if self._artifact_index is None:
                raise ContractError("artifact_index: evaluation is not complete")
            return self._artifact_index

    def complete(
        self,
        session_result: SessionResult,
        evaluation_spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
        patch_artifact: PatchArtifact | None,
    ) -> CompletedEvaluation:
        with self._lock:
            if self._completed is not None:
                return self._completed
            if self._failed:
                raise ContractError("evaluation publication previously failed")
            self._validate_session_terminal(session_result)
            publication_started_at_ms = require_int(
                self._clock_ms(), "clock_ms", minimum=0
            )
            patch = (
                None
                if evaluation_spec.frozen_patch is None
                else evaluation_spec.frozen_patch.to_dict()
            )
            try:
                self._journal.append(
                    "evaluation_started",
                    {
                        "attempt_id": session_result.attempt_id,
                        "session_result_hash": session_result.content_hash,
                        "evaluation_spec_hash": evaluation_spec.content_hash,
                        "patch": patch,
                        "publication_started_at_ms": publication_started_at_ms,
                    },
                )
                completed = self._evaluator.evaluate(
                    session_result,
                    evaluation_spec,
                    frozen_patch,
                    patch_artifact,
                )
                hashes = self._artifact_store.write_evaluation(
                    session_result.attempt_id,
                    completed,
                    retry_index=self._retry_index,
                )
                publication_completed_at_ms = require_int(
                    self._clock_ms(), "clock_ms", minimum=0
                )
                if publication_completed_at_ms < publication_started_at_ms:
                    raise ContractError("clock_ms: publication clock moved backwards")
                appended = self._journal.append_batch(
                    (
                        (
                            "evaluation_completed",
                            {
                                "attempt_id": session_result.attempt_id,
                                "evaluation_result_hash": completed.result.content_hash,
                                "public_evaluation_hash": hashes.public_evaluation_hash,
                                "private_evaluation_hash": hashes.private_evaluation_hash,
                                "attempt_validity": completed.result.attempt_validity,
                                "evaluation_outcome": completed.result.evaluation_outcome,
                                "publication_completed_at_ms": publication_completed_at_ms,
                            },
                        ),
                        (
                            "terminal_emitted",
                            {
                                "attempt_id": session_result.attempt_id,
                                "session_result_hash": session_result.content_hash,
                                "evaluation_result_hash": completed.result.content_hash,
                                "attempt_validity": completed.result.attempt_validity,
                                "agent_terminal": completed.result.agent_terminal,
                                "evaluation_outcome": completed.result.evaluation_outcome,
                            },
                        ),
                    )
                )
                terminal = appended[-1]
                index = self._artifact_store.build_index(
                    session_result.attempt_id,
                    terminal.event_hash,
                    retry_index=self._retry_index,
                )
            except Exception:
                self._failed = True
                raise
            self._artifact_index = index
            self._completed = completed
            return completed

    def _validate_session_terminal(self, session_result: SessionResult) -> None:
        if not isinstance(session_result, SessionResult):
            raise ContractError("session_result: expected SessionResult")
        if session_result.session_id != self._journal.session_id:
            raise ContractError("session_result: session does not match journal")
        records = self._journal.records
        terminals = [
            event
            for event in records
            if event.event_type == "session_terminal_emitted"
        ]
        if len(terminals) != 1 or records[-1] != terminals[0]:
            raise ContractError(
                "session terminal: expected one final Session terminal event"
            )
        if terminals[0].public_payload.get("session_result_hash") != session_result.content_hash:
            raise ContractError("session terminal: SessionResult hash mismatch")


def _validate_evidence(
    spec: EvaluationSpec,
    evidence: PrivateEvaluationEvidence,
) -> tuple[TestExecutionSummary, TestExecutionSummary]:
    return _validate_replay_evidence(
        spec,
        evidence,
        expected_patch=spec.frozen_patch,
    )


def _validate_replay_evidence(
    spec: EvaluationSpec,
    evidence: PrivateEvaluationEvidence,
    *,
    expected_patch: ContentIdentity | None,
) -> tuple[TestExecutionSummary, TestExecutionSummary]:
    if not isinstance(evidence, PrivateEvaluationEvidence):
        raise EvaluationInfrastructureError("invalid_evaluation_evidence")
    if not evidence.cleanup_completed:
        raise EvaluationInfrastructureError("evaluation_cleanup_failed")
    if evidence.source != spec.source:
        raise EvaluationInfrastructureError("source_identity_mismatch")
    if evidence.patch != expected_patch:
        raise EvaluationInfrastructureError("patch_identity_mismatch")
    if evidence.hidden_test_asset != spec.hidden_test_asset:
        raise EvaluationInfrastructureError("hidden_test_identity_mismatch")

    expected = {
        **{selector: "fail_to_pass" for selector in spec.fail_to_pass},
        **{selector: "pass_to_pass" for selector in spec.pass_to_pass},
    }
    observed: dict[str, SelectorExecution] = {}
    for item in evidence.selectors:
        if item.selector_id in observed:
            raise EvaluationInfrastructureError("selector_set_mismatch")
        observed[item.selector_id] = item
    if set(observed) != set(expected):
        raise EvaluationInfrastructureError("selector_set_mismatch")
    for selector_id, group in expected.items():
        item = observed[selector_id]
        if item.group != group:
            raise EvaluationInfrastructureError("selector_group_mismatch")
        if item.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        if item.summary.collected == 0 or item.summary.executed == 0:
            raise EvaluationInfrastructureError("test_not_executed")

    return (
        _aggregate(observed[item] for item in spec.fail_to_pass),
        _aggregate(observed[item] for item in spec.pass_to_pass),
    )


def _aggregate(items) -> TestExecutionSummary:
    values = tuple(items)
    return TestExecutionSummary(
        collected=sum(item.summary.collected for item in values),
        executed=sum(item.summary.executed for item in values),
        passed=sum(item.summary.passed for item in values),
        failed=sum(item.summary.failed for item in values),
        skipped=sum(item.summary.skipped for item in values),
    )


def validate_evaluation_semantics(
    result: EvaluationResultV06,
    spec: EvaluationSpec,
    evidence: PrivateEvaluationEvidence | None,
) -> None:
    """Rebuild evaluator-owned semantics from private evidence."""
    if not isinstance(result, EvaluationResultV06):
        raise ContractError("evaluation semantics: expected EvaluationResultV06")
    if not isinstance(spec, EvaluationSpec):
        raise ContractError("evaluation semantics: expected EvaluationSpec")
    if evidence is not None and not isinstance(evidence, PrivateEvaluationEvidence):
        raise ContractError(
            "evaluation semantics: expected PrivateEvaluationEvidence"
        )

    evidence_outcomes = {"resolved", "f2p_failed", "p2p_regression"}
    if result.evaluation_outcome in evidence_outcomes:
        if result.attempt_validity != "valid":
            raise ContractError(
                "evaluation semantics: scored outcome must be valid"
            )
        if evidence is None:
            raise ContractError(
                "evaluation semantics: scored outcome requires private evidence"
            )
        try:
            fail_to_pass, pass_to_pass = _validate_evidence(spec, evidence)
        except EvaluationInfrastructureError as exc:
            raise ContractError(
                f"evaluation semantics: invalid private evidence ({exc.invalid_reason})"
            ) from exc
        if (
            result.fail_to_pass != fail_to_pass
            or result.pass_to_pass != pass_to_pass
        ):
            raise ContractError(
                "evaluation semantics: private aggregates do not match result"
            )
        expected_outcome = (
            "f2p_failed"
            if fail_to_pass.failed
            else "p2p_regression"
            if pass_to_pass.failed
            else "resolved"
        )
        if result.evaluation_outcome != expected_outcome:
            raise ContractError(
                "evaluation semantics: outcome does not match private evidence"
            )
        return

    if evidence is not None:
        raise ContractError(
            "evaluation semantics: unscored outcome must not contain private evidence"
        )
    zero = _zero_summary()
    if result.fail_to_pass != zero or result.pass_to_pass != zero:
        raise ContractError(
            "evaluation semantics: unscored outcome requires zero summaries"
        )
    expected_validity = (
        "infrastructure_invalid"
        if result.evaluation_outcome in {"evaluation_error", "not_evaluated"}
        else "valid"
    )
    if result.attempt_validity != expected_validity:
        raise ContractError(
            "evaluation semantics: outcome and attempt validity disagree"
        )


def validate_session_evaluation_binding(
    session_result: SessionResult,
    evaluation_result: EvaluationResultV06,
) -> None:
    """Bind evaluator attribution to the authoritative Session terminal."""
    if not isinstance(session_result, SessionResult):
        raise ContractError("session attribution: expected SessionResult")
    if not isinstance(evaluation_result, EvaluationResultV06):
        raise ContractError("session attribution: expected EvaluationResultV06")
    if (
        evaluation_result.session_id != session_result.session_id
        or evaluation_result.attempt_id != session_result.attempt_id
    ):
        raise ContractError("session attribution: session/attempt mismatch")
    attribution = termination_attribution(session_result.terminal_reason)
    if evaluation_result.agent_terminal != attribution.agent_terminal:
        raise ContractError("session attribution: agent terminal mismatch")
    if evaluation_result.patch != session_result.final_patch:
        raise ContractError("session attribution: patch mismatch")

    if attribution.attempt_validity == "infrastructure_invalid":
        expected_reason = _SESSION_INVALID_REASONS[
            session_result.terminal_reason
        ]
        if (
            evaluation_result.attempt_validity != "infrastructure_invalid"
            or evaluation_result.evaluation_outcome != "not_evaluated"
            or evaluation_result.invalid_reason != expected_reason
        ):
            raise ContractError(
                "session attribution: infrastructure terminal must remain "
                "not_evaluated"
            )
        return

    expected_validity = (
        "infrastructure_invalid"
        if evaluation_result.evaluation_outcome == "evaluation_error"
        else "valid"
    )
    if (
        evaluation_result.evaluation_outcome == "not_evaluated"
        or evaluation_result.attempt_validity != expected_validity
    ):
        raise ContractError(
            "session attribution: scorable terminal has invalid evaluation state"
        )


def validate_no_patch_artifact(
    result: EvaluationResultV06,
    patch_bytes: bytes | None,
) -> None:
    if not isinstance(result, EvaluationResultV06):
        raise ContractError("patch semantics: expected EvaluationResultV06")
    if patch_bytes is not None and not isinstance(patch_bytes, bytes):
        raise ContractError("patch semantics: expected bytes")
    if result.evaluation_outcome == "no_patch" and patch_bytes != b"":
        raise ContractError(
            "no_patch outcome requires an empty final.patch artifact"
        )


def _zero_summary() -> TestExecutionSummary:
    return TestExecutionSummary(0, 0, 0, 0, 0)


def _optional_identity(value: object, path: str) -> ContentIdentity | None:
    if value is None:
        return None
    return ContentIdentity.from_dict(value, path=path)


__all__ = [
    "AttemptEvaluationCoordinator",
    "CompletedEvaluation",
    "EvaluationInfrastructureError",
    "FreshEvaluationBackend",
    "FreshEvaluator",
    "PrivateEvaluationEvidence",
    "ReplayEvaluationEvidence",
    "SelectorExecution",
    "StrictPatchApplyError",
    "validate_evaluation_semantics",
    "validate_no_patch_artifact",
    "validate_session_evaluation_binding",
]
