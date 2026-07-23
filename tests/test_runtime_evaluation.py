from __future__ import annotations

from dataclasses import replace
import unittest

from op_bench.runtime.contracts import TestExecutionSummary
from op_bench.runtime.evaluation import (
    EvaluationInfrastructureError,
    FreshEvaluator,
    PrivateEvaluationEvidence,
    SelectorExecution,
    StrictPatchApplyError,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import (
    FrozenPatch,
    build_patch_artifact,
    raw_patch_identity,
)
from tests.test_runtime_contracts import SHA_A, SHA_B, identity
from tests.test_runtime_wire_contracts import evaluation_spec, session_result


class StepClock:
    def __init__(self, *values: int) -> None:
        self.values = iter(values or (100, 125))

    def __call__(self) -> int:
        return next(self.values)


def execution(
    selector_id: str,
    group: str,
    *,
    passed: bool = True,
    collected: int = 1,
    skipped: int = 0,
) -> SelectorExecution:
    executed = collected - skipped
    failed = 0 if passed else executed
    passed_count = executed if passed else 0
    return SelectorExecution(
        selector_id=selector_id,
        group=group,
        command_digest=SHA_A,
        exit_code=0 if passed else 1,
        timed_out=False,
        summary=TestExecutionSummary(
            collected=collected,
            executed=executed,
            passed=passed_count,
            failed=failed,
            skipped=skipped,
        ),
        stdout="private stdout",
        stderr="private stderr",
    )


def evidence(*, f2p_passed: bool = True, p2p_passed: bool = True) -> PrivateEvaluationEvidence:
    spec = evaluation_spec()
    return PrivateEvaluationEvidence(
        source=spec.source,
        patch=spec.frozen_patch,
        hidden_test_asset=spec.hidden_test_asset,
        selectors=(
            execution("hidden::f2p", "fail_to_pass", passed=f2p_passed),
            execution("public::smoke", "pass_to_pass", passed=p2p_passed),
        ),
        cleanup_completed=True,
    )


class ScriptedBackend:
    def __init__(
        self,
        value: PrivateEvaluationEvidence | Exception,
    ) -> None:
        self.value = value
        self.calls = 0

    def evaluate(self, spec, frozen_patch):
        self.calls += 1
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class FreshEvaluatorTests(unittest.TestCase):
    def make_inputs(self, patch_bytes: bytes = b"fixture patch\n"):
        spec = evaluation_spec()
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        spec = replace(spec, frozen_patch=patch)
        frozen = FrozenPatch(
            workspace=identity("workspace", "workspace-001", SHA_B),
            source=spec.source,
            base_commit="fixture-base",
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=(() if not patch_bytes else ("src/operator.py",)),
            empty=not patch_bytes,
        )
        artifact = build_patch_artifact(frozen, artifact_id="attempt-001/final.patch")
        result = replace(session_result(), final_patch=patch)
        return spec, frozen, artifact, result

    def evaluate(self, backend_value, *, patch_bytes=b"fixture patch\n", terminal_reason=None):
        spec, frozen, artifact, result = self.make_inputs(patch_bytes)
        if terminal_reason is not None:
            result = replace(result, terminal_reason=terminal_reason)
        if isinstance(backend_value, PrivateEvaluationEvidence):
            backend_value = replace(
                backend_value,
                source=spec.source,
                patch=spec.frozen_patch,
                hidden_test_asset=spec.hidden_test_asset,
            )
        backend = ScriptedBackend(backend_value)
        completed = FreshEvaluator(backend, clock_ms=StepClock()).evaluate(
            result,
            spec,
            frozen,
            artifact,
        )
        return completed, backend

    def test_classifies_resolved_f2p_failed_and_p2p_regression(self) -> None:
        cases = (
            (evidence(), "agent_finished", "resolved", "finished"),
            (evidence(f2p_passed=False), "agent_exited", "f2p_failed", "exited"),
            (evidence(p2p_passed=False), "timeout", "p2p_regression", "timeout"),
        )

        for private, reason, outcome, terminal in cases:
            with self.subTest(outcome=outcome):
                completed, backend = self.evaluate(private, terminal_reason=reason)
                self.assertEqual(completed.result.attempt_validity, "valid")
                self.assertEqual(completed.result.agent_terminal, terminal)
                self.assertEqual(completed.result.evaluation_outcome, outcome)
                self.assertEqual(completed.result.evaluation, evaluation_spec().evaluation)
                self.assertEqual(completed.result.scoring, evaluation_spec().scoring)
                self.assertEqual(
                    completed.evaluation_spec_hash,
                    completed.evaluation_spec.content_hash,
                )
                self.assertEqual(
                    completed.evaluation_spec.evaluation,
                    completed.result.evaluation,
                )
                self.assertEqual(
                    completed.evaluation_spec.scoring,
                    completed.result.scoring,
                )
                self.assertEqual(completed.result.duration_ms, 25)
                self.assertEqual(backend.calls, 1)

    def test_empty_patch_is_no_patch_without_backend_execution(self) -> None:
        completed, backend = self.evaluate(evidence(), patch_bytes=b"")

        self.assertEqual(completed.result.attempt_validity, "valid")
        self.assertEqual(completed.result.evaluation_outcome, "no_patch")
        self.assertIsNotNone(completed.result.patch)
        self.assertEqual(completed.result.fail_to_pass.collected, 0)
        self.assertIsNone(completed.private_evidence)
        self.assertEqual(backend.calls, 0)

    def test_strict_patch_failure_is_valid_invalid_patch(self) -> None:
        completed, backend = self.evaluate(StrictPatchApplyError("does not apply"))

        self.assertEqual(completed.result.attempt_validity, "valid")
        self.assertEqual(completed.result.evaluation_outcome, "invalid_patch")
        self.assertIsNone(completed.result.invalid_reason)
        self.assertEqual(backend.calls, 1)

    def test_backend_failure_is_infrastructure_invalid_evaluation_error(self) -> None:
        completed, backend = self.evaluate(
            EvaluationInfrastructureError("source_identity_mismatch", "private detail")
        )

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(completed.result.agent_terminal, "finished")
        self.assertEqual(completed.result.evaluation_outcome, "evaluation_error")
        self.assertEqual(completed.result.invalid_reason, "source_identity_mismatch")
        self.assertNotIn("private detail", str(completed.result.to_dict()))
        self.assertEqual(backend.calls, 1)

    def test_infrastructure_session_is_not_evaluated_and_can_have_no_patch(self) -> None:
        spec = replace(evaluation_spec(), frozen_patch=None)
        result = replace(
            session_result(),
            terminal_reason="platform_error",
            final_patch=None,
        )
        backend = ScriptedBackend(evidence())

        completed = FreshEvaluator(backend, clock_ms=StepClock()).evaluate(
            result,
            spec,
            None,
            None,
        )

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertIsNone(completed.result.agent_terminal)
        self.assertEqual(completed.result.evaluation_outcome, "not_evaluated")
        self.assertEqual(completed.result.invalid_reason, "session_platform_error")
        self.assertIsNone(completed.result.patch)
        self.assertEqual(backend.calls, 0)

    def test_patch_identity_handoff_precedes_backend_execution(self) -> None:
        spec, frozen, artifact, result = self.make_inputs()
        backend = ScriptedBackend(evidence())

        with self.assertRaisesRegex(ContractError, "patch identity mismatch"):
            FreshEvaluator(backend, clock_ms=StepClock()).evaluate(
                result,
                replace(spec, frozen_patch=identity("patch", "other.patch", SHA_A)),
                frozen,
                artifact,
            )

        self.assertEqual(backend.calls, 0)

    def test_missing_or_unexecuted_selector_is_evaluation_error(self) -> None:
        invalid_evidence = (
            replace(evidence(), selectors=evidence().selectors[:1]),
            replace(
                evidence(),
                selectors=(
                    execution(
                        "hidden::f2p",
                        "fail_to_pass",
                        collected=1,
                        skipped=1,
                    ),
                    execution("public::smoke", "pass_to_pass"),
                ),
            ),
        )

        for value in invalid_evidence:
            with self.subTest(selectors=value.selectors):
                completed, _ = self.evaluate(value)
                self.assertEqual(
                    completed.result.attempt_validity,
                    "infrastructure_invalid",
                )
                self.assertEqual(
                    completed.result.evaluation_outcome,
                    "evaluation_error",
                )
                self.assertIn(
                    completed.result.invalid_reason,
                    {"selector_set_mismatch", "test_not_executed"},
                )
                self.assertRegex(
                    completed.evaluation_spec_hash,
                    r"^sha256:[0-9a-f]{64}$",
                )


if __name__ == "__main__":
    unittest.main()
