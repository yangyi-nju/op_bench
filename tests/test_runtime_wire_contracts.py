from __future__ import annotations

import unittest

from op_bench.runtime.contracts import (
    ActionObservation,
    ActionRequest,
    BudgetDelta,
    EvaluationResultV06,
    EvaluationSpec,
    EventRecord,
    IntegrityCheck,
    IntegrityReport,
    SessionResult,
    SessionSpec,
    TestExecutionSummary,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_contracts import (
    SHA_A,
    SHA_B,
    agent_task_view,
    budget_policy,
    capability_policy,
    identity,
    public_test,
    runtime_profile,
)


def action_request() -> ActionRequest:
    return ActionRequest(
        session_id="session-001",
        action_id="action-001",
        action_name="workspace_read",
        arguments={"path": "torch/example.py", "start_line": 1},
        client_sequence=1,
        deadline_ms=1_900_000_000_000,
    )


def budget_delta() -> BudgetDelta:
    return BudgetDelta(
        wall_clock_ms=25,
        actions=1,
        tests=0,
        commands=0,
        output_bytes=120,
        provider_tokens=0,
    )


def action_observation() -> ActionObservation:
    return ActionObservation(
        session_id="session-001",
        action_id="action-001",
        ok=True,
        error_code="ok",
        message="file read",
        data={"content": "value = 1\n"},
        started_at_ms=100,
        ended_at_ms=125,
        budget_delta=budget_delta(),
        mutation_state="unchanged",
    )


def event_record() -> EventRecord:
    return EventRecord(
        session_id="session-001",
        sequence=1,
        occurred_at_ms=100,
        event_type="session_created",
        public_payload={"attempt_id": "attempt-001"},
        previous_event_hash=None,
        event_hash=SHA_A,
    )


def session_spec() -> SessionSpec:
    return SessionSpec(
        session_id="session-001",
        attempt_id="attempt-001",
        workspace=identity("workspace", "workspace-001", SHA_A),
        agent_task_view=identity("task_view", "view-001", SHA_B),
        capability_policy=capability_policy(),
        budget_policy=budget_policy(),
        deadline_ms=1_900_000_000_000,
        adapter_config=identity("agent_config", "adapter-config", SHA_A),
        runtime=runtime_profile(),
        artifact_root_id="attempts/attempt-001",
        resume_policy="retry_infrastructure",
    )


def evaluation_spec() -> EvaluationSpec:
    return EvaluationSpec(
        session_id="session-001",
        attempt_id="attempt-001",
        task=identity("task", "pytorch__example", SHA_A),
        source=identity("source", "pytorch@abc", SHA_B),
        frozen_patch=identity("patch", "final.patch", SHA_A),
        hidden_test_asset=identity("test", "hidden.patch", SHA_B),
        public_tests=(public_test(),),
        fail_to_pass=("hidden::f2p",),
        pass_to_pass=("public::smoke",),
        runtime=runtime_profile(),
        timeout_ms=900_000,
        scoring=identity("scoring", "opbench-scoring-v1", SHA_A),
    )


def session_result() -> SessionResult:
    return SessionResult(
        session_id="session-001",
        attempt_id="attempt-001",
        terminal_reason="agent_finished",
        final_patch=identity("patch", "final.patch", SHA_A),
        started_at_ms=100,
        ended_at_ms=500,
    )


def test_summary() -> TestExecutionSummary:
    return TestExecutionSummary(collected=1, executed=1, passed=1, failed=0, skipped=0)


def evaluation_result() -> EvaluationResultV06:
    return EvaluationResultV06(
        session_id="session-001",
        attempt_id="attempt-001",
        attempt_validity="valid",
        agent_terminal="finished",
        evaluation_outcome="resolved",
        invalid_reason=None,
        patch=identity("patch", "final.patch", SHA_A),
        fail_to_pass=test_summary(),
        pass_to_pass=test_summary(),
        duration_ms=400,
    )


def integrity_report() -> IntegrityReport:
    return IntegrityReport(
        run_id="cohort-001",
        status="passed",
        checks=(
            IntegrityCheck(
                check_id="patch_hash",
                status="passed",
                message="session and evaluator patch hashes match",
                expected_hash=SHA_A,
                actual_hash=SHA_A,
            ),
        ),
    )


class WireContractRoundTripTests(unittest.TestCase):
    def test_every_wire_contract_round_trips(self) -> None:
        values = (
            action_request(),
            budget_delta(),
            action_observation(),
            event_record(),
            session_spec(),
            evaluation_spec(),
            session_result(),
            test_summary(),
            evaluation_result(),
            integrity_report().checks[0],
            integrity_report(),
        )

        for value in values:
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                self.assertEqual(encoded["schema_version"], "v1")
                self.assertEqual(type(value).from_dict(encoded), value)
                self.assertEqual(type(value).from_dict(encoded).content_hash, value.content_hash)

    def test_result_axes_are_independent(self) -> None:
        result = EvaluationResultV06(
            session_id="session-002",
            attempt_id="attempt-002",
            attempt_validity="infrastructure_invalid",
            agent_terminal="finished",
            evaluation_outcome="not_evaluated",
            invalid_reason="remote_runtime_unavailable",
            patch=None,
            fail_to_pass=TestExecutionSummary(0, 0, 0, 0, 0),
            pass_to_pass=TestExecutionSummary(0, 0, 0, 0, 0),
            duration_ms=0,
        )

        self.assertEqual(result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(result.agent_terminal, "finished")
        self.assertEqual(result.evaluation_outcome, "not_evaluated")

    def test_json_payloads_are_defensively_frozen(self) -> None:
        constructors = (
            lambda payload: ActionRequest(
                session_id="session-001",
                action_id="action-001",
                action_name="workspace_read",
                arguments=payload,
                client_sequence=1,
                deadline_ms=1_900_000_000_000,
            ),
            lambda payload: ActionObservation(
                session_id="session-001",
                action_id="action-001",
                ok=True,
                error_code="ok",
                message="ok",
                data=payload,
                started_at_ms=1,
                ended_at_ms=2,
                budget_delta=budget_delta(),
                mutation_state="unchanged",
            ),
            lambda payload: EventRecord(
                session_id="session-001",
                sequence=1,
                occurred_at_ms=1,
                event_type="session_created",
                public_payload=payload,
                previous_event_hash=None,
                event_hash=SHA_A,
            ),
        )

        for construct in constructors:
            source = {"nested": {"values": [1]}}
            value = construct(source)
            before = value.content_hash
            source["nested"]["values"].append(2)

            self.assertEqual(value.content_hash, before)
            self.assertEqual(value.to_dict()[next(
                name for name in ("arguments", "data", "public_payload") if hasattr(value, name)
            )], {"nested": {"values": [1]}})
            payload = next(
                getattr(value, name)
                for name in ("arguments", "data", "public_payload")
                if hasattr(value, name)
            )
            with self.assertRaises(TypeError):
                payload["nested"]["new"] = "mutation"


class WireContractNegativeTests(unittest.TestCase):
    def test_unknown_version_is_rejected_for_every_wire_contract(self) -> None:
        values = (
            action_request(),
            budget_delta(),
            action_observation(),
            event_record(),
            session_spec(),
            evaluation_spec(),
            session_result(),
            test_summary(),
            evaluation_result(),
            integrity_report().checks[0],
            integrity_report(),
        )

        for value in values:
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                encoded["schema_version"] = "v2"
                with self.assertRaisesRegex(ContractError, "schema_version: expected 'v1'"):
                    type(value).from_dict(encoded)

    def test_action_rejects_unknown_name_and_noncanonical_arguments(self) -> None:
        encoded = action_request().to_dict()
        encoded["action_name"] = "network_probe"
        with self.assertRaisesRegex(ContractError, "action_name: unsupported value"):
            ActionRequest.from_dict(encoded)

        encoded = action_request().to_dict()
        encoded["arguments"] = {"timeout": 1.5}
        with self.assertRaisesRegex(ContractError, r"\$\.timeout: floats are not canonical"):
            ActionRequest.from_dict(encoded)

    def test_action_sequence_and_deadline_are_positive_integers(self) -> None:
        encoded = action_request().to_dict()
        encoded["client_sequence"] = True
        with self.assertRaisesRegex(ContractError, "client_sequence: expected integer"):
            ActionRequest.from_dict(encoded)

        encoded = action_request().to_dict()
        encoded["deadline_ms"] = 0
        with self.assertRaisesRegex(ContractError, "deadline_ms: must be >= 1"):
            ActionRequest.from_dict(encoded)

    def test_observation_rejects_invalid_error_code_and_time_order(self) -> None:
        encoded = action_observation().to_dict()
        encoded["error_code"] = "random_exception"
        with self.assertRaisesRegex(ContractError, "error_code: unsupported value"):
            ActionObservation.from_dict(encoded)

        encoded = action_observation().to_dict()
        encoded["ended_at_ms"] = 99
        with self.assertRaisesRegex(ContractError, "ended_at_ms: must be >= started_at_ms"):
            ActionObservation.from_dict(encoded)

    def test_event_rejects_invalid_hash_or_sequence(self) -> None:
        encoded = event_record().to_dict()
        encoded["event_hash"] = "not-a-hash"
        with self.assertRaisesRegex(ContractError, "event_hash: does not match required pattern"):
            EventRecord.from_dict(encoded)

        encoded = event_record().to_dict()
        encoded["sequence"] = 0
        with self.assertRaisesRegex(ContractError, "sequence: must be >= 1"):
            EventRecord.from_dict(encoded)

    def test_session_rejects_identity_type_swap(self) -> None:
        encoded = session_spec().to_dict()
        encoded["workspace"] = identity("source", "wrong", SHA_A).to_dict()
        with self.assertRaisesRegex(ContractError, "workspace: expected identity_type 'workspace'"):
            SessionSpec.from_dict(encoded)

    def test_test_summary_rejects_inconsistent_counts(self) -> None:
        with self.assertRaisesRegex(ContractError, "executed: must equal passed \+ failed"):
            TestExecutionSummary(collected=1, executed=1, passed=0, failed=0, skipped=0)

        with self.assertRaisesRegex(ContractError, "collected: must equal executed \+ skipped"):
            TestExecutionSummary(collected=2, executed=1, passed=1, failed=0, skipped=0)

    def test_invalid_attempt_requires_a_reason(self) -> None:
        encoded = evaluation_result().to_dict()
        encoded["attempt_validity"] = "infrastructure_invalid"

        with self.assertRaisesRegex(ContractError, "invalid_reason: required"):
            EvaluationResultV06.from_dict(encoded)

    def test_valid_attempt_rejects_an_invalid_reason(self) -> None:
        encoded = evaluation_result().to_dict()
        encoded["invalid_reason"] = "should-not-exist"

        with self.assertRaisesRegex(ContractError, "invalid_reason: must be null"):
            EvaluationResultV06.from_dict(encoded)

    def test_agent_terminal_axis_uses_only_approved_agent_outcomes(self) -> None:
        for invalid in ("budget_exhausted", "provider_error", "runtime_error", "platform_error"):
            with self.subTest(invalid=invalid):
                encoded = evaluation_result().to_dict()
                encoded["agent_terminal"] = invalid
                with self.assertRaisesRegex(ContractError, "agent_terminal: unsupported value"):
                    EvaluationResultV06.from_dict(encoded)

        encoded = evaluation_result().to_dict()
        encoded["agent_terminal"] = "budget"
        self.assertEqual(EvaluationResultV06.from_dict(encoded).agent_terminal, "budget")


if __name__ == "__main__":
    unittest.main()
