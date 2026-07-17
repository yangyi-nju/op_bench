from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import ACTION_NAMES, ActionRequest, SessionSpec
from op_bench.runtime.events import EventJournal, verify_action_pairing, verify_event_chain
from op_bench.runtime.session import (
    AttemptSession,
    SessionStateError,
    termination_attribution,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import AuthoritativeWorkspace, WorkspacePolicyError
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_actions_service import FakeCommandBackend
from tests.test_runtime_contracts import SHA_A, budget_policy, capability_policy, identity
from tests.test_runtime_wire_contracts import session_spec
from tests.test_runtime_workspace import policy as workspace_policy


class MutableClock:
    def __init__(self, value: int = 1_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value


class AttemptSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.base_commit = initialize_git_repo(self.root)
        self.clock = MutableClock()
        self.source = identity("source", "fixture@session", SHA_A)
        self.capabilities = replace(
            capability_policy(),
            allowed_actions=ACTION_NAMES,
            writable_paths=("src/",),
            allowed_command_prefixes=(),
            registered_tests=(),
        )
        self.budget = replace(
            budget_policy(),
            wall_clock_ms=10_000,
            max_actions=10,
            max_tests=0,
            max_commands=0,
            max_output_bytes=10_000,
        )
        self.session = self.make_session()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_session(
        self,
        *,
        session_id: str = "session-lifecycle",
        budget=None,
        deadline_ms: int = 5_000,
        freeze_patch=None,
        events_path: Path | None = None,
    ) -> AttemptSession:
        selected_budget = budget or self.budget
        workspace = AuthoritativeWorkspace.open(
            self.root,
            source=self.source,
            policy=workspace_policy(),
        )
        journal = EventJournal(session_id, clock_ms=self.clock, events_path=events_path)
        service = CanonicalActionService(
            session_id=session_id,
            workspace=workspace,
            capability_policy=self.capabilities,
            budget_policy=selected_budget,
            command_backend=FakeCommandBackend(),
            test_registry={},
            clock_ms=self.clock,
            event_journal=journal,
        )
        base = session_spec()
        spec = SessionSpec(
            session_id=session_id,
            attempt_id=f"attempt-{session_id}",
            workspace=workspace.identity,
            agent_task_view=base.agent_task_view,
            capability_policy=self.capabilities,
            budget_policy=selected_budget,
            deadline_ms=deadline_ms,
            adapter_config=base.adapter_config,
            runtime=base.runtime,
            artifact_root_id=f"attempts/{session_id}",
            resume_policy="retry_infrastructure",
        )
        return AttemptSession(
            spec=spec,
            action_service=service,
            journal=journal,
            freeze_patch=freeze_patch or workspace.freeze,
            clock_ms=self.clock,
        )

    @staticmethod
    def request(
        action_id: str,
        action_name: str,
        arguments: dict[str, object],
        sequence: int,
        *,
        deadline_ms: int = 5_000,
        session_id: str = "session-lifecycle",
    ) -> ActionRequest:
        return ActionRequest(
            session_id=session_id,
            action_id=action_id,
            action_name=action_name,
            arguments=arguments,
            client_sequence=sequence,
            deadline_ms=deadline_ms,
        )

    def test_frozen_state_machine_and_idempotent_lifecycle(self) -> None:
        self.assertEqual(self.session.state, "created")
        self.assertEqual(
            [event.event_type for event in self.session.events],
            ["session_created"],
        )
        with self.assertRaisesRegex(SessionStateError, "created.*start"):
            self.session.start()
        with self.assertRaisesRegex(SessionStateError, "running"):
            self.session.execute_action(
                self.request("early", "workspace_read", {"path": "src/operator.py"}, 1)
            )

        self.assertEqual(self.session.prepare(), "preparing")
        self.assertEqual(self.session.prepare(), "preparing")
        self.assertEqual(self.session.mark_ready(), "ready")
        self.assertEqual(self.session.mark_ready(), "ready")
        self.assertEqual(self.session.start(), "running")
        self.assertEqual(self.session.start(), "running")
        self.assertEqual(self.session.mark_agent_launched(), "running")
        self.assertEqual(self.session.mark_agent_launched(), "running")

        observation = self.session.execute_action(
            self.request("read", "workspace_read", {"path": "src/operator.py"}, 1)
        )
        self.assertTrue(observation.ok)
        self.assertEqual(self.session.request_stop("cancelled"), "stopping")
        self.assertEqual(self.session.request_stop("cancelled"), "stopping")
        with self.assertRaisesRegex(SessionStateError, "running"):
            self.session.execute_action(
                self.request("late", "workspace_read", {"path": "src/operator.py"}, 2)
            )

        result = self.session.finalize()
        self.assertIs(self.session.finalize(), result)
        self.assertEqual(result.terminal_reason, "cancelled")
        self.assertIsNotNone(result.final_patch)
        self.assertEqual(self.session.state, "terminal")
        before = self.session.events
        self.assertEqual(self.session.request_stop("platform_error"), "terminal")
        self.assertEqual(self.session.events, before)
        self.assertEqual(verify_event_chain(self.session.events), ())
        self.assertEqual(verify_action_pairing(self.session.events), ())
        self.assertEqual(
            [event.event_type for event in self.session.events].count("terminal_emitted"),
            1,
        )
        self.assertEqual(
            [event.event_type for event in self.session.events].count("agent_launched"),
            1,
        )

    def test_finish_action_requests_stop_and_repeated_finalize_keeps_patch(self) -> None:
        self.session.prepare()
        self.session.mark_ready()
        self.session.start()
        finish = self.session.execute_action(
            self.request("finish", "session_finish", {}, 1)
        )

        self.assertTrue(finish.ok)
        self.assertEqual(self.session.state, "stopping")
        result = self.session.finalize()

        self.assertEqual(result.terminal_reason, "agent_finished")
        self.assertEqual(result.final_patch.to_dict(), finish.data["patch"])
        self.assertIs(self.session.finalize(), result)
        self.assertEqual(
            [event.event_type for event in self.session.events].count("terminal_emitted"),
            1,
        )

    def test_budget_and_deadline_observations_request_server_owned_stop(self) -> None:
        zero_budget = replace(self.budget, max_actions=0)
        budget_session = self.make_session(
            session_id="session-budget",
            budget=zero_budget,
        )
        budget_session.prepare()
        budget_session.mark_ready()
        budget_session.start()
        denied = budget_session.execute_action(
            self.request(
                "read",
                "workspace_read",
                {"path": "src/operator.py"},
                1,
                session_id="session-budget",
            )
        )
        self.assertEqual(denied.error_code, "budget_exhausted")
        self.assertEqual(budget_session.state, "stopping")
        self.assertEqual(budget_session.finalize().terminal_reason, "budget_exhausted")

        deadline_session = self.make_session(
            session_id="session-deadline",
            deadline_ms=1_001,
        )
        deadline_session.prepare()
        deadline_session.mark_ready()
        deadline_session.start()
        self.clock.value = 1_001
        self.assertEqual(deadline_session.poll_limits(), "stopping")
        self.assertEqual(deadline_session.finalize().terminal_reason, "timeout")

    def test_session_deadline_is_enforced_before_action_admission(self) -> None:
        session = self.make_session(
            session_id="session-deadline-gate",
            deadline_ms=1_001,
        )
        session.prepare()
        session.mark_ready()
        session.start()
        self.clock.value = 1_001

        with self.assertRaisesRegex(SessionStateError, "deadline"):
            session.execute_action(
                self.request(
                    "late-read",
                    "workspace_read",
                    {"path": "src/operator.py"},
                    1,
                    deadline_ms=2_000,
                    session_id="session-deadline-gate",
                )
            )

        self.assertEqual(session.action_service.usage.actions, 0)
        self.assertEqual(session.state, "stopping")
        self.assertEqual(session.finalize().terminal_reason, "timeout")

    def test_poll_limits_covers_resource_usage_and_unsealed_stopping_state(self) -> None:
        one_action = replace(self.budget, max_actions=1)
        session = self.make_session(
            session_id="session-action-limit",
            budget=one_action,
        )
        session.prepare()
        session.mark_ready()
        session.start()
        self.assertTrue(
            session.execute_action(
                self.request(
                    "read-one",
                    "workspace_read",
                    {"path": "src/operator.py"},
                    1,
                    session_id="session-action-limit",
                )
            ).ok
        )
        self.assertEqual(session.poll_limits(), "stopping")
        self.assertEqual(session.finalize().terminal_reason, "budget_exhausted")

        stopping = self.make_session(
            session_id="session-stopping-timeout",
            deadline_ms=1_001,
        )
        stopping.prepare()
        stopping.mark_ready()
        stopping.start()
        stopping.request_stop("agent_finished")
        self.clock.value = 1_001
        self.assertEqual(stopping.poll_limits(), "stopping")
        self.assertEqual(stopping.finalize().terminal_reason, "timeout")

    def test_finalize_recovers_after_freeze_start_persistence_failure_without_hanging(self) -> None:
        session = self.make_session(session_id="session-freeze-start-journal-failure")
        session.request_stop("agent_finished")
        original = session._journal.append
        failed = [False]

        def fail_once(event_type, payload):
            if event_type == "patch_freeze_started" and not failed[0]:
                failed[0] = True
                raise ContractError("fixture persistence failure")
            return original(event_type, payload)

        with mock.patch.object(session._journal, "append", side_effect=fail_once):
            with self.assertRaisesRegex(SessionStateError, "persistence"):
                session.finalize()
            results: list[object] = []
            thread = threading.Thread(target=lambda: results.append(session.finalize()), daemon=True)
            thread.start()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive(), "retrying finalize must not deadlock")

        self.assertEqual(results[0].terminal_reason, "platform_error")
        self.assertEqual(session.state, "terminal")
        self.assertEqual(
            [event.event_type for event in session.events].count("terminal_emitted"),
            1,
        )

    def test_terminal_persistence_failure_is_retryable_without_refreezing(self) -> None:
        freeze_calls = [0]

        def freeze():
            freeze_calls[0] += 1
            return self.session.action_service._workspace.freeze()

        session = self.make_session(
            session_id="session-terminal-journal-failure",
            freeze_patch=freeze,
        )
        session.request_stop("agent_finished")
        original = session._journal.append
        failed = [False]

        def fail_once(event_type, payload):
            if event_type == "terminal_emitted" and not failed[0]:
                failed[0] = True
                raise ContractError("fixture persistence failure")
            return original(event_type, payload)

        with mock.patch.object(session._journal, "append", side_effect=fail_once):
            with self.assertRaisesRegex(SessionStateError, "persistence"):
                session.finalize()
            results: list[object] = []
            thread = threading.Thread(target=lambda: results.append(session.finalize()), daemon=True)
            thread.start()
            thread.join(timeout=1)
            self.assertFalse(thread.is_alive(), "terminal retry must not deadlock")

        self.assertEqual(results[0].terminal_reason, "platform_error")
        self.assertEqual(freeze_calls, [1])
        self.assertEqual(
            [event.event_type for event in session.events].count("terminal_emitted"),
            1,
        )

    def test_uncertain_but_complete_terminal_commit_reconciles_as_success(self) -> None:
        session = self.make_session(
            session_id="session-terminal-sync",
            events_path=Path(self.temporary.name) / "terminal-events.jsonl",
        )
        session.request_stop("agent_finished")

        with (
            mock.patch(
                "op_bench.runtime.events.os.fsync",
                side_effect=(
                    None,
                    None,
                    None,
                    None,
                    None,
                    OSError("fixture parent fsync failure"),
                    None,
                    None,
                ),
            ),
            mock.patch(
                "op_bench.runtime.events.os.ftruncate",
                side_effect=OSError("fixture rollback failure"),
            ),
        ):
            result = session.finalize()

        self.assertEqual(result.terminal_reason, "agent_finished")
        self.assertEqual(session.state, "terminal")
        self.assertEqual(
            [event.event_type for event in session.events].count("terminal_emitted"),
            1,
        )

    def test_freeze_failure_is_workspace_error_and_has_no_patch(self) -> None:
        def fail_freeze():
            raise WorkspacePolicyError("fixture freeze failed at /Users/private")

        session = self.make_session(
            session_id="session-freeze-failure",
            freeze_patch=fail_freeze,
        )
        session.request_stop("agent_finished")

        result = session.finalize()

        self.assertEqual(result.terminal_reason, "workspace_error")
        self.assertIsNone(result.final_patch)
        encoded = str([event.to_dict() for event in session.events])
        self.assertNotIn("/Users/", encoded)
        self.assertIn("patch_freeze_failed", encoded)

    def test_termination_attribution_separates_infrastructure_from_agent_outcome(self) -> None:
        expected = {
            "agent_finished": ("valid", "finished", True),
            "agent_exited": ("valid", "exited", True),
            "budget_exhausted": ("valid", "budget", True),
            "timeout": ("valid", "timeout", True),
            "cancelled": ("valid", "cancelled", True),
            "workspace_error": ("infrastructure_invalid", None, False),
            "runtime_error": ("infrastructure_invalid", None, False),
            "provider_error": ("infrastructure_invalid", None, False),
            "platform_error": ("infrastructure_invalid", None, False),
        }
        for reason, values in expected.items():
            with self.subTest(reason=reason):
                actual = termination_attribution(reason)
                self.assertEqual(
                    (actual.attempt_validity, actual.agent_terminal, actual.scorable),
                    values,
                )

    def test_constructor_rejects_mismatched_authorities(self) -> None:
        other = EventJournal("other-session", clock_ms=self.clock)
        with self.assertRaisesRegex(ContractError, "journal.*session"):
            AttemptSession(
                spec=self.session.spec,
                action_service=self.session.action_service,
                journal=other,
                freeze_patch=lambda: None,
                clock_ms=self.clock,
            )


if __name__ == "__main__":
    unittest.main()
