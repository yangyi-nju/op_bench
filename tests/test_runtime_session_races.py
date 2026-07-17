from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.actions import CanonicalActionService
from op_bench.runtime.contracts import ACTION_NAMES, ActionRequest, SessionSpec
from op_bench.runtime.events import EventJournal, verify_event_chain
from op_bench.runtime.session import AttemptSession
from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    FrozenPatch,
    raw_patch_identity,
)
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_actions_service import FakeCommandBackend
from tests.test_runtime_contracts import SHA_A, budget_policy, capability_policy, identity
from tests.test_runtime_wire_contracts import session_spec
from tests.test_runtime_workspace import policy as workspace_policy


PRIORITY = (
    "platform_error",
    "workspace_error",
    "runtime_error",
    "provider_error",
    "cancelled",
    "timeout",
    "budget_exhausted",
    "agent_finished",
    "agent_exited",
)


class AttemptSessionRaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        self.base_commit = initialize_git_repo(self.root)
        self.source = identity("source", "fixture@session-races", SHA_A)
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_session(
        self,
        suffix: str,
        *,
        prepare: bool = True,
        start: bool = True,
    ):
        session_id = f"session-race-{suffix}"
        workspace = AuthoritativeWorkspace.open(
            self.root,
            source=self.source,
            policy=workspace_policy(),
        )
        journal = EventJournal(session_id, clock_ms=lambda: 1_000)
        service = CanonicalActionService(
            session_id=session_id,
            workspace=workspace,
            capability_policy=self.capabilities,
            budget_policy=self.budget,
            command_backend=FakeCommandBackend(),
            test_registry={},
            clock_ms=lambda: 1_000,
            event_journal=journal,
        )
        base = session_spec()
        spec = SessionSpec(
            session_id=session_id,
            attempt_id=f"attempt-race-{suffix}",
            workspace=workspace.identity,
            agent_task_view=base.agent_task_view,
            capability_policy=self.capabilities,
            budget_policy=self.budget,
            deadline_ms=5_000,
            adapter_config=base.adapter_config,
            runtime=base.runtime,
            artifact_root_id=f"attempts/race-{suffix}",
            resume_policy="retry_infrastructure",
        )
        frozen = FrozenPatch(
            workspace=workspace.identity,
            source=self.source,
            base_commit=self.base_commit,
            patch=raw_patch_identity(b"", identifier="final.patch"),
            patch_bytes=b"",
            changed_paths=(),
            empty=True,
        )
        freeze_calls = [0]
        freeze_lock = threading.Lock()

        def freeze() -> FrozenPatch:
            with freeze_lock:
                freeze_calls[0] += 1
            return frozen

        session = AttemptSession(
            spec=spec,
            action_service=service,
            journal=journal,
            freeze_patch=freeze,
            clock_ms=lambda: 1_000,
        )
        if prepare:
            session.prepare()
            session.mark_ready()
            if start:
                session.start()
        return session, freeze_calls

    def test_every_stop_reason_pair_uses_fixed_priority_before_finalize_seal(self) -> None:
        case = 0
        for left_index, left in enumerate(PRIORITY):
            for right in PRIORITY[left_index + 1 :]:
                case += 1
                with self.subTest(left=left, right=right):
                    session, freeze_calls = self.make_session(str(case))
                    barrier = threading.Barrier(3)

                    def request(reason: str) -> None:
                        barrier.wait()
                        session.request_stop(reason)

                    threads = (
                        threading.Thread(target=request, args=(left,)),
                        threading.Thread(target=request, args=(right,)),
                    )
                    for thread in threads:
                        thread.start()
                    barrier.wait()
                    for thread in threads:
                        thread.join(timeout=2)
                        self.assertFalse(thread.is_alive())

                    results: list[object] = []
                    finalize_barrier = threading.Barrier(7)

                    def finalize() -> None:
                        finalize_barrier.wait()
                        results.append(session.finalize())

                    finalizers = [threading.Thread(target=finalize) for _ in range(6)]
                    for thread in finalizers:
                        thread.start()
                    finalize_barrier.wait()
                    for thread in finalizers:
                        thread.join(timeout=2)
                        self.assertFalse(thread.is_alive())

                    self.assertEqual(len(results), 6)
                    self.assertTrue(all(result is results[0] for result in results))
                    self.assertEqual(results[0].terminal_reason, left)
                    self.assertEqual(freeze_calls, [1])
                    self.assertEqual(
                        [event.event_type for event in session.events].count("terminal_emitted"),
                        1,
                    )
                    self.assertEqual(verify_event_chain(session.events), ())

    def test_signal_after_finalize_seal_cannot_change_terminal_or_event_tail(self) -> None:
        session, freeze_calls = self.make_session("sealed")
        session.request_stop("agent_finished")
        result = session.finalize()
        events = session.events

        self.assertEqual(session.request_stop("platform_error"), "terminal")
        self.assertIs(session.finalize(), result)
        self.assertEqual(result.terminal_reason, "agent_finished")
        self.assertEqual(session.events, events)
        self.assertEqual(freeze_calls, [1])

    def test_finalize_waits_for_admitted_action_through_publication(self) -> None:
        session, _ = self.make_session("active-action")
        entered = threading.Event()
        release = threading.Event()
        original = session._journal.record_action_requested

        def blocked(request):
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return original(request)

        request = ActionRequest(
            session_id=session.spec.session_id,
            action_id="write-during-stop",
            action_name="workspace_write",
            arguments={"path": "src/operator.py", "content": "VALUE = 2\n"},
            client_sequence=1,
            deadline_ms=5_000,
        )
        action_results: list[object] = []
        final_results: list[object] = []
        with mock.patch.object(
            session._journal, "record_action_requested", side_effect=blocked
        ):
            action = threading.Thread(
                target=lambda: action_results.append(session.execute_action(request))
            )
            action.start()
            self.assertTrue(entered.wait(timeout=1))
            session.request_stop("agent_exited")
            finalizer = threading.Thread(target=lambda: final_results.append(session.finalize()))
            finalizer.start()
            finalizer.join(timeout=0.1)
            waiting = finalizer.is_alive()
            release.set()
            action.join(timeout=2)
            finalizer.join(timeout=2)

        self.assertTrue(waiting, "freeze must wait for an admitted Action")
        self.assertFalse(action.is_alive())
        self.assertFalse(finalizer.is_alive())
        self.assertTrue(action_results[0].ok)
        self.assertEqual(final_results[0].terminal_reason, "agent_exited")
        self.assertEqual(session.events[-1].event_type, "terminal_emitted")
        self.assertEqual((self.root / "src/operator.py").read_text(), "VALUE = 2\n")

    def test_finalize_waits_for_pending_stop_event_publisher(self) -> None:
        session, _ = self.make_session("pending-stop-event")
        entered = threading.Event()
        release = threading.Event()
        original = session._journal.append

        def blocked(event_type, payload):
            if event_type == "cancel_requested":
                entered.set()
                self.assertTrue(release.wait(timeout=2))
            return original(event_type, payload)

        with mock.patch.object(session._journal, "append", side_effect=blocked):
            stopper = threading.Thread(target=lambda: session.request_stop("cancelled"))
            stopper.start()
            self.assertTrue(entered.wait(timeout=1))
            final_results: list[object] = []
            finalizer = threading.Thread(target=lambda: final_results.append(session.finalize()))
            finalizer.start()
            finalizer.join(timeout=0.1)
            waiting = finalizer.is_alive()
            release.set()
            stopper.join(timeout=2)
            finalizer.join(timeout=2)

        self.assertTrue(waiting, "terminal must wait for pending lifecycle publication")
        self.assertFalse(stopper.is_alive())
        self.assertFalse(finalizer.is_alive())
        self.assertEqual(final_results[0].terminal_reason, "cancelled")
        self.assertEqual(session.events[-1].event_type, "terminal_emitted")

    def test_action_waits_until_session_started_event_is_durable(self) -> None:
        session, _ = self.make_session("start-publication", start=False)
        entered = threading.Event()
        release = threading.Event()
        original = session._journal.append

        def blocked(event_type, payload):
            if event_type == "session_started":
                entered.set()
                self.assertTrue(release.wait(timeout=2))
            return original(event_type, payload)

        request = ActionRequest(
            session_id=session.spec.session_id,
            action_id="read-after-start",
            action_name="workspace_read",
            arguments={"path": "src/operator.py"},
            client_sequence=1,
            deadline_ms=5_000,
        )
        action_results: list[object] = []
        with mock.patch.object(session._journal, "append", side_effect=blocked):
            starter = threading.Thread(target=session.start)
            starter.start()
            self.assertTrue(entered.wait(timeout=1))
            action = threading.Thread(
                target=lambda: action_results.append(session.execute_action(request))
            )
            action.start()
            action.join(timeout=0.1)
            waiting = action.is_alive()
            release.set()
            starter.join(timeout=2)
            action.join(timeout=2)

        self.assertTrue(waiting, "Action must wait for session_started persistence")
        self.assertFalse(starter.is_alive())
        self.assertFalse(action.is_alive())
        self.assertTrue(action_results[0].ok)
        event_types = [event.event_type for event in session.events]
        self.assertLess(event_types.index("session_started"), event_types.index("action_requested"))

    def test_lifecycle_transition_waits_for_preceding_event_publication(self) -> None:
        session, _ = self.make_session(
            "ordered-lifecycle",
            prepare=False,
            start=False,
        )
        entered = threading.Event()
        release = threading.Event()
        original = session._journal.append

        def blocked(event_type, payload):
            if event_type == "session_prepared":
                entered.set()
                self.assertTrue(release.wait(timeout=2))
            return original(event_type, payload)

        with mock.patch.object(session._journal, "append", side_effect=blocked):
            preparer = threading.Thread(target=session.prepare)
            preparer.start()
            self.assertTrue(entered.wait(timeout=1))
            transitions = threading.Thread(
                target=lambda: (session.mark_ready(), session.start())
            )
            transitions.start()
            transitions.join(timeout=0.1)
            waiting = transitions.is_alive()
            release.set()
            preparer.join(timeout=2)
            transitions.join(timeout=2)

        self.assertTrue(waiting, "later lifecycle transitions must await prior durability")
        self.assertFalse(preparer.is_alive())
        self.assertFalse(transitions.is_alive())
        self.assertEqual(session.state, "running")
        event_types = [event.event_type for event in session.events]
        self.assertLess(event_types.index("session_prepared"), event_types.index("session_started"))


if __name__ == "__main__":
    unittest.main()
