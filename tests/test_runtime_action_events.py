from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.actions import CanonicalActionService, RegisteredTest
from op_bench.runtime.artifacts import PublicArtifactStore
from op_bench.runtime.contracts import ACTION_NAMES, ActionRequest
from op_bench.runtime.events import EventJournal, verify_action_pairing, verify_event_chain
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import AuthoritativeWorkspace
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_actions_service import FakeCommandBackend
from tests.test_runtime_contracts import SHA_A, budget_policy, capability_policy, identity
from tests.test_runtime_workspace import policy as workspace_policy


class CanonicalActionEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)
        self.journal = EventJournal(
            "session-actions",
            clock_ms=lambda: 1_000,
            events_path=Path(self.temporary.name) / "events.jsonl",
            artifact_store=PublicArtifactStore(Path(self.temporary.name) / "artifacts"),
            max_inline_bytes=80,
        )
        self.backend = FakeCommandBackend()
        self.service = self.make_service(self.journal)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def make_service(self, journal: EventJournal) -> CanonicalActionService:
        return CanonicalActionService(
            session_id="session-actions",
            workspace=AuthoritativeWorkspace.open(
                self.root,
                source=identity("source", "fixture@action-events", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=ACTION_NAMES,
                writable_paths=("src/",),
                allowed_command_prefixes=("git diff",),
                registered_tests=("public::smoke",),
                max_read_bytes=1_024,
                max_output_bytes=1_024,
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=100_000,
                max_actions=20,
                max_tests=2,
                max_commands=2,
                max_output_bytes=10_000,
            ),
            command_backend=self.backend,
            test_registry={
                "public::smoke": RegisteredTest(
                    selector_id="public::smoke",
                    command=("python", "-m", "unittest", "private.runtime.selector"),
                    cwd=".",
                    timeout_ms=500,
                )
            },
            clock_ms=lambda: 1_000,
            event_journal=journal,
        )

    @staticmethod
    def request(
        action_id: str,
        action_name: str,
        arguments: dict[str, object],
        sequence: int,
        *,
        session_id: str = "session-actions",
        deadline_ms: int = 2_000,
    ) -> ActionRequest:
        return ActionRequest(
            session_id=session_id,
            action_id=action_id,
            action_name=action_name,
            arguments=arguments,
            client_sequence=sequence,
            deadline_ms=deadline_ms,
        )

    def test_unique_success_and_rejections_each_have_one_event_pair(self) -> None:
        read = self.request("read", "workspace_read", {"path": "src/operator.py"}, 1)
        denied = self.request(
            "denied",
            "workspace_write",
            {"path": "tests/test_operator.py", "content": "bad\n"},
            2,
        )

        read_observation = self.service.execute(read)
        event_count = len(self.journal.records)
        self.assertIs(self.service.execute(read), read_observation)
        self.assertEqual(len(self.journal.records), event_count)
        self.assertEqual(self.service.execute(denied).error_code, "path_denied")

        self.assertEqual(verify_event_chain(self.journal.records), ())
        self.assertEqual(verify_action_pairing(self.journal.records), ())
        action_events = [
            event.event_type
            for event in self.journal.records
            if event.event_type in {"action_requested", "action_observed"}
        ]
        self.assertEqual(
            action_events,
            ["action_requested", "action_observed"] * 2,
        )

    def test_wrong_session_and_conflicting_redelivery_do_not_pollute_active_events(self) -> None:
        wrong = self.request(
            "shared-id",
            "workspace_read",
            {"path": "src/operator.py"},
            1,
            session_id="other-session",
        )
        self.assertEqual(self.service.execute(wrong).error_code, "session_not_running")
        self.assertEqual(self.journal.records, ())

        accepted = self.request(
            "shared-id",
            "workspace_read",
            {"path": "src/operator.py"},
            1,
        )
        self.assertTrue(self.service.execute(accepted).ok)
        before = self.journal.records
        conflict = replace(accepted, arguments={"path": "src/helper.py"})
        self.assertEqual(self.service.execute(conflict).error_code, "conflict")
        self.assertEqual(self.journal.records, before)
        self.assertEqual(verify_action_pairing(self.journal.records), ())

    def test_test_and_finish_events_hide_registry_command_and_large_data(self) -> None:
        test = self.request("test", "test_run", {"selector_id": "public::smoke"}, 1)
        finish = self.request("finish", "session_finish", {}, 2)

        self.assertTrue(self.service.execute(test).ok)
        self.assertTrue(self.service.execute(finish).ok)

        encoded = "\n".join(str(event.to_dict()) for event in self.journal.records)
        self.assertNotIn("private.runtime.selector", encoded)
        self.assertNotIn("/Users/", encoded)
        self.assertNotIn("AuthoritativeWorkspace", encoded)
        self.assertIn("data_artifact", encoded)
        self.assertEqual(verify_action_pairing(self.journal.records), ())
        self.assertIn("test_started", [event.event_type for event in self.journal.records])
        self.assertIn("test_completed", [event.event_type for event in self.journal.records])
        self.assertIn("finish_requested", [event.event_type for event in self.journal.records])

    def test_request_event_failure_prevents_mutation_and_returns_stable_platform_error(self) -> None:
        request = self.request(
            "write",
            "workspace_write",
            {"path": "src/operator.py", "content": "VALUE = 2\n"},
            1,
        )

        with mock.patch.object(
            self.journal,
            "record_action_requested",
            side_effect=ContractError("private persistence at /Users/runtime"),
        ):
            observation = self.service.execute(request)

        self.assertEqual(observation.error_code, "platform_error")
        self.assertEqual(observation.message, "action trajectory persistence failed")
        self.assertEqual((self.root / "src/operator.py").read_text(), "VALUE = 1\n")
        self.assertNotIn("/Users/", str(observation.to_dict()))
        self.assertEqual(self.journal.records, ())

    def test_observation_event_failure_preserves_mutation_state_and_seals_service(self) -> None:
        request = self.request(
            "write",
            "workspace_write",
            {"path": "src/operator.py", "content": "VALUE = 2\n"},
            1,
        )

        with mock.patch.object(
            self.journal,
            "record_action_observed",
            side_effect=ContractError("private persistence at /Users/runtime"),
        ):
            observation = self.service.execute(request)

        self.assertEqual(observation.error_code, "platform_error")
        self.assertEqual(observation.mutation_state, "mutated")
        self.assertEqual((self.root / "src/operator.py").read_text(), "VALUE = 2\n")
        self.assertNotIn("/Users/", str(observation.to_dict()))
        self.assertTrue(verify_action_pairing(self.journal.records))

        rejected = self.service.execute(
            self.request("after-failure", "workspace_read", {"path": "src/operator.py"}, 2)
        )
        self.assertEqual(rejected.error_code, "session_not_running")

    def test_observation_batch_failure_rolls_back_and_records_returned_platform_error(self) -> None:
        request = self.request("read-atomic", "workspace_read", {"path": "src/operator.py"}, 1)

        with mock.patch(
            "op_bench.runtime.events.os.fsync",
            side_effect=(
                None,
                None,
                OSError("fixture fsync failure"),
                None,
                None,
                None,
                None,
                None,
            ),
        ):
            observation = self.service.execute(request)

        self.assertEqual(observation.error_code, "platform_error")
        reopened = EventJournal(
            "session-actions",
            clock_ms=lambda: 1_000,
            events_path=Path(self.temporary.name) / "events.jsonl",
            artifact_store=PublicArtifactStore(Path(self.temporary.name) / "artifacts"),
            max_inline_bytes=80,
        )
        observed = [
            event for event in reopened.records if event.event_type == "action_observed"
        ]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].public_payload["observation_hash"], observation.content_hash)
        self.assertEqual(observed[0].public_payload["error_code"], "platform_error")
        self.assertEqual(verify_action_pairing(reopened.records), ())

    def test_uncertain_but_complete_observation_commit_is_reconciled_without_duplicate(self) -> None:
        request = self.request(
            "denied-uncertain",
            "workspace_write",
            {"path": "tests/denied.py", "content": "bad\n"},
            1,
        )

        with (
            mock.patch(
                "op_bench.runtime.events.os.fsync",
                side_effect=(
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
            observation = self.service.execute(request)

        self.assertEqual(observation.error_code, "path_denied")
        self.assertEqual(verify_action_pairing(self.journal.records), ())
        observed = [
            event for event in self.journal.records if event.event_type == "action_observed"
        ]
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0].public_payload["observation_hash"], observation.content_hash)

    def test_malformed_finish_at_exhausted_budget_is_cached_after_publication(self) -> None:
        self.service.budget_policy = replace(self.service.budget_policy, max_actions=0)
        request = self.request(
            "finish-malformed",
            "session_finish",
            {"unexpected": True},
            1,
        )

        first = self.service.execute(request)
        count = len(self.journal.records)
        second = self.service.execute(request)

        self.assertIs(second, first)
        self.assertEqual(len(self.journal.records), count)
        self.assertEqual(verify_action_pairing(self.journal.records), ())


if __name__ == "__main__":
    unittest.main()
