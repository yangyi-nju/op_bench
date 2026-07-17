from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.artifacts import ArtifactReference, PublicArtifactStore
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.events import (
    EventJournal,
    verify_action_pairing,
    verify_event_chain,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_wire_contracts import action_observation, action_request


class StepClock:
    def __init__(self, value: int = 100) -> None:
        self.value = value

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class EventJournalTests(unittest.TestCase):
    def test_sequence_previous_hash_and_hash_preimage_are_canonical(self) -> None:
        journal = EventJournal("session-001", clock_ms=StepClock())

        first = journal.append("session_created", {"attempt_id": "attempt-001"})
        second = journal.append("session_prepared", {})

        self.assertEqual((first.sequence, second.sequence), (1, 2))
        self.assertIsNone(first.previous_event_hash)
        self.assertEqual(second.previous_event_hash, first.event_hash)
        self.assertEqual(verify_event_chain(journal.records), ())
        self.assertEqual(
            first.event_hash,
            EventJournal.event_hash_for(
                session_id="session-001",
                sequence=1,
                occurred_at_ms=100,
                event_type="session_created",
                public_payload={"attempt_id": "attempt-001"},
                previous_event_hash=None,
            ),
        )

    def test_chain_verifier_detects_sequence_payload_previous_and_hash_mutation(self) -> None:
        journal = EventJournal("session-001", clock_ms=StepClock())
        first = journal.append("session_created", {"attempt_id": "attempt-001"})
        second = journal.append("session_prepared", {})

        fixtures = (
            replace(second, sequence=3),
            replace(second, public_payload={"changed": True}),
            replace(second, previous_event_hash=None),
            replace(second, event_hash="sha256:" + "f" * 64),
        )
        for changed in fixtures:
            with self.subTest(changed=changed):
                issues = verify_event_chain((first, changed))
                self.assertTrue(issues)

    def test_duplicate_terminal_is_rejected_without_growing_the_journal(self) -> None:
        journal = EventJournal("session-001", clock_ms=StepClock())
        journal.append("session_created", {"attempt_id": "attempt-001"})
        terminal = journal.append("terminal_emitted", {"reason": "agent_finished"})

        with self.assertRaisesRegex(ContractError, "terminal event already exists"):
            journal.append("terminal_emitted", {"reason": "timeout"})

        self.assertEqual(journal.records[-1], terminal)
        self.assertEqual(len(journal.records), 2)

    def test_durable_journal_reopens_and_fails_closed_on_a_corrupt_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            journal = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            journal.append("session_created", {"attempt_id": "attempt-001"})
            journal.append("session_prepared", {})

            reopened = EventJournal("session-001", clock_ms=StepClock(200), events_path=path)
            third = reopened.append("session_started", {})
            self.assertEqual(third.sequence, 3)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)

            with path.open("a", encoding="utf-8") as handle:
                handle.write('{"truncated":')
            with self.assertRaisesRegex(ContractError, "final newline"):
                EventJournal("session-001", clock_ms=StepClock(), events_path=path)

    def test_reopen_rejects_missing_newline_wrong_session_and_unsafe_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "events.jsonl"
            journal = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            record = journal.append("session_created", {"attempt_id": "attempt-001"})

            path.write_bytes(path.read_bytes().removesuffix(b"\n"))
            with self.assertRaisesRegex(ContractError, "final newline"):
                EventJournal("session-001", clock_ms=StepClock(), events_path=path)

            path.write_bytes((canonical_json(record.to_dict()) + "\n").encode("utf-8"))
            with self.assertRaisesRegex(ContractError, "session"):
                EventJournal("session-002", clock_ms=StepClock(), events_path=path)

            unsafe = record.to_dict()
            unsafe["public_payload"] = {"path": "/Users/private/secret"}
            unsafe["event_hash"] = EventJournal.event_hash_for(
                session_id="session-001",
                sequence=1,
                occurred_at_ms=record.occurred_at_ms,
                event_type=record.event_type,
                public_payload=unsafe["public_payload"],
                previous_event_hash=None,
            )
            path.write_bytes((canonical_json(unsafe) + "\n").encode("utf-8"))
            with self.assertRaisesRegex(ContractError, "public artifact"):
                EventJournal("session-001", clock_ms=StepClock(), events_path=path)

    def test_two_live_journals_serialize_against_the_durable_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            left = EventJournal("session-001", clock_ms=lambda: 100, events_path=path)
            right = EventJournal("session-001", clock_ms=lambda: 101, events_path=path)
            barrier = threading.Barrier(3)
            errors: list[BaseException] = []

            def append(journal: EventJournal, event_type: str) -> None:
                barrier.wait()
                try:
                    journal.append(event_type, {})
                except BaseException as exc:  # noqa: BLE001 - race evidence.
                    errors.append(exc)

            threads = (
                threading.Thread(target=append, args=(left, "session_created")),
                threading.Thread(target=append, args=(right, "session_prepared")),
            )
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

            self.assertEqual(errors, [])
            reopened = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            self.assertEqual([event.sequence for event in reopened.records], [1, 2])
            self.assertEqual(verify_event_chain(reopened.records), ())

    def test_live_journal_records_refresh_and_deny_lost_durable_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            left = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            right = EventJournal("session-001", clock_ms=StepClock(200), events_path=path)
            unseen = EventJournal("session-001", clock_ms=StepClock(300), events_path=path)
            first = left.append("session_created", {})

            self.assertEqual(right.records, (first,))
            path.unlink()
            with self.assertRaisesRegex(ContractError, "durable history"):
                unseen.append("session_prepared", {})

    def test_uncertain_partial_append_poisoned_instead_of_starting_a_new_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            journal = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            journal.append("session_created", {})
            real_write = __import__("os").write
            calls = [0]

            def partial_then_fail(descriptor: int, content) -> int:
                calls[0] += 1
                if calls[0] == 1:
                    data = bytes(content)
                    return real_write(descriptor, data[: max(1, len(data) // 2)])
                raise OSError("fixture write failure")

            with (
                mock.patch(
                    "op_bench.runtime.events.os.write",
                    side_effect=partial_then_fail,
                ),
                mock.patch(
                    "op_bench.runtime.events.os.ftruncate",
                    side_effect=OSError("fixture rollback failure"),
                ),
            ):
                with self.assertRaisesRegex(ContractError, "uncertain|poison"):
                    journal.append("session_prepared", {})

            with self.assertRaisesRegex(ContractError, "poison"):
                journal.append("session_started", {})

    def test_closed_journal_never_degrades_to_memory_only_append(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            journal = EventJournal("session-001", clock_ms=StepClock(), events_path=path)
            journal.append("session_created", {})
            before = path.read_bytes()
            journal.close()

            with self.assertRaisesRegex(ContractError, "closed"):
                journal.append("session_prepared", {})
            self.assertEqual(path.read_bytes(), before)

    def test_journal_rejects_directory_and_fifo_without_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "directory.jsonl"
            directory.mkdir()
            fifo = root / "fifo.jsonl"
            __import__("os").mkfifo(fifo)

            for path in (directory, fifo):
                with self.subTest(path=path.name):
                    with self.assertRaisesRegex(
                        ContractError, "invalid file|regular file"
                    ):
                        EventJournal("session-001", clock_ms=StepClock(), events_path=path)

    def test_journal_constructor_failure_closes_file_and_parent_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            closed: list[int] = []
            real_close = __import__("os").close

            def close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with (
                mock.patch(
                    "op_bench.runtime.events.os.fsync",
                    side_effect=OSError("fixture fsync failure"),
                ),
                mock.patch("op_bench.runtime.events.os.close", side_effect=close),
            ):
                with self.assertRaises(OSError):
                    EventJournal(
                        "session-001",
                        clock_ms=StepClock(),
                        events_path=Path(tmp) / "events.jsonl",
                    )

            self.assertEqual(len(closed), 2)
            self.assertEqual(len(set(closed)), 2)

    def test_journal_parent_binding_survives_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound = root / "bound"
            bound.mkdir()
            moved = root / "moved"
            outside = root / "outside"
            outside.mkdir()
            journal = EventJournal(
                "session-001",
                clock_ms=StepClock(),
                events_path=bound / "events.jsonl",
            )

            bound.rename(moved)
            bound.symlink_to(outside, target_is_directory=True)
            journal.append("session_created", {})

            self.assertFalse((outside / "events.jsonl").exists())
            self.assertTrue((moved / "events.jsonl").is_file())

    def test_action_helpers_pair_once_and_spill_large_observation_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PublicArtifactStore(Path(tmp) / "artifacts")
            journal = EventJournal(
                "session-001",
                clock_ms=StepClock(),
                artifact_store=store,
                max_inline_bytes=80,
            )
            request = action_request()
            observation = replace(
                action_observation(),
                data={"content": "x" * 256},
            )

            journal.record_action_requested(request)
            journal.record_action_observed(request, observation)

            self.assertEqual(verify_action_pairing(journal.records), ())
            self.assertEqual(
                [event.event_type for event in journal.records],
                ["action_requested", "action_observed", "budget_updated"],
            )
            observed_payload = journal.records[1].public_payload
            self.assertNotIn("data", observed_payload)
            reference = observed_payload["data_artifact"]
            self.assertEqual(reference["media_type"], "application/json")
            self.assertEqual(
                json.loads(store.read_bytes(reference).decode("utf-8")),
                {"content": "x" * 256},
            )

            journal.record_action_requested(request)
            self.assertNotEqual(verify_action_pairing(journal.records), ())

    def test_action_pairing_rejects_mismatched_request_identity(self) -> None:
        journal = EventJournal("session-001", clock_ms=StepClock())
        requested = action_request()
        observed_request = replace(
            requested,
            action_name="workspace_write",
            arguments={"path": "src/operator.py", "content": "VALUE = 2\n"},
        )

        journal.record_action_requested(requested)
        journal.record_action_observed(observed_request, action_observation())

        issues = verify_action_pairing(journal.records)
        self.assertTrue(any("request_hash" in issue for issue in issues))
        self.assertTrue(any("action_name" in issue for issue in issues))

    def test_test_finish_and_budget_helpers_emit_required_event_types(self) -> None:
        journal = EventJournal("session-001", clock_ms=StepClock())
        test_request = replace(
            action_request(),
            action_id="test-action",
            action_name="test_run",
            arguments={"selector_id": "public::smoke"},
        )
        exhausted = replace(
            action_observation(),
            action_id="test-action",
            ok=False,
            error_code="budget_exhausted",
            message="budget exhausted",
            data={},
        )
        finish_request = replace(
            action_request(),
            action_id="finish-action",
            action_name="session_finish",
            arguments={},
        )

        journal.record_action_requested(test_request)
        journal.record_action_observed(test_request, exhausted)
        journal.record_action_requested(finish_request)

        self.assertEqual(
            [event.event_type for event in journal.records],
            [
                "action_requested",
                "test_started",
                "action_observed",
                "test_completed",
                "budget_updated",
                "budget_exhausted",
                "finish_requested",
                "action_requested",
            ],
        )

    def test_public_values_are_scanned_before_event_or_artifact_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = PublicArtifactStore(root / "artifacts")
            journal = EventJournal(
                "session-001",
                clock_ms=StepClock(),
                events_path=root / "events.jsonl",
                artifact_store=store,
            )

            with self.assertRaisesRegex(ContractError, "public artifact"):
                journal.append("session_created", {"path": "/Users/private/task"})
            with self.assertRaisesRegex(ContractError, "public artifact"):
                store.put_json("unsafe", {"api_key": "fixture-secret"})

            self.assertEqual((root / "events.jsonl").read_bytes(), b"")
            self.assertEqual(
                tuple(path for path in (root / "artifacts").glob("**/*") if path.is_file()),
                (),
            )


class PublicArtifactStoreTests(unittest.TestCase):
    def test_json_artifacts_are_content_addressed_deduplicated_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PublicArtifactStore(Path(tmp) / "artifacts")
            payload = {"content": "safe output", "exit_code": 0}

            first = store.put_json("first", payload)
            second = store.put_json("second", payload)

            self.assertEqual(first, second)
            self.assertEqual(first.size_bytes, len(canonical_json(payload).encode("utf-8")))
            self.assertEqual(store.read_bytes(first), canonical_json(payload).encode("utf-8"))
            self.assertEqual(len(tuple((Path(tmp) / "artifacts/public").iterdir())), 1)

            stored_path = Path(tmp) / "artifacts" / first.artifact_id
            stored_path.write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "artifact (?:size|digest) mismatch"):
                store.read_bytes(first)

    def test_store_rejects_a_symlink_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            real = root / "real"
            real.mkdir()
            link = root / "linked"
            link.symlink_to(real, target_is_directory=True)

            with self.assertRaisesRegex(ContractError, "artifact root.*symlink"):
                PublicArtifactStore(link)

    def test_directory_binding_prevents_post_construction_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store_root = root / "artifacts"
            store = PublicArtifactStore(store_root)
            moved = root / "bound-public"
            outside = root / "outside"
            outside.mkdir()
            (store_root / "public").rename(moved)
            (store_root / "public").symlink_to(outside, target_is_directory=True)

            reference = store.put_json("safe", {"content": "safe"})

            self.assertEqual(tuple(outside.iterdir()), ())
            self.assertTrue((moved / reference.artifact_id.removeprefix("public/")).is_file())
            self.assertEqual(store.read_bytes(reference), b'{"content":"safe"}')

    def test_read_rejects_noncanonical_or_publicly_unsafe_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            store = PublicArtifactStore(root)

            for content, message in (
                (b'{"safe": true}', "canonical"),
                (canonical_json({"path": "/Users/private/secret"}).encode("utf-8"), "public artifact"),
            ):
                with self.subTest(message=message):
                    digest = hashlib.sha256(content).hexdigest()
                    reference = ArtifactReference(
                        artifact_id=f"public/sha256-{digest}.json",
                        digest=f"sha256:{digest}",
                        size_bytes=len(content),
                        media_type="application/json",
                    )
                    (root / reference.artifact_id).write_bytes(content)
                    with self.assertRaisesRegex(ContractError, message):
                        store.read_bytes(reference)

    def test_failed_write_removes_private_temporary_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "artifacts"
            store = PublicArtifactStore(root)

            with mock.patch(
                "op_bench.runtime.artifacts.os.write",
                side_effect=OSError("fixture write failure"),
            ):
                with self.assertRaises(OSError):
                    store.put_json("safe", {"content": "safe"})

            self.assertEqual(tuple((root / "public").iterdir()), ())

    def test_closed_artifact_store_fails_with_contract_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PublicArtifactStore(Path(tmp) / "artifacts")
            reference = store.put_json("safe", {"content": "safe"})
            store.close()

            with self.assertRaisesRegex(ContractError, "closed"):
                store.read_bytes(reference)
            with self.assertRaisesRegex(ContractError, "closed"):
                store.put_json("other", {"content": "other"})

    def test_constructor_failure_closes_both_bound_directory_descriptors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            closed: list[int] = []
            real_close = __import__("os").close

            def close(descriptor: int) -> None:
                closed.append(descriptor)
                real_close(descriptor)

            with (
                mock.patch(
                    "op_bench.runtime.artifacts.os.fsync",
                    side_effect=OSError("fixture fsync failure"),
                ),
                mock.patch("op_bench.runtime.artifacts.os.close", side_effect=close),
            ):
                with self.assertRaises(OSError):
                    PublicArtifactStore(Path(tmp) / "artifacts")

            self.assertEqual(len(closed), 2)
            self.assertEqual(len(set(closed)), 2)


if __name__ == "__main__":
    unittest.main()
