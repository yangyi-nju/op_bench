from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from op_bench.runtime.contracts import EvaluationResultV06, SessionResult
from op_bench.runtime.resume import (
    AttemptLedger,
    AttemptLedgerRecord,
    ResumeDecision,
)
from op_bench.runtime.session import termination_attribution
from op_bench.runtime.validation import ContractError
from tests.test_runtime_manifest import manifest
from tests.test_runtime_wire_contracts import evaluation_result, session_result


ATTEMPT_A = "attempt:v1:" + "a" * 64
ATTEMPT_B = "attempt:v1:" + "b" * 64
EVALUATION_SPEC_HASH = "sha256:" + "e" * 64


def result(
    *,
    attempt_id: str = ATTEMPT_A,
    session_id: str = "session-retry-1",
    terminal_reason: str = "platform_error",
) -> SessionResult:
    return replace(
        session_result(),
        session_id=session_id,
        attempt_id=attempt_id,
        terminal_reason=terminal_reason,
        final_patch=None,
    )


def evaluated(
    session: SessionResult,
    *,
    attempt_validity: str | None = None,
) -> EvaluationResultV06:
    attribution = termination_attribution(session.terminal_reason)
    validity = attempt_validity or attribution.attempt_validity
    if validity == "valid":
        outcome = "no_patch" if session.final_patch is None else "resolved"
        invalid_reason = None
    else:
        outcome = (
            "not_evaluated"
            if attribution.attempt_validity == "infrastructure_invalid"
            else "evaluation_error"
        )
        invalid_reason = (
            f"session_{session.terminal_reason}"
            if attribution.attempt_validity == "infrastructure_invalid"
            else "fixture_evaluation_infrastructure_error"
        )
    return replace(
        evaluation_result(),
        session_id=session.session_id,
        attempt_id=session.attempt_id,
        attempt_validity=validity,
        agent_terminal=attribution.agent_terminal,
        evaluation_outcome=outcome,
        invalid_reason=invalid_reason,
        patch=session.final_patch,
    )


def append_evaluated(
    ledger: AttemptLedger,
    *,
    session_result: SessionResult,
    retry_index: int,
    recorded_at_ms: int,
    attempt_validity: str | None = None,
    evaluation_spec_hash: str = EVALUATION_SPEC_HASH,
) -> AttemptLedgerRecord:
    return ledger.append(
        session_result=session_result,
        evaluation_result=evaluated(
            session_result,
            attempt_validity=attempt_validity,
        ),
        evaluation_spec_hash=evaluation_spec_hash,
        retry_index=retry_index,
        recorded_at_ms=recorded_at_ms,
    )


class AttemptLedgerTests(unittest.TestCase):
    def test_append_is_fsynced_canonical_idempotent_and_strictly_monotonic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            ledger = AttemptLedger(path)
            first_result = result()

            with mock.patch("op_bench.runtime.resume.os.fsync") as fsync:
                first = append_evaluated(
                    ledger,
                    session_result=first_result,
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_000,
                )
            self.assertEqual(fsync.call_count, 2)
            self.assertEqual(first.session_result_hash, first_result.content_hash)
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                json.dumps(first.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n",
            )

            duplicate = append_evaluated(
                ledger,
                session_result=first_result,
                attempt_validity="infrastructure_invalid",
                retry_index=1,
                recorded_at_ms=1_000,
            )
            self.assertIs(duplicate, first)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

            with self.assertRaisesRegex(ContractError, "session_id.*conflicting"):
                append_evaluated(
                    ledger,
                    session_result=first_result,
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_001,
                )
            with self.assertRaisesRegex(ContractError, "session attribution"):
                ledger.append(
                    session_result=first_result,
                    evaluation_result=replace(
                        evaluated(first_result),
                        invalid_reason="different_evaluation_failure",
                    ),
                    evaluation_spec_hash=EVALUATION_SPEC_HASH,
                    retry_index=1,
                    recorded_at_ms=1_000,
                )
            with self.assertRaisesRegex(ContractError, "retry_index.*expected 2"):
                append_evaluated(
                    ledger,
                    session_result=result(session_id="session-retry-3"),
                    attempt_validity="infrastructure_invalid",
                    retry_index=3,
                    recorded_at_ms=1_002,
                )

            valid_result = result(
                session_id="session-retry-2",
                terminal_reason="agent_finished",
            )
            valid = append_evaluated(
                ledger,
                session_result=valid_result,
                attempt_validity="valid",
                retry_index=2,
                recorded_at_ms=1_003,
            )
            self.assertEqual(ledger.latest_valid(ATTEMPT_A), valid)
            self.assertEqual(len(ledger.records(ATTEMPT_A)), 2)

            with self.assertRaisesRegex(ContractError, "valid result already exists"):
                append_evaluated(
                    ledger,
                    session_result=result(session_id="session-retry-3"),
                    attempt_validity="infrastructure_invalid",
                    retry_index=3,
                    recorded_at_ms=1_004,
                )

            reopened = AttemptLedger(path)
            self.assertEqual(reopened.records(), ledger.records())
            self.assertEqual(reopened.latest_valid(ATTEMPT_A), valid)

    def test_resume_policy_truth_table_is_read_only_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty = AttemptLedger(root / "empty.jsonl")
            self.assertEqual(
                empty.decide(ATTEMPT_A, "skip_valid"),
                ResumeDecision("run", 1, "no prior attempt"),
            )
            self.assertEqual(
                empty.decide(ATTEMPT_A, "retry_infrastructure"),
                ResumeDecision("run", 1, "no prior attempt"),
            )
            self.assertEqual(
                empty.decide(ATTEMPT_A, "never"),
                ResumeDecision("run", 1, "no prior attempt"),
            )

            invalid = AttemptLedger(root / "invalid.jsonl")
            append_evaluated(
                invalid,
                session_result=result(terminal_reason="agent_finished"),
                attempt_validity="infrastructure_invalid",
                retry_index=1,
                recorded_at_ms=1_000,
            )
            before = (root / "invalid.jsonl").read_bytes()
            self.assertEqual(invalid.decide(ATTEMPT_A, "skip_valid").action, "blocked")
            retry = invalid.decide(ATTEMPT_A, "retry_infrastructure")
            self.assertEqual((retry.action, retry.retry_index), ("run", 2))
            self.assertEqual(invalid.decide(ATTEMPT_A, "never").action, "blocked")
            self.assertEqual((root / "invalid.jsonl").read_bytes(), before)

            valid = AttemptLedger(root / "valid.jsonl")
            append_evaluated(
                valid,
                session_result=result(terminal_reason="agent_finished"),
                attempt_validity="valid",
                retry_index=1,
                recorded_at_ms=1_000,
            )
            self.assertEqual(valid.decide(ATTEMPT_A, "skip_valid").action, "skip")
            self.assertEqual(
                valid.decide(ATTEMPT_A, "retry_infrastructure").action,
                "skip",
            )
            self.assertEqual(valid.decide(ATTEMPT_A, "never").action, "blocked")

    def test_retry_history_remains_append_only_while_logical_result_is_one_valid_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            ledger = AttemptLedger(path)
            for retry in (1, 2):
                append_evaluated(
                    ledger,
                    session_result=result(
                        session_id=f"session-retry-{retry}",
                        terminal_reason="agent_finished",
                    ),
                    attempt_validity="infrastructure_invalid",
                    retry_index=retry,
                    recorded_at_ms=1_000 + retry,
                )
            final = append_evaluated(
                ledger,
                session_result=result(
                    session_id="session-retry-3",
                    terminal_reason="agent_finished",
                ),
                attempt_validity="valid",
                retry_index=3,
                recorded_at_ms=1_003,
            )

            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 3)
            self.assertEqual(len(ledger.records(ATTEMPT_A)), 3)
            self.assertEqual(ledger.latest_valid(ATTEMPT_A), final)
            self.assertEqual(ledger.decide(ATTEMPT_A, "retry_infrastructure").action, "skip")

    def test_invalid_records_and_corrupt_or_symlink_ledgers_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ledger = AttemptLedger(root / "attempts.jsonl")
            session = result(terminal_reason="agent_finished")
            with self.assertRaisesRegex(ContractError, "agent_terminal.*SessionResult"):
                ledger.append(
                    session_result=session,
                    evaluation_result=replace(evaluated(session), agent_terminal=None),
                    evaluation_spec_hash=EVALUATION_SPEC_HASH,
                    retry_index=1,
                    recorded_at_ms=1_000,
                )
            with self.assertRaisesRegex(
                ContractError,
                "patch.*SessionResult.*EvaluationResult",
            ):
                ledger.append(
                    session_result=session,
                    evaluation_result=replace(
                        evaluated(session),
                        patch=evaluation_result().patch,
                    ),
                    evaluation_spec_hash=EVALUATION_SPEC_HASH,
                    retry_index=1,
                    recorded_at_ms=1_000,
                )
            with self.assertRaisesRegex(ContractError, "attempt_id"):
                append_evaluated(
                    ledger,
                    session_result=result(attempt_id="not-an-attempt"),
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_000,
                )

            corrupt = root / "corrupt.jsonl"
            corrupt.write_text('{"truncated":', encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "attempt ledger.*(?:line 1|final newline)"):
                AttemptLedger(corrupt)

            real = root / "real.jsonl"
            real.write_text("", encoding="utf-8")
            linked = root / "linked.jsonl"
            linked.symlink_to(real)
            with self.assertRaisesRegex(ContractError, "ledger path.*symlink"):
                AttemptLedger(linked)

    def test_reopen_rejects_missing_final_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            ledger = AttemptLedger(path)
            append_evaluated(
                ledger,
                session_result=result(),
                attempt_validity="infrastructure_invalid",
                retry_index=1,
                recorded_at_ms=1_000,
            )
            path.write_bytes(path.read_bytes().removesuffix(b"\n"))

            with self.assertRaisesRegex(ContractError, "final newline"):
                AttemptLedger(path)

    def test_two_live_ledgers_cannot_both_append_the_same_retry_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            left = AttemptLedger(path)
            right = AttemptLedger(path)
            barrier = threading.Barrier(3)
            successes: list[AttemptLedgerRecord] = []
            errors: list[BaseException] = []

            def append(ledger: AttemptLedger, session_id: str) -> None:
                barrier.wait()
                try:
                    successes.append(
                        append_evaluated(
                            ledger,
                            session_result=result(session_id=session_id),
                            attempt_validity="infrastructure_invalid",
                            retry_index=1,
                            recorded_at_ms=1_000,
                        )
                    )
                except BaseException as exc:  # noqa: BLE001 - race evidence.
                    errors.append(exc)

            threads = (
                threading.Thread(target=append, args=(left, "session-concurrent-left")),
                threading.Thread(target=append, args=(right, "session-concurrent-right")),
            )
            for thread in threads:
                thread.start()
            barrier.wait()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

            self.assertEqual(len(successes), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ContractError)
            self.assertEqual(len(AttemptLedger(path).records()), 1)

    def test_live_ledger_decisions_refresh_and_deny_lost_durable_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "attempts.jsonl"
            left = AttemptLedger(path)
            right = AttemptLedger(path)
            unseen = AttemptLedger(path)
            append_evaluated(
                left,
                session_result=result(terminal_reason="agent_finished"),
                attempt_validity="valid",
                retry_index=1,
                recorded_at_ms=1_000,
            )

            self.assertEqual(right.decide(ATTEMPT_A, "skip_valid").action, "skip")
            path.unlink()
            with self.assertRaisesRegex(ContractError, "durable history"):
                unseen.records()

    def test_closed_ledger_fails_instead_of_using_stale_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ledger = AttemptLedger(Path(tmp) / "attempts.jsonl")
            ledger.close()

            with self.assertRaisesRegex(ContractError, "closed"):
                ledger.records()
            with self.assertRaisesRegex(ContractError, "closed"):
                append_evaluated(
                    ledger,
                    session_result=result(),
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_000,
                )

    def test_ledger_rejects_directory_and_fifo_without_blocking(self) -> None:
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
                        AttemptLedger(path)

    def test_uncertain_complete_ledger_commit_reconciles_and_partial_tail_poisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            committed = AttemptLedger(Path(tmp) / "committed.jsonl")
            with (
                mock.patch(
                    "op_bench.runtime.resume.os.fsync",
                    side_effect=(
                        None,
                        OSError("fixture parent fsync failure"),
                        None,
                        None,
                    ),
                ),
                mock.patch(
                    "op_bench.runtime.resume.os.ftruncate",
                    side_effect=OSError("fixture rollback failure"),
                ),
            ):
                record = append_evaluated(
                    committed,
                    session_result=result(),
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_000,
                )
            self.assertEqual(committed.records(), (record,))

            poisoned = AttemptLedger(Path(tmp) / "poisoned.jsonl")
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
                    "op_bench.runtime.resume.os.write",
                    side_effect=partial_then_fail,
                ),
                mock.patch(
                    "op_bench.runtime.resume.os.ftruncate",
                    side_effect=OSError("fixture rollback failure"),
                ),
            ):
                with self.assertRaisesRegex(ContractError, "uncertain|poison"):
                    append_evaluated(
                        poisoned,
                        session_result=result(session_id="session-poisoned-1"),
                        attempt_validity="infrastructure_invalid",
                        retry_index=1,
                        recorded_at_ms=1_000,
                    )
            with self.assertRaisesRegex(ContractError, "poison"):
                append_evaluated(
                    poisoned,
                    session_result=result(session_id="session-poisoned-2"),
                    attempt_validity="infrastructure_invalid",
                    retry_index=1,
                    recorded_at_ms=1_001,
                )

    def test_ledger_parent_binding_survives_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bound = root / "bound"
            bound.mkdir()
            moved = root / "moved"
            outside = root / "outside"
            outside.mkdir()
            ledger = AttemptLedger(bound / "attempts.jsonl")

            bound.rename(moved)
            bound.symlink_to(outside, target_is_directory=True)
            append_evaluated(
                ledger,
                session_result=result(),
                attempt_validity="infrastructure_invalid",
                retry_index=1,
                recorded_at_ms=1_000,
            )

            self.assertFalse((outside / "attempts.jsonl").exists())
            self.assertTrue((moved / "attempts.jsonl").is_file())

    def test_attempt_identity_changes_for_frozen_effective_axes(self) -> None:
        base = manifest()
        variants = (
            manifest(tasks=(replace(base.tasks[0], statement_body="changed task"),)),
            manifest(agents=(replace(base.agents[0], task_prompt=replace(
                base.agents[0].task_prompt,
                identifier="changed-prompt",
            )),)),
            manifest(capability=replace(base.capability_policy, max_read_bytes=123_456)),
            manifest(budget=replace(base.budget_policy, max_actions=123)),
            manifest(action_protocol="action-v2"),
            manifest(tasks=(replace(
                base.tasks[0],
                runtime=replace(base.tasks[0].runtime, timeout_ms=123_456),
            ),)),
        )

        baseline_id = base.expected_attempts[0].attempt_id
        self.assertTrue(all(item.expected_attempts[0].attempt_id != baseline_id for item in variants))
        self.assertEqual(len({baseline_id, *(item.expected_attempts[0].attempt_id for item in variants)}), 7)

    def test_record_round_trip_rejects_hash_and_session_mismatch(self) -> None:
        record = AttemptLedgerRecord(
            attempt_id=ATTEMPT_A,
            session_id="session-retry-1",
            retry_index=1,
            attempt_validity="infrastructure_invalid",
            session_result=result(),
            session_result_hash=result().content_hash,
            evaluation_result=evaluated(result()),
            evaluation_result_hash=evaluated(result()).content_hash,
            evaluation_spec_hash=EVALUATION_SPEC_HASH,
            recorded_at_ms=1_000,
        )
        self.assertEqual(AttemptLedgerRecord.from_dict(record.to_dict()), record)

        changed_hash = record.to_dict()
        changed_hash["session_result_hash"] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(ContractError, "session_result_hash"):
            AttemptLedgerRecord.from_dict(changed_hash)

        changed_session = record.to_dict()
        changed_session["session_id"] = "other-session"
        with self.assertRaisesRegex(ContractError, "session_id.*SessionResult"):
            AttemptLedgerRecord.from_dict(changed_session)

        changed_evaluation_hash = record.to_dict()
        changed_evaluation_hash["evaluation_result_hash"] = "sha256:" + "f" * 64
        with self.assertRaisesRegex(ContractError, "evaluation_result_hash"):
            AttemptLedgerRecord.from_dict(changed_evaluation_hash)

        changed_spec_hash = record.to_dict()
        changed_spec_hash["evaluation_spec_hash"] = "not-a-hash"
        with self.assertRaisesRegex(ContractError, "evaluation_spec_hash"):
            AttemptLedgerRecord.from_dict(changed_spec_hash)

    def test_infrastructure_session_cannot_be_recorded_as_valid(self) -> None:
        session = result(terminal_reason="platform_error")
        fabricated = evaluated(session, attempt_validity="valid")

        with self.assertRaisesRegex(ContractError, "session attribution"):
            AttemptLedgerRecord(
                attempt_id=ATTEMPT_A,
                session_id=session.session_id,
                retry_index=1,
                attempt_validity="valid",
                session_result=session,
                session_result_hash=session.content_hash,
                evaluation_result=fabricated,
                evaluation_result_hash=fabricated.content_hash,
                evaluation_spec_hash=EVALUATION_SPEC_HASH,
                recorded_at_ms=1_000,
            )


if __name__ == "__main__":
    unittest.main()
