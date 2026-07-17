from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.events import EventJournal
from op_bench.runtime.integrity import verify_run_artifacts
from op_bench.runtime.resources import (
    runtime_raw_handle_hash,
    runtime_resource_id,
)
from tests.test_runtime_integrity import (
    CompleteRun,
    build_complete_run,
    build_retry_run,
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def write_rechained_events(path: Path, payloads: list[dict[str, object]]) -> None:
    previous = None
    for sequence, payload in enumerate(payloads, start=1):
        payload["sequence"] = sequence
        payload["previous_event_hash"] = previous
        payload["event_hash"] = EventJournal.event_hash_for(
            session_id=payload["session_id"],
            sequence=sequence,
            occurred_at_ms=payload["occurred_at_ms"],
            event_type=payload["event_type"],
            public_payload=payload["public_payload"],
            previous_event_hash=previous,
        )
        previous = payload["event_hash"]
    path.write_text(
        "".join(canonical_json(payload) + "\n" for payload in payloads),
        encoding="utf-8",
    )


def attempt_path(
    complete: CompleteRun,
    name: str,
    *,
    retry_index: int = 1,
) -> Path:
    return (
        complete.root
        / "attempts"
        / complete.attempt_id
        / "retries"
        / f"retry-{retry_index:04d}"
        / name
    )


def remove_event_and_rechain(complete: CompleteRun, event_type: str) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    remaining = [
        payload for payload in payloads if payload["event_type"] != event_type
    ]
    if len(remaining) == len(payloads):
        raise AssertionError(f"fixture event missing: {event_type}")
    write_rechained_events(path, remaining)


def move_event_before_and_rechain(
    complete: CompleteRun,
    event_type: str,
    before_event_type: str,
) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    moved = next(
        payload for payload in payloads if payload["event_type"] == event_type
    )
    payloads.remove(moved)
    destination = next(
        index
        for index, payload in enumerate(payloads)
        if payload["event_type"] == before_event_type
    )
    payloads.insert(destination, moved)
    write_rechained_events(path, payloads)


def insert_event_before_and_rechain(
    complete: CompleteRun,
    event_type: str,
    public_payload: dict[str, object],
    before_event_type: str = "patch_freeze_started",
) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    destination = next(
        index
        for index, payload in enumerate(payloads)
        if payload["event_type"] == before_event_type
    )
    template = payloads[destination]
    payloads.insert(
        destination,
        {
            "contract_type": "event_record",
            "schema_version": "v1",
            "session_id": template["session_id"],
            "sequence": 0,
            "occurred_at_ms": template["occurred_at_ms"],
            "event_type": event_type,
            "public_payload": public_payload,
            "previous_event_hash": None,
            "event_hash": "sha256:" + "0" * 64,
        },
    )
    write_rechained_events(path, payloads)


def change_event_payload_and_rechain(
    complete: CompleteRun,
    event_type: str,
    public_payload: dict[str, object],
) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    selected = next(
        payload for payload in payloads if payload["event_type"] == event_type
    )
    selected["public_payload"] = public_payload
    write_rechained_events(path, payloads)


def update_event_payload_and_rechain(
    complete: CompleteRun,
    event_type: str,
    updates: dict[str, object],
    *,
    action_id: str | None = None,
) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    selected = next(
        payload
        for payload in payloads
        if payload["event_type"] == event_type
        and (
            action_id is None
            or payload["public_payload"].get("action_id") == action_id
        )
    )
    selected["public_payload"].update(updates)
    write_rechained_events(path, payloads)


def move_action_bundle_before_and_rechain(
    complete: CompleteRun,
    action_id: str,
    before_event_type: str,
) -> None:
    path = attempt_path(complete, "events.jsonl")
    payloads = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    bundle = [
        payload
        for payload in payloads
        if payload["public_payload"].get("action_id") == action_id
    ]
    if not bundle:
        raise AssertionError(f"fixture action bundle missing: {action_id}")
    remaining = [payload for payload in payloads if payload not in bundle]
    destination = next(
        index
        for index, payload in enumerate(remaining)
        if payload["event_type"] == before_event_type
    )
    remaining[destination:destination] = bundle
    write_rechained_events(path, remaining)


class RuntimeIntegrityMutationTests(unittest.TestCase):
    def assert_mutation_fails(self, expected_check: str, mutate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(Path(tmp) / "run")
            mutate(complete)

            report = verify_run_artifacts(complete.root, complete.manifest)

            checks = {check.check_id: check for check in report.checks}
            self.assertEqual(report.status, "failed")
            self.assertEqual(checks[expected_check].status, "failed")

    def test_manifest_matrix_and_retry_mutations_fail_closed(self) -> None:
        def changed_manifest(complete: CompleteRun) -> None:
            path = complete.root / "run_manifest.json"
            payload = read_json(path)
            payload["created_at"] = "2026-07-18T00:00:00Z"
            write_json(path, payload)

        def missing_attempt(complete: CompleteRun) -> None:
            (complete.root / "attempts" / complete.attempt_id).rename(
                complete.root / "missing-attempt-fixture"
            )

        def unexpected_attempt(complete: CompleteRun) -> None:
            (complete.root / "attempts" / ("attempt:v1:" + "f" * 64)).mkdir()

        def hidden_attempt(complete: CompleteRun) -> None:
            (complete.root / "attempts" / ".hidden-attempt").mkdir()

        def unexpected_retry(complete: CompleteRun) -> None:
            (
                complete.root
                / "attempts"
                / complete.attempt_id
                / "retries"
                / "retry-0002"
            ).mkdir()

        def invalid_retry(complete: CompleteRun) -> None:
            path = complete.root / "attempts.jsonl"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["retry_index"] = 2
            write_json(path, payload)

        for name, check, mutate in (
            ("manifest", "manifest_identity", changed_manifest),
            ("missing", "expected_observed_matrix", missing_attempt),
            ("unexpected", "expected_observed_matrix", unexpected_attempt),
            ("hidden", "expected_observed_matrix", hidden_attempt),
            ("unexpected-retry", "retry_audit", unexpected_retry),
            ("retry", "retry_audit", invalid_retry),
        ):
            with self.subTest(name=name):
                self.assert_mutation_fails(check, mutate)

    def test_runtime_resource_ownership_and_cleanup_mutations_fail_closed(self) -> None:
        def missing_public_ledger(complete: CompleteRun) -> None:
            attempt_path(complete, "runtime_resources.jsonl").unlink()

        def changed_private_handle(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_runtime_resources.json")
            payload = read_json(path)
            payload["handles"][0]["raw_handle"] += "-changed"
            write_json(path, payload)

        def cross_profile_private_store(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_runtime_resources.json")
            payload = read_json(path)
            payload["runtime_profile_hash"] = "sha256:" + "f" * 64
            write_json(path, payload)

        def cross_attempt_private_store(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_runtime_resources.json")
            payload = read_json(path)
            payload["attempt_id"] = "attempt:v1:" + "f" * 64
            write_json(path, payload)

        def unreferenced_private_handle(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_runtime_resources.json")
            payload = read_json(path)
            profile_hash = payload["runtime_profile_hash"]
            raw_handle = "pid:unreferenced-fixture"
            payload["handles"].append(
                {
                    "resource_id": runtime_resource_id(
                        complete.attempt_id,
                        1,
                        profile_hash,
                        "process",
                        1,
                    ),
                    "resource_type": "process",
                    "ordinal": 1,
                    "raw_handle": raw_handle,
                    "raw_handle_hash": runtime_raw_handle_hash(raw_handle),
                }
            )
            payload["handles"].sort(key=lambda item: item["resource_id"])
            write_json(path, payload)

        def active_resource(complete: CompleteRun) -> None:
            path = attempt_path(complete, "runtime_resources.jsonl")
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b"".join(lines[:-1]))

        def cleanup_failed(complete: CompleteRun) -> None:
            ledger_path = attempt_path(complete, "runtime_resources.jsonl")
            records = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            final = records[-1]
            final["transition"] = "cleanup_failed"
            unhashed = dict(final)
            del unhashed["record_hash"]
            final["record_hash"] = canonical_sha256(unhashed)
            ledger_path.write_text(
                "".join(canonical_json(record) + "\n" for record in records),
                encoding="utf-8",
            )

            cleanup_path = attempt_path(complete, "runtime_cleanup.json")
            cleanup = read_json(cleanup_path)
            cleanup["entries"][0].update(
                {
                    "status": "cleanup_failed",
                    "error_code": "runtime_cleanup_failed",
                }
            )
            cleanup["all_released"] = False
            write_json(cleanup_path, cleanup)

        def missing_cleanup_report(complete: CompleteRun) -> None:
            attempt_path(complete, "runtime_cleanup.json").unlink()

        for name, check, mutate in (
            ("missing-ledger", "runtime_resource_ownership", missing_public_ledger),
            ("private-handle", "runtime_resource_ownership", changed_private_handle),
            ("cross-profile", "runtime_resource_ownership", cross_profile_private_store),
            ("cross-attempt", "runtime_resource_ownership", cross_attempt_private_store),
            ("unreferenced", "runtime_resource_ownership", unreferenced_private_handle),
            ("active", "runtime_cleanup", active_resource),
            ("cleanup-failed", "runtime_cleanup", cleanup_failed),
            ("missing-cleanup", "runtime_cleanup", missing_cleanup_report),
        ):
            with self.subTest(name=name):
                self.assert_mutation_fails(check, mutate)

    def test_runtime_resource_evidence_is_required_for_every_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            complete = build_retry_run(Path(temporary) / "run")
            attempt_path(
                complete,
                "runtime_resources.jsonl",
                retry_index=1,
            ).unlink()

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(report.status, "failed")
            self.assertEqual(
                checks["runtime_resource_ownership"].status,
                "failed",
            )

    def test_task_view_event_and_patch_mutations_fail_closed(self) -> None:
        def changed_task_view(complete: CompleteRun) -> None:
            path = attempt_path(complete, "agent_task_view.json")
            payload = read_json(path)
            payload["statement_body"] = str(payload["statement_body"]) + " changed"
            write_json(path, payload)

        def deleted_event(complete: CompleteRun) -> None:
            path = attempt_path(complete, "events.jsonl")
            lines = path.read_bytes().splitlines(keepends=True)
            path.write_bytes(b"".join(lines[:1] + lines[2:]))

        def reordered_event(complete: CompleteRun) -> None:
            path = attempt_path(complete, "events.jsonl")
            lines = path.read_bytes().splitlines(keepends=True)
            lines[1], lines[2] = lines[2], lines[1]
            path.write_bytes(b"".join(lines))

        def changed_event(complete: CompleteRun) -> None:
            path = attempt_path(complete, "events.jsonl")
            lines = path.read_text(encoding="utf-8").splitlines()
            payload = json.loads(lines[0])
            payload["public_payload"]["attempt_id"] = "changed-attempt"
            lines[0] = canonical_json(payload)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        def changed_patch(complete: CompleteRun) -> None:
            attempt_path(complete, "final.patch").write_bytes(b"changed patch\n")

        for name, check, mutate in (
            ("task-view", "task_view_identity", changed_task_view),
            ("event-deleted", "event_chain", deleted_event),
            ("event-reordered", "event_chain", reordered_event),
            ("event-changed", "event_chain", changed_event),
            ("patch", "session_patch_evaluation_identity", changed_patch),
        ):
            with self.subTest(name=name):
                self.assert_mutation_fails(check, mutate)

    def test_evaluation_results_and_summary_mutations_fail_closed(self) -> None:
        def changed_public_evaluation(complete: CompleteRun) -> None:
            path = attempt_path(complete, "public_evaluation.json")
            payload = read_json(path)
            result = payload["evaluation_result"]
            result["duration_ms"] += 1
            payload["evaluation_result_hash"] = canonical_sha256(result)
            write_json(path, payload)

        def changed_private_evidence(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_evaluation.json")
            payload = read_json(path)
            payload["private_evidence"]["selectors"][0]["stdout"] = "changed private output"
            write_json(path, payload)

        def changed_private_spec(complete: CompleteRun) -> None:
            path = attempt_path(complete, "private_evaluation.json")
            payload = read_json(path)
            payload["evaluation_spec"]["timeout_ms"] += 1
            write_json(path, payload)

        def duplicate_result(complete: CompleteRun) -> None:
            path = complete.root / "results.jsonl"
            raw = path.read_bytes()
            path.write_bytes(raw + raw)

        def changed_result(complete: CompleteRun) -> None:
            path = complete.root / "results.jsonl"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["duration_ms"] += 1
            write_json(path, payload)

        def changed_summary(complete: CompleteRun) -> None:
            path = complete.root / "summary.json"
            payload = read_json(path)
            payload["results_hash"] = "sha256:" + "f" * 64
            write_json(path, payload)

        for name, check, mutate in (
            (
                "public-evaluation",
                "public_private_evaluation_identity",
                changed_public_evaluation,
            ),
            (
                "private-evidence",
                "public_private_evaluation_identity",
                changed_private_evidence,
            ),
            (
                "private-spec",
                "public_private_evaluation_identity",
                changed_private_spec,
            ),
            ("duplicate-result", "results_rebuild", duplicate_result),
            ("changed-result", "results_rebuild", changed_result),
            ("summary", "summary_rebuild", changed_summary),
        ):
            with self.subTest(name=name):
                self.assert_mutation_fails(check, mutate)

    def test_rehashed_spec_still_has_to_match_frozen_task_authority(self) -> None:
        def changed_timeout_with_consistent_hashes(complete: CompleteRun) -> None:
            private_path = attempt_path(complete, "private_evaluation.json")
            private_payload = read_json(private_path)
            private_payload["evaluation_spec"]["timeout_ms"] -= 1
            changed_spec_hash = canonical_sha256(
                private_payload["evaluation_spec"]
            )
            private_payload["evaluation_spec_hash"] = changed_spec_hash
            write_json(private_path, private_payload)

            public_path = attempt_path(complete, "public_evaluation.json")
            public_payload = read_json(public_path)
            public_payload["evaluation_spec_hash"] = changed_spec_hash
            write_json(public_path, public_payload)

            ledger_path = complete.root / "attempts.jsonl"
            ledger_payload = json.loads(
                ledger_path.read_text(encoding="utf-8")
            )
            ledger_payload["evaluation_spec_hash"] = changed_spec_hash
            write_json(ledger_path, ledger_payload)

            events_path = attempt_path(complete, "events.jsonl")
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            started = next(
                event
                for event in events
                if event["event_type"] == "evaluation_started"
            )
            started["public_payload"]["evaluation_spec_hash"] = changed_spec_hash
            completed = next(
                event
                for event in events
                if event["event_type"] == "evaluation_completed"
            )
            completed["public_payload"]["public_evaluation_hash"] = (
                canonical_sha256(public_payload)
            )
            completed["public_payload"]["private_evaluation_hash"] = (
                canonical_sha256(private_payload)
            )
            write_rechained_events(events_path, events)

        self.assert_mutation_fails(
            "public_private_evaluation_identity",
            changed_timeout_with_consistent_hashes,
        )

    def test_rehashed_private_evidence_must_match_public_result(self) -> None:
        def changed_selector_summary_with_consistent_hashes(
            complete: CompleteRun,
        ) -> None:
            private_path = attempt_path(complete, "private_evaluation.json")
            private_payload = read_json(private_path)
            summary = private_payload["private_evidence"]["selectors"][0][
                "summary"
            ]
            summary["passed"] = 0
            summary["failed"] = 1
            write_json(private_path, private_payload)

            events_path = attempt_path(complete, "events.jsonl")
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            completed = next(
                event
                for event in events
                if event["event_type"] == "evaluation_completed"
            )
            completed["public_payload"]["private_evaluation_hash"] = (
                canonical_sha256(private_payload)
            )
            write_rechained_events(events_path, events)

        self.assert_mutation_fails(
            "public_private_evaluation_identity",
            changed_selector_summary_with_consistent_hashes,
        )

    def test_rehashed_result_must_match_session_attribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(
                Path(tmp) / "run",
                terminal_reason="platform_error",
                freeze_failed=True,
            )
            public_path = attempt_path(complete, "public_evaluation.json")
            public_payload = read_json(public_path)
            result = public_payload["evaluation_result"]
            result["attempt_validity"] = "valid"
            result["evaluation_outcome"] = "no_patch"
            result["invalid_reason"] = None
            result_hash = canonical_sha256(result)
            public_payload["evaluation_result_hash"] = result_hash
            write_json(public_path, public_payload)

            private_path = attempt_path(complete, "private_evaluation.json")
            private_payload = read_json(private_path)
            private_payload["evaluation_result"] = result
            private_payload["evaluation_result_hash"] = result_hash
            write_json(private_path, private_payload)

            ledger_path = complete.root / "attempts.jsonl"
            ledger_payload = json.loads(
                ledger_path.read_text(encoding="utf-8")
            )
            ledger_payload["attempt_validity"] = "valid"
            ledger_payload["evaluation_result"] = result
            ledger_payload["evaluation_result_hash"] = result_hash
            write_json(ledger_path, ledger_payload)

            events_path = attempt_path(complete, "events.jsonl")
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            completed = next(
                event
                for event in events
                if event["event_type"] == "evaluation_completed"
            )
            completed["public_payload"].update(
                {
                    "attempt_validity": "valid",
                    "evaluation_outcome": "no_patch",
                    "evaluation_result_hash": result_hash,
                    "public_evaluation_hash": canonical_sha256(public_payload),
                    "private_evaluation_hash": canonical_sha256(private_payload),
                }
            )
            terminal = next(
                event
                for event in events
                if event["event_type"] == "terminal_emitted"
            )
            terminal["public_payload"].update(
                {
                    "attempt_validity": "valid",
                    "evaluation_outcome": "no_patch",
                    "evaluation_result_hash": result_hash,
                }
            )
            write_rechained_events(events_path, events)

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(
                checks["session_patch_evaluation_identity"].status,
                "failed",
            )

    def test_nonselected_retry_private_evaluation_is_still_protected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_retry_run(Path(tmp) / "run")
            path = attempt_path(
                complete,
                "private_evaluation.json",
                retry_index=1,
            )
            payload = read_json(path)
            payload["evaluation_result"]["invalid_reason"] = "changed_nonselected_retry"
            payload["evaluation_result_hash"] = canonical_sha256(
                payload["evaluation_result"]
            )
            write_json(path, payload)

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(report.status, "failed")
            self.assertEqual(
                checks["public_private_evaluation_identity"].status,
                "failed",
            )

    def test_missing_required_runtime_event_classes_fail_lifecycle_grammar(self) -> None:
        for event_type in (
            "session_prepared",
            "session_started",
            "agent_launched",
            "test_completed",
            "budget_updated",
            "patch_freeze_started",
            "patch_freeze_completed",
        ):
            with self.subTest(event_type=event_type), tempfile.TemporaryDirectory() as tmp:
                complete = build_complete_run(Path(tmp) / "run")
                remove_event_and_rechain(complete, event_type)

                report = verify_run_artifacts(complete.root, complete.manifest)
                checks = {check.check_id: check for check in report.checks}

                self.assertEqual(checks["event_chain"].status, "passed")
                self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_semantically_reordered_runtime_events_fail_lifecycle_grammar(self) -> None:
        for event_type, before_event_type in (
            ("test_completed", "test_started"),
            ("budget_updated", "action_observed"),
            ("patch_freeze_started", "action_requested"),
        ):
            with (
                self.subTest(
                    event_type=event_type,
                    before_event_type=before_event_type,
                ),
                tempfile.TemporaryDirectory() as tmp,
            ):
                complete = build_complete_run(Path(tmp) / "run")
                move_event_before_and_rechain(
                    complete,
                    event_type,
                    before_event_type,
                )

                report = verify_run_artifacts(complete.root, complete.manifest)
                checks = {check.check_id: check for check in report.checks}

                self.assertEqual(checks["event_chain"].status, "passed")
                self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_runtime_action_bundle_cannot_cross_session_evaluation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(Path(tmp) / "run")
            move_action_bundle_before_and_rechain(
                complete,
                "test-action-1",
                "evaluation_started",
            )

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(checks["event_chain"].status, "passed")
            self.assertEqual(checks["action_pairing"].status, "passed")
            self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_action_budget_stop_must_follow_matching_budget_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(
                Path(tmp) / "run",
                terminal_reason="budget_exhausted",
                action_derived_budget=True,
            )
            move_event_before_and_rechain(
                complete,
                "budget_exhausted",
                "budget_updated",
            )

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(checks["event_chain"].status, "passed")
            self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_observed_infrastructure_error_has_terminal_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(Path(tmp) / "run")
            update_event_payload_and_rechain(
                complete,
                "action_observed",
                {"ok": False, "error_code": "runtime_error"},
                action_id="test-action-1",
            )

            report = verify_run_artifacts(complete.root, complete.manifest)
            checks = {check.check_id: check for check in report.checks}

            self.assertEqual(checks["event_chain"].status, "passed")
            self.assertEqual(checks["action_pairing"].status, "passed")
            self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_action_auxiliary_payloads_bind_the_observation(self) -> None:
        for name, event_type, updates in (
            (
                "test-completed",
                "test_completed",
                {"ok": False, "error_code": "runtime_error"},
            ),
            (
                "budget-updated",
                "budget_updated",
                {"budget_delta": {"actions": 999}},
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                complete = build_complete_run(Path(tmp) / "run")
                update_event_payload_and_rechain(
                    complete,
                    event_type,
                    updates,
                    action_id="test-action-1",
                )

                report = verify_run_artifacts(complete.root, complete.manifest)
                checks = {check.check_id: check for check in report.checks}

                self.assertEqual(checks["event_chain"].status, "passed")
                self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_terminal_reason_requires_matching_stop_evidence(self) -> None:
        for terminal_reason, event_type in (
            ("agent_finished", "finish_requested"),
            ("agent_exited", "agent_exited"),
            ("budget_exhausted", "budget_exhausted"),
            ("timeout", "timeout_requested"),
            ("cancelled", "cancel_requested"),
        ):
            with self.subTest(terminal_reason=terminal_reason), tempfile.TemporaryDirectory() as tmp:
                complete = build_complete_run(
                    Path(tmp) / "run",
                    terminal_reason=terminal_reason,
                )
                remove_event_and_rechain(complete, event_type)

                report = verify_run_artifacts(complete.root, complete.manifest)
                checks = {check.check_id: check for check in report.checks}

                self.assertEqual(checks["event_chain"].status, "passed")
                self.assertEqual(checks["lifecycle_terminal"].status, "failed")

    def test_orphan_duplicate_and_malformed_stop_events_fail_grammar(self) -> None:
        mutations = (
            (
                "higher-priority-timeout",
                "agent_finished",
                lambda complete: insert_event_before_and_rechain(
                    complete,
                    "timeout_requested",
                    {"reason": "timeout"},
                ),
            ),
            (
                "higher-priority-cancel",
                "timeout",
                lambda complete: insert_event_before_and_rechain(
                    complete,
                    "cancel_requested",
                    {"reason": "cancelled"},
                ),
            ),
            (
                "duplicate-timeout",
                "timeout",
                lambda complete: insert_event_before_and_rechain(
                    complete,
                    "timeout_requested",
                    {"reason": "timeout"},
                ),
            ),
            (
                "malformed-cancel",
                "cancelled",
                lambda complete: change_event_payload_and_rechain(
                    complete,
                    "cancel_requested",
                    {"reason": "timeout"},
                ),
            ),
        )
        for name, terminal_reason, mutate in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                complete = build_complete_run(
                    Path(tmp) / "run",
                    terminal_reason=terminal_reason,
                )
                mutate(complete)

                report = verify_run_artifacts(complete.root, complete.manifest)
                checks = {check.check_id: check for check in report.checks}

                self.assertEqual(checks["event_chain"].status, "passed")
                self.assertEqual(checks["lifecycle_terminal"].status, "failed")


if __name__ == "__main__":
    unittest.main()
