from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.contracts import IntegrityReport
from op_bench.runtime.evaluation import (
    AttemptEvaluationCoordinator,
    FreshEvaluator,
    PrivateEvaluationEvidence,
)
from op_bench.runtime.events import EventJournal
from op_bench.runtime.integrity import (
    persist_integrity_reports,
    selected_attempts_from_ledger,
    verify_run_artifacts,
)
from op_bench.runtime.resume import AttemptLedger
from op_bench.runtime.resources import (
    AttemptResourceLedger,
    RuntimeCleanupEntry,
    RuntimeCleanupReport,
    RuntimeLeaseStore,
)
from op_bench.runtime.run_artifacts import AttemptArtifactStore
from op_bench.runtime.session import termination_attribution
from op_bench.runtime.summary import SelectedAttempt, write_rebuilt_outputs
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_evaluation import execution
from tests.test_runtime_manifest import manifest
from tests.test_runtime_wire_contracts import (
    action_observation,
    action_request,
    evaluation_spec,
    session_result,
)


CHECK_IDS = (
    "manifest_identity",
    "expected_observed_matrix",
    "retry_audit",
    "task_view_identity",
    "event_chain",
    "action_pairing",
    "lifecycle_terminal",
    "runtime_resource_ownership",
    "runtime_cleanup",
    "session_patch_evaluation_identity",
    "public_private_evaluation_identity",
    "evaluation_protocol_scoring_identity",
    "results_rebuild",
    "summary_rebuild",
)


class StepClock:
    def __init__(self, start: int) -> None:
        self.value = start

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class EvidenceBackend:
    def evaluate(self, spec, frozen_patch):
        return PrivateEvaluationEvidence(
            source=spec.source,
            patch=frozen_patch.patch,
            hidden_test_asset=spec.hidden_test_asset,
            selectors=(
                execution("hidden::f2p", "fail_to_pass"),
                execution("public::smoke", "pass_to_pass"),
            ),
            cleanup_completed=True,
        )


def write_runtime_resource_evidence(
    store: AttemptArtifactStore,
    *,
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
) -> None:
    ledger = AttemptResourceLedger(
        store.runtime_resources_path(attempt_id, retry_index=retry_index),
        attempt_id=attempt_id,
        retry_index=retry_index,
        runtime_profile_hash=runtime_profile_hash,
        clock_ms=StepClock(4_000 + retry_index * 10),
    )
    lease_store = RuntimeLeaseStore(
        store.private_runtime_resources_path(attempt_id, retry_index=retry_index),
        attempt_id=attempt_id,
        retry_index=retry_index,
        runtime_profile_hash=runtime_profile_hash,
    )
    declared = ledger.declare("workspace", 1)
    handle = lease_store.put_exact(
        declared.resource_id,
        "workspace",
        1,
        f"/private/runtime/{attempt_id[-12:]}/retry-{retry_index:04d}",
    )
    ledger.created(declared.resource_id, handle.raw_handle_hash)
    ledger.released(declared.resource_id)
    store.write_runtime_cleanup(
        attempt_id,
        RuntimeCleanupReport(
            attempt_id=attempt_id,
            retry_index=retry_index,
            runtime_profile_hash=runtime_profile_hash,
            entries=(
                RuntimeCleanupEntry(
                    resource_id=declared.resource_id,
                    resource_type="workspace",
                    status="released",
                    error_code=None,
                ),
            ),
            all_released=True,
        ),
        retry_index=retry_index,
    )
    store.write_runtime_conformance(
        attempt_id,
        {
            "report_type": "runtime_conformance",
            "schema_version": "v1",
            "status": "not_applicable",
            "entries": [],
        },
        retry_index=retry_index,
    )
    ledger.close()
    lease_store.close()


def append_runtime_evidence(
    journal: EventJournal,
    *,
    retry_index: int,
    patch,
    freeze_failed: bool,
    terminal_reason: str,
    action_derived_budget: bool = False,
) -> None:
    journal.append("session_prepared", {})
    journal.append("session_started", {})
    journal.append("agent_launched", {})
    test_request = replace(
        action_request(),
        session_id=journal.session_id,
        action_id=f"test-action-{retry_index}",
        action_name="test_run",
        arguments={"selector_id": "public::smoke"},
        client_sequence=1,
    )
    test_observation = replace(
        action_observation(),
        session_id=journal.session_id,
        action_id=test_request.action_id,
        ok=not action_derived_budget,
        error_code=("budget_exhausted" if action_derived_budget else "ok"),
        data={},
    )
    journal.record_action_requested(test_request)
    journal.record_action_observed(test_request, test_observation)
    if terminal_reason == "agent_finished":
        finish_request = replace(
            action_request(),
            session_id=journal.session_id,
            action_id=f"finish-action-{retry_index}",
            action_name="session_finish",
            arguments={},
            client_sequence=2,
        )
        finish_observation = replace(
            action_observation(),
            session_id=journal.session_id,
            action_id=finish_request.action_id,
            data={},
        )
        journal.record_action_requested(finish_request)
        journal.record_action_observed(finish_request, finish_observation)
    elif terminal_reason == "agent_exited":
        journal.append("agent_exited", {"exit_code": 0})
    elif terminal_reason == "budget_exhausted" and not action_derived_budget:
        journal.append("budget_exhausted", {"reason": terminal_reason})
    elif terminal_reason == "timeout":
        journal.append("timeout_requested", {"reason": terminal_reason})
    elif terminal_reason == "cancelled":
        journal.append("cancel_requested", {"reason": terminal_reason})
    journal.append("patch_freeze_started", {})
    if freeze_failed:
        journal.append("patch_freeze_failed", {"error_code": "workspace_error"})
    else:
        journal.append(
            "patch_freeze_completed",
            {"patch": patch.to_dict(), "empty": False},
        )


@dataclass(frozen=True)
class CompleteRun:
    root: Path
    manifest: object
    attempt_id: str
    session_result: object
    evaluation_spec_hash: str
    retry_count: int = 1


def build_complete_run(
    root: Path,
    *,
    terminal_reason: str = "agent_finished",
    action_derived_budget: bool = False,
    freeze_failed: bool = False,
) -> CompleteRun:
    frozen_manifest = manifest(repeat_count=1)
    expected = frozen_manifest.expected_attempts[0]
    task = frozen_manifest.tasks[0]
    store = AttemptArtifactStore(root, frozen_manifest)
    store.write_run_manifest()
    journal = EventJournal(
        "session-integrity-1",
        clock_ms=StepClock(100),
        events_path=store.events_path(expected.attempt_id),
    )
    journal.append("session_created", {"attempt_id": expected.attempt_id})

    if freeze_failed:
        patch = None
        frozen = None
        patch_artifact = None
    else:
        patch_bytes = b"fixture patch\n"
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        frozen = FrozenPatch(
            workspace=identity("workspace", "integrity-workspace", SHA_B),
            source=task.source,
            base_commit="fixture-base",
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=("src/operator.py",),
            empty=False,
        )
        patch_artifact = build_patch_artifact(
            frozen,
            artifact_id=f"{expected.attempt_id}/final.patch",
        )
    session = replace(
        session_result(),
        session_id=journal.session_id,
        attempt_id=expected.attempt_id,
        terminal_reason=terminal_reason,
        final_patch=patch,
    )
    spec = replace(
        evaluation_spec(),
        session_id=session.session_id,
        attempt_id=session.attempt_id,
        task=task.task,
        source=task.source,
        frozen_patch=patch,
        hidden_test_asset=task.hidden_test_asset,
        runtime=task.runtime,
        evaluation=identity(
            "evaluation",
            frozen_manifest.evaluation_protocol,
            SHA_B,
        ),
        scoring=frozen_manifest.scoring,
    )
    store.write_session_inputs(
        expected.attempt_id,
        frozen_manifest.task_views[0],
        session,
        frozen,
        patch_artifact,
    )
    append_runtime_evidence(
        journal,
        retry_index=1,
        patch=patch,
        freeze_failed=freeze_failed,
        terminal_reason=terminal_reason,
        action_derived_budget=action_derived_budget,
    )
    journal.append(
        "session_terminal_emitted",
        {
            "attempt_id": session.attempt_id,
            "terminal_reason": session.terminal_reason,
            "session_result_hash": session.content_hash,
            "final_patch": (
                None
                if session.final_patch is None
                else session.final_patch.to_dict()
            ),
            "session_validity": termination_attribution(
                session.terminal_reason
            ).attempt_validity,
        },
    )
    coordinator = AttemptEvaluationCoordinator(
        FreshEvaluator(EvidenceBackend(), clock_ms=StepClock(1_000)),
        journal,
        store,
        clock_ms=StepClock(2_000),
    )
    completed = coordinator.complete(session, spec, frozen, patch_artifact)
    write_runtime_resource_evidence(
        store,
        attempt_id=expected.attempt_id,
        retry_index=1,
        runtime_profile_hash=task.runtime.content_hash,
    )

    ledger = AttemptLedger(root / "attempts.jsonl")
    ledger.append(
        session_result=session,
        evaluation_result=completed.result,
        evaluation_spec_hash=completed.evaluation_spec_hash,
        retry_index=1,
        recorded_at_ms=3_000,
    )
    write_rebuilt_outputs(
        store,
        (
            SelectedAttempt(
                attempt_id=expected.attempt_id,
                retry_index=1,
                evaluation_spec_hash=completed.evaluation_spec_hash,
                evaluation_result=completed.result,
            ),
        ),
    )
    ledger.close()
    journal.close()
    store.close()
    return CompleteRun(
        root=root,
        manifest=frozen_manifest,
        attempt_id=expected.attempt_id,
        session_result=session,
        evaluation_spec_hash=completed.evaluation_spec_hash,
    )


def build_retry_run(root: Path) -> CompleteRun:
    frozen_manifest = manifest(repeat_count=1)
    expected = frozen_manifest.expected_attempts[0]
    task = frozen_manifest.tasks[0]
    store = AttemptArtifactStore(root, frozen_manifest)
    store.write_run_manifest()
    ledger = AttemptLedger(root / "attempts.jsonl")
    selected_session = None
    selected_completed = None

    for retry_index in (1, 2):
        journal = EventJournal(
            f"session-integrity-retry-{retry_index}",
            clock_ms=StepClock(100 * retry_index),
            events_path=store.events_path(
                expected.attempt_id,
                retry_index=retry_index,
            ),
        )
        journal.append("session_created", {"attempt_id": expected.attempt_id})
        if retry_index == 1:
            frozen = None
            patch_artifact = None
            session = replace(
                session_result(),
                session_id=journal.session_id,
                attempt_id=expected.attempt_id,
                terminal_reason="platform_error",
                final_patch=None,
            )
            spec = replace(
                evaluation_spec(),
                session_id=session.session_id,
                attempt_id=session.attempt_id,
                task=task.task,
                source=task.source,
                frozen_patch=None,
                hidden_test_asset=task.hidden_test_asset,
                runtime=task.runtime,
                evaluation=identity(
                    "evaluation",
                    frozen_manifest.evaluation_protocol,
                    SHA_B,
                ),
                scoring=frozen_manifest.scoring,
            )
        else:
            patch_bytes = b"fixture retry patch\n"
            patch = raw_patch_identity(patch_bytes, identifier="final.patch")
            frozen = FrozenPatch(
                workspace=identity("workspace", "retry-workspace", SHA_B),
                source=task.source,
                base_commit="fixture-base",
                patch=patch,
                patch_bytes=patch_bytes,
                changed_paths=("src/operator.py",),
                empty=False,
            )
            patch_artifact = build_patch_artifact(
                frozen,
                artifact_id=f"{expected.attempt_id}/retry-2/final.patch",
            )
            session = replace(
                session_result(),
                session_id=journal.session_id,
                attempt_id=expected.attempt_id,
                final_patch=patch,
            )
            spec = replace(
                evaluation_spec(),
                session_id=session.session_id,
                attempt_id=session.attempt_id,
                task=task.task,
                source=task.source,
                frozen_patch=patch,
                hidden_test_asset=task.hidden_test_asset,
                runtime=task.runtime,
                evaluation=identity(
                    "evaluation",
                    frozen_manifest.evaluation_protocol,
                    SHA_B,
                ),
                scoring=frozen_manifest.scoring,
            )
        store.write_session_inputs(
            expected.attempt_id,
            frozen_manifest.task_views[0],
            session,
            frozen,
            patch_artifact,
            retry_index=retry_index,
        )
        append_runtime_evidence(
            journal,
            retry_index=retry_index,
            patch=None if frozen is None else frozen.patch,
            freeze_failed=frozen is None,
            terminal_reason=session.terminal_reason,
        )
        journal.append(
            "session_terminal_emitted",
            {
                "attempt_id": session.attempt_id,
                "terminal_reason": session.terminal_reason,
                "session_result_hash": session.content_hash,
                "final_patch": (
                    None
                    if session.final_patch is None
                    else session.final_patch.to_dict()
                ),
                "session_validity": (
                    "infrastructure_invalid"
                    if retry_index == 1
                    else "valid"
                ),
            },
        )
        coordinator = AttemptEvaluationCoordinator(
            FreshEvaluator(EvidenceBackend(), clock_ms=StepClock(1_000 * retry_index)),
            journal,
            store,
            retry_index=retry_index,
            clock_ms=StepClock(2_000 * retry_index),
        )
        completed = coordinator.complete(session, spec, frozen, patch_artifact)
        write_runtime_resource_evidence(
            store,
            attempt_id=expected.attempt_id,
            retry_index=retry_index,
            runtime_profile_hash=task.runtime.content_hash,
        )
        ledger.append(
            session_result=session,
            evaluation_result=completed.result,
            evaluation_spec_hash=completed.evaluation_spec_hash,
            retry_index=retry_index,
            recorded_at_ms=3_000 + retry_index,
        )
        if retry_index == 2:
            selected_session = session
            selected_completed = completed
        journal.close()

    write_rebuilt_outputs(
        store,
        (
            SelectedAttempt(
                attempt_id=expected.attempt_id,
                retry_index=2,
                evaluation_spec_hash=selected_completed.evaluation_spec_hash,
                evaluation_result=selected_completed.result,
            ),
        ),
    )
    ledger.close()
    store.close()
    return CompleteRun(
        root=root,
        manifest=frozen_manifest,
        attempt_id=expected.attempt_id,
        session_result=selected_session,
        evaluation_spec_hash=selected_completed.evaluation_spec_hash,
        retry_count=2,
    )


class RuntimeIntegrityTests(unittest.TestCase):
    def test_complete_cohort_passes_every_stable_check_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(Path(tmp) / "run")

            report = verify_run_artifacts(complete.root, complete.manifest)

            self.assertEqual(tuple(check.check_id for check in report.checks), CHECK_IDS)
            self.assertEqual(report.status, "passed")
            self.assertTrue(all(check.status == "passed" for check in report.checks))
            self.assertFalse((complete.root / "integrity.json").exists())
            self.assertFalse(
                (complete.root / "attempts" / complete.attempt_id / "integrity.json").exists()
            )

            ledger = AttemptLedger(complete.root / "attempts.jsonl")
            selected = selected_attempts_from_ledger(ledger, complete.manifest)
            ledger.close()
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].attempt_id, complete.attempt_id)
            self.assertEqual(selected[0].evaluation_spec_hash, complete.evaluation_spec_hash)

    def test_reports_are_persisted_canonically_and_reopen_through_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(Path(tmp) / "run")
            report = verify_run_artifacts(complete.root, complete.manifest)

            persist_integrity_reports(complete.root, complete.manifest, report)

            cohort = IntegrityReport.from_dict(
                json.loads((complete.root / "integrity.json").read_text(encoding="utf-8"))
            )
            attempt = IntegrityReport.from_dict(
                json.loads(
                    (
                        complete.root
                        / "attempts"
                        / complete.attempt_id
                        / "integrity.json"
                    ).read_text(encoding="utf-8")
                )
            )
            self.assertEqual(cohort, report)
            self.assertEqual(attempt.status, "passed")
            self.assertTrue(attempt.checks)
            self.assertLess(len(attempt.checks), len(cohort.checks))

    def test_infrastructure_retry_then_valid_attempt_preserves_and_verifies_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            complete = build_retry_run(Path(tmp) / "run")

            report = verify_run_artifacts(complete.root, complete.manifest)
            ledger = AttemptLedger(complete.root / "attempts.jsonl")
            records = ledger.records(complete.attempt_id)
            ledger.close()

            self.assertEqual(report.status, "passed")
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].attempt_validity, "infrastructure_invalid")
            self.assertEqual(records[1].attempt_validity, "valid")
            self.assertTrue(
                (
                    complete.root
                    / "attempts"
                    / complete.attempt_id
                    / "retries"
                    / "retry-0001"
                    / "private_evaluation.json"
                ).is_file()
            )
            summary = json.loads(
                (complete.root / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["totals"]["retries"], 1)

    def test_all_agent_terminal_reasons_have_valid_complete_evidence(self) -> None:
        for terminal_reason in (
            "agent_finished",
            "agent_exited",
            "budget_exhausted",
            "timeout",
            "cancelled",
        ):
            with self.subTest(terminal_reason=terminal_reason), tempfile.TemporaryDirectory() as tmp:
                complete = build_complete_run(
                    Path(tmp) / "run",
                    terminal_reason=terminal_reason,
                )

                report = verify_run_artifacts(complete.root, complete.manifest)

                self.assertEqual(report.status, "passed")

        with tempfile.TemporaryDirectory() as tmp:
            complete = build_complete_run(
                Path(tmp) / "run",
                terminal_reason="budget_exhausted",
                action_derived_budget=True,
            )

            report = verify_run_artifacts(complete.root, complete.manifest)

            self.assertEqual(report.status, "passed")


if __name__ == "__main__":
    unittest.main()
