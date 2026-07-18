from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.backends import (
    LocalProcessBackend,
    RuntimeBackendUnavailable,
    RuntimeCleanupResult,
    _cleanup_report,
)
from op_bench.runtime.codex_adapter import CodexAdapterResult
from op_bench.runtime.integrity import verify_run_artifacts
from op_bench.runtime.resume import AttemptLedger
from op_bench.runtime.workspace import WorkspacePolicyError
from tests.runtime_orchestrator_fixture import (
    PatchAdapter,
    build_orchestrator_fixture,
    orchestrator_for,
    request_for,
)


class PrepareUnavailableBackend(LocalProcessBackend):
    def prepare(self, profile, attempt_context):
        declared = attempt_context.resource_ledger.declare("workspace", 1)
        attempt_context.resource_ledger.create_failed(declared.resource_id)
        raise RuntimeBackendUnavailable("workspace_prepare_failed")


class EvaluationUnavailableBackend(LocalProcessBackend):
    def prepare(self, profile, attempt_context):
        ordinal = 1 + sum(
            record.resource_type == "workspace" and record.transition == "declared"
            for record in attempt_context.resource_ledger.records
        )
        declared = attempt_context.resource_ledger.declare("workspace", ordinal)
        attempt_context.resource_ledger.create_failed(declared.resource_id)
        raise RuntimeBackendUnavailable("workspace_prepare_failed")


class ActionUnavailableBackend(LocalProcessBackend):
    def run(self, lease, command, cwd, timeout_ms):
        if "-m" in command and "unittest" in command:
            raise RuntimeBackendUnavailable("process_launch_failed")
        return super().run(lease, command, cwd, timeout_ms)


class CleanupFailureBackend(LocalProcessBackend):
    def cleanup(self, lease):
        state = self._state_for(lease)
        if state.cleanup_result is not None:
            return state.cleanup_result
        workspace = next(
            handle for handle in lease.handles if handle.resource_type == "workspace"
        )
        state.context.resource_ledger.cleanup_failed(workspace.resource_id)
        result = RuntimeCleanupResult(_cleanup_report(state.context))
        state.cleanup_result = result
        return result


class TerminalAdapter:
    def __init__(self, status: str, terminal_reason: str, exit_code: int | None) -> None:
        self.status = status
        self.terminal_reason = terminal_reason
        self.exit_code = exit_code

    def run(self, context):
        return CodexAdapterResult(
            status=self.status,
            terminal_reason=self.terminal_reason,
            exit_code=self.exit_code,
            observation_count=0,
            finish_count=0,
        )


class RaisingAdapter:
    def run(self, context):
        raise RuntimeError("private Provider failure detail")


class InterruptingAdapter:
    def run(self, context):
        raise KeyboardInterrupt()


class V06OrchestratorFailureTests(unittest.TestCase):
    def test_keyboard_interrupt_is_durable_and_next_run_retries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))

            with self.assertRaises(KeyboardInterrupt):
                orchestrator_for(
                    fixture,
                    backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                    adapter=InterruptingAdapter(),
                ).run(request_for(fixture))

            ledger = AttemptLedger(fixture.output_root / "attempts.jsonl")
            try:
                interrupted = ledger.records(fixture.expected.attempt_id)
                self.assertEqual(len(interrupted), 1)
                self.assertEqual(interrupted[0].retry_index, 1)
                self.assertEqual(
                    interrupted[0].attempt_validity,
                    "infrastructure_invalid",
                )
            finally:
                ledger.close()
            retry_one = (
                fixture.output_root
                / "attempts"
                / fixture.expected.attempt_id
                / "retries"
                / "retry-0001"
            )
            self.assertTrue((retry_one / "runtime_cleanup.json").is_file())

            adapter = PatchAdapter()
            resumed = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(resumed.integrity.status, "passed")
            self.assertEqual(adapter.run_count, 1)
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["retry_index"], 2)
            self.assertEqual(row["attempt_validity"], "valid")
            self.assertEqual(row["evaluation_outcome"], "resolved")
            self.assertEqual(
                verify_run_artifacts(fixture.output_root, fixture.manifest).status,
                "passed",
            )
    def test_action_backend_failure_is_runtime_error_not_provider_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()
            result = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: ActionUnavailableBackend(),
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(result.integrity.status, "passed", result.integrity.to_dict())
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(row["evaluation_outcome"], "not_evaluated")
            self.assertEqual(row["invalid_reason"], "session_runtime_error")
            self.assertIsNone(row["agent_terminal"])

    def test_cleanup_failure_preserves_session_and_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()
            result = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: CleanupFailureBackend(),
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(result.integrity.status, "failed")
            failed_checks = {
                check.check_id
                for check in result.integrity.checks
                if check.status == "failed"
            }
            self.assertEqual(failed_checks, {"runtime_cleanup"})
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["agent_terminal"], "finished")
            self.assertEqual(row["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(row["evaluation_outcome"], "evaluation_error")
            self.assertEqual(row["invalid_reason"], "evaluation_cleanup_failed")
            cleanup = result.cleanup_reports[fixture.expected.attempt_id]
            self.assertFalse(cleanup.all_released)
            self.assertGreaterEqual(
                sum(entry.status == "cleanup_failed" for entry in cleanup.entries),
                1,
            )

    def test_patch_freeze_failure_converges_without_a_patch_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()
            with mock.patch.object(
                __import__(
                    "op_bench.runtime.workspace",
                    fromlist=["AuthoritativeWorkspace"],
                ).AuthoritativeWorkspace,
                "freeze",
                side_effect=WorkspacePolicyError("private freeze detail"),
            ):
                result = orchestrator_for(
                    fixture,
                    backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                    adapter=adapter,
                ).run(request_for(fixture))

            self.assertEqual(result.integrity.status, "passed", result.integrity.to_dict())
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(row["evaluation_outcome"], "not_evaluated")
            self.assertEqual(row["invalid_reason"], "session_workspace_error")
            retry_root = (
                fixture.output_root
                / "attempts"
                / fixture.expected.attempt_id
                / "retries"
                / "retry-0001"
            )
            self.assertFalse((retry_root / "final.patch").exists())
            self.assertNotIn("private freeze detail", repr(result))

    def test_workspace_authority_failure_cleans_prepared_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()

            with mock.patch.object(
                __import__(
                    "op_bench.runtime.workspace",
                    fromlist=["AuthoritativeWorkspace"],
                ).AuthoritativeWorkspace,
                "open",
                side_effect=WorkspacePolicyError("private authority detail"),
            ):
                result = orchestrator_for(
                    fixture,
                    backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                    adapter=adapter,
                ).run(request_for(fixture))

            self.assertEqual(result.integrity.status, "passed", result.integrity.to_dict())
            self.assertEqual(adapter.run_count, 0)
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(row["evaluation_outcome"], "not_evaluated")
            self.assertEqual(row["invalid_reason"], "session_workspace_error")
            self.assertNotIn("private authority detail", repr(result))
            cleanup = result.cleanup_reports[fixture.expected.attempt_id]
            self.assertTrue(cleanup.all_released)
            self.assertTrue(
                all(entry.status == "released" for entry in cleanup.entries)
            )
            self.assertEqual(
                list(fixture.target_binding.local_workspace_parent.iterdir()),
                [],
            )

    def test_evaluation_prepare_failure_preserves_session_and_closes_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()

            result = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: (
                    LocalProcessBackend()
                    if phase == "agent"
                    else EvaluationUnavailableBackend()
                ),
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(result.integrity.status, "passed", result.integrity.to_dict())
            row = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(row["agent_terminal"], "finished")
            self.assertEqual(row["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(row["evaluation_outcome"], "evaluation_error")
            self.assertEqual(row["invalid_reason"], "workspace_prepare_failed")
            cleanup = result.cleanup_reports[fixture.expected.attempt_id]
            self.assertTrue(cleanup.all_released)
            self.assertEqual(
                sorted(
                    (entry.resource_type, entry.status)
                    for entry in cleanup.entries
                ),
                [
                    ("process", "released"),
                    ("workspace", "create_failed"),
                    ("workspace", "released"),
                ],
            )

    def test_adapter_terminal_and_exception_modes_have_stable_attribution(self) -> None:
        cases = (
            (
                "provider_failure",
                TerminalAdapter("provider_failure", "provider_error", 1),
                "infrastructure_invalid",
                None,
                "not_evaluated",
                "session_provider_error",
            ),
            (
                "timeout",
                TerminalAdapter("timeout", "timeout", None),
                "valid",
                "timeout",
                "no_patch",
                None,
            ),
            (
                "agent_exited",
                TerminalAdapter("missing_finish", "agent_exited", 0),
                "valid",
                "exited",
                "no_patch",
                None,
            ),
            (
                "raised_provider_exception",
                RaisingAdapter(),
                "infrastructure_invalid",
                None,
                "not_evaluated",
                "session_provider_error",
            ),
        )
        for (
            name,
            adapter,
            validity,
            agent_terminal,
            outcome,
            invalid_reason,
        ) in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = build_orchestrator_fixture(Path(temporary))
                result = orchestrator_for(
                    fixture,
                    backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                    adapter=adapter,
                ).run(request_for(fixture))

                self.assertEqual(result.integrity.status, "passed", result.integrity.to_dict())
                row = json.loads(
                    (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
                )
                self.assertEqual(row["attempt_validity"], validity)
                self.assertEqual(row["agent_terminal"], agent_terminal)
                self.assertEqual(row["evaluation_outcome"], outcome)
                self.assertEqual(row["invalid_reason"], invalid_reason)
                self.assertNotIn("private Provider failure detail", repr(result))

    def test_backend_prepare_failure_is_complete_and_retries_to_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(Path(temporary))
            adapter = PatchAdapter()

            def unavailable_factory(profile, target, phase):
                return (
                    PrepareUnavailableBackend()
                    if phase == "agent"
                    else LocalProcessBackend()
                )

            failed = orchestrator_for(
                fixture,
                backend_factory=unavailable_factory,
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(
                failed.integrity.status,
                "passed",
                failed.integrity.to_dict(),
            )
            self.assertEqual(adapter.run_count, 0)
            result = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(result["attempt_validity"], "infrastructure_invalid")
            self.assertEqual(result["agent_terminal"], None)
            self.assertEqual(result["evaluation_outcome"], "not_evaluated")
            self.assertEqual(result["invalid_reason"], "session_platform_error")
            cleanup = failed.cleanup_reports[fixture.expected.attempt_id]
            self.assertTrue(cleanup.all_released)
            self.assertEqual(
                [(entry.resource_type, entry.status) for entry in cleanup.entries],
                [("workspace", "create_failed")],
            )
            ledger = AttemptLedger(fixture.output_root / "attempts.jsonl")
            try:
                self.assertEqual(len(ledger.records(fixture.expected.attempt_id)), 1)
                self.assertEqual(
                    ledger.records(fixture.expected.attempt_id)[0].retry_index,
                    1,
                )
            finally:
                ledger.close()

            succeeded = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                adapter=adapter,
            ).run(request_for(fixture))

            self.assertEqual(succeeded.integrity.status, "passed")
            self.assertEqual(adapter.run_count, 1)
            result = json.loads(
                (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(result["retry_index"], 2)
            self.assertEqual(result["attempt_validity"], "valid")
            self.assertEqual(result["evaluation_outcome"], "resolved")
            ledger = AttemptLedger(fixture.output_root / "attempts.jsonl")
            try:
                records = ledger.records(fixture.expected.attempt_id)
                self.assertEqual([item.retry_index for item in records], [1, 2])
                self.assertEqual(
                    [item.attempt_validity for item in records],
                    ["infrastructure_invalid", "valid"],
                )
            finally:
                ledger.close()
            self.assertEqual(
                verify_run_artifacts(
                    fixture.output_root,
                    fixture.manifest,
                ).status,
                "passed",
            )


if __name__ == "__main__":
    unittest.main()
