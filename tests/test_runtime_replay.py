from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.evaluation import (
    EvaluationInfrastructureError,
    FreshEvaluator,
    PrivateEvaluationEvidence,
    ReplayEvaluationEvidence,
    SelectorExecution,
)
from op_bench.runtime.backends import RuntimeBackendUnavailable
from op_bench.runtime.replay import (
    ExactReplayObserver,
    ReplayRunner,
    build_replay_inventory,
)
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_wire_contracts import evaluation_spec, test_summary


class ReplayBackend:
    def __init__(self, *, f2p_failed: bool = False, reason: str | None = None) -> None:
        self.f2p_failed = f2p_failed
        self.reason = reason
        self.calls = []

    def evaluate_replay(self, spec, frozen_patch):
        self.calls.append((spec, frozen_patch))
        if self.reason is not None:
            raise EvaluationInfrastructureError(self.reason)
        failing = replace(test_summary(), passed=0, failed=1)
        return PrivateEvaluationEvidence(
            source=spec.source,
            patch=None if frozen_patch is None else frozen_patch.patch,
            hidden_test_asset=spec.hidden_test_asset,
            selectors=(
                SelectorExecution(
                    selector_id="hidden::f2p",
                    group="fail_to_pass",
                    command_digest=SHA_B,
                    exit_code=1 if self.f2p_failed else 0,
                    timed_out=False,
                    summary=failing if self.f2p_failed else test_summary(),
                    stdout="",
                    stderr="",
                ),
                SelectorExecution(
                    selector_id="public::smoke",
                    group="pass_to_pass",
                    command_digest=SHA_B,
                    exit_code=0,
                    timed_out=False,
                    summary=test_summary(),
                    stdout="",
                    stderr="",
                ),
            ),
            cleanup_completed=True,
        )

    def evaluate(self, spec, frozen_patch):
        raise AssertionError("Agent evaluate path must not be used for replay")


class UnavailableRuntimeBackend:
    prepare_calls = 0

    def prepare(self, profile, attempt_context):
        type(self).prepare_calls += 1
        raise RuntimeBackendUnavailable("remote_workspace_create_failed")

    def run(self, lease, command, cwd, timeout_ms):
        raise AssertionError("unavailable backend must not run")

    def collect(self, lease):
        raise AssertionError("unavailable backend must not collect")

    def cleanup(self, lease):
        raise AssertionError("unavailable backend has no lease")


class FreshEvaluatorReplayTests(unittest.TestCase):
    def test_cleanup_failure_is_not_reclassified_or_cached_as_unavailable(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = build_replay_inventory(root)[:2]
        evaluators = []

        class UnavailableThenCleanupFailureEvaluator:
            def __init__(self, **kwargs):
                del kwargs
                self.last_backend_unavailable_reason = None
                evaluators.append(self)

            def evaluate_replay(self, spec, frozen_patch):
                del spec, frozen_patch
                self.last_backend_unavailable_reason = "remote_source_sync_failed"
                raise EvaluationInfrastructureError("evaluation_cleanup_failed")

            def evaluate(self, spec, frozen_patch):
                return self.evaluate_replay(spec, frozen_patch)

        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary)
            workspaces = private / "workspaces"
            workspaces.mkdir()
            identity_file = private / "identity"
            identity_file.write_text("fixture", encoding="utf-8")
            target = private / "target.json"
            target.write_text(
                "{\"backend\":\"remote_docker\","
                f"\"local_workspace_parent\":\"{workspaces}\","
                "\"host_alias\":\"exact.invalid\","
                "\"remote_user\":\"runner\","
                f"\"identity_file\":\"{identity_file}\"}}",
                encoding="utf-8",
            )
            with mock.patch(
                "op_bench.runtime.replay.RuntimeFreshEvaluationBackend",
                UnavailableThenCleanupFailureEvaluator,
            ):
                with ExactReplayObserver(root, target) as observer:
                    observed = tuple(observer(case) for case in cases)

        self.assertEqual(len(evaluators), 2)
        self.assertEqual(
            [item.observed_outcome for item in observed],
            ["evaluation_error", "evaluation_error"],
        )
        self.assertEqual(
            [item.invalid_reason for item in observed],
            ["evaluation_cleanup_failed", "evaluation_cleanup_failed"],
        )

    def test_exact_observer_caches_connection_level_unavailability(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = build_replay_inventory(root)[:2]
        with tempfile.TemporaryDirectory() as temporary:
            private = Path(temporary)
            workspaces = private / "workspaces"
            workspaces.mkdir()
            identity_file = private / "identity"
            identity_file.write_text("fixture", encoding="utf-8")
            target = private / "target.json"
            target.write_text(
                "{\"backend\":\"remote_docker\","
                f"\"local_workspace_parent\":\"{workspaces}\","
                "\"host_alias\":\"exact.invalid\","
                "\"remote_user\":\"runner\","
                f"\"identity_file\":\"{identity_file}\"}}",
                encoding="utf-8",
            )
            UnavailableRuntimeBackend.prepare_calls = 0
            with ExactReplayObserver(
                root,
                target,
                backend_factory=lambda profile, binding: UnavailableRuntimeBackend(),
            ) as observer:
                for case in cases:
                    with self.assertRaisesRegex(
                        RuntimeBackendUnavailable,
                        "remote_workspace_create_failed",
                    ):
                        observer(case)

            self.assertEqual(UnavailableRuntimeBackend.prepare_calls, 1)

    def test_baseline_without_agent_patch_observes_historical_failure(self) -> None:
        backend = ReplayBackend(f2p_failed=True)
        evaluator = FreshEvaluator(backend, clock_ms=lambda: 100)

        observed = evaluator.evaluate_replay(evaluation_spec(), None, None)

        self.assertIsInstance(observed, ReplayEvaluationEvidence)
        self.assertEqual(observed.observed_outcome, "f2p_failed")
        self.assertIsNone(observed.private_evidence.patch)
        self.assertEqual(backend.calls[0][1], None)

    def test_patch_replay_is_strictly_identity_bound_and_resolves(self) -> None:
        backend = ReplayBackend()
        evaluator = FreshEvaluator(backend, clock_ms=lambda: 100)
        patch_bytes = b"fixture replay patch\n"
        patch = raw_patch_identity(patch_bytes, identifier="replay.patch")
        spec = replace(evaluation_spec(), frozen_patch=patch)
        frozen = FrozenPatch(
            workspace=identity("workspace", "replay-workspace", SHA_B),
            source=spec.source,
            base_commit="fixture-base",
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=("src/operator.py",),
            empty=False,
        )

        observed = evaluator.evaluate_replay(
            spec,
            frozen,
            build_patch_artifact(frozen, artifact_id="replay/final.patch"),
        )

        self.assertEqual(observed.observed_outcome, "resolved")
        self.assertEqual(observed.private_evidence.patch, patch)

    def test_unavailable_backend_is_replay_evaluation_error_not_agent_result(self) -> None:
        observed = FreshEvaluator(
            ReplayBackend(reason="exact_runtime_unavailable"),
            clock_ms=lambda: 100,
        ).evaluate_replay(evaluation_spec(), None, None)

        self.assertEqual(observed.observed_outcome, "evaluation_error")
        self.assertEqual(observed.invalid_reason, "exact_runtime_unavailable")
        self.assertIsNone(observed.private_evidence)

    def test_replay_runner_keeps_passed_failed_and_blocked_separate(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cases = build_replay_inventory(root)[:3]

        def observe(case):
            if case == cases[0]:
                return case.expected_outcome
            if case == cases[1]:
                return "resolved" if case.expected_outcome != "resolved" else "f2p_failed"
            return None

        with tempfile.TemporaryDirectory() as temporary:
            report = ReplayRunner(root, cases).run(
                Path(temporary),
                observer=observe,
            )

            self.assertEqual(report.summary.total, 3)
            self.assertEqual(report.summary.passed, 1)
            self.assertEqual(report.summary.failed, 1)
            self.assertEqual(report.summary.blocked, 1)
            self.assertEqual(len(report.differences), 1)


if __name__ == "__main__":
    unittest.main()
