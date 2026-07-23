from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.contracts import EvaluationResultV06
from op_bench.runtime.evaluation import (
    AttemptEvaluationCoordinator,
    FreshEvaluator,
    PrivateEvaluationEvidence,
)
from op_bench.runtime.events import EventJournal, verify_event_chain
from op_bench.runtime.run_artifacts import AttemptArtifactStore
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_evaluation import execution
from tests.test_runtime_manifest import manifest
from tests.test_runtime_wire_contracts import evaluation_spec, session_result


class StepClock:
    def __init__(self, start: int = 100) -> None:
        self.value = start

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class EvidenceBackend:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, spec, frozen_patch):
        self.calls += 1
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


class AttemptEvaluationCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = manifest(repeat_count=1)
        self.expected = self.manifest.expected_attempts[0]
        self.task = self.manifest.tasks[0]
        self.store = AttemptArtifactStore(self.root / "run", self.manifest)
        self.store.write_run_manifest()
        self.journal = EventJournal(
            "session-lifecycle",
            clock_ms=StepClock(),
            events_path=self.store.events_path(self.expected.attempt_id),
        )
        self.journal.append(
            "session_created",
            {"attempt_id": self.expected.attempt_id},
        )
        self.backend = EvidenceBackend()
        self.evaluator = FreshEvaluator(self.backend, clock_ms=StepClock(1_000))
        self.coordinator = AttemptEvaluationCoordinator(
            self.evaluator,
            self.journal,
            self.store,
            clock_ms=StepClock(2_000),
        )

    def tearDown(self) -> None:
        self.journal.close()
        self.store.close()
        self.temporary.cleanup()

    def inputs(self):
        patch_bytes = b"fixture patch\n"
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        frozen = FrozenPatch(
            workspace=identity("workspace", "lifecycle-workspace", SHA_B),
            source=self.task.source,
            base_commit="fixture-base",
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=("src/operator.py",),
            empty=False,
        )
        artifact = build_patch_artifact(
            frozen,
            artifact_id=f"{self.expected.attempt_id}/final.patch",
        )
        result = replace(
            session_result(),
            session_id=self.journal.session_id,
            attempt_id=self.expected.attempt_id,
            final_patch=patch,
        )
        spec = replace(
            evaluation_spec(),
            session_id=result.session_id,
            attempt_id=result.attempt_id,
            task=self.task.task,
            source=self.task.source,
            frozen_patch=patch,
            hidden_test_asset=self.task.hidden_test_asset,
            runtime=self.task.runtime,
            evaluation=identity(
                "evaluation",
                self.manifest.evaluation_protocol,
                SHA_B,
            ),
            scoring=self.manifest.scoring,
        )
        return result, spec, frozen, artifact

    def prepare_session_artifacts(self, result, frozen, artifact) -> None:
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.manifest.task_views[0],
            result,
            frozen,
            artifact,
        )

    def test_emits_evaluation_and_final_terminal_once_with_hash_bindings(self) -> None:
        result, spec, frozen, artifact = self.inputs()
        self.prepare_session_artifacts(result, frozen, artifact)
        session_terminal = self.journal.append(
            "session_terminal_emitted",
            {"session_result_hash": result.content_hash},
        )

        completed = self.coordinator.complete(result, spec, frozen, artifact)

        self.assertEqual(
            [event.event_type for event in self.journal.records[-3:]],
            ["evaluation_started", "evaluation_completed", "terminal_emitted"],
        )
        started, evaluated, terminal = self.journal.records[-3:]
        self.assertEqual(started.previous_event_hash, session_terminal.event_hash)
        self.assertEqual(started.public_payload["session_result_hash"], result.content_hash)
        self.assertEqual(started.public_payload["evaluation_spec_hash"], spec.content_hash)
        self.assertEqual(
            evaluated.public_payload["evaluation_result_hash"],
            completed.result.content_hash,
        )
        self.assertEqual(
            terminal.public_payload["evaluation_result_hash"],
            completed.result.content_hash,
        )
        self.assertEqual(verify_event_chain(self.journal.records), ())
        self.assertEqual(self.backend.calls, 1)
        self.assertIs(self.coordinator.complete(result, spec, frozen, artifact), completed)
        self.assertEqual(self.backend.calls, 1)
        self.assertEqual(
            [event.event_type for event in self.journal.records].count("terminal_emitted"),
            1,
        )
        self.assertEqual(
            self.coordinator.artifact_index.terminal_event_hash,
            terminal.event_hash,
        )

    def test_refuses_evaluation_before_matching_session_terminal(self) -> None:
        result, spec, frozen, artifact = self.inputs()
        self.prepare_session_artifacts(result, frozen, artifact)

        with self.assertRaisesRegex(ContractError, "session terminal"):
            self.coordinator.complete(result, spec, frozen, artifact)

        self.assertEqual(self.backend.calls, 0)
        self.assertNotIn(
            "evaluation_started",
            [event.event_type for event in self.journal.records],
        )

    def test_infrastructure_session_still_gets_not_evaluated_final_result(self) -> None:
        result, spec, _, _ = self.inputs()
        result = replace(
            result,
            terminal_reason="platform_error",
            final_patch=None,
        )
        spec = replace(spec, frozen_patch=None)
        self.prepare_session_artifacts(result, None, None)
        self.journal.append(
            "session_terminal_emitted",
            {"session_result_hash": result.content_hash},
        )

        completed = self.coordinator.complete(result, spec, None, None)

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(completed.result.evaluation_outcome, "not_evaluated")
        self.assertIsNone(completed.result.agent_terminal)
        self.assertEqual(self.backend.calls, 0)
        self.assertEqual(self.journal.records[-1].event_type, "terminal_emitted")
        self.assertEqual(
            self.store.read_public_evaluation(self.expected.attempt_id),
            completed.result,
        )


if __name__ == "__main__":
    unittest.main()
