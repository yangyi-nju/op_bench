from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import EvaluationResultV06, TestExecutionSummary
from op_bench.runtime.evaluation import CompletedEvaluation, PrivateEvaluationEvidence
from op_bench.runtime.events import EventJournal
from op_bench.runtime.run_artifacts import AttemptArtifactStore
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_evaluation import execution
from tests.test_runtime_manifest import manifest
from tests.test_runtime_wire_contracts import evaluation_spec, session_result, test_summary


class StepClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class AttemptArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manifest = manifest(repeat_count=1)
        self.expected = self.manifest.expected_attempts[0]
        self.task = self.manifest.tasks[0]
        self.view = self.manifest.task_views[0]
        self.store = AttemptArtifactStore(self.root / "run", self.manifest)
        self.store.write_run_manifest()
        self.journal = EventJournal(
            "session-artifact",
            clock_ms=StepClock(),
            events_path=self.store.events_path(self.expected.attempt_id),
        )
        self.journal.append(
            "session_created",
            {"attempt_id": self.expected.attempt_id},
        )

    def tearDown(self) -> None:
        self.journal.close()
        self.store.close()
        self.temporary.cleanup()

    def attempt_inputs(self, patch_bytes: bytes = b"fixture patch\n"):
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        frozen = FrozenPatch(
            workspace=identity("workspace", "artifact-workspace", SHA_B),
            source=self.task.source,
            base_commit="fixture-base",
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=(() if not patch_bytes else ("src/operator.py",)),
            empty=not patch_bytes,
        )
        artifact = build_patch_artifact(
            frozen,
            artifact_id=f"{self.expected.attempt_id}/final.patch",
        )
        result = replace(
            session_result(),
            session_id="session-artifact",
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
        private = PrivateEvaluationEvidence(
            source=spec.source,
            patch=patch,
            hidden_test_asset=spec.hidden_test_asset,
            selectors=(
                execution("hidden::f2p", "fail_to_pass"),
                execution("public::smoke", "pass_to_pass"),
            ),
            cleanup_completed=True,
        )
        evaluation = EvaluationResultV06(
            session_id=result.session_id,
            attempt_id=result.attempt_id,
            attempt_validity="valid",
            agent_terminal="finished",
            evaluation_outcome="resolved",
            invalid_reason=None,
            patch=patch,
            fail_to_pass=test_summary(),
            pass_to_pass=test_summary(),
            duration_ms=25,
            evaluation=spec.evaluation,
            scoring=spec.scoring,
        )
        completed = CompletedEvaluation(
            result=evaluation,
            private_evidence=private,
            evaluation_spec=spec,
        )
        return frozen, artifact, result, completed

    def write_complete_attempt(self):
        frozen, artifact, result, completed = self.attempt_inputs()
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            result,
            frozen,
            artifact,
        )
        self.journal.append(
            "session_terminal_emitted",
            {"session_result_hash": result.content_hash},
        )
        hashes = self.store.write_evaluation(
            self.expected.attempt_id,
            completed,
        )
        return frozen, artifact, result, completed, hashes

    def test_writes_canonical_retry_layout_idempotently(self) -> None:
        frozen, artifact, result, completed, hashes = self.write_complete_attempt()
        attempt_root = self.root / "run" / "attempts" / self.expected.attempt_id
        attempt = attempt_root / "retries" / "retry-0001"
        expected_files = {
            "agent_task_view.json",
            "session_result.json",
            "events.jsonl",
            "final.patch",
            "public_evaluation.json",
            "private_evaluation.json",
        }

        self.assertEqual({item.name for item in attempt_root.iterdir()}, {"retries"})
        self.assertEqual({item.name for item in attempt.iterdir()}, expected_files)
        for name in expected_files - {"final.patch"}:
            raw = (attempt / name).read_bytes()
            self.assertTrue(raw.endswith(b"\n"))
            for line in raw.splitlines():
                self.assertEqual(
                    canonical_json(json.loads(line)).encode("utf-8"),
                    line,
                )
        self.assertEqual((attempt / "final.patch").read_bytes(), frozen.patch_bytes)
        self.assertEqual(self.store.read_public_evaluation(self.expected.attempt_id), completed.result)
        self.assertEqual(
            self.store.read_private_evaluation(self.expected.attempt_id),
            completed,
        )
        self.assertRegex(hashes.public_evaluation_hash, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(hashes.private_evaluation_hash, r"^sha256:[0-9a-f]{64}$")

        before = {path.name: path.read_bytes() for path in attempt.iterdir()}
        self.store.write_run_manifest()
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            result,
            frozen,
            artifact,
        )
        self.store.write_evaluation(self.expected.attempt_id, completed)
        self.assertEqual(
            {path.name: path.read_bytes() for path in attempt.iterdir()},
            before,
        )

        with self.assertRaisesRegex(ContractError, "conflicting artifact"):
            self.store.write_session_inputs(
                self.expected.attempt_id,
                self.view,
                replace(result, ended_at_ms=result.ended_at_ms + 1),
                frozen,
                artifact,
            )

    def test_public_inputs_reject_host_paths_and_credentials_but_private_output_is_allowed(self) -> None:
        frozen, artifact, result, completed = self.attempt_inputs()
        host_patch = b'+path = "/Users/private/source.py"\n'
        host_identity = raw_patch_identity(host_patch, identifier="final.patch")
        host_frozen = replace(
            frozen,
            patch=host_identity,
            patch_bytes=host_patch,
        )
        host_artifact = build_patch_artifact(
            host_frozen,
            artifact_id=f"{self.expected.attempt_id}/final.patch",
        )
        with self.assertRaisesRegex(ContractError, "sensitive value"):
            self.store.write_session_inputs(
                self.expected.attempt_id,
                self.view,
                replace(result, final_patch=host_identity),
                host_frozen,
                host_artifact,
            )

        secret_patch = b'+api_key = "sk-abcdefghijklmnop"\n'
        secret_identity = raw_patch_identity(secret_patch, identifier="final.patch")
        secret_frozen = replace(
            frozen,
            patch=secret_identity,
            patch_bytes=secret_patch,
        )
        secret_artifact = build_patch_artifact(
            secret_frozen,
            artifact_id=f"{self.expected.attempt_id}/final.patch",
        )
        with self.assertRaisesRegex(ContractError, "sensitive value"):
            self.store.write_session_inputs(
                self.expected.attempt_id,
                self.view,
                replace(result, final_patch=secret_identity),
                secret_frozen,
                secret_artifact,
            )

        private = completed.private_evidence
        self.assertIsNotNone(private)
        private_with_secret = replace(
            private,
            selectors=(
                replace(private.selectors[0], stdout="api_key=sk-abcdefghijklmnop"),
                private.selectors[1],
            ),
        )
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            result,
            frozen,
            artifact,
        )
        self.store.write_evaluation(
            self.expected.attempt_id,
            replace(completed, private_evidence=private_with_secret),
        )
        stored = self.store.read_private_evaluation(self.expected.attempt_id)
        self.assertIn("api_key=", stored.private_evidence.selectors[0].stdout)
        self.assertNotIn(
            "api_key=",
            str(self.store.read_public_evaluation(self.expected.attempt_id).to_dict()),
        )

    def test_rejects_symlink_root_and_protected_file_replacement(self) -> None:
        real = self.root / "real-run"
        real.mkdir()
        linked = self.root / "linked-run"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ContractError, "symlink"):
            AttemptArtifactStore(linked, self.manifest)

        self.write_complete_attempt()
        public_path = (
            self.root
            / "run"
            / "attempts"
            / self.expected.attempt_id
            / "retries"
            / "retry-0001"
            / "public_evaluation.json"
        )
        replacement = self.root / "replacement.json"
        replacement.write_text("{}\n", encoding="utf-8")
        public_path.unlink()
        public_path.symlink_to(replacement)
        with self.assertRaisesRegex(ContractError, "regular file"):
            self.store.read_public_evaluation(self.expected.attempt_id)

    def test_retries_use_separate_immutable_artifact_directories(self) -> None:
        frozen, artifact, first, _ = self.attempt_inputs()
        second = replace(first, session_id="session-artifact-retry-2")

        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            first,
            frozen,
            artifact,
            retry_index=1,
        )
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            second,
            frozen,
            artifact,
            retry_index=2,
        )

        self.assertEqual(
            self.store.read_session_result(self.expected.attempt_id, retry_index=1),
            first,
        )
        self.assertEqual(
            self.store.read_session_result(self.expected.attempt_id, retry_index=2),
            second,
        )

    def test_same_protocol_name_with_different_evaluation_digest_is_rejected(self) -> None:
        frozen, artifact, result, completed = self.attempt_inputs()
        changed_identity = replace(
            completed.evaluation_spec.evaluation,
            digest="sha256:" + "f" * 64,
        )
        changed = CompletedEvaluation(
            result=replace(completed.result, evaluation=changed_identity),
            private_evidence=completed.private_evidence,
            evaluation_spec=replace(
                completed.evaluation_spec,
                evaluation=changed_identity,
            ),
        )
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            result,
            frozen,
            artifact,
        )

        with self.assertRaisesRegex(ContractError, "evaluation identity mismatch"):
            self.store.write_evaluation(self.expected.attempt_id, changed)

    def test_evaluation_spec_must_exactly_match_frozen_task_authority(self) -> None:
        frozen, artifact, result, completed = self.attempt_inputs()
        selector = completed.evaluation_spec.public_tests[0]
        mutations = (
            (
                "public-tests",
                replace(
                    completed.evaluation_spec,
                    public_tests=(
                        replace(selector, description="Easier public selector"),
                    ),
                ),
            ),
            (
                "fail-to-pass",
                replace(
                    completed.evaluation_spec,
                    fail_to_pass=("hidden::different",),
                ),
            ),
            (
                "pass-to-pass",
                replace(
                    completed.evaluation_spec,
                    pass_to_pass=("public::different",),
                ),
            ),
            (
                "timeout",
                replace(
                    completed.evaluation_spec,
                    timeout_ms=completed.evaluation_spec.timeout_ms - 1,
                ),
            ),
        )

        for retry_index, (name, changed_spec) in enumerate(mutations, start=1):
            self.store.write_session_inputs(
                self.expected.attempt_id,
                self.view,
                result,
                frozen,
                artifact,
                retry_index=retry_index,
            )
            with self.subTest(name=name), self.assertRaisesRegex(
                ContractError,
                "evaluation_spec: task evidence mismatch",
            ):
                self.store.write_evaluation(
                    self.expected.attempt_id,
                    replace(completed, evaluation_spec=changed_spec),
                    retry_index=retry_index,
                )

    def test_private_evidence_must_semantically_match_public_result(self) -> None:
        frozen, artifact, result, completed = self.attempt_inputs()
        private = completed.private_evidence
        self.assertIsNotNone(private)
        zero = TestExecutionSummary(0, 0, 0, 0, 0)
        mutations = (
            (
                "aggregate",
                replace(
                    completed,
                    private_evidence=replace(
                        private,
                        selectors=(
                            replace(
                                private.selectors[0],
                                summary=TestExecutionSummary(1, 1, 0, 1, 0),
                            ),
                            private.selectors[1],
                        ),
                    ),
                ),
            ),
            (
                "group",
                replace(
                    completed,
                    private_evidence=replace(
                        private,
                        selectors=(
                            replace(private.selectors[0], group="pass_to_pass"),
                            replace(private.selectors[1], group="fail_to_pass"),
                        ),
                    ),
                ),
            ),
            (
                "missing",
                replace(completed, private_evidence=None),
            ),
            (
                "unexpected",
                replace(
                    completed,
                    result=replace(
                        completed.result,
                        evaluation_outcome="no_patch",
                        fail_to_pass=zero,
                        pass_to_pass=zero,
                    ),
                ),
            ),
        )

        for retry_index, (name, changed) in enumerate(mutations, start=1):
            self.store.write_session_inputs(
                self.expected.attempt_id,
                self.view,
                result,
                frozen,
                artifact,
                retry_index=retry_index,
            )
            with self.subTest(name=name), self.assertRaisesRegex(
                ContractError,
                "evaluation semantics",
            ):
                self.store.write_evaluation(
                    self.expected.attempt_id,
                    changed,
                    retry_index=retry_index,
                )

    def test_infrastructure_session_cannot_publish_a_valid_result(self) -> None:
        _, _, result, completed = self.attempt_inputs()
        session = replace(
            result,
            terminal_reason="platform_error",
            final_patch=None,
        )
        spec = replace(completed.evaluation_spec, frozen_patch=None)
        zero = TestExecutionSummary(0, 0, 0, 0, 0)
        fabricated = CompletedEvaluation(
            result=replace(
                completed.result,
                attempt_validity="valid",
                agent_terminal=None,
                evaluation_outcome="no_patch",
                invalid_reason=None,
                patch=None,
                fail_to_pass=zero,
                pass_to_pass=zero,
            ),
            private_evidence=None,
            evaluation_spec=spec,
        )
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            session,
            None,
            None,
        )

        with self.assertRaisesRegex(ContractError, "session attribution"):
            self.store.write_evaluation(
                self.expected.attempt_id,
                fabricated,
            )

    def test_no_patch_outcome_requires_an_empty_patch_artifact(self) -> None:
        frozen, artifact, result, completed = self.attempt_inputs()
        zero = TestExecutionSummary(0, 0, 0, 0, 0)
        fabricated = replace(
            completed,
            result=replace(
                completed.result,
                evaluation_outcome="no_patch",
                fail_to_pass=zero,
                pass_to_pass=zero,
            ),
            private_evidence=None,
        )
        self.store.write_session_inputs(
            self.expected.attempt_id,
            self.view,
            result,
            frozen,
            artifact,
        )

        with self.assertRaisesRegex(ContractError, "no_patch.*empty final.patch"):
            self.store.write_evaluation(
                self.expected.attempt_id,
                fabricated,
            )


if __name__ == "__main__":
    unittest.main()
