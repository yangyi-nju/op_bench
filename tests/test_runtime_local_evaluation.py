from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.contracts import ContentIdentity, TestSelector
from op_bench.runtime.evaluation import FreshEvaluator, PrivateEvaluationEvidence
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitEvaluationBackend,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.runtime_git_fixture import (
    git,
    git_authority_pollution,
    initialize_evaluation_git_fixture,
)
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_wire_contracts import evaluation_spec, session_result


F2P = "test_calc.NormalizeTests.test_nan_is_preserved"
P2P = "test_calc.NormalizeTests.test_number_is_preserved"


class StepClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        current = self.value
        self.value += 25
        return current


class LocalGitEvaluationBackendTests(unittest.TestCase):
    def test_archive_identity_ignores_ambient_git_authority(self) -> None:
        source = self.fixture.repository
        expected = git_archive_source_identity(
            source,
            self.fixture.revision,
            "fixture@expected",
        )
        decoy_fixture = initialize_evaluation_git_fixture(self.root / "decoy")
        (decoy_fixture.repository / "foreign.txt").write_text(
            "foreign\n",
            encoding="utf-8",
        )
        git(decoy_fixture.repository, "add", "foreign.txt")
        git(
            decoy_fixture.repository,
            "commit",
            "--quiet",
            "-m",
            "foreign authority",
        )

        with mock.patch.dict(
            os.environ,
            git_authority_pollution(
                self.root,
                source,
                decoy_fixture.repository,
            ),
            clear=False,
        ):
            observed = git_archive_source_identity(
                source,
                "HEAD",
                "fixture@expected",
            )

        self.assertEqual(observed, expected)

    def test_evaluation_git_commands_ignore_ambient_git_authority(self) -> None:
        decoy_fixture = initialize_evaluation_git_fixture(self.root / "decoy-run")
        (decoy_fixture.repository / "foreign.txt").write_text(
            "foreign\n",
            encoding="utf-8",
        )
        git(decoy_fixture.repository, "add", "foreign.txt")
        git(
            decoy_fixture.repository,
            "commit",
            "--quiet",
            "-m",
            "foreign authority",
        )

        with mock.patch.dict(
            os.environ,
            git_authority_pollution(
                self.root,
                self.fixture.repository,
                decoy_fixture.repository,
            ),
            clear=False,
        ):
            completed = self.evaluate(self.fixture.gold_patch)

        self.assertEqual(completed.result.attempt_validity, "valid")
        self.assertEqual(completed.result.evaluation_outcome, "resolved")
        self.assertEqual(list(self.evaluation_root.iterdir()), [])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = initialize_evaluation_git_fixture(self.root / "source")
        self.source = git_archive_source_identity(
            self.fixture.repository,
            self.fixture.revision,
            "fixture@evaluation-base",
        )
        hidden_digest = "sha256:" + hashlib.sha256(
            self.fixture.hidden_test_patch
        ).hexdigest()
        self.hidden = ContentIdentity(
            "test",
            "fixture:hidden-test.patch",
            hidden_digest,
            "content_sha256",
        )
        selectors = (
            TestSelector(
                selector_id=F2P,
                visibility="evaluation_only",
                command_template="{python} -m unittest {test}",
                description="Evaluation-only F2P control.",
            ),
            TestSelector(
                selector_id=P2P,
                visibility="evaluation_only",
                command_template="{python} -m unittest {test}",
                description="Evaluation-only P2P control.",
            ),
        )
        self.evaluation_root = self.root / "evaluator-workspaces"
        self.evaluation_root.mkdir()
        self.backend = LocalGitEvaluationBackend(
            source=LocalGitSource(
                identity=self.source,
                repository=self.fixture.repository,
                revision=self.fixture.revision,
            ),
            hidden_asset=EvaluationOnlyTestAsset(
                identity=self.hidden,
                patch_bytes=self.fixture.hidden_test_patch,
                selectors=selectors,
            ),
            python_executable=sys.executable,
            workspace_parent=self.evaluation_root,
        )
        self.spec = replace(
            evaluation_spec(),
            source=self.source,
            hidden_test_asset=self.hidden,
            public_tests=(),
            fail_to_pass=(F2P,),
            pass_to_pass=(P2P,),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def evaluate(self, patch_bytes: bytes):
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        spec = replace(self.spec, frozen_patch=patch)
        frozen = FrozenPatch(
            workspace=identity("workspace", "evaluator-agent-workspace", SHA_B),
            source=self.source,
            base_commit=self.fixture.revision,
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=("calc.py",),
            empty=False,
        )
        artifact = build_patch_artifact(frozen, artifact_id="attempt-001/final.patch")
        result = replace(session_result(), final_patch=patch)
        return FreshEvaluator(self.backend, clock_ms=StepClock()).evaluate(
            result,
            spec,
            frozen,
            artifact,
        )

    def test_bad_gold_and_regression_controls_use_a_fresh_source_copy(self) -> None:
        poison = self.root / "agent-workspace"
        poison.mkdir()
        (poison / "calc.py").write_text("def normalize(value):\n    return value\n")
        (poison / "test_calc.py").write_text("raise AssertionError('poison read')\n")

        cases = (
            (self.fixture.bad_patch, "f2p_failed"),
            (self.fixture.gold_patch, "resolved"),
            (self.fixture.regression_patch, "p2p_regression"),
        )
        for patch_bytes, expected in cases:
            with self.subTest(expected=expected):
                completed = self.evaluate(patch_bytes)
                self.assertEqual(completed.result.evaluation_outcome, expected)
                self.assertEqual(completed.result.attempt_validity, "valid")
                self.assertIsNotNone(completed.private_evidence)
                self.assertEqual(
                    PrivateEvaluationEvidence.from_dict(
                        completed.private_evidence.to_dict()
                    ),
                    completed.private_evidence,
                )
                self.assertEqual(list(self.evaluation_root.iterdir()), [])

    def test_nonapplying_patch_has_no_fuzz_fallback(self) -> None:
        completed = self.evaluate(self.fixture.invalid_patch)

        self.assertEqual(completed.result.attempt_validity, "valid")
        self.assertEqual(completed.result.evaluation_outcome, "invalid_patch")
        self.assertEqual(list(self.evaluation_root.iterdir()), [])

    def test_changed_source_content_is_infrastructure_invalid(self) -> None:
        wrong_source = replace(self.source, digest="sha256:" + "f" * 64)
        backend = LocalGitEvaluationBackend(
            source=LocalGitSource(
                identity=wrong_source,
                repository=self.fixture.repository,
                revision=self.fixture.revision,
            ),
            hidden_asset=self.backend.hidden_asset,
            python_executable=sys.executable,
            workspace_parent=self.evaluation_root,
        )
        patch = raw_patch_identity(self.fixture.gold_patch, identifier="final.patch")
        spec = replace(self.spec, source=wrong_source, frozen_patch=patch)
        frozen = FrozenPatch(
            workspace=identity("workspace", "source-mismatch", SHA_B),
            source=wrong_source,
            base_commit=self.fixture.revision,
            patch=patch,
            patch_bytes=self.fixture.gold_patch,
            changed_paths=("calc.py",),
            empty=False,
        )

        completed = FreshEvaluator(backend, clock_ms=StepClock()).evaluate(
            replace(session_result(), final_patch=patch),
            spec,
            frozen,
            build_patch_artifact(frozen, artifact_id="attempt-001/final.patch"),
        )

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(completed.result.evaluation_outcome, "evaluation_error")
        self.assertEqual(completed.result.invalid_reason, "source_identity_mismatch")
        self.assertEqual(list(self.evaluation_root.iterdir()), [])

    def test_source_archive_failure_has_stable_infrastructure_attribution(self) -> None:
        with mock.patch(
            "op_bench.runtime.local_evaluation.git_archive_source_identity",
            side_effect=ContractError("source archive failed"),
        ):
            completed = self.evaluate(self.fixture.gold_patch)

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(completed.result.evaluation_outcome, "evaluation_error")
        self.assertEqual(
            completed.result.invalid_reason,
            "source_identity_verification_failed",
        )
        self.assertEqual(list(self.evaluation_root.iterdir()), [])

    def test_agent_stdout_cannot_forge_structured_test_counts(self) -> None:
        completed = self.evaluate(self.fixture.forged_output_patch)

        self.assertEqual(completed.result.evaluation_outcome, "resolved")
        self.assertEqual(completed.result.fail_to_pass.collected, 1)
        self.assertEqual(completed.result.fail_to_pass.executed, 1)
        self.assertEqual(completed.result.pass_to_pass.collected, 1)
        self.assertEqual(completed.result.pass_to_pass.executed, 1)

    def test_workspace_module_cannot_shadow_evaluator_owned_unittest_runner(self) -> None:
        completed = self.evaluate(self.fixture.unittest_shadow_patch)

        self.assertEqual(completed.result.evaluation_outcome, "resolved")
        self.assertEqual(completed.result.fail_to_pass.passed, 1)
        self.assertEqual(completed.result.pass_to_pass.passed, 1)

    def test_materialization_timeout_has_stable_evaluation_timeout_attribution(self) -> None:
        with mock.patch(
            "op_bench.runtime.local_evaluation._run",
            side_effect=subprocess.TimeoutExpired(("git", "clone"), 1),
        ):
            completed = self.evaluate(self.fixture.gold_patch)

        self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
        self.assertEqual(completed.result.evaluation_outcome, "evaluation_error")
        self.assertEqual(completed.result.invalid_reason, "evaluation_timeout")


if __name__ == "__main__":
    unittest.main()
