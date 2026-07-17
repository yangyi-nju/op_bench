from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    FrozenPatch,
    PatchArtifact,
    assert_patch_identity_handoff,
    build_patch_artifact,
)
from op_bench.runtime.validation import ContractError
from tests.runtime_git_fixture import initialize_git_repo
from tests.test_runtime_contracts import SHA_A, SHA_B, identity
from tests.test_runtime_wire_contracts import evaluation_spec, session_result
from tests.test_runtime_workspace import policy


class FrozenPatchIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)
        workspace = AuthoritativeWorkspace.open(
            self.root,
            source=identity("source", "fixture@base", SHA_A),
            policy=policy(),
        )
        workspace.write("src/operator.py", b"VALUE = 2\n")
        self.frozen = workspace.freeze()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_session_artifact_and_evaluation_share_exact_patch_identity(self) -> None:
        artifact = build_patch_artifact(self.frozen, artifact_id="attempt-1/final.patch")
        session = replace(session_result(), final_patch=self.frozen.patch)
        evaluation = replace(evaluation_spec(), frozen_patch=self.frozen.patch)

        assert_patch_identity_handoff(
            frozen=self.frozen,
            session_result=session,
            patch_artifact=artifact,
            evaluation_spec=evaluation,
        )

        self.assertEqual(session.final_patch, artifact.patch)
        self.assertEqual(artifact.patch, evaluation.frozen_patch)
        self.assertEqual(artifact.workspace, self.frozen.workspace)
        self.assertEqual(artifact.size_bytes, len(self.frozen.patch_bytes))

    def test_handoff_rejects_any_of_three_identity_mismatches(self) -> None:
        artifact = build_patch_artifact(self.frozen, artifact_id="attempt-1/final.patch")
        session = replace(session_result(), final_patch=self.frozen.patch)
        evaluation = replace(evaluation_spec(), frozen_patch=self.frozen.patch)
        wrong = identity("patch", "other.patch", SHA_B)
        mutations = (
            (replace(session, final_patch=wrong), artifact, evaluation),
            (session, replace(artifact, patch=wrong), evaluation),
            (session, artifact, replace(evaluation, frozen_patch=wrong)),
        )

        for changed_session, changed_artifact, changed_evaluation in mutations:
            with self.subTest(
                session=changed_session.final_patch,
                artifact=changed_artifact.patch,
                evaluation=changed_evaluation.frozen_patch,
            ):
                with self.assertRaisesRegex(ContractError, "patch identity mismatch"):
                    assert_patch_identity_handoff(
                        frozen=self.frozen,
                        session_result=changed_session,
                        patch_artifact=changed_artifact,
                        evaluation_spec=changed_evaluation,
                    )

    def test_frozen_patch_rejects_raw_byte_tampering(self) -> None:
        with self.assertRaisesRegex(ContractError, "patch bytes do not match"):
            replace(self.frozen, patch_bytes=self.frozen.patch_bytes + b"\n")

    def test_patch_artifact_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ContractError, "artifact_id"):
            PatchArtifact(
                artifact_id="",
                workspace=self.frozen.workspace,
                patch=self.frozen.patch,
                size_bytes=len(self.frozen.patch_bytes),
                changed_paths=self.frozen.changed_paths,
                empty=False,
            )
        with self.assertRaisesRegex(ContractError, "workspace"):
            replace(build_patch_artifact(self.frozen, artifact_id="final.patch"), workspace=identity("source", "bad"))


if __name__ == "__main__":
    unittest.main()
