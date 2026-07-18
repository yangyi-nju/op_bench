from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.backends import (
    LocalProcessBackend,
    RuntimeAttemptContext,
    RuntimeCommandResult,
    RuntimeTargetBinding,
)
from op_bench.runtime.contracts import ContentIdentity, TestSelector
from op_bench.runtime.evaluation import FreshEvaluator
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.resources import (
    AttemptResourceLedger,
    RuntimeLeaseStore,
    verify_runtime_cleanup,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.runtime_evaluation import RuntimeFreshEvaluationBackend
from op_bench.runtime.workspace import FrozenPatch, build_patch_artifact, raw_patch_identity
from tests.runtime_git_fixture import initialize_evaluation_git_fixture
from tests.test_runtime_contracts import SHA_B, identity
from tests.test_runtime_wire_contracts import evaluation_spec, session_result


ATTEMPT_ID = "attempt:v1:" + "e" * 64
F2P = "test_calc.NormalizeTests.test_nan_is_preserved"
P2P = "test_calc.NormalizeTests.test_number_is_preserved"


class StepClock:
    def __init__(self) -> None:
        self.value = 100

    def __call__(self) -> int:
        current = self.value
        self.value += 25
        return current


class RecordingLocalBackend(LocalProcessBackend):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list[tuple[str, ...]] = []

    def run(self, lease, command, cwd, timeout_ms):
        self.commands.append(command)
        return super().run(lease, command, cwd, timeout_ms)


class SelectorTimeoutBackend(RecordingLocalBackend):
    def run(self, lease, command, cwd, timeout_ms):
        if "--selector" in command:
            self.commands.append(command)
            return super(RecordingLocalBackend, self).run(
                lease,
                (sys.executable, "-c", "import time; time.sleep(30)"),
                cwd,
                50,
            )
        return super().run(lease, command, cwd, timeout_ms)


class RuntimeEvaluationFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.git = initialize_evaluation_git_fixture(root / "source")
        self.source = git_archive_source_identity(
            self.git.repository,
            self.git.revision,
            "fixture@runtime-evaluation-base",
        )
        self.hidden = ContentIdentity(
            "test",
            "fixture:hidden-test.patch",
            "sha256:" + hashlib.sha256(self.git.hidden_test_patch).hexdigest(),
            "content_sha256",
        )
        self.hidden_asset = EvaluationOnlyTestAsset(
            identity=self.hidden,
            patch_bytes=self.git.hidden_test_patch,
            selectors=(
                TestSelector(
                    selector_id=F2P,
                    visibility="evaluation_only",
                    command_template="{python} -m unittest {test}",
                    description="F2P control",
                ),
                TestSelector(
                    selector_id=P2P,
                    visibility="evaluation_only",
                    command_template="{python} -m unittest {test}",
                    description="P2P control",
                ),
            ),
        )
        self.profile = load_runtime_profile_registry(
            Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
        ).get("local-cpu-process-v1")
        self.workspaces = root / "workspaces"
        self.workspaces.mkdir()
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.ledger = AttemptResourceLedger(
            self.evidence / "runtime_resources.jsonl",
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
            clock_ms=StepClock(),
        )
        self.store = RuntimeLeaseStore(
            self.evidence / "private_runtime_resources.json",
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
        )
        self.context = RuntimeAttemptContext(
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
            frozen_source_directory=self.git.repository,
            frozen_source_revision=self.git.revision,
            resource_ledger=self.ledger,
            lease_store=self.store,
            target_binding=RuntimeTargetBinding(
                backend="local",
                local_workspace_parent=self.workspaces,
            ),
        )
        self.runtime = RecordingLocalBackend()
        self.backend = RuntimeFreshEvaluationBackend(
            source=LocalGitSource(
                identity=self.source,
                repository=self.git.repository,
                revision=self.git.revision,
            ),
            hidden_asset=self.hidden_asset,
            python_executable=sys.executable,
            runtime_backend=self.runtime,
            runtime_profile=self.profile,
            attempt_context=self.context,
        )

    def evaluate(self, patch_bytes: bytes, *, source=None):
        patch = raw_patch_identity(patch_bytes, identifier="final.patch")
        selected_source = self.source if source is None else source
        spec = replace(
            evaluation_spec(),
            attempt_id=ATTEMPT_ID,
            source=selected_source,
            hidden_test_asset=self.hidden,
            public_tests=(),
            fail_to_pass=(F2P,),
            pass_to_pass=(P2P,),
            frozen_patch=patch,
            runtime=self.profile,
            timeout_ms=30_000,
        )
        frozen = FrozenPatch(
            workspace=identity("workspace", "poison-agent-workspace", SHA_B),
            source=selected_source,
            base_commit=self.git.revision,
            patch=patch,
            patch_bytes=patch_bytes,
            changed_paths=("calc.py",),
            empty=False,
        )
        return FreshEvaluator(self.backend, clock_ms=StepClock()).evaluate(
            replace(
                session_result(),
                attempt_id=ATTEMPT_ID,
                final_patch=patch,
            ),
            spec,
            frozen,
            build_patch_artifact(frozen, artifact_id=f"{ATTEMPT_ID}/final.patch"),
        )


class RuntimeFreshEvaluationBackendTests(unittest.TestCase):
    def test_source_and_patches_are_staged_without_runtime_git(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))

            completed = fixture.evaluate(fixture.git.gold_patch)

            self.assertEqual(completed.result.evaluation_outcome, "resolved")
            self.assertFalse(
                any(command[0] == "git" for command in fixture.runtime.commands)
            )
            self.assertFalse(
                any(
                    "git','archive" in part or 'git","archive' in part
                    for command in fixture.runtime.commands
                    for part in command
                )
            )

    def test_baseline_replay_runs_hidden_tests_without_an_agent_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            spec = replace(
                evaluation_spec(),
                attempt_id=ATTEMPT_ID,
                source=fixture.source,
                hidden_test_asset=fixture.hidden,
                public_tests=(),
                fail_to_pass=(F2P,),
                pass_to_pass=(P2P,),
                runtime=fixture.profile,
                timeout_ms=30_000,
            )

            observed = FreshEvaluator(
                fixture.backend,
                clock_ms=StepClock(),
            ).evaluate_replay(spec, None, None)

            self.assertEqual(observed.observed_outcome, "f2p_failed")
            self.assertIsNone(observed.private_evidence.patch)
            self.assertFalse(
                any("agent.patch" in part for command in fixture.runtime.commands for part in command)
            )
            self.assertEqual(list(fixture.workspaces.iterdir()), [])

    def test_bad_gold_regression_are_fresh_and_f2p_precedes_p2p(self) -> None:
        for patch_name, expected in (
            ("bad_patch", "f2p_failed"),
            ("gold_patch", "resolved"),
            ("regression_patch", "p2p_regression"),
        ):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                fixture = RuntimeEvaluationFixture(Path(temporary))
                poison = fixture.root / "agent-workspace"
                poison.mkdir()
                (poison / "calc.py").write_text(
                    "raise AssertionError('poison workspace read')\n",
                    encoding="utf-8",
                )

                completed = fixture.evaluate(getattr(fixture.git, patch_name))

                self.assertEqual(completed.result.evaluation_outcome, expected)
                self.assertTrue(completed.private_evidence.cleanup_completed)
                selectors = [
                    command[command.index("--selector") + 1]
                    for command in fixture.runtime.commands
                    if "--selector" in command
                ]
                self.assertEqual(selectors, [F2P, P2P])
                self.assertEqual(list(fixture.workspaces.iterdir()), [])
                self.assertTrue(
                    all(
                        record.transition in {"released", "create_failed"}
                        for record in _final_records(fixture.ledger)
                    )
                )

    def test_invalid_patch_is_strict_and_still_cleans_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))

            completed = fixture.evaluate(fixture.git.invalid_patch)

            self.assertEqual(completed.result.evaluation_outcome, "invalid_patch")
            self.assertEqual(list(fixture.workspaces.iterdir()), [])
            self.assertEqual(_final_records(fixture.ledger)[0].transition, "released")

    def test_source_mismatch_is_infrastructure_invalid_before_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            wrong_source = replace(
                fixture.source,
                digest="sha256:" + "f" * 64,
            )

            completed = fixture.evaluate(fixture.git.gold_patch, source=wrong_source)

            self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
            self.assertEqual(completed.result.invalid_reason, "source_binding_mismatch")
            self.assertFalse(
                any("agent.patch" in part for command in fixture.runtime.commands for part in command)
            )
            self.assertEqual(list(fixture.workspaces.iterdir()), [])

    def test_legacy_logical_source_identity_verifies_exact_revision_without_hash_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            logical_source = ContentIdentity(
                identity_type="source",
                identifier="fixture@logical-revision",
                digest="sha256:" + "7" * 64,
                digest_kind="canonical_config",
            )
            fixture.backend.source = LocalGitSource(
                identity=logical_source,
                repository=fixture.git.repository,
                revision=fixture.git.revision,
            )

            completed = fixture.evaluate(
                fixture.git.gold_patch,
                source=logical_source,
            )

            self.assertEqual(completed.result.evaluation_outcome, "resolved")
            self.assertEqual(completed.result.attempt_validity, "valid")

    def test_exact_script_selector_template_is_supported_without_parsing_test_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            selectors = tuple(
                replace(
                    selector,
                    command_template=(
                        "{python} -c \"import sys; sys.exit(0)\" {test}"
                    ),
                )
                for selector in fixture.hidden_asset.selectors
            )
            fixture.backend.hidden_asset = replace(
                fixture.hidden_asset,
                selectors=selectors,
            )

            completed = fixture.evaluate(fixture.git.gold_patch)

            self.assertEqual(completed.result.evaluation_outcome, "resolved")
            direct = [
                command
                for command in fixture.runtime.commands
                if "import sys; sys.exit(0)" in command
            ]
            self.assertEqual(len(direct), 2)
            self.assertEqual(direct[0][-1], F2P)
            self.assertEqual(direct[1][-1], P2P)

    def test_cleanup_failure_is_evaluation_infrastructure_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            with mock.patch(
                "op_bench.runtime.backends.shutil.rmtree",
                side_effect=OSError("private cleanup failure"),
            ):
                completed = fixture.evaluate(fixture.git.gold_patch)

            self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
            self.assertEqual(completed.result.invalid_reason, "evaluation_cleanup_failed")
            workspace = next(
                record
                for record in _final_records(fixture.ledger)
                if record.resource_type == "workspace"
            )
            self.assertEqual(workspace.transition, "cleanup_failed")
            with self.assertRaisesRegex(ContractError, "cleanup_failed"):
                verify_runtime_cleanup(
                    fixture.ledger.records,
                    fixture.backend.last_cleanup_result.report,
                )

    def test_selector_timeout_is_stable_and_cleans_every_resource(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = RuntimeEvaluationFixture(Path(temporary))
            fixture.runtime = SelectorTimeoutBackend()
            fixture.backend.runtime_backend = fixture.runtime

            completed = fixture.evaluate(fixture.git.gold_patch)

            self.assertEqual(completed.result.attempt_validity, "infrastructure_invalid")
            self.assertEqual(completed.result.invalid_reason, "evaluation_timeout")
            self.assertEqual(list(fixture.workspaces.iterdir()), [])
            self.assertTrue(
                all(
                    record.transition in {"released", "create_failed"}
                    for record in _final_records(fixture.ledger)
                )
            )


def _final_records(ledger: AttemptResourceLedger):
    final = {}
    for record in ledger.records:
        final[record.resource_id] = record
    return tuple(final[key] for key in sorted(final))


if __name__ == "__main__":
    unittest.main()
