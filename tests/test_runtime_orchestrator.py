from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from op_bench.runtime.backends import LocalProcessBackend, RuntimeTargetBinding
from op_bench.runtime.codex_adapter import CodexAdapterResult
from op_bench.runtime.contracts import ActionRequest, ContentIdentity, TestSelector
from op_bench.runtime.integrity import verify_run_artifacts
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.orchestrator import V06Orchestrator, V06RunRequest
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.resources import (
    parse_runtime_lease_store,
    parse_runtime_resource_ledger,
    verify_runtime_cleanup,
    verify_runtime_resource_ownership,
)
from tests.runtime_git_fixture import git, initialize_evaluation_git_fixture
from tests.test_runtime_contracts import (
    agent_spec,
    budget_policy,
    capability_policy,
    full_task_spec,
    identity,
)
from tests.test_runtime_manifest import manifest


F2P = "test_calc.NormalizeTests.test_nan_is_preserved"
P2P = "test_public.PublicTests.test_number_is_preserved"


class StepClock:
    def __init__(self, start: int = 1_000) -> None:
        self.value = start

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class ScriptedPatchAdapter:
    def __init__(self) -> None:
        self.run_count = 0

    def run(self, context) -> CodexAdapterResult:
        self.run_count += 1
        actions = (
            ("workspace_read", {"path": "calc.py"}),
            (
                "workspace_write",
                {
                    "path": "calc.py",
                    "content": "def normalize(value):\n    return value\n",
                },
            ),
            ("test_run", {"selector_id": P2P}),
            ("vcs_diff", {}),
            ("session_finish", {}),
        )
        observations = []
        for sequence, (name, arguments) in enumerate(actions, start=1):
            request = ActionRequest(
                session_id=context.session_id,
                action_id=f"scripted-action-{sequence}",
                action_name=name,
                arguments=arguments,
                client_sequence=sequence,
                deadline_ms=context.launch_input.task_view.budget_policy.wall_clock_ms,
            )
            observation = context.action_client.execute(request.to_dict())
            observations.append(observation)
            self.assert_observation_identity(observation, request)
        return CodexAdapterResult(
            status="completed",
            terminal_reason="agent_finished",
            exit_code=0,
            observation_count=len(observations),
            finish_count=1,
        )

    @staticmethod
    def assert_observation_identity(observation, request) -> None:
        if observation["session_id"] != request.session_id:
            raise AssertionError("session mismatch")
        if observation["action_id"] != request.action_id:
            raise AssertionError("action mismatch")
        if not observation["ok"]:
            raise AssertionError(observation)


class V06OrchestratorTests(unittest.TestCase):
    def test_local_happy_path_is_complete_and_resume_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            git_fixture = initialize_evaluation_git_fixture(root / "source")
            (git_fixture.repository / "test_public.py").write_text(
                "import unittest\n\n"
                "from calc import normalize\n\n"
                "class PublicTests(unittest.TestCase):\n"
                "    def test_number_is_preserved(self):\n"
                "        self.assertEqual(normalize(1), 1)\n",
                encoding="utf-8",
            )
            git(git_fixture.repository, "add", "test_public.py")
            git(git_fixture.repository, "commit", "--quiet", "-m", "add public test")
            revision = git(
                git_fixture.repository,
                "rev-parse",
                "HEAD",
            ).stdout.decode("ascii").strip()
            source_identity = git_archive_source_identity(
                git_fixture.repository,
                revision,
                "fixture@orchestrator-base",
            )
            hidden_identity = ContentIdentity(
                identity_type="test",
                identifier="fixture:hidden-test.patch",
                digest="sha256:" + hashlib.sha256(git_fixture.hidden_test_patch).hexdigest(),
                digest_kind="content_sha256",
            )
            hidden_selector = TestSelector(
                selector_id=F2P,
                visibility="evaluation_only",
                command_template="{python} -m unittest {test}",
                description="F2P control",
            )
            public_selector = TestSelector(
                selector_id=P2P,
                visibility="registered",
                command_template="{python} -m unittest {test}",
                description="Public regression control",
            )
            hidden_asset = EvaluationOnlyTestAsset(
                identity=hidden_identity,
                patch_bytes=git_fixture.hidden_test_patch,
                selectors=(hidden_selector,),
            )

            registry = load_runtime_profile_registry(
                Path(__file__).resolve().parents[1]
                / "configs"
                / "runtime_profiles.v1.json"
            )
            profile = registry.get("local-cpu-process-v1")
            capability = replace(
                capability_policy(),
                allowed_actions=(
                    "workspace_read",
                    "workspace_write",
                    "test_run",
                    "vcs_diff",
                    "session_finish",
                ),
                writable_paths=("calc.py",),
                registered_tests=(P2P,),
                network_access="provider_only",
            )
            budget = replace(budget_policy(), wall_clock_ms=60_000)
            task = replace(
                full_task_spec(),
                source=source_identity,
                runtime=profile,
                public_tests=(public_selector,),
                hidden_tests=(hidden_selector,),
                fail_to_pass=(F2P,),
                pass_to_pass=(P2P,),
                patch_scope=("calc.py",),
                hidden_test_asset=hidden_identity,
            )
            scripted_agent = replace(
                agent_spec(),
                agent=identity("agent", "scripted", "sha256:" + "d" * 64),
                adapter=identity(
                    "adapter",
                    "scripted_canonical",
                    "sha256:" + "e" * 64,
                ),
            )
            frozen_manifest = manifest(
                tasks=(task,),
                agents=(scripted_agent,),
                capability=capability,
                budget=budget,
                repeat_count=1,
            )
            expected = frozen_manifest.expected_attempts[0]
            workspaces = root / "workspaces"
            workspaces.mkdir()
            output = root / "run"
            adapter = ScriptedPatchAdapter()
            backend_phases: list[str] = []

            def backend_factory(profile, target_binding, phase):
                backend_phases.append(phase)
                return LocalProcessBackend()

            orchestrator = V06Orchestrator(
                source_resolver=lambda task: LocalGitSource(
                    identity=source_identity,
                    repository=git_fixture.repository,
                    revision=revision,
                ),
                hidden_asset_resolver=lambda task: hidden_asset,
                backend_factory=backend_factory,
                adapter_factory=lambda agent, adapter_id: adapter,
                python_executable=sys.executable,
            )
            request = V06RunRequest(
                manifest=frozen_manifest,
                selected_attempt_ids=(expected.attempt_id,),
                runtime_profile_registry=registry,
                runtime_profile_id=profile.profile_id,
                target_binding=RuntimeTargetBinding(
                    backend="local",
                    local_workspace_parent=workspaces,
                ),
                output_root=output,
                resume_policy="retry_infrastructure",
                adapter_id="scripted_canonical",
                enable_external_canary=False,
                clock_ms=StepClock(),
            )

            first = orchestrator.run(request)

            self.assertEqual(first.ran_attempt_ids, (expected.attempt_id,))
            self.assertEqual(first.skipped_attempt_ids, ())
            self.assertEqual(first.integrity.status, "passed")
            self.assertEqual(adapter.run_count, 1)
            self.assertEqual(backend_phases, ["agent", "evaluation"])
            self.assertEqual(list(workspaces.iterdir()), [])

            events_path = (
                output
                / "attempts"
                / expected.attempt_id
                / "retries"
                / "retry-0001"
                / "events.jsonl"
            )
            event_types = [
                json.loads(line)["event_type"]
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(event_types[0:4], [
                "session_created",
                "session_prepared",
                "session_started",
                "agent_launched",
            ])
            self.assertEqual(event_types[-5:], [
                "patch_freeze_completed",
                "session_terminal_emitted",
                "evaluation_started",
                "evaluation_completed",
                "terminal_emitted",
            ])
            self.assertEqual(event_types.count("action_requested"), 5)
            self.assertEqual(event_types.count("action_observed"), 5)

            retry_root = events_path.parent
            records = parse_runtime_resource_ledger(
                (retry_root / "runtime_resources.jsonl").read_bytes(),
                attempt_id=expected.attempt_id,
                retry_index=1,
                runtime_profile_hash=profile.content_hash,
            )
            handles = parse_runtime_lease_store(
                (retry_root / "private_runtime_resources.json").read_bytes(),
                attempt_id=expected.attempt_id,
                retry_index=1,
                runtime_profile_hash=profile.content_hash,
            )
            cleanup = first.cleanup_reports[expected.attempt_id]
            verify_runtime_resource_ownership(records, handles)
            verify_runtime_cleanup(records, cleanup)
            self.assertTrue(cleanup.all_released)
            self.assertEqual(
                json.loads((output / "results.jsonl").read_text(encoding="utf-8"))[
                    "evaluation_outcome"
                ],
                "resolved",
            )
            self.assertEqual(verify_run_artifacts(output, frozen_manifest).status, "passed")

            before = {
                str(path.relative_to(output)): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }
            second = orchestrator.run(request)
            after = {
                str(path.relative_to(output)): path.read_bytes()
                for path in output.rglob("*")
                if path.is_file()
            }

            self.assertEqual(second.ran_attempt_ids, ())
            self.assertEqual(second.skipped_attempt_ids, (expected.attempt_id,))
            self.assertEqual(adapter.run_count, 1)
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
