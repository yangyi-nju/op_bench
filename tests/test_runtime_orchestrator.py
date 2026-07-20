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
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ActionRequest, ContentIdentity, TestSelector
from op_bench.runtime.integrity import verify_run_artifacts
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.legacy import agent_spec_for_v1_adapter
from op_bench.runtime.mcp import McpAdapterTrace
from op_bench.runtime.orchestrator import V1_ADAPTER_IDS, V06Orchestrator, V06RunRequest
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.resources import (
    parse_runtime_lease_store,
    parse_runtime_resource_ledger,
    verify_runtime_cleanup,
    verify_runtime_resource_ownership,
)
from tests.runtime_git_fixture import git, initialize_evaluation_git_fixture
from tests.runtime_orchestrator_fixture import (
    PatchAdapter,
    build_orchestrator_fixture,
    orchestrator_for,
    request_for,
)
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


class McpScenarioAdapter:
    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.run_count = 0

    @staticmethod
    def trace(*, calls: int, terminal: str = "client_closed", initialized: bool = True):
        return McpAdapterTrace(
            adapter_id="codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
            negotiated_protocol_version="2025-06-18" if initialized else None,
            initialize_count=1 if initialized else 0,
            tools_list_count=1 if initialized else 0,
            tools_call_count=calls,
            protocol_error_count=0,
            server_terminal_status=terminal,
        )

    def run(self, context) -> CodexAdapterResult:
        self.run_count += 1
        if self.scenario == "infrastructure_then_resolved" and self.run_count == 1:
            return CodexAdapterResult(
                status="provider_failure",
                terminal_reason="provider_error",
                exit_code=1,
                observation_count=0,
                finish_count=0,
                adapter_trace=self.trace(
                    calls=0,
                    terminal="start_failed",
                    initialized=False,
                ),
            )

        if self.scenario == "timeout":
            actions = ()
            status = "timeout"
            terminal_reason = "timeout"
            trace_terminal = "terminated"
        elif self.scenario == "no_patch":
            actions = (("session_finish", {}),)
            status = "completed"
            terminal_reason = "agent_finished"
            trace_terminal = "client_closed"
        else:
            content = (
                "def normalize(value):\n"
                "    # Deliberately preserve the defect for F2P attribution.\n"
                "    return 0 if value != value else value\n"
                if self.scenario == "f2p_failed"
                else "def normalize(value):\n    return value\n"
            )
            actions = (
                ("workspace_read", {"path": "calc.py"}),
                ("workspace_write", {"path": "calc.py", "content": content}),
                ("test_run", {"selector_id": P2P}),
                ("vcs_diff", {}),
                *(() if self.scenario == "missing_finish" else (("session_finish", {}),)),
            )
            status = "missing_finish" if self.scenario == "missing_finish" else "completed"
            terminal_reason = "agent_exited" if status == "missing_finish" else "agent_finished"
            trace_terminal = "client_closed"

        for sequence, (name, arguments) in enumerate(actions, start=1):
            request = ActionRequest(
                session_id=context.session_id,
                action_id=f"mcp-scenario-{self.run_count}-{sequence}",
                action_name=name,
                arguments=arguments,
                client_sequence=sequence,
                deadline_ms=context.launch_input.task_view.budget_policy.wall_clock_ms,
            )
            observation = context.action_client.execute(request.to_dict())
            if not observation["ok"]:
                raise AssertionError(observation)
        finish_count = sum(1 for name, _ in actions if name == "session_finish")
        return CodexAdapterResult(
            status=status,
            terminal_reason=terminal_reason,
            exit_code=None if status == "timeout" else 0,
            observation_count=len(actions),
            finish_count=finish_count,
            adapter_trace=self.trace(
                calls=len(actions),
                terminal=trace_terminal,
            ),
        )


class V06OrchestratorTests(unittest.TestCase):
    def test_v1_adapter_registry_includes_the_independent_mcp_adapter(self) -> None:
        self.assertEqual(
            V1_ADAPTER_IDS,
            ("scripted_canonical", "codex_canonical", "codex_mcp_canonical"),
        )

    def test_mcp_trace_is_persisted_and_bound_under_lifecycle_integrity(self) -> None:
        mcp_agent = agent_spec_for_v1_adapter(
            "codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
        )
        trace = McpAdapterTrace(
            adapter_id="codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
            negotiated_protocol_version="2025-06-18",
            initialize_count=1,
            tools_list_count=1,
            tools_call_count=5,
            protocol_error_count=0,
            server_terminal_status="client_closed",
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(
                Path(temporary),
                selected_agent=mcp_agent,
            )
            adapter = PatchAdapter(adapter_trace=trace)

            result = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                adapter=adapter,
            ).run(
                request_for(
                    fixture,
                    adapter_id="codex_mcp_canonical",
                    enable_external_canary=True,
                )
            )

            retry_root = (
                fixture.output_root
                / "attempts"
                / fixture.expected.attempt_id
                / "retries"
                / "retry-0001"
            )
            trace_path = retry_root / "adapter_trace.json"
            self.assertTrue(trace_path.is_file())
            self.assertEqual(json.loads(trace_path.read_text()), trace.to_dict())
            self.assertEqual(result.integrity.status, "passed")
            self.assertEqual(len(result.integrity.checks), 14)

            original = trace_path.read_bytes()
            mutations = (
                {**trace.to_dict(), "model_id": "gpt-5.6-terra"},
                {**trace.to_dict(), "tools_call_count": 4},
                {**trace.to_dict(), "codex_cli_version": "codex-cli 0.146.0"},
            )
            for payload in mutations:
                with self.subTest(payload=payload):
                    trace_path.write_bytes(
                        (canonical_json(payload) + "\n").encode("utf-8")
                    )
                    report = verify_run_artifacts(
                        fixture.output_root,
                        fixture.manifest,
                    )
                    checks = {check.check_id: check for check in report.checks}
                    self.assertEqual(checks["lifecycle_terminal"].status, "failed")
                    self.assertEqual(len(report.checks), 14)
                    trace_path.write_bytes(original)

            trace_path.unlink()
            report = verify_run_artifacts(fixture.output_root, fixture.manifest)
            checks = {check.check_id: check for check in report.checks}
            self.assertEqual(checks["lifecycle_terminal"].status, "failed")
            self.assertEqual(len(report.checks), 14)

    def test_local_happy_path_is_complete_and_resume_is_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            git_fixture = initialize_evaluation_git_fixture(root / "source")
            calc_path = git_fixture.repository / "calc.py"
            calc_path.write_text(
                calc_path.read_text(encoding="utf-8")
                + "\n# Public source padding for large Action data.\n"
                + "# "
                + "x" * 5_000
                + "\n",
                encoding="utf-8",
            )
            (git_fixture.repository / "test_public.py").write_text(
                "import unittest\n\n"
                "from calc import normalize\n\n"
                "class PublicTests(unittest.TestCase):\n"
                "    def test_number_is_preserved(self):\n"
                "        self.assertEqual(normalize(1), 1)\n",
                encoding="utf-8",
            )
            git(git_fixture.repository, "add", "calc.py", "test_public.py")
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
            self.assertEqual(first.integrity.status, "passed", first.integrity)
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
            event_payloads = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
            ]
            event_types = [item["event_type"] for item in event_payloads]
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
            first_observation = next(
                item
                for item in event_payloads
                if item["event_type"] == "action_observed"
            )
            reference = first_observation["public_payload"]["data_artifact"]
            action_artifact = retry_root / "action_artifacts" / reference["artifact_id"]
            original_action_artifact = action_artifact.read_bytes()
            self.assertGreater(reference["size_bytes"], 4_096)
            action_artifact.write_bytes(b"{}")
            tampered = verify_run_artifacts(output, frozen_manifest)
            tampered_checks = {item.check_id: item for item in tampered.checks}
            self.assertEqual(tampered_checks["lifecycle_terminal"].status, "failed")
            action_artifact.write_bytes(original_action_artifact)

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

    def test_mcp_infrastructure_invalid_attempt_gets_a_new_retry(self) -> None:
        mcp_agent = agent_spec_for_v1_adapter(
            "codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = build_orchestrator_fixture(
                Path(temporary),
                selected_agent=mcp_agent,
            )
            adapter = McpScenarioAdapter("infrastructure_then_resolved")
            orchestrator = orchestrator_for(
                fixture,
                backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                adapter=adapter,
            )
            request = request_for(
                fixture,
                adapter_id="codex_mcp_canonical",
                enable_external_canary=True,
            )

            first = orchestrator.run(request)
            second = orchestrator.run(request)

            self.assertEqual(first.ran_attempt_ids, (fixture.expected.attempt_id,))
            self.assertEqual(second.ran_attempt_ids, (fixture.expected.attempt_id,))
            self.assertEqual(second.skipped_attempt_ids, ())
            self.assertEqual(adapter.run_count, 2)
            records = [
                json.loads(line)
                for line in (fixture.output_root / "attempts.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                [(item["retry_index"], item["attempt_validity"]) for item in records],
                [(1, "infrastructure_invalid"), (2, "valid")],
            )
            self.assertEqual(second.integrity.status, "passed")

    def test_valid_mcp_logical_outcomes_are_skipped_byte_for_byte(self) -> None:
        mcp_agent = agent_spec_for_v1_adapter(
            "codex_mcp_canonical",
            model_id="gpt-5.6-sol",
            codex_cli_version="codex-cli 0.145.0-alpha.18",
        )
        expected = {
            "timeout": ("timeout", "no_patch"),
            "missing_finish": ("exited", "resolved"),
            "no_patch": ("finished", "no_patch"),
            "f2p_failed": ("finished", "f2p_failed"),
        }
        for scenario, (terminal_reason, evaluation_outcome) in expected.items():
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                fixture = build_orchestrator_fixture(
                    Path(temporary),
                    selected_agent=mcp_agent,
                )
                adapter = McpScenarioAdapter(scenario)
                orchestrator = orchestrator_for(
                    fixture,
                    backend_factory=lambda profile, target, phase: LocalProcessBackend(),
                    adapter=adapter,
                )
                request = request_for(
                    fixture,
                    adapter_id="codex_mcp_canonical",
                    enable_external_canary=True,
                )

                first = orchestrator.run(request)
                before = {
                    str(path.relative_to(fixture.output_root)): path.read_bytes()
                    for path in fixture.output_root.rglob("*")
                    if path.is_file()
                }
                second = orchestrator.run(request)
                after = {
                    str(path.relative_to(fixture.output_root)): path.read_bytes()
                    for path in fixture.output_root.rglob("*")
                    if path.is_file()
                }

                result = json.loads(
                    (fixture.output_root / "results.jsonl").read_text(encoding="utf-8")
                )
                self.assertEqual(first.integrity.status, "passed")
                self.assertEqual(second.ran_attempt_ids, ())
                self.assertEqual(second.skipped_attempt_ids, (fixture.expected.attempt_id,))
                self.assertEqual(adapter.run_count, 1)
                self.assertEqual(result["attempt_validity"], "valid")
                self.assertEqual(result["agent_terminal"], terminal_reason)
                self.assertEqual(result["evaluation_outcome"], evaluation_outcome)
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
