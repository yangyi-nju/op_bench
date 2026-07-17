from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
import sys

from op_bench.runtime.backends import RuntimeTargetBinding
from op_bench.runtime.codex_adapter import CodexAdapterResult
from op_bench.runtime.contracts import ActionRequest, ContentIdentity, TestSelector
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.profiles import load_runtime_profile_registry
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


class PatchAdapter:
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
        for sequence, (name, arguments) in enumerate(actions, start=1):
            request = ActionRequest(
                session_id=context.session_id,
                action_id=f"fixture-action-{sequence}",
                action_name=name,
                arguments=arguments,
                client_sequence=sequence,
                deadline_ms=context.launch_input.task_view.budget_policy.wall_clock_ms,
            )
            observation = context.action_client.execute(request.to_dict())
            if not observation["ok"]:
                raise AssertionError(observation)
        return CodexAdapterResult(
            status="completed",
            terminal_reason="agent_finished",
            exit_code=0,
            observation_count=len(actions),
            finish_count=1,
        )


@dataclass(frozen=True)
class OrchestratorFixture:
    source: LocalGitSource
    hidden_asset: EvaluationOnlyTestAsset
    registry: object
    profile: object
    manifest: object
    expected: object
    target_binding: RuntimeTargetBinding
    output_root: Path


def build_orchestrator_fixture(root: Path) -> OrchestratorFixture:
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
    revision = git(git_fixture.repository, "rev-parse", "HEAD").stdout.decode(
        "ascii"
    ).strip()
    source_identity = git_archive_source_identity(
        git_fixture.repository,
        revision,
        "fixture@orchestrator-base",
    )
    source = LocalGitSource(
        identity=source_identity,
        repository=git_fixture.repository,
        revision=revision,
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
        Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
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
        budget=replace(budget_policy(), wall_clock_ms=60_000),
        repeat_count=1,
    )
    workspaces = root / "workspaces"
    workspaces.mkdir()
    return OrchestratorFixture(
        source=source,
        hidden_asset=hidden_asset,
        registry=registry,
        profile=profile,
        manifest=frozen_manifest,
        expected=frozen_manifest.expected_attempts[0],
        target_binding=RuntimeTargetBinding(
            backend="local",
            local_workspace_parent=workspaces,
        ),
        output_root=root / "run",
    )

def orchestrator_for(fixture: OrchestratorFixture, *, backend_factory, adapter):
    from op_bench.runtime.orchestrator import V06Orchestrator

    return V06Orchestrator(
        source_resolver=lambda task: fixture.source,
        hidden_asset_resolver=lambda task: fixture.hidden_asset,
        backend_factory=backend_factory,
        adapter_factory=lambda agent, adapter_id: adapter,
        python_executable=sys.executable,
    )


def request_for(fixture: OrchestratorFixture, *, clock=None):
    from op_bench.runtime.orchestrator import V06RunRequest

    return V06RunRequest(
        manifest=fixture.manifest,
        selected_attempt_ids=(fixture.expected.attempt_id,),
        runtime_profile_registry=fixture.registry,
        runtime_profile_id=fixture.profile.profile_id,
        target_binding=fixture.target_binding,
        output_root=fixture.output_root,
        resume_policy="retry_infrastructure",
        adapter_id="scripted_canonical",
        enable_external_canary=False,
        clock_ms=clock or StepClock(),
    )
