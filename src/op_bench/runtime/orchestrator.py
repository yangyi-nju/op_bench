from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
import shlex

from op_bench.runtime.actions import (
    CanonicalActionService,
    RegisteredTest,
)
from op_bench.runtime.adapters import AdapterActionChannel, AdapterContext
from op_bench.runtime.artifacts import PublicArtifactStore
from op_bench.runtime.backends import (
    RuntimeAttemptContext,
    RuntimeBackend,
    RuntimeCommandBackend,
    RuntimeTargetBinding,
)
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import (
    ActionRequest,
    AgentSpec,
    EvaluationSpec,
    IntegrityReport,
    RuntimeProfile,
    SessionResult,
)
from op_bench.runtime.evaluation import AttemptEvaluationCoordinator, FreshEvaluator
from op_bench.runtime.events import EventJournal
from op_bench.runtime.integrity import (
    persist_integrity_reports,
    selected_attempts_from_ledger,
    verify_run_artifacts,
)
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.manifest import RunManifest
from op_bench.runtime.profiles import RuntimeProfileRegistry
from op_bench.runtime.process_group import ProcessGroupCleanupError
from op_bench.runtime.resources import (
    AttemptResourceLedger,
    RuntimeCleanupEntry,
    RuntimeCleanupReport,
    RuntimeLeaseStore,
)
from op_bench.runtime.resume import AttemptLedger
from op_bench.runtime.run_artifacts import AttemptArtifactStore, retry_directory_name
from op_bench.runtime.runtime_evaluation import RuntimeFreshEvaluationBackend
from op_bench.runtime.session import AttemptSession
from op_bench.runtime.summary import write_rebuilt_outputs
from op_bench.runtime.task_view import AgentLaunchInput, agent_task_view_identity
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_str,
)
from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    WorkspaceError,
    WorkspacePolicy,
    build_patch_artifact,
)


V1_ADAPTER_IDS = (
    "scripted_canonical",
    "codex_canonical",
    "codex_mcp_canonical",
)


class _InfrastructureNotEvaluatedBackend:
    def evaluate(self, spec, frozen_patch):
        del spec, frozen_patch
        raise ContractError("infrastructure-invalid session cannot be evaluated")


@dataclass(frozen=True)
class V06RunRequest:
    manifest: RunManifest
    selected_attempt_ids: tuple[str, ...]
    runtime_profile_registry: RuntimeProfileRegistry
    runtime_profile_id: str
    target_binding: RuntimeTargetBinding
    output_root: Path
    resume_policy: str
    adapter_id: str
    enable_external_canary: bool
    clock_ms: Callable[[], int]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RunManifest):
            raise ContractError("manifest: expected RunManifest")
        if not isinstance(self.selected_attempt_ids, tuple) or not self.selected_attempt_ids:
            raise ContractError("selected_attempt_ids: expected non-empty tuple")
        expected_order = tuple(item.attempt_id for item in self.manifest.expected_attempts)
        if self.selected_attempt_ids != expected_order:
            raise ContractError(
                "selected_attempt_ids: must match the frozen Manifest matrix"
            )
        if not isinstance(self.runtime_profile_registry, RuntimeProfileRegistry):
            raise ContractError(
                "runtime_profile_registry: expected RuntimeProfileRegistry"
            )
        require_str(self.runtime_profile_id, "runtime_profile_id")
        if not isinstance(self.target_binding, RuntimeTargetBinding):
            raise ContractError("target_binding: expected RuntimeTargetBinding")
        if not isinstance(self.output_root, Path) or not self.output_root.is_absolute():
            raise ContractError("output_root: expected absolute Path")
        require_enum(
            self.resume_policy,
            "resume_policy",
            ("skip_valid", "retry_infrastructure", "never"),
        )
        require_enum(self.adapter_id, "adapter_id", V1_ADAPTER_IDS)
        require_bool(self.enable_external_canary, "enable_external_canary")
        if self.adapter_id in {
            "codex_canonical",
            "codex_mcp_canonical",
        } and not self.enable_external_canary:
            raise ContractError(
                f"{self.adapter_id} requires enable_external_canary"
            )
        if not callable(self.clock_ms):
            raise ContractError("clock_ms: expected callable")


@dataclass(frozen=True)
class V06RunResult:
    ran_attempt_ids: tuple[str, ...]
    skipped_attempt_ids: tuple[str, ...]
    blocked_attempt_ids: tuple[str, ...]
    cleanup_reports: Mapping[str, RuntimeCleanupReport]
    integrity: IntegrityReport

    def __post_init__(self) -> None:
        for name, values in (
            ("ran_attempt_ids", self.ran_attempt_ids),
            ("skipped_attempt_ids", self.skipped_attempt_ids),
            ("blocked_attempt_ids", self.blocked_attempt_ids),
        ):
            if not isinstance(values, tuple):
                raise ContractError(f"{name}: expected tuple")
            for index, value in enumerate(values):
                require_str(value, f"{name}[{index}]")
        if not isinstance(self.cleanup_reports, Mapping):
            raise ContractError("cleanup_reports: expected mapping")
        if not isinstance(self.integrity, IntegrityReport):
            raise ContractError("integrity: expected IntegrityReport")


class V06Orchestrator:
    def __init__(
        self,
        *,
        source_resolver: object,
        hidden_asset_resolver: object,
        backend_factory: object,
        adapter_factory: object,
        python_executable: str,
        source_overlay_resolver: object | None = None,
    ) -> None:
        for value, path in (
            (source_resolver, "source_resolver"),
            (hidden_asset_resolver, "hidden_asset_resolver"),
            (backend_factory, "backend_factory"),
            (adapter_factory, "adapter_factory"),
        ):
            if not callable(value):
                raise ContractError(f"{path}: expected callable")
        self._source_resolver = source_resolver
        self._hidden_asset_resolver = hidden_asset_resolver
        if source_overlay_resolver is None:
            self._source_overlay_resolver = lambda task: task.patch_scope
        elif callable(source_overlay_resolver):
            self._source_overlay_resolver = source_overlay_resolver
        else:
            raise ContractError("source_overlay_resolver: expected callable")
        self._backend_factory = backend_factory
        self._adapter_factory = adapter_factory
        self._python_executable = require_str(
            python_executable,
            "python_executable",
        )

    def run(self, request: V06RunRequest) -> V06RunResult:
        if not isinstance(request, V06RunRequest):
            raise ContractError("request: expected V06RunRequest")
        profile = self._validate_request(request)
        request.output_root.mkdir(parents=True, exist_ok=True)
        store = AttemptArtifactStore(request.output_root, request.manifest)
        ledger = AttemptLedger(request.output_root / "attempts.jsonl")
        ran: list[str] = []
        skipped: list[str] = []
        blocked: list[str] = []
        cleanup_reports: dict[str, RuntimeCleanupReport] = {}
        try:
            store.write_run_manifest()
            for expected in request.manifest.expected_attempts:
                decision = ledger.decide(expected.attempt_id, request.resume_policy)
                if decision.action == "skip":
                    skipped.append(expected.attempt_id)
                    cleanup_reports[expected.attempt_id] = store.read_runtime_cleanup(
                        expected.attempt_id,
                        retry_index=decision.retry_index,
                    )
                    continue
                if decision.action == "blocked":
                    blocked.append(expected.attempt_id)
                    continue
                report = self._run_attempt(
                    request=request,
                    profile=profile,
                    expected=expected,
                    retry_index=decision.retry_index,
                    store=store,
                    ledger=ledger,
                )
                cleanup_reports[expected.attempt_id] = report
                ran.append(expected.attempt_id)

            selected = selected_attempts_from_ledger(ledger, request.manifest)
            write_rebuilt_outputs(store, selected)
        finally:
            ledger.close()
            store.close()

        integrity = verify_run_artifacts(request.output_root, request.manifest)
        if integrity.status == "passed":
            persist_integrity_reports(
                request.output_root,
                request.manifest,
                integrity,
            )
        return V06RunResult(
            ran_attempt_ids=tuple(ran),
            skipped_attempt_ids=tuple(skipped),
            blocked_attempt_ids=tuple(blocked),
            cleanup_reports=dict(cleanup_reports),
            integrity=integrity,
        )

    def _validate_request(self, request: V06RunRequest) -> RuntimeProfile:
        profile = request.runtime_profile_registry.get(request.runtime_profile_id)
        if request.target_binding.backend != profile.backend:
            raise ContractError("target_binding: backend does not match Runtime Profile")
        expected_ids = {item.attempt_id for item in request.manifest.expected_attempts}
        if set(request.selected_attempt_ids) != expected_ids:
            raise ContractError("selected_attempt_ids: incomplete Manifest matrix")
        for task in request.manifest.tasks:
            if task.runtime != profile:
                raise ContractError(
                    "runtime_profile_id: selected Task has a different Runtime Profile"
                )
        for agent in request.manifest.agents:
            if agent.adapter.identifier != request.adapter_id:
                raise ContractError("adapter_id: does not match Manifest Agent")
        return profile

    def _run_attempt(
        self,
        *,
        request: V06RunRequest,
        profile: RuntimeProfile,
        expected,
        retry_index: int,
        store: AttemptArtifactStore,
        ledger: AttemptLedger,
    ) -> RuntimeCleanupReport:
        task = next(item for item in request.manifest.tasks if item.task == expected.task)
        task_view = next(
            item for item in request.manifest.task_views if item.task == expected.task
        )
        agent = next(
            item for item in request.manifest.agents if item.agent == expected.agent
        )
        source = self._source_resolver(task)
        hidden_asset = self._hidden_asset_resolver(task)
        self._validate_private_inputs(task, source, hidden_asset)

        construction_cleanup = ExitStack()
        try:
            resource_ledger = AttemptResourceLedger(
                store.runtime_resources_path(
                    expected.attempt_id,
                    retry_index=retry_index,
                ),
                attempt_id=expected.attempt_id,
                retry_index=retry_index,
                runtime_profile_hash=profile.content_hash,
                clock_ms=request.clock_ms,
            )
            construction_cleanup.callback(resource_ledger.close)
            lease_store = RuntimeLeaseStore(
                store.private_runtime_resources_path(
                    expected.attempt_id,
                    retry_index=retry_index,
                ),
                attempt_id=expected.attempt_id,
                retry_index=retry_index,
                runtime_profile_hash=profile.content_hash,
            )
            construction_cleanup.callback(lease_store.close)
            attempt_context = RuntimeAttemptContext(
                attempt_id=expected.attempt_id,
                retry_index=retry_index,
                runtime_profile_hash=profile.content_hash,
                frozen_source_directory=source.repository,
                frozen_source_revision=source.revision,
                resource_ledger=resource_ledger,
                lease_store=lease_store,
                target_binding=request.target_binding,
            )
            session_id = _session_id(expected.attempt_id, retry_index)
            action_artifacts = PublicArtifactStore(
                store.action_artifacts_path(
                    expected.attempt_id,
                    retry_index=retry_index,
                )
            )
            construction_cleanup.callback(action_artifacts.close)
            journal = EventJournal(
                session_id,
                clock_ms=request.clock_ms,
                events_path=store.events_path(
                    expected.attempt_id,
                    retry_index=retry_index,
                ),
                artifact_store=action_artifacts,
            )
            construction_cleanup.callback(journal.close)
        except BaseException:
            construction_cleanup.close()
            raise
        attempt_resource_cleanup = construction_cleanup.pop_all()
        agent_backend: RuntimeBackend | None = None
        agent_lease = None
        workspace: AuthoritativeWorkspace | None = None
        cleanup_report: RuntimeCleanupReport | None = None
        interrupted: KeyboardInterrupt | None = None
        cleanup_uncertain: ProcessGroupCleanupError | None = None
        try:
            agent_backend = self._runtime_backend(
                profile,
                request.target_binding,
                "agent",
            )
            try:
                agent_lease = agent_backend.prepare(profile, attempt_context)
            except Exception as exc:
                from op_bench.runtime.backends import RuntimeBackendUnavailable

                if not isinstance(exc, RuntimeBackendUnavailable):
                    raise
                return self._complete_pre_session_failure(
                    request=request,
                    profile=profile,
                    expected=expected,
                    task=task,
                    task_view=task_view,
                    retry_index=retry_index,
                    terminal_reason="platform_error",
                    resource_ledger=resource_ledger,
                    journal=journal,
                    store=store,
                    ledger=ledger,
                )
            workspace_path = _workspace_path(agent_lease)
            try:
                workspace = AuthoritativeWorkspace.open(
                    workspace_path,
                    source=task.source,
                    policy=_workspace_policy(request, task),
                )
            except WorkspaceError:
                agent_backend.cleanup(agent_lease)
                return self._complete_pre_session_failure(
                    request=request,
                    profile=profile,
                    expected=expected,
                    task=task,
                    task_view=task_view,
                    retry_index=retry_index,
                    terminal_reason="workspace_error",
                    resource_ledger=resource_ledger,
                    journal=journal,
                    store=store,
                    ledger=ledger,
                )
            action_service = CanonicalActionService(
                session_id=session_id,
                workspace=workspace,
                capability_policy=request.manifest.capability_policy,
                budget_policy=request.manifest.budget_policy,
                command_backend=RuntimeCommandBackend(agent_backend, agent_lease),
                test_registry=_registered_tests(
                    request,
                    task,
                    self._python_executable,
                ),
                clock_ms=request.clock_ms,
                event_journal=journal,
            )
            started = request.clock_ms()
            session = AttemptSession(
                spec=_session_spec(
                    request,
                    expected,
                    agent,
                    workspace,
                    session_id,
                    retry_index,
                    started,
                ),
                action_service=action_service,
                journal=journal,
                freeze_patch=workspace.freeze,
                clock_ms=request.clock_ms,
            )
            session.prepare()
            session.mark_ready()
            session.start()
            session.mark_agent_launched()
            adapter = self._adapter_factory(agent, request.adapter_id)
            with AdapterActionChannel(
                lambda payload: session.execute_action(
                    ActionRequest.from_dict(payload)
                ).to_dict()
            ) as action_client:
                try:
                    adapter_result = adapter.run(
                        AdapterContext(
                            launch_input=AgentLaunchInput(
                                task_view=task_view,
                                task_view_identity=agent_task_view_identity(task_view),
                            ),
                            session_id=session_id,
                            action_client=action_client,
                        )
                    )
                except KeyboardInterrupt as exc:
                    interrupted = exc
                    adapter_result = None
                except ProcessGroupCleanupError as exc:
                    cleanup_uncertain = exc
                    adapter_result = None
                except Exception:  # noqa: BLE001 - private Adapter boundary.
                    adapter_result = None
            if adapter_result is None:
                session.request_stop(
                    "runtime_error" if cleanup_uncertain is not None else "provider_error"
                )
            else:
                adapter_trace = getattr(adapter_result, "adapter_trace", None)
                if adapter_trace is not None:
                    if request.adapter_id != "codex_mcp_canonical":
                        raise ContractError(
                            "adapter trace is only valid for codex_mcp_canonical"
                        )
                    store.write_adapter_trace(
                        expected.attempt_id,
                        adapter_trace,
                        retry_index=retry_index,
                    )
                _converge_adapter_result(session, adapter_result)
            session_result = session.finalize()
            if session_result.final_patch is None:
                frozen = None
                patch_artifact = None
            else:
                frozen = workspace.freeze()
                patch_artifact = build_patch_artifact(
                    frozen,
                    artifact_id=(
                        f"{expected.attempt_id}/{retry_directory_name(retry_index)}/final.patch"
                    ),
                )
            store.write_session_inputs(
                expected.attempt_id,
                task_view,
                session_result,
                frozen,
                patch_artifact,
                retry_index=retry_index,
            )

            workspace.close()
            workspace = None
            agent_cleanup = agent_backend.cleanup(agent_lease)
            cleanup_report = agent_cleanup.report
            if cleanup_uncertain is None:
                evaluation_backend = self._runtime_backend(
                    profile,
                    request.target_binding,
                    "evaluation",
                )
                evaluation_implementation = RuntimeFreshEvaluationBackend(
                    source=source,
                    hidden_asset=hidden_asset,
                    python_executable=self._python_executable,
                    runtime_backend=evaluation_backend,
                    runtime_profile=profile,
                    attempt_context=attempt_context,
                    source_overlay_paths=self._source_overlay_resolver(task),
                )
            else:
                evaluation_implementation = _InfrastructureNotEvaluatedBackend()
            evaluation_spec = EvaluationSpec(
                session_id=session_id,
                attempt_id=expected.attempt_id,
                task=task.task,
                source=task.source,
                frozen_patch=session_result.final_patch,
                hidden_test_asset=task.hidden_test_asset,
                public_tests=task.public_tests,
                fail_to_pass=task.fail_to_pass,
                pass_to_pass=task.pass_to_pass,
                runtime=profile,
                timeout_ms=profile.timeout_ms,
                evaluation=request.manifest.evaluation,
                scoring=request.manifest.scoring,
            )
            coordinator = AttemptEvaluationCoordinator(
                FreshEvaluator(
                    evaluation_implementation,
                    clock_ms=request.clock_ms,
                ),
                journal,
                store,
                retry_index=retry_index,
                clock_ms=request.clock_ms,
            )
            completed = coordinator.complete(
                session_result,
                evaluation_spec,
                frozen,
                patch_artifact,
            )
            cleanup_report = _cleanup_report_from_records(
                expected.attempt_id,
                retry_index,
                profile.content_hash,
                resource_ledger,
            )
            store.write_runtime_cleanup(
                expected.attempt_id,
                cleanup_report,
                retry_index=retry_index,
            )
            store.write_runtime_conformance(
                expected.attempt_id,
                {
                    "report_type": "runtime_conformance",
                    "schema_version": "v1",
                    "status": "not_applicable",
                    "entries": [],
                },
                retry_index=retry_index,
            )
            ledger.append(
                session_result=session_result,
                evaluation_result=completed.result,
                evaluation_spec_hash=completed.evaluation_spec_hash,
                retry_index=retry_index,
                recorded_at_ms=request.clock_ms(),
            )
            if interrupted is not None:
                raise interrupted
            if cleanup_uncertain is not None:
                raise cleanup_uncertain
            return cleanup_report
        finally:
            if workspace is not None:
                workspace.close()
            if agent_backend is not None and agent_lease is not None and cleanup_report is None:
                try:
                    cleanup_report = agent_backend.cleanup(agent_lease).report
                except Exception:
                    pass
            attempt_resource_cleanup.close()

    def _complete_pre_session_failure(
        self,
        *,
        request: V06RunRequest,
        profile: RuntimeProfile,
        expected,
        task,
        task_view,
        retry_index: int,
        terminal_reason: str,
        resource_ledger: AttemptResourceLedger,
        journal: EventJournal,
        store: AttemptArtifactStore,
        ledger: AttemptLedger,
    ) -> RuntimeCleanupReport:
        started_at_ms = request.clock_ms()
        session_result = SessionResult(
            session_id=journal.session_id,
            attempt_id=expected.attempt_id,
            terminal_reason=terminal_reason,
            final_patch=None,
            started_at_ms=started_at_ms,
            ended_at_ms=request.clock_ms(),
        )
        journal.append(
            "session_created",
            {"attempt_id": expected.attempt_id},
        )
        journal.append("patch_freeze_started", {})
        journal.append(
            "patch_freeze_failed",
            {"error_code": "workspace_error"},
        )
        journal.append(
            "session_terminal_emitted",
            {
                "attempt_id": expected.attempt_id,
                "terminal_reason": terminal_reason,
                "session_result_hash": session_result.content_hash,
                "final_patch": None,
                "session_validity": "infrastructure_invalid",
            },
        )
        store.write_session_inputs(
            expected.attempt_id,
            task_view,
            session_result,
            None,
            None,
            retry_index=retry_index,
        )
        evaluation_spec = EvaluationSpec(
            session_id=journal.session_id,
            attempt_id=expected.attempt_id,
            task=task.task,
            source=task.source,
            frozen_patch=None,
            hidden_test_asset=task.hidden_test_asset,
            public_tests=task.public_tests,
            fail_to_pass=task.fail_to_pass,
            pass_to_pass=task.pass_to_pass,
            runtime=profile,
            timeout_ms=profile.timeout_ms,
            evaluation=request.manifest.evaluation,
            scoring=request.manifest.scoring,
        )
        coordinator = AttemptEvaluationCoordinator(
            FreshEvaluator(_SkippedEvaluationBackend(), clock_ms=request.clock_ms),
            journal,
            store,
            retry_index=retry_index,
            clock_ms=request.clock_ms,
        )
        completed = coordinator.complete(
            session_result,
            evaluation_spec,
            None,
            None,
        )
        cleanup_report = _cleanup_report_from_records(
            expected.attempt_id,
            retry_index,
            profile.content_hash,
            resource_ledger,
        )
        store.write_runtime_cleanup(
            expected.attempt_id,
            cleanup_report,
            retry_index=retry_index,
        )
        store.write_runtime_conformance(
            expected.attempt_id,
            {
                "report_type": "runtime_conformance",
                "schema_version": "v1",
                "status": "not_applicable",
                "entries": [],
            },
            retry_index=retry_index,
        )
        ledger.append(
            session_result=session_result,
            evaluation_result=completed.result,
            evaluation_spec_hash=completed.evaluation_spec_hash,
            retry_index=retry_index,
            recorded_at_ms=request.clock_ms(),
        )
        return cleanup_report

    def _runtime_backend(
        self,
        profile: RuntimeProfile,
        target_binding: RuntimeTargetBinding,
        phase: str,
    ) -> RuntimeBackend:
        backend = self._backend_factory(profile, target_binding, phase)
        for method in ("prepare", "run", "collect", "cleanup"):
            if not callable(getattr(backend, method, None)):
                raise ContractError("backend_factory: returned invalid RuntimeBackend")
        return backend

    @staticmethod
    def _validate_private_inputs(task, source, hidden_asset) -> None:
        if not isinstance(source, LocalGitSource):
            raise ContractError("source_resolver: expected LocalGitSource")
        if source.identity != task.source:
            raise ContractError("source_resolver: Source identity mismatch")
        observed = git_archive_source_identity(
            source.repository,
            source.revision,
            source.identity.identifier,
        )
        if (
            source.identity.digest_kind == "content_sha256"
            and observed != source.identity
        ):
            raise ContractError("source_resolver: Source bytes mismatch")
        if source.identity.digest_kind not in {
            "content_sha256",
            "canonical_config",
        }:
            raise ContractError("source_resolver: unsupported Source identity kind")
        if not isinstance(hidden_asset, EvaluationOnlyTestAsset):
            raise ContractError(
                "hidden_asset_resolver: expected EvaluationOnlyTestAsset"
            )
        if hidden_asset.identity != task.hidden_test_asset:
            raise ContractError("hidden_asset_resolver: hidden test identity mismatch")


def _workspace_path(lease) -> Path:
    workspace = [
        handle for handle in lease.handles if handle.resource_type == "workspace"
    ]
    if len(workspace) != 1:
        raise ContractError(
            "Runtime lease does not expose one local authoritative workspace"
        )
    path = Path(workspace[0].raw_handle)
    if path.is_symlink() or not path.is_dir():
        raise ContractError("Runtime workspace handle is not a real directory")
    return path


def _workspace_policy(request: V06RunRequest, task) -> WorkspacePolicy:
    capability = request.manifest.capability_policy
    return WorkspacePolicy(
        policy_id=f"{capability.policy_id}:workspace-v1",
        writable_paths=capability.writable_paths,
        patch_paths=task.patch_scope,
        allowed_modes=(0o600, 0o644, 0o755),
        max_read_bytes=capability.max_read_bytes,
        max_write_bytes=capability.max_write_bytes,
        max_file_bytes=max(capability.max_read_bytes, capability.max_write_bytes),
        max_patch_bytes=capability.max_write_bytes,
        allow_binary=False,
    )


def _registered_tests(
    request: V06RunRequest,
    task,
    python_executable: str,
) -> dict[str, RegisteredTest]:
    selectors = {selector.selector_id: selector for selector in task.public_tests}
    result: dict[str, RegisteredTest] = {}
    for selector_id in request.manifest.capability_policy.registered_tests:
        try:
            selector = selectors[selector_id]
        except KeyError as exc:
            raise ContractError(
                f"registered_tests: unknown public selector {selector_id!r}"
            ) from exc
        try:
            template = tuple(shlex.split(selector.command_template))
        except ValueError as exc:
            raise ContractError("registered test command template is invalid") from exc
        command = tuple(
            python_executable
            if part == "{python}"
            else selector_id
            if part == "{test}"
            else part
            for part in template
        )
        if any("{" in part or "}" in part for part in command):
            raise ContractError("registered test command template has unknown field")
        result[selector_id] = RegisteredTest(
            selector_id=selector_id,
            command=command,
            cwd=".",
            timeout_ms=task.runtime.timeout_ms,
        )
    return result


def _session_spec(
    request: V06RunRequest,
    expected,
    agent: AgentSpec,
    workspace: AuthoritativeWorkspace,
    session_id: str,
    retry_index: int,
    started_at_ms: int,
):
    from op_bench.runtime.contracts import SessionSpec

    return SessionSpec(
        session_id=session_id,
        attempt_id=expected.attempt_id,
        workspace=workspace.identity,
        agent_task_view=expected.task_view,
        capability_policy=request.manifest.capability_policy,
        budget_policy=request.manifest.budget_policy,
        deadline_ms=started_at_ms + request.manifest.budget_policy.wall_clock_ms,
        adapter_config=agent.config,
        runtime=next(
            task.runtime
            for task in request.manifest.tasks
            if task.task == expected.task
        ),
        artifact_root_id=(
            f"attempts/{expected.attempt_id}/retries/{retry_directory_name(retry_index)}"
        ),
        resume_policy=request.resume_policy,
    )


def _converge_adapter_result(session: AttemptSession, result: object) -> None:
    terminal = getattr(result, "terminal_reason", None)
    if terminal == "agent_finished":
        if session.state == "running":
            session.request_stop("agent_exited")
        return
    if terminal == "agent_exited":
        exit_code = getattr(result, "exit_code", 0)
        session.mark_agent_exited(0 if exit_code is None else exit_code)
        return
    if terminal in {"timeout", "provider_error", "runtime_error", "platform_error"}:
        session.request_stop(terminal)
        return
    raise ContractError("Adapter returned an invalid terminal result")


def _session_id(attempt_id: str, retry_index: int) -> str:
    digest = canonical_sha256(
        {
            "identity_version": "session-v1",
            "attempt_id": attempt_id,
            "retry_index": retry_index,
        }
    )
    return "session:v1:" + digest.removeprefix("sha256:")


class _SkippedEvaluationBackend:
    def evaluate(self, spec, frozen_patch):
        raise AssertionError("infrastructure-invalid Session must not be evaluated")


def _cleanup_report_from_records(
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
    ledger: AttemptResourceLedger,
) -> RuntimeCleanupReport:
    final = {}
    for record in ledger.records:
        final[record.resource_id] = record
    if not final:
        raise ContractError(
            "Runtime failure did not record an Attempt-owned resource"
        )
    entries = tuple(
        RuntimeCleanupEntry(
            resource_id=resource_id,
            resource_type=record.resource_type,
            status=record.transition,
            error_code=(
                None
                if record.transition == "released"
                else "resource_create_failed"
                if record.transition == "create_failed"
                else "resource_cleanup_failed"
            ),
        )
        for resource_id, record in sorted(final.items())
    )
    return RuntimeCleanupReport(
        attempt_id=attempt_id,
        retry_index=retry_index,
        runtime_profile_hash=runtime_profile_hash,
        entries=entries,
        all_released=all(
            entry.status in {"released", "create_failed"}
            for entry in entries
        ),
    )


__all__ = [
    "V1_ADAPTER_IDS",
    "V06Orchestrator",
    "V06RunRequest",
    "V06RunResult",
]
