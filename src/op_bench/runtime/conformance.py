from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.source_materialization import _git_environment
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_int,
    require_str,
)


CONFORMANCE_IGNORED_FIELDS_V1 = (
    "duration_ms",
    "ended_at_ms",
    "machine_local_path",
    "private_target_hash",
    "raw_handle_hash",
    "started_at_ms",
    "wall_clock_ms",
)


@dataclass(frozen=True)
class ConformanceSnapshot:
    contract_version: str
    action_observations: tuple[dict[str, object], ...]
    budget_usage: dict[str, object]
    patch_identity: dict[str, object]
    session_terminal: dict[str, object]
    evaluation_outcome: str
    cleanup_status: dict[str, object]
    event_sequence: tuple[dict[str, object], ...]
    workspace_tree: tuple[dict[str, object], ...]
    finish_count: int

    def __post_init__(self) -> None:
        if self.contract_version != "v1":
            raise ContractError("contract_version: expected 'v1'")
        if not isinstance(self.action_observations, tuple):
            raise ContractError("action_observations: expected tuple")
        actions: list[str] = []
        names: list[str] = []
        frozen_actions: list[dict[str, object]] = []
        for index, item in enumerate(self.action_observations):
            if not isinstance(item, dict):
                raise ContractError(
                    f"action_observations[{index}]: expected object"
                )
            action_id = require_str(
                item.get("action_id"),
                f"action_observations[{index}].action_id",
            )
            action_name = require_str(
                item.get("action_name"),
                f"action_observations[{index}].action_name",
            )
            if not isinstance(item.get("observation"), dict):
                raise ContractError(
                    f"action_observations[{index}].observation: expected object"
                )
            if action_id in actions:
                raise ContractError(f"duplicate action_id {action_id!r}")
            actions.append(action_id)
            names.append(action_name)
            frozen_actions.append(_json_copy(item, f"action_observations[{index}]"))
        if names.count("session_finish") != 1 or not names or names[-1] != "session_finish":
            raise ContractError("action_observations: requires one final session_finish")
        object.__setattr__(self, "action_observations", tuple(frozen_actions))
        for value, path in (
            (self.budget_usage, "budget_usage"),
            (self.patch_identity, "patch_identity"),
            (self.session_terminal, "session_terminal"),
            (self.cleanup_status, "cleanup_status"),
        ):
            if not isinstance(value, dict):
                raise ContractError(f"{path}: expected object")
            object.__setattr__(self, path, _json_copy(value, path))
        for value, path in (
            (self.event_sequence, "event_sequence"),
            (self.workspace_tree, "workspace_tree"),
        ):
            if not isinstance(value, tuple):
                raise ContractError(f"{path}: expected tuple")
            frozen = []
            for index, item in enumerate(value):
                if not isinstance(item, dict):
                    raise ContractError(f"{path}[{index}]: expected object")
                frozen.append(_json_copy(item, f"{path}[{index}]"))
            object.__setattr__(self, path, tuple(frozen))
        require_int(self.finish_count, "finish_count", minimum=0)
        if self.finish_count != names.count("session_finish"):
            raise ContractError("finish_count: does not match action observations")
        tree_paths = [
            require_str(item.get("path"), f"workspace_tree[{index}].path")
            for index, item in enumerate(self.workspace_tree)
        ]
        if tree_paths != sorted(set(tree_paths)):
            raise ContractError("workspace_tree: expected sorted unique paths")
        require_str(self.evaluation_outcome, "evaluation_outcome")
        if self.session_terminal.get("status") != "terminal":
            raise ContractError("session_terminal: expected terminal status")
        require_str(
            self.session_terminal.get("terminal_reason"),
            "session_terminal.terminal_reason",
        )
        if self.cleanup_status.get("all_released") is not True:
            raise ContractError("cleanup_status: active resources or cleanup failure")
        entries = self.cleanup_status.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ContractError("cleanup_status.entries: expected non-empty list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ContractError(f"cleanup_status.entries[{index}]: expected object")
            if entry.get("status") not in {"released", "create_failed"}:
                raise ContractError("cleanup_status: active resources or cleanup failure")

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "action_observations": _json_copy(
                list(self.action_observations),
                "action_observations",
            ),
            "budget_usage": _json_copy(self.budget_usage, "budget_usage"),
            "patch_identity": _json_copy(self.patch_identity, "patch_identity"),
            "session_terminal": _json_copy(
                self.session_terminal,
                "session_terminal",
            ),
            "evaluation_outcome": self.evaluation_outcome,
            "cleanup_status": _json_copy(self.cleanup_status, "cleanup_status"),
            "event_sequence": _json_copy(
                list(self.event_sequence),
                "event_sequence",
            ),
            "workspace_tree": _json_copy(
                list(self.workspace_tree),
                "workspace_tree",
            ),
            "finish_count": self.finish_count,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "ConformanceSnapshot":
        data = require_exact_fields(
            value,
            "conformance_snapshot",
            (
                "contract_version",
                "action_observations",
                "budget_usage",
                "patch_identity",
                "session_terminal",
                "evaluation_outcome",
                "cleanup_status",
                "event_sequence",
                "workspace_tree",
                "finish_count",
            ),
        )
        actions = data["action_observations"]
        if not isinstance(actions, list):
            raise ContractError("action_observations: expected list")
        events = data["event_sequence"]
        tree = data["workspace_tree"]
        if not isinstance(events, list):
            raise ContractError("event_sequence: expected list")
        if not isinstance(tree, list):
            raise ContractError("workspace_tree: expected list")
        return cls(
            contract_version=require_str(data["contract_version"], "contract_version"),
            action_observations=tuple(actions),
            budget_usage=data["budget_usage"],
            patch_identity=data["patch_identity"],
            session_terminal=data["session_terminal"],
            evaluation_outcome=require_str(
                data["evaluation_outcome"],
                "evaluation_outcome",
            ),
            cleanup_status=data["cleanup_status"],
            event_sequence=tuple(events),
            workspace_tree=tuple(tree),
            finish_count=require_int(data["finish_count"], "finish_count", minimum=0),
        )


@dataclass(frozen=True)
class ConformanceComparison:
    left_snapshot_hash: str
    right_snapshot_hash: str
    left_normalized_hash: str
    right_normalized_hash: str
    equal: bool
    differences: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "left_snapshot_hash": self.left_snapshot_hash,
            "right_snapshot_hash": self.right_snapshot_hash,
            "left_normalized_hash": self.left_normalized_hash,
            "right_normalized_hash": self.right_normalized_hash,
            "equal": self.equal,
            "differences": list(self.differences),
        }


def normalize_conformance_snapshot(
    snapshot: ConformanceSnapshot,
) -> dict[str, object]:
    if not isinstance(snapshot, ConformanceSnapshot):
        raise ContractError("snapshot: expected ConformanceSnapshot")
    payload = snapshot.to_dict()
    for index, item in enumerate(payload["action_observations"], start=1):
        canonical_action_id = f"action-{index}"
        item["action_id"] = canonical_action_id
        observation = item.get("observation")
        if isinstance(observation, dict) and "action_id" in observation:
            observation["action_id"] = canonical_action_id
    return _strip_ignored(payload)


def compare_conformance_snapshots(
    left: ConformanceSnapshot,
    right: ConformanceSnapshot,
) -> ConformanceComparison:
    if not isinstance(left, ConformanceSnapshot) or not isinstance(
        right,
        ConformanceSnapshot,
    ):
        raise ContractError("snapshots: expected ConformanceSnapshot values")
    left_normalized = normalize_conformance_snapshot(left)
    right_normalized = normalize_conformance_snapshot(right)
    differences: list[str] = []
    _difference_paths(left_normalized, right_normalized, "$", differences)
    ordered = tuple(sorted(set(differences)))
    return ConformanceComparison(
        left_snapshot_hash=left.content_hash,
        right_snapshot_hash=right.content_hash,
        left_normalized_hash=canonical_sha256(left_normalized),
        right_normalized_hash=canonical_sha256(right_normalized),
        equal=not ordered,
        differences=ordered,
    )


def _strip_ignored(value):
    if isinstance(value, dict):
        return {
            key: _strip_ignored(item)
            for key, item in sorted(value.items())
            if key not in CONFORMANCE_IGNORED_FIELDS_V1
        }
    if isinstance(value, list):
        return [_strip_ignored(item) for item in value]
    return value


def _difference_paths(left, right, path: str, result: list[str]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}"
            if key not in left or key not in right:
                result.append(child)
            else:
                _difference_paths(left[key], right[key], child, result)
        return
    if isinstance(left, list) and isinstance(right, list):
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                result.append(child)
            else:
                _difference_paths(left[index], right[index], child, result)
        return
    if left != right:
        result.append(path)


def _json_copy(value, path: str):
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{path}: expected JSON value") from exc


@dataclass(frozen=True)
class ConformanceEntry:
    entry_id: str
    transport: str
    backend_semantics: str
    status: str
    normalized_snapshot_hash: str | None
    differences: tuple[str, ...]
    reason_code: str | None = None
    runtime_profile_id: str | None = None
    runtime_profile_hash: str | None = None
    transport_counters: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "entry_id": self.entry_id,
            "transport": self.transport,
            "backend_semantics": self.backend_semantics,
            "status": self.status,
            "normalized_snapshot_hash": self.normalized_snapshot_hash,
            "differences": list(self.differences),
            "reason_code": self.reason_code,
            "runtime_profile_id": self.runtime_profile_id,
            "runtime_profile_hash": self.runtime_profile_hash,
            "transport_counters": _json_copy(
                self.transport_counters or {},
                "transport_counters",
            ),
        }


@dataclass(frozen=True)
class ConformanceExecution:
    snapshot: ConformanceSnapshot
    transport_counters: dict[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ConformanceSnapshot):
            raise ContractError("snapshot: expected ConformanceSnapshot")
        if not isinstance(self.transport_counters, dict):
            raise ContractError("transport_counters: expected object")
        object.__setattr__(
            self,
            "transport_counters",
            _json_copy(self.transport_counters, "transport_counters"),
        )


@dataclass(frozen=True)
class ConformanceRunReport:
    status: str
    entries: tuple[ConformanceEntry, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "report_type": "runtime_conformance",
            "schema_version": "v1",
            "status": self.status,
            "entries": [entry.to_dict() for entry in self.entries],
        }


class RuntimeConformanceRunner:
    def __init__(
        self,
        *,
        fixture_source: Path,
        runtime_profile,
        semantic_override: dict[str, str] | None = None,
        external_backend_factory=None,
    ) -> None:
        from op_bench.runtime.contracts import RuntimeProfile

        if not isinstance(fixture_source, Path):
            raise ContractError("fixture_source: expected Path")
        if (
            fixture_source.is_symlink()
            or not fixture_source.is_dir()
            or not (fixture_source / ".git").is_dir()
        ):
            raise ContractError("fixture_source: expected real Git repository")
        if not isinstance(runtime_profile, RuntimeProfile):
            raise ContractError("runtime_profile: expected RuntimeProfile")
        if runtime_profile.backend != "local":
            raise ContractError("runtime_profile: deterministic matrix requires local")
        self.fixture_source = fixture_source
        self.runtime_profile = runtime_profile
        self.semantic_override = dict(semantic_override or {})
        if external_backend_factory is not None and not callable(
            external_backend_factory
        ):
            raise ContractError("external_backend_factory: expected callable")
        self.external_backend_factory = external_backend_factory

    def run(
        self,
        output_dir: Path,
        *,
        include_external: bool = False,
        target_config: Path | None = None,
        external_profile=None,
        transport: str | None = None,
    ) -> ConformanceRunReport:
        if not isinstance(output_dir, Path):
            raise ContractError("output_dir: expected Path")
        output_dir.mkdir(parents=True, exist_ok=True)
        if transport is None or transport == "in-process":
            selected_transports = ("cli", "mcp")
        elif transport == "mcp-stdio":
            selected_transports = ("cli", "mcp-stdio")
        else:
            raise ContractError("transport: expected in-process or mcp-stdio")
        snapshots: list[tuple[str, str, str, ConformanceExecution]] = []
        for selected_transport in selected_transports:
            for semantics in ("local_process", "scripted_remote"):
                entry_id = f"{selected_transport}-{semantics}"
                execution = self._run_entry(
                    entry_id,
                    selected_transport,
                    semantics,
                )
                override = self.semantic_override.get(entry_id)
                if override is not None:
                    execution = ConformanceExecution(
                        snapshot=ConformanceSnapshot.from_dict(
                            {
                                **execution.snapshot.to_dict(),
                                "evaluation_outcome": override,
                            }
                        ),
                        transport_counters=execution.transport_counters,
                    )
                snapshots.append(
                    (entry_id, selected_transport, semantics, execution)
                )
        reference = snapshots[0][3].snapshot
        entries: list[ConformanceEntry] = []
        for entry_id, selected_transport, semantics, execution in snapshots:
            comparison = compare_conformance_snapshots(
                reference,
                execution.snapshot,
            )
            counter_differences = _transport_counter_differences(
                selected_transport,
                execution.transport_counters,
            )
            differences = tuple(
                sorted(set(comparison.differences) | set(counter_differences))
            )
            entries.append(
                ConformanceEntry(
                    entry_id=entry_id,
                    transport=selected_transport,
                    backend_semantics=semantics,
                    status="passed" if not differences else "failed",
                    normalized_snapshot_hash=comparison.right_normalized_hash,
                    differences=differences,
                    runtime_profile_id=self.runtime_profile.profile_id,
                    runtime_profile_hash=self.runtime_profile.content_hash,
                    transport_counters=execution.transport_counters,
                )
            )
        if include_external:
            entries.append(
                self._run_external_entry(
                    reference,
                    target_config=target_config,
                    external_profile=external_profile,
                )
            )
        status = "failed" if any(item.status == "failed" for item in entries) else (
            "blocked" if any(item.status == "blocked" for item in entries) else "passed"
        )
        report = ConformanceRunReport(status=status, entries=tuple(entries))
        raw = canonical_json(report.to_dict()) + "\n"
        (output_dir / "runtime_conformance.json").write_text(raw, encoding="utf-8")
        return report

    def _run_external_entry(
        self,
        reference: ConformanceSnapshot,
        *,
        target_config: Path | None,
        external_profile,
    ) -> ConformanceEntry:
        from op_bench.runtime.backends import (
            RuntimeBackendUnavailable,
            load_runtime_target_binding,
        )
        from op_bench.runtime.contracts import RuntimeProfile

        common = {
            "entry_id": "external-exact-target",
            "transport": "cli",
            "backend_semantics": "external",
        }
        if target_config is None:
            return ConformanceEntry(
                **common,
                status="blocked",
                normalized_snapshot_hash=None,
                differences=(),
                reason_code="target_config_missing",
            )
        if not isinstance(external_profile, RuntimeProfile):
            return ConformanceEntry(
                **common,
                status="blocked",
                normalized_snapshot_hash=None,
                differences=(),
                reason_code="external_profile_missing",
            )
        profile_fields = {
            "runtime_profile_id": external_profile.profile_id,
            "runtime_profile_hash": external_profile.content_hash,
        }
        try:
            with tempfile.TemporaryDirectory(
                prefix="opbench-conformance-external-"
            ) as temporary:
                local_parent = Path(temporary) / "workspaces"
                local_parent.mkdir()
                binding = load_runtime_target_binding(
                    Path(target_config).resolve(strict=True),
                    local_workspace_parent=local_parent,
                )
                if binding.backend != external_profile.backend:
                    raise ContractError(
                        "target_config: backend does not match external Profile"
                    )
                backend = (
                    self.external_backend_factory(external_profile, binding)
                    if self.external_backend_factory is not None
                    else _runtime_backend(external_profile)
                )
                execution = self._run_entry(
                    "external-exact-target",
                    "cli",
                    "external",
                    runtime_profile=external_profile,
                    target_binding=binding,
                    backend=backend,
                )
        except RuntimeBackendUnavailable as exc:
            return ConformanceEntry(
                **common,
                status="blocked",
                normalized_snapshot_hash=None,
                differences=(),
                reason_code=exc.reason_code,
                **profile_fields,
            )
        except (ContractError, OSError, ValueError):
            return ConformanceEntry(
                **common,
                status="failed",
                normalized_snapshot_hash=None,
                differences=(),
                reason_code="external_conformance_error",
                **profile_fields,
            )
        comparison = compare_conformance_snapshots(reference, execution.snapshot)
        return ConformanceEntry(
            **common,
            status="passed" if comparison.equal else "failed",
            normalized_snapshot_hash=comparison.right_normalized_hash,
            differences=comparison.differences,
            reason_code=None,
            transport_counters=execution.transport_counters,
            **profile_fields,
        )

    def _run_entry(
        self,
        entry_id: str,
        transport_name: str,
        semantics: str,
        *,
        runtime_profile=None,
        target_binding=None,
        backend=None,
    ) -> ConformanceExecution:
        from op_bench.runtime.adapters import AdapterActionChannel
        from op_bench.runtime.action_cli import ActionCliTransport
        from op_bench.runtime.actions import CanonicalActionService, RegisteredTest
        from op_bench.runtime.backends import (
            LocalProcessBackend,
            RuntimeAttemptContext,
            RuntimeCommandBackend,
            RuntimeTargetBinding,
            ScriptedRuntimeBackend,
        )
        from op_bench.runtime.contracts import (
            ActionRequest,
            BudgetPolicy,
            CapabilityPolicy,
        )
        from op_bench.runtime.events import EventJournal
        from op_bench.runtime.local_evaluation import git_archive_source_identity
        from op_bench.runtime.mcp import CanonicalMcpTransport
        from op_bench.runtime.resources import AttemptResourceLedger, RuntimeLeaseStore
        from op_bench.runtime.workspace import AuthoritativeWorkspace, WorkspacePolicy

        with tempfile.TemporaryDirectory(prefix="opbench-conformance-") as temporary:
            root = Path(temporary)
            selected_profile = runtime_profile or self.runtime_profile
            runtime_parent = root / "runtime-workspaces"
            runtime_parent.mkdir()
            evidence = root / "evidence"
            evidence.mkdir()
            attempt_id = "attempt:v1:" + canonical_sha256(
                {"entry_id": entry_id}
            ).removeprefix("sha256:")
            ledger = AttemptResourceLedger(
                evidence / "runtime_resources.jsonl",
                attempt_id=attempt_id,
                retry_index=1,
                runtime_profile_hash=selected_profile.content_hash,
                clock_ms=_CounterClock(),
            )
            lease_store = RuntimeLeaseStore(
                evidence / "private_runtime_resources.json",
                attempt_id=attempt_id,
                retry_index=1,
                runtime_profile_hash=selected_profile.content_hash,
            )
            context = RuntimeAttemptContext(
                attempt_id=attempt_id,
                retry_index=1,
                runtime_profile_hash=selected_profile.content_hash,
                frozen_source_directory=self.fixture_source,
                frozen_source_revision=_git_head(self.fixture_source),
                resource_ledger=ledger,
                lease_store=lease_store,
                target_binding=(
                    target_binding
                    if target_binding is not None
                    else RuntimeTargetBinding(
                        backend="local",
                        local_workspace_parent=runtime_parent,
                    )
                ),
            )
            selected_backend = backend
            if selected_backend is None and semantics == "local_process":
                selected_backend = LocalProcessBackend()
            elif selected_backend is None:
                selected_backend = ScriptedRuntimeBackend.success(
                    stdout="conformance-test\n",
                    exit_code=0,
                )
            lease = None
            workspace_authority = None
            event_journal = None
            cleanup = None
            try:
                lease = selected_backend.prepare(selected_profile, context)
                if semantics in {"local_process", "external"}:
                    workspace_handles = [
                        item
                        for item in lease.handles
                        if item.resource_type == "workspace"
                    ]
                    if len(workspace_handles) != 1:
                        raise ContractError(
                            "Runtime lease must expose one authoritative workspace"
                        )
                    action_workspace = Path(workspace_handles[0].raw_handle)
                    if action_workspace.is_symlink() or not action_workspace.is_dir():
                        raise ContractError(
                            "Runtime workspace handle is not a real directory"
                        )
                else:
                    action_workspace = root / "action-workspace"
                    shutil.copytree(
                        self.fixture_source,
                        action_workspace,
                        symlinks=False,
                    )
                command_backend = RuntimeCommandBackend(selected_backend, lease)
                source = git_archive_source_identity(
                    action_workspace,
                    "HEAD",
                    "conformance-fixture-source",
                )
                workspace_authority = AuthoritativeWorkspace.open(
                    action_workspace,
                    source=source,
                    policy=WorkspacePolicy(
                        policy_id="conformance-workspace-v1",
                        writable_paths=("src/",),
                        patch_paths=("src/",),
                        allowed_modes=(0o644, 0o755),
                        max_read_bytes=16_384,
                        max_write_bytes=16_384,
                        max_file_bytes=32_768,
                        max_patch_bytes=65_536,
                        allow_binary=False,
                    ),
                )
                event_journal = EventJournal(
                    "session-conformance-v1",
                    clock_ms=_CounterClock(),
                )
                service = CanonicalActionService(
                    session_id="session-conformance-v1",
                    workspace=workspace_authority,
                    capability_policy=CapabilityPolicy(
                        policy_id="conformance-capability-v1",
                        allowed_actions=(
                            "workspace_list",
                            "workspace_read",
                            "workspace_write",
                            "test_run",
                            "vcs_diff",
                            "session_finish",
                        ),
                        writable_paths=("src/",),
                        allowed_command_prefixes=(),
                        registered_tests=("conformance::smoke",),
                        max_read_bytes=16_384,
                        max_write_bytes=16_384,
                        max_output_bytes=65_536,
                        network_access="denied",
                    ),
                    budget_policy=BudgetPolicy(
                        policy_id="conformance-budget-v1",
                        wall_clock_ms=60_000,
                        max_actions=10,
                        max_tests=2,
                        max_commands=2,
                        max_output_bytes=131_072,
                        provider_token_limit=None,
                    ),
                    command_backend=command_backend,
                    test_registry={
                        "conformance::smoke": RegisteredTest(
                            selector_id="conformance::smoke",
                            command=(
                                (
                                    sys.executable
                                    if selected_profile.backend == "local"
                                    else "python"
                                ),
                                "-c",
                                (
                                    "import pathlib,sys;"
                                    "value=pathlib.Path('src/operator.py').read_text();"
                                    "sys.exit(3) if 'VALUE = 2' not in value else "
                                    "print('conformance-test')"
                                ),
                            ),
                            cwd=".",
                            timeout_ms=5_000,
                        )
                    },
                    clock_ms=lambda: 1_000,
                    event_journal=event_journal,
                )
                steps = (
                    ("workspace_list", {"path": ".", "recursive": False}),
                    ("workspace_read", {"path": "src/operator.py"}),
                    (
                        "workspace_write",
                        {"path": "src/operator.py", "content": "VALUE = 2\n"},
                    ),
                    ("test_run", {"selector_id": "conformance::smoke"}),
                    ("vcs_diff", {}),
                    ("session_finish", {}),
                )
                actions: list[dict[str, object]] = []
                transport_counters: dict[str, object] = {}
                if transport_name == "mcp-stdio":
                    with AdapterActionChannel(
                        lambda payload: service.execute(
                            ActionRequest.from_dict(payload)
                        ).to_dict()
                    ) as action_client:
                        actions, transport_counters = _run_mcp_stdio_steps(
                            root,
                            action_client=action_client,
                            steps=steps,
                        )
                else:
                    selected_transport = (
                        ActionCliTransport(service)
                        if transport_name == "cli"
                        else CanonicalMcpTransport(service)
                    )
                    for sequence, (action_name, arguments) in enumerate(
                        steps,
                        start=1,
                    ):
                        envelope = {
                            "session_id": "session-conformance-v1",
                            "action_id": f"action-{sequence}",
                            "client_sequence": sequence,
                            "deadline_ms": 10_000,
                            "arguments": arguments,
                        }
                        if transport_name == "cli":
                            observation = selected_transport.execute(
                                {
                                    "contract_type": "action_request",
                                    "schema_version": "v1",
                                    "action_name": action_name,
                                    **envelope,
                                }
                            )
                        else:
                            observation = selected_transport.call_tool(
                                action_name,
                                envelope,
                            )
                        actions.append(
                            {
                                "action_id": envelope["action_id"],
                                "action_name": action_name,
                                "observation": observation,
                            }
                        )
                frozen = workspace_authority.freeze()
                event_sequence = _semantic_event_sequence(
                    service.audit_exchanges,
                    event_journal.records,
                )
                workspace_tree = _workspace_tree(workspace_authority)
                workspace_authority.close()
                workspace_authority = None
                cleanup = selected_backend.cleanup(lease)
                usage = service.usage
                final_states = {}
                for record in ledger.records:
                    final_states[record.resource_id] = (
                        record.resource_type,
                        record.transition,
                    )
                return ConformanceExecution(
                    snapshot=ConformanceSnapshot(
                        contract_version="v1",
                        action_observations=tuple(actions),
                        budget_usage={
                            "actions": usage.actions,
                            "tests": usage.tests,
                            "commands": usage.commands,
                            "output_bytes": usage.output_bytes,
                            "wall_clock_ms": usage.wall_clock_ms,
                        },
                        patch_identity={
                            "patch": frozen.patch.to_dict(),
                            "size_bytes": len(frozen.patch_bytes),
                            "changed_paths": list(frozen.changed_paths),
                            "empty": frozen.empty,
                            "patch_bytes_sha256": (
                                "sha256:" + hashlib.sha256(frozen.patch_bytes).hexdigest()
                            ),
                        },
                        session_terminal={
                            "status": "terminal",
                            "terminal_reason": "agent_finished",
                            "started_at_ms": 100,
                            "ended_at_ms": 200,
                        },
                        evaluation_outcome="resolved",
                        cleanup_status={
                            "all_released": cleanup.report.all_released,
                            "entries": [
                                {"resource_type": kind, "status": status}
                                for kind, status in sorted(final_states.values())
                            ],
                        },
                        event_sequence=event_sequence,
                        workspace_tree=workspace_tree,
                        finish_count=sum(
                            1
                            for action in actions
                            if action["action_name"] == "session_finish"
                        ),
                    ),
                    transport_counters=transport_counters,
                )
            finally:
                try:
                    if workspace_authority is not None:
                        workspace_authority.close()
                finally:
                    try:
                        if lease is not None and cleanup is None:
                            selected_backend.cleanup(lease)
                    finally:
                        if event_journal is not None:
                            event_journal.close()
                        ledger.close()
                        lease_store.close()


def _transport_counter_differences(
    transport_name: str,
    counters: dict[str, object],
) -> tuple[str, ...]:
    if transport_name != "mcp-stdio":
        return () if counters == {} else ("$.transport_counters",)
    expected = {
        "initialize_count": 1,
        "tools_list_count": 1,
        "tools_call_count": 6,
        "protocol_error_count": 0,
        "server_terminal_status": "client_closed",
    }
    differences: list[str] = []
    _difference_paths(expected, counters, "$.transport_counters", differences)
    return tuple(sorted(set(differences)))


def _run_mcp_stdio_steps(
    root: Path,
    *,
    action_client,
    steps: tuple[tuple[str, dict[str, object]], ...],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    from op_bench.runtime.contracts import ActionObservation
    from op_bench.runtime.mcp import McpAdapterTrace, canonical_mcp_tools
    from op_bench.runtime.mcp_stdio import render_mcp_stdio_launcher
    from op_bench.runtime.process_actions import ProcessActionExchange
    from op_bench.runtime.process_group import run_process_group

    launcher = root / "opbench_mcp_server.py"
    trace_path = root / "mcp_trace.json"
    transport_token = "c" * 64
    exchange = ProcessActionExchange(
        action_client=action_client,
        session_id="session-conformance-v1",
        exchange_root=root / "mcp-action-exchange",
        timeout_ms=10_000,
        transport_token=transport_token,
    )
    exchange.start()
    try:
        launcher.write_text(
            render_mcp_stdio_launcher(canonical_mcp_tools()),
            encoding="utf-8",
        )
        launcher.chmod(0o700)
        messages: list[dict[str, object]] = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "opbench-conformance", "version": "v1"},
                },
            },
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        ]
        messages.extend(
            {
                "jsonrpc": "2.0",
                "id": index,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            for index, (name, arguments) in enumerate(steps, start=3)
        )
        token_descriptor, token_writer = os.pipe()
        try:
            os.write(token_writer, transport_token.encode("ascii"))
        finally:
            os.close(token_writer)
        try:
            result = run_process_group(
                (
                    sys.executable,
                    "-I",
                    str(launcher),
                    "--action-client",
                    str(exchange.client_path),
                    "--python-executable",
                    sys.executable,
                    "--trace-path",
                    str(trace_path),
                    "--model-id",
                    "conformance-model-v1",
                    "--codex-cli-version",
                    "conformance-cli-v1",
                    "--bridge-token-fd",
                    str(token_descriptor),
                ),
                cwd=root,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "LANG": "C.UTF-8",
                    "LC_ALL": "C.UTF-8",
                },
                timeout_ms=30_000,
                input_text="".join(
                    f"{canonical_json(message)}\n" for message in messages
                ),
                pass_fds=(token_descriptor,),
            )
        finally:
            os.close(token_descriptor)
    finally:
        exchange.close()
    if result.terminal_status != "completed" or result.returncode != 0:
        raise ContractError("mcp-stdio conformance subprocess failed")
    try:
        responses = [json.loads(line) for line in result.stdout.splitlines()]
    except ValueError:
        raise ContractError("mcp-stdio conformance returned invalid JSON") from None
    if len(responses) != len(steps) + 2 or any(
        not isinstance(response, dict) or "error" in response
        for response in responses
    ):
        raise ContractError("mcp-stdio conformance returned an RPC error")
    if responses[0].get("id") != 1 or responses[1].get("id") != 2:
        raise ContractError("mcp-stdio conformance response ordering changed")
    expected_tools = [tool.to_dict() for tool in canonical_mcp_tools()]
    if responses[1].get("result") != {"tools": expected_tools}:
        raise ContractError("mcp-stdio conformance tool registry changed")

    actions: list[dict[str, object]] = []
    for (action_name, _), response in zip(steps, responses[2:]):
        result_payload = response.get("result")
        if not isinstance(result_payload, dict):
            raise ContractError("mcp-stdio conformance tool result is missing")
        observation = ActionObservation.from_dict(
            result_payload.get("structuredContent")
        ).to_dict()
        if result_payload.get("isError") is not (not observation["ok"]):
            raise ContractError("mcp-stdio conformance error flag is inconsistent")
        content = result_payload.get("content")
        if (
            not isinstance(content, list)
            or len(content) != 1
            or not isinstance(content[0], dict)
            or content[0].get("text") != canonical_json(observation)
        ):
            raise ContractError("mcp-stdio conformance text result is inconsistent")
        actions.append(
            {
                "action_id": observation["action_id"],
                "action_name": action_name,
                "observation": observation,
            }
        )
    if exchange.server_failure is not None:
        raise ContractError("mcp-stdio conformance action exchange failed")
    if exchange.observation_count != len(steps) or exchange.finish_count != 1:
        raise ContractError("mcp-stdio conformance action counts changed")
    try:
        encoded_trace = trace_path.read_bytes()
        trace = McpAdapterTrace.from_dict(json.loads(encoded_trace))
    except (OSError, ValueError, ContractError):
        raise ContractError("mcp-stdio conformance trace is invalid") from None
    if encoded_trace != (canonical_json(trace.to_dict()) + "\n").encode("utf-8"):
        raise ContractError("mcp-stdio conformance trace is not canonical")
    return actions, {
        "initialize_count": trace.initialize_count,
        "tools_list_count": trace.tools_list_count,
        "tools_call_count": trace.tools_call_count,
        "protocol_error_count": trace.protocol_error_count,
        "server_terminal_status": trace.server_terminal_status,
    }


def _semantic_event_sequence(exchanges, records) -> tuple[dict[str, object], ...]:
    action_ids = [exchange.request.action_id for exchange in exchanges]
    if len(action_ids) != len(set(action_ids)):
        raise ContractError("conformance events contain duplicate Action IDs")
    indices = {action_id: index for index, action_id in enumerate(action_ids, start=1)}
    requested: dict[str, int] = {action_id: 0 for action_id in action_ids}
    observed: dict[str, int] = {action_id: 0 for action_id in action_ids}
    projected: list[dict[str, object]] = []
    selected_types = {
        "finish_requested",
        "action_requested",
        "test_started",
        "action_observed",
        "test_completed",
        "budget_updated",
    }
    for record in records:
        if record.event_type not in selected_types:
            continue
        payload = record.to_dict()["public_payload"]
        action_id = payload.get("action_id")
        if action_id not in indices:
            raise ContractError("conformance event is not paired to an Action")
        if record.event_type == "action_requested":
            requested[action_id] += 1
        elif record.event_type == "action_observed":
            observed[action_id] += 1
        item: dict[str, object] = {
            "event_type": record.event_type,
            "action_index": indices[action_id],
        }
        for key in (
            "action_name",
            "ok",
            "error_code",
            "budget_delta",
            "mutation_state",
        ):
            if key in payload:
                item[key] = payload[key]
        projected.append(item)
    if any(count != 1 for count in requested.values()) or any(
        count != 1 for count in observed.values()
    ):
        raise ContractError("conformance event request/observation pairing changed")
    return tuple(projected)


def _workspace_tree(workspace) -> tuple[dict[str, object], ...]:
    entries = workspace.list_entries(
        "src",
        recursive=True,
        max_entries=1_000,
        max_depth=32,
    )
    result: list[dict[str, object]] = []
    for entry in entries:
        if entry.entry_type != "file":
            continue
        read = workspace.read(entry.path)
        result.append(
            {
                "path": entry.path,
                "mode": read.mode,
                "size_bytes": len(read.content),
                "content_sha256": "sha256:" + hashlib.sha256(read.content).hexdigest(),
            }
        )
    return tuple(sorted(result, key=lambda item: item["path"]))


def initialize_builtin_conformance_fixture(root: Path) -> Path:
    if not isinstance(root, Path) or root.exists() or root.is_symlink():
        raise ContractError("built-in fixture root must be a new Path")
    root.mkdir(parents=True)
    (root / "src").mkdir()
    (root / "tests").mkdir()
    (root / "src" / "operator.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "src" / "helper.py").write_text(
        "def helper():\n    return 1\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_operator.py").write_text(
        "def test_operator():\n    assert True\n",
        encoding="utf-8",
    )
    git_env = {
        **_git_environment(),
        "GIT_AUTHOR_DATE": "2026-07-20T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-07-20T00:00:00Z",
    }
    prefix = (
        "git",
        "-c",
        "core.autocrlf=false",
        "-c",
        "user.name=OpBench Conformance",
        "-c",
        "user.email=opbench@example.invalid",
        "-C",
        str(root),
    )
    for arguments in (
        ("init", "--quiet", "--initial-branch=main"),
        ("add", "--all"),
        ("commit", "--quiet", "-m", "conformance fixture"),
    ):
        subprocess.run(
            (*prefix, *arguments),
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=git_env,
        )
    return root


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "--verify", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    return completed.stdout.decode("ascii").strip()


def _runtime_backend(profile):
    from op_bench.runtime.backends import (
        DockerRuntimeBackend,
        LocalProcessBackend,
        RemoteDockerRuntimeBackend,
    )

    backend_types = {
        "local": LocalProcessBackend,
        "docker": DockerRuntimeBackend,
        "remote_docker": RemoteDockerRuntimeBackend,
    }
    try:
        return backend_types[profile.backend]()
    except KeyError as exc:
        raise ContractError("Runtime Profile backend is not executable") from exc


class _CounterClock:
    def __init__(self) -> None:
        self.value = 1_000

    def __call__(self) -> int:
        value = self.value
        self.value += 1
        return value


__all__ = [
    "CONFORMANCE_IGNORED_FIELDS_V1",
    "ConformanceComparison",
    "ConformanceEntry",
    "ConformanceExecution",
    "ConformanceRunReport",
    "ConformanceSnapshot",
    "RuntimeConformanceRunner",
    "compare_conformance_snapshots",
    "initialize_builtin_conformance_fixture",
    "normalize_conformance_snapshot",
]
