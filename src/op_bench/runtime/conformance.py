from __future__ import annotations

from dataclasses import dataclass
import json
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
            ),
        )
        actions = data["action_observations"]
        if not isinstance(actions, list):
            raise ContractError("action_observations: expected list")
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
    return _strip_ignored(snapshot.to_dict())


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
        }


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
    ) -> ConformanceRunReport:
        if not isinstance(output_dir, Path):
            raise ContractError("output_dir: expected Path")
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshots: list[tuple[str, str, str, ConformanceSnapshot]] = []
        for transport in ("cli", "mcp"):
            for semantics in ("local_process", "scripted_remote"):
                entry_id = f"{transport}-{semantics}"
                snapshot = self._run_entry(entry_id, transport, semantics)
                override = self.semantic_override.get(entry_id)
                if override is not None:
                    snapshot = ConformanceSnapshot.from_dict(
                        {
                            **snapshot.to_dict(),
                            "evaluation_outcome": override,
                        }
                    )
                snapshots.append((entry_id, transport, semantics, snapshot))
        reference = snapshots[0][3]
        entries: list[ConformanceEntry] = []
        for entry_id, transport, semantics, snapshot in snapshots:
            comparison = compare_conformance_snapshots(reference, snapshot)
            entries.append(
                ConformanceEntry(
                    entry_id=entry_id,
                    transport=transport,
                    backend_semantics=semantics,
                    status="passed" if comparison.equal else "failed",
                    normalized_snapshot_hash=comparison.right_normalized_hash,
                    differences=comparison.differences,
                    runtime_profile_id=self.runtime_profile.profile_id,
                    runtime_profile_hash=self.runtime_profile.content_hash,
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
                snapshot = self._run_entry(
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
        comparison = compare_conformance_snapshots(reference, snapshot)
        return ConformanceEntry(
            **common,
            status="passed" if comparison.equal else "failed",
            normalized_snapshot_hash=comparison.right_normalized_hash,
            differences=comparison.differences,
            reason_code=None,
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
    ) -> ConformanceSnapshot:
        from op_bench.runtime.action_cli import ActionCliTransport
        from op_bench.runtime.actions import CanonicalActionService, RegisteredTest
        from op_bench.runtime.backends import (
            LocalProcessBackend,
            RuntimeAttemptContext,
            RuntimeCommandBackend,
            RuntimeTargetBinding,
            ScriptedRuntimeBackend,
        )
        from op_bench.runtime.contracts import BudgetPolicy, CapabilityPolicy
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
                )
                transport = (
                    ActionCliTransport(service)
                    if transport_name == "cli"
                    else CanonicalMcpTransport(service)
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
                for sequence, (action_name, arguments) in enumerate(steps, start=1):
                    envelope = {
                        "session_id": "session-conformance-v1",
                        "action_id": f"action-{sequence}",
                        "client_sequence": sequence,
                        "deadline_ms": 10_000,
                        "arguments": arguments,
                    }
                    if transport_name == "cli":
                        observation = transport.execute(
                            {
                                "contract_type": "action_request",
                                "schema_version": "v1",
                                "action_name": action_name,
                                **envelope,
                            }
                        )
                    else:
                        observation = transport.call_tool(action_name, envelope)
                    actions.append(
                        {
                            "action_id": envelope["action_id"],
                            "action_name": action_name,
                            "observation": observation,
                        }
                    )
                workspace_authority.close()
                workspace_authority = None
                cleanup = selected_backend.cleanup(lease)
                finish = actions[-1]["observation"]
                usage = service.usage
                final_states = {}
                for record in ledger.records:
                    final_states[record.resource_id] = (
                        record.resource_type,
                        record.transition,
                    )
                return ConformanceSnapshot(
                    contract_version="v1",
                    action_observations=tuple(actions),
                    budget_usage={
                        "actions": usage.actions,
                        "tests": usage.tests,
                        "commands": usage.commands,
                        "output_bytes": usage.output_bytes,
                        "wall_clock_ms": usage.wall_clock_ms,
                    },
                    patch_identity=finish["data"]["patch"],
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
                        ledger.close()
                        lease_store.close()


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
    "ConformanceRunReport",
    "ConformanceSnapshot",
    "RuntimeConformanceRunner",
    "compare_conformance_snapshots",
    "normalize_conformance_snapshot",
]
