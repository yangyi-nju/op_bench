from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any, ClassVar

from op_bench.runtime.canonical import JsonValue, canonical_json, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_bool,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_optional_int,
    require_str,
    require_str_tuple,
)


SCHEMA_VERSION = "v1"
SHA256_PATTERN = r"sha256:[0-9a-f]{64}"

IDENTITY_TYPES = (
    "dataset",
    "task",
    "source",
    "environment",
    "image",
    "hardware",
    "agent",
    "model",
    "adapter",
    "prompt",
    "policy",
    "runtime",
    "patch",
    "test",
    "admission",
    "attachment",
    "agent_config",
    "evaluation",
    "scoring",
    "workspace",
    "task_view",
)
DIGEST_KINDS = (
    "content_sha256",
    "canonical_config",
    "replay_spec_v1",
    "image_id",
    "declared",
)
ACTION_NAMES = (
    "workspace_list",
    "workspace_search",
    "workspace_read",
    "workspace_write",
    "workspace_apply_patch",
    "command_run",
    "test_run",
    "vcs_diff",
    "session_finish",
)
NETWORK_POLICIES = ("denied", "provider_only", "task_runtime")
RUNTIME_BACKENDS = ("local", "docker", "remote_docker")
RUNTIME_TIERS = (
    "local_fixture",
    "cpu_python_overlay",
    "cuda_python_overlay",
    "cuda_kernel_build",
)
SOURCE_LOADING_MODES = ("none", "python_overlay", "inplace_build")
MOUNT_SOURCE_ACCESS_MODES = ("authoritative_workspace", "remote_sync")
ARTIFACT_ACCESS_MODES = ("controller_only",)
ROOT_FILESYSTEM_MODES = ("host", "mutable_container", "read_only_container")
CLEANUP_SCOPES = ("attempt_owned_only",)
TEST_VISIBILITIES = ("public", "registered", "hidden", "evaluation_only")
FEEDBACK_POLICIES = ("visible", "none")
ERROR_CODES = (
    "ok",
    "invalid_request",
    "unsupported_action",
    "session_not_running",
    "capability_denied",
    "path_denied",
    "selector_denied",
    "budget_exhausted",
    "timeout",
    "conflict",
    "workspace_error",
    "runtime_error",
    "platform_error",
)
MUTATION_STATES = ("none", "unchanged", "mutated", "frozen")
EVENT_TYPES = (
    "session_created",
    "session_prepared",
    "session_started",
    "agent_launched",
    "agent_exited",
    "action_requested",
    "action_observed",
    "test_started",
    "test_completed",
    "budget_updated",
    "budget_exhausted",
    "finish_requested",
    "timeout_requested",
    "cancel_requested",
    "patch_freeze_started",
    "patch_freeze_completed",
    "patch_freeze_failed",
    "session_terminal_emitted",
    "evaluation_started",
    "evaluation_completed",
    "terminal_emitted",
)
RESUME_POLICIES = ("skip_valid", "retry_infrastructure", "never")
TERMINAL_REASONS = (
    "agent_finished",
    "agent_exited",
    "budget_exhausted",
    "timeout",
    "cancelled",
    "workspace_error",
    "runtime_error",
    "provider_error",
    "platform_error",
)
ATTEMPT_VALIDITIES = ("valid", "infrastructure_invalid")
AGENT_TERMINALS = (
    "finished",
    "exited",
    "timeout",
    "budget",
    "cancelled",
)
EVALUATION_OUTCOMES = (
    "resolved",
    "f2p_failed",
    "p2p_regression",
    "invalid_patch",
    "no_patch",
    "evaluation_error",
    "not_evaluated",
)
CHECK_STATUSES = ("passed", "failed")
INTEGRITY_STATUSES = ("passed", "failed")


class Contract:
    contract_type: ClassVar[str]
    schema_version: ClassVar[str] = SCHEMA_VERSION

    def to_dict(self) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
        }
        for item in fields(self):
            payload[item.name] = _wire_value(getattr(self, item.name))
        return payload

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict())


def _wire_value(value: object) -> JsonValue:
    if isinstance(value, Contract):
        return value.to_dict()
    if isinstance(value, tuple):
        return [_wire_value(item) for item in value]
    if isinstance(value, list):
        return [_wire_value(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _wire_value(item) for key, item in value.items()}
    return value


def _contract_data(
    value: object,
    *,
    contract_type: str,
    field_names: tuple[str, ...],
    path: str | None = None,
) -> dict[str, Any]:
    contract_path = path or contract_type
    data = require_exact_fields(
        value,
        contract_path,
        ("contract_type", "schema_version", *field_names),
    )
    actual_type = require_str(data["contract_type"], f"{contract_path}.contract_type")
    if actual_type != contract_type:
        raise ContractError(
            f"{contract_path}.contract_type: expected {contract_type!r}, got {actual_type!r}"
        )
    actual_version = require_str(data["schema_version"], f"{contract_path}.schema_version")
    if actual_version != SCHEMA_VERSION:
        raise ContractError(
            f"{contract_path}.schema_version: expected {SCHEMA_VERSION!r}, got {actual_version!r}"
        )
    return data


def _require_instance(value: object, cls: type, path: str) -> None:
    if not isinstance(value, cls):
        raise ContractError(f"{path}: expected {cls.__name__}")


def _require_identity_type(value: "ContentIdentity", expected: str, path: str) -> None:
    if value.identity_type != expected:
        raise ContractError(f"{path}: expected identity_type {expected!r}")


def _parse_contract_tuple(value: object, cls: type, path: str) -> tuple[Any, ...]:
    items = require_list(value, path)
    return tuple(cls.from_dict(item, path=f"{path}[{index}]") for index, item in enumerate(items))


@dataclass(frozen=True)
class ContentIdentity(Contract):
    contract_type: ClassVar[str] = "content_identity"

    identity_type: str
    identifier: str
    digest: str
    digest_kind: str

    def __post_init__(self) -> None:
        require_enum(self.identity_type, "identity_type", IDENTITY_TYPES)
        require_str(self.identifier, "identifier")
        require_str(self.digest, "digest", pattern=SHA256_PATTERN)
        require_enum(self.digest_kind, "digest_kind", DIGEST_KINDS)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "ContentIdentity":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=("identity_type", "identifier", "digest", "digest_kind"),
            path=path,
        )
        return cls(
            identity_type=require_enum(data["identity_type"], "identity_type", IDENTITY_TYPES),
            identifier=require_str(data["identifier"], "identifier"),
            digest=require_str(data["digest"], "digest", pattern=SHA256_PATTERN),
            digest_kind=require_enum(data["digest_kind"], "digest_kind", DIGEST_KINDS),
        )


@dataclass(frozen=True)
class CapabilityPolicy(Contract):
    contract_type: ClassVar[str] = "capability_policy"

    policy_id: str
    allowed_actions: tuple[str, ...]
    writable_paths: tuple[str, ...]
    allowed_command_prefixes: tuple[str, ...]
    registered_tests: tuple[str, ...]
    max_read_bytes: int
    max_write_bytes: int
    max_output_bytes: int
    network_access: str

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        _validate_str_values(self.allowed_actions, "allowed_actions", allowed=ACTION_NAMES)
        _validate_str_values(self.writable_paths, "writable_paths")
        _validate_str_values(self.allowed_command_prefixes, "allowed_command_prefixes")
        _validate_str_values(self.registered_tests, "registered_tests")
        require_int(self.max_read_bytes, "max_read_bytes", minimum=1)
        require_int(self.max_write_bytes, "max_write_bytes", minimum=1)
        require_int(self.max_output_bytes, "max_output_bytes", minimum=1)
        require_enum(self.network_access, "network_access", NETWORK_POLICIES)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "CapabilityPolicy":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=(
                "policy_id",
                "allowed_actions",
                "writable_paths",
                "allowed_command_prefixes",
                "registered_tests",
                "max_read_bytes",
                "max_write_bytes",
                "max_output_bytes",
                "network_access",
            ),
            path=path,
        )
        return cls(
            policy_id=require_str(data["policy_id"], "policy_id"),
            allowed_actions=require_str_tuple(data["allowed_actions"], "allowed_actions", allowed=ACTION_NAMES),
            writable_paths=require_str_tuple(data["writable_paths"], "writable_paths"),
            allowed_command_prefixes=require_str_tuple(
                data["allowed_command_prefixes"], "allowed_command_prefixes"
            ),
            registered_tests=require_str_tuple(data["registered_tests"], "registered_tests"),
            max_read_bytes=require_int(data["max_read_bytes"], "max_read_bytes", minimum=1),
            max_write_bytes=require_int(data["max_write_bytes"], "max_write_bytes", minimum=1),
            max_output_bytes=require_int(data["max_output_bytes"], "max_output_bytes", minimum=1),
            network_access=require_enum(data["network_access"], "network_access", NETWORK_POLICIES),
        )


@dataclass(frozen=True)
class BudgetPolicy(Contract):
    contract_type: ClassVar[str] = "budget_policy"

    policy_id: str
    wall_clock_ms: int
    max_actions: int
    max_tests: int
    max_commands: int
    max_output_bytes: int
    provider_token_limit: int | None

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        require_int(self.wall_clock_ms, "wall_clock_ms", minimum=1)
        require_int(self.max_actions, "max_actions", minimum=0)
        require_int(self.max_tests, "max_tests", minimum=0)
        require_int(self.max_commands, "max_commands", minimum=0)
        require_int(self.max_output_bytes, "max_output_bytes", minimum=0)
        require_optional_int(self.provider_token_limit, "provider_token_limit", minimum=1)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "BudgetPolicy":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=(
                "policy_id",
                "wall_clock_ms",
                "max_actions",
                "max_tests",
                "max_commands",
                "max_output_bytes",
                "provider_token_limit",
            ),
            path=path,
        )
        return cls(
            policy_id=require_str(data["policy_id"], "policy_id"),
            wall_clock_ms=require_int(data["wall_clock_ms"], "wall_clock_ms", minimum=1),
            max_actions=require_int(data["max_actions"], "max_actions", minimum=0),
            max_tests=require_int(data["max_tests"], "max_tests", minimum=0),
            max_commands=require_int(data["max_commands"], "max_commands", minimum=0),
            max_output_bytes=require_int(data["max_output_bytes"], "max_output_bytes", minimum=0),
            provider_token_limit=require_optional_int(
                data["provider_token_limit"], "provider_token_limit", minimum=1
            ),
        )


@dataclass(frozen=True)
class MountPolicy(Contract):
    contract_type: ClassVar[str] = "mount_policy"

    policy_id: str
    workspace_target: str
    source_access: str
    artifact_access: str
    root_filesystem: str

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        _validate_logical_workspace_target(self.workspace_target)
        require_enum(
            self.source_access,
            "source_access",
            MOUNT_SOURCE_ACCESS_MODES,
        )
        require_enum(
            self.artifact_access,
            "artifact_access",
            ARTIFACT_ACCESS_MODES,
        )
        require_enum(
            self.root_filesystem,
            "root_filesystem",
            ROOT_FILESYSTEM_MODES,
        )

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "MountPolicy":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            policy_id=require_str(data["policy_id"], "policy_id"),
            workspace_target=require_str(data["workspace_target"], "workspace_target"),
            source_access=require_enum(
                data["source_access"],
                "source_access",
                MOUNT_SOURCE_ACCESS_MODES,
            ),
            artifact_access=require_enum(
                data["artifact_access"],
                "artifact_access",
                ARTIFACT_ACCESS_MODES,
            ),
            root_filesystem=require_enum(
                data["root_filesystem"],
                "root_filesystem",
                ROOT_FILESYSTEM_MODES,
            ),
        )


@dataclass(frozen=True)
class ResourcePolicy(Contract):
    contract_type: ClassVar[str] = "resource_policy"

    policy_id: str
    cpu_millis: int | None
    memory_bytes: int | None
    pids_limit: int | None
    gpu_count: int

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        require_optional_int(self.cpu_millis, "cpu_millis", minimum=1)
        require_optional_int(self.memory_bytes, "memory_bytes", minimum=1)
        require_optional_int(self.pids_limit, "pids_limit", minimum=1)
        require_int(self.gpu_count, "gpu_count", minimum=0)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "ResourcePolicy":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            policy_id=require_str(data["policy_id"], "policy_id"),
            cpu_millis=require_optional_int(data["cpu_millis"], "cpu_millis", minimum=1),
            memory_bytes=require_optional_int(
                data["memory_bytes"],
                "memory_bytes",
                minimum=1,
            ),
            pids_limit=require_optional_int(data["pids_limit"], "pids_limit", minimum=1),
            gpu_count=require_int(data["gpu_count"], "gpu_count", minimum=0),
        )


@dataclass(frozen=True)
class CleanupPolicy(Contract):
    contract_type: ClassVar[str] = "cleanup_policy"

    policy_id: str
    scope: str
    grace_ms: int
    timeout_ms: int
    remove_workspace: bool
    remove_process: bool
    remove_container: bool

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        require_enum(self.scope, "scope", CLEANUP_SCOPES)
        require_int(self.grace_ms, "grace_ms", minimum=0)
        require_int(self.timeout_ms, "timeout_ms", minimum=1)
        require_bool(self.remove_workspace, "remove_workspace")
        require_bool(self.remove_process, "remove_process")
        require_bool(self.remove_container, "remove_container")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "CleanupPolicy":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            policy_id=require_str(data["policy_id"], "policy_id"),
            scope=require_enum(data["scope"], "scope", CLEANUP_SCOPES),
            grace_ms=require_int(data["grace_ms"], "grace_ms", minimum=0),
            timeout_ms=require_int(data["timeout_ms"], "timeout_ms", minimum=1),
            remove_workspace=require_bool(data["remove_workspace"], "remove_workspace"),
            remove_process=require_bool(data["remove_process"], "remove_process"),
            remove_container=require_bool(data["remove_container"], "remove_container"),
        )


@dataclass(frozen=True)
class RuntimeProfile(Contract):
    contract_type: ClassVar[str] = "runtime_profile"

    profile_id: str
    backend: str
    runtime_tier: str
    source_loading_mode: str
    platform: str
    image: ContentIdentity
    hardware: ContentIdentity
    requires_gpu: bool
    network_policy: str
    timeout_ms: int
    mount_policy: MountPolicy
    resource_policy: ResourcePolicy
    cleanup_policy: CleanupPolicy

    def __post_init__(self) -> None:
        require_str(self.profile_id, "profile_id")
        require_enum(self.backend, "backend", RUNTIME_BACKENDS)
        require_enum(self.runtime_tier, "runtime_tier", RUNTIME_TIERS)
        require_enum(self.source_loading_mode, "source_loading_mode", SOURCE_LOADING_MODES)
        require_str(self.platform, "platform")
        _require_instance(self.image, ContentIdentity, "image")
        _require_identity_type(self.image, "image", "image")
        _require_instance(self.hardware, ContentIdentity, "hardware")
        _require_identity_type(self.hardware, "hardware", "hardware")
        require_bool(self.requires_gpu, "requires_gpu")
        require_enum(self.network_policy, "network_policy", NETWORK_POLICIES)
        require_int(self.timeout_ms, "timeout_ms", minimum=1)
        _require_instance(self.mount_policy, MountPolicy, "mount_policy")
        _require_instance(self.resource_policy, ResourcePolicy, "resource_policy")
        _require_instance(self.cleanup_policy, CleanupPolicy, "cleanup_policy")
        if self.requires_gpu and self.resource_policy.gpu_count < 1:
            raise ContractError("requires_gpu: requires gpu_count >= 1")
        if not self.requires_gpu and self.resource_policy.gpu_count != 0:
            raise ContractError("gpu_count: requires requires_gpu=true")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "RuntimeProfile":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=(
                "profile_id",
                "backend",
                "runtime_tier",
                "source_loading_mode",
                "platform",
                "image",
                "hardware",
                "requires_gpu",
                "network_policy",
                "timeout_ms",
                "mount_policy",
                "resource_policy",
                "cleanup_policy",
            ),
            path=path,
        )
        return cls(
            profile_id=require_str(data["profile_id"], "profile_id"),
            backend=require_enum(data["backend"], "backend", RUNTIME_BACKENDS),
            runtime_tier=require_enum(data["runtime_tier"], "runtime_tier", RUNTIME_TIERS),
            source_loading_mode=require_enum(
                data["source_loading_mode"], "source_loading_mode", SOURCE_LOADING_MODES
            ),
            platform=require_str(data["platform"], "platform"),
            image=ContentIdentity.from_dict(data["image"], path="runtime_profile.image"),
            hardware=ContentIdentity.from_dict(
                data["hardware"], path="runtime_profile.hardware"
            ),
            requires_gpu=require_bool(data["requires_gpu"], "requires_gpu"),
            network_policy=require_enum(data["network_policy"], "network_policy", NETWORK_POLICIES),
            timeout_ms=require_int(data["timeout_ms"], "timeout_ms", minimum=1),
            mount_policy=MountPolicy.from_dict(
                data["mount_policy"], path="runtime_profile.mount_policy"
            ),
            resource_policy=ResourcePolicy.from_dict(
                data["resource_policy"], path="runtime_profile.resource_policy"
            ),
            cleanup_policy=CleanupPolicy.from_dict(
                data["cleanup_policy"], path="runtime_profile.cleanup_policy"
            ),
        )


@dataclass(frozen=True)
class TestSelector(Contract):
    contract_type: ClassVar[str] = "test_selector"

    selector_id: str
    visibility: str
    command_template: str
    description: str

    def __post_init__(self) -> None:
        require_str(self.selector_id, "selector_id")
        require_enum(self.visibility, "visibility", TEST_VISIBILITIES)
        require_str(self.command_template, "command_template")
        require_str(self.description, "description")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "TestSelector":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=("selector_id", "visibility", "command_template", "description"),
            path=path,
        )
        return cls(
            selector_id=require_str(data["selector_id"], "selector_id"),
            visibility=require_enum(data["visibility"], "visibility", TEST_VISIBILITIES),
            command_template=require_str(data["command_template"], "command_template"),
            description=require_str(data["description"], "description"),
        )


@dataclass(frozen=True)
class FullTaskSpec(Contract):
    contract_type: ClassVar[str] = "full_task_spec"

    task: ContentIdentity
    source: ContentIdentity
    environment: ContentIdentity
    runtime: RuntimeProfile
    statement_title: str
    statement_body: str
    framework: str
    operator_name: str
    public_tests: tuple[TestSelector, ...]
    hidden_tests: tuple[TestSelector, ...]
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    patch_scope: tuple[str, ...]
    gold_patch: ContentIdentity
    hidden_test_asset: ContentIdentity
    admission: ContentIdentity

    def __post_init__(self) -> None:
        for value, cls, path, identity_type in (
            (self.task, ContentIdentity, "task", "task"),
            (self.source, ContentIdentity, "source", "source"),
            (self.environment, ContentIdentity, "environment", "environment"),
            (self.gold_patch, ContentIdentity, "gold_patch", "patch"),
            (self.hidden_test_asset, ContentIdentity, "hidden_test_asset", "test"),
            (self.admission, ContentIdentity, "admission", "admission"),
        ):
            _require_instance(value, cls, path)
            _require_identity_type(value, identity_type, path)
        _require_instance(self.runtime, RuntimeProfile, "runtime")
        require_str(self.statement_title, "statement_title")
        require_str(self.statement_body, "statement_body")
        require_str(self.framework, "framework")
        require_str(self.operator_name, "operator_name")
        _validate_contract_values(self.public_tests, TestSelector, "public_tests")
        _validate_contract_values(self.hidden_tests, TestSelector, "hidden_tests")
        for selector in self.public_tests:
            if selector.visibility not in {"public", "registered"}:
                raise ContractError("public_tests: selector is not public or registered")
        for selector in self.hidden_tests:
            if selector.visibility not in {"hidden", "evaluation_only"}:
                raise ContractError("hidden_tests: selector is not hidden or evaluation_only")
        selector_ids: set[str] = set()
        for selector in (*self.public_tests, *self.hidden_tests):
            if selector.selector_id in selector_ids:
                raise ContractError(
                    "duplicate selector_id across public_tests and hidden_tests: "
                    f"{selector.selector_id!r}"
                )
            selector_ids.add(selector.selector_id)
        _validate_str_values(self.fail_to_pass, "fail_to_pass", allow_empty=False)
        _validate_str_values(self.pass_to_pass, "pass_to_pass", allow_empty=False)
        _validate_disjoint_scoring_groups(self.fail_to_pass, self.pass_to_pass)
        _validate_str_values(self.patch_scope, "patch_scope", allow_empty=False)
        known_selectors = {item.selector_id for item in (*self.public_tests, *self.hidden_tests)}
        for path, selectors in (("fail_to_pass", self.fail_to_pass), ("pass_to_pass", self.pass_to_pass)):
            for selector in selectors:
                if selector not in known_selectors:
                    raise ContractError(f"{path}: unknown selector {selector!r}")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "FullTaskSpec":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            task=ContentIdentity.from_dict(data["task"], path="full_task_spec.task"),
            source=ContentIdentity.from_dict(data["source"], path="full_task_spec.source"),
            environment=ContentIdentity.from_dict(
                data["environment"], path="full_task_spec.environment"
            ),
            runtime=RuntimeProfile.from_dict(data["runtime"], path="runtime_profile"),
            statement_title=require_str(data["statement_title"], "statement_title"),
            statement_body=require_str(data["statement_body"], "statement_body"),
            framework=require_str(data["framework"], "framework"),
            operator_name=require_str(data["operator_name"], "operator_name"),
            public_tests=_parse_contract_tuple(data["public_tests"], TestSelector, "public_tests"),
            hidden_tests=_parse_contract_tuple(data["hidden_tests"], TestSelector, "hidden_tests"),
            fail_to_pass=require_str_tuple(data["fail_to_pass"], "fail_to_pass", allow_empty=False),
            pass_to_pass=require_str_tuple(data["pass_to_pass"], "pass_to_pass", allow_empty=False),
            patch_scope=require_str_tuple(data["patch_scope"], "patch_scope", allow_empty=False),
            gold_patch=ContentIdentity.from_dict(data["gold_patch"], path="full_task_spec.gold_patch"),
            hidden_test_asset=ContentIdentity.from_dict(
                data["hidden_test_asset"], path="full_task_spec.hidden_test_asset"
            ),
            admission=ContentIdentity.from_dict(data["admission"], path="full_task_spec.admission"),
        )


@dataclass(frozen=True)
class AgentTaskView(Contract):
    contract_type: ClassVar[str] = "agent_task_view"

    task: ContentIdentity
    statement_title: str
    statement_body: str
    framework: str
    operator_name: str
    runtime_hint: str
    public_tests: tuple[TestSelector, ...]
    capability_policy: CapabilityPolicy
    budget_policy: BudgetPolicy
    termination_notes: str
    attachments: tuple[ContentIdentity, ...]

    def __post_init__(self) -> None:
        _require_instance(self.task, ContentIdentity, "task")
        _require_identity_type(self.task, "task", "task")
        for path, value in (
            ("statement_title", self.statement_title),
            ("statement_body", self.statement_body),
            ("framework", self.framework),
            ("operator_name", self.operator_name),
            ("runtime_hint", self.runtime_hint),
            ("termination_notes", self.termination_notes),
        ):
            require_str(value, path)
        _validate_contract_values(self.public_tests, TestSelector, "public_tests")
        for selector in self.public_tests:
            if selector.visibility not in {"public", "registered"}:
                raise ContractError("public_tests: selector is not public or registered")
        _require_instance(self.capability_policy, CapabilityPolicy, "capability_policy")
        _require_instance(self.budget_policy, BudgetPolicy, "budget_policy")
        _validate_contract_values(self.attachments, ContentIdentity, "attachments")
        for attachment in self.attachments:
            _require_identity_type(attachment, "attachment", "attachments")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "AgentTaskView":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            task=ContentIdentity.from_dict(data["task"], path="agent_task_view.task"),
            statement_title=require_str(data["statement_title"], "statement_title"),
            statement_body=require_str(data["statement_body"], "statement_body"),
            framework=require_str(data["framework"], "framework"),
            operator_name=require_str(data["operator_name"], "operator_name"),
            runtime_hint=require_str(data["runtime_hint"], "runtime_hint"),
            public_tests=_parse_contract_tuple(data["public_tests"], TestSelector, "public_tests"),
            capability_policy=CapabilityPolicy.from_dict(
                data["capability_policy"], path="agent_task_view.capability_policy"
            ),
            budget_policy=BudgetPolicy.from_dict(
                data["budget_policy"], path="agent_task_view.budget_policy"
            ),
            termination_notes=require_str(data["termination_notes"], "termination_notes"),
            attachments=_parse_contract_tuple(data["attachments"], ContentIdentity, "attachments"),
        )


@dataclass(frozen=True)
class AgentSpec(Contract):
    contract_type: ClassVar[str] = "agent_spec"

    agent: ContentIdentity
    model: ContentIdentity
    adapter: ContentIdentity
    system_prompt: ContentIdentity
    task_prompt: ContentIdentity
    config: ContentIdentity
    feedback_policy: str

    def __post_init__(self) -> None:
        for value, path, identity_type in (
            (self.agent, "agent", "agent"),
            (self.model, "model", "model"),
            (self.adapter, "adapter", "adapter"),
            (self.system_prompt, "system_prompt", "prompt"),
            (self.task_prompt, "task_prompt", "prompt"),
            (self.config, "config", "agent_config"),
        ):
            _require_instance(value, ContentIdentity, path)
            _require_identity_type(value, identity_type, path)
        require_enum(self.feedback_policy, "feedback_policy", FEEDBACK_POLICIES)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "AgentSpec":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            agent=ContentIdentity.from_dict(data["agent"], path="agent_spec.agent"),
            model=ContentIdentity.from_dict(data["model"], path="agent_spec.model"),
            adapter=ContentIdentity.from_dict(data["adapter"], path="agent_spec.adapter"),
            system_prompt=ContentIdentity.from_dict(
                data["system_prompt"], path="agent_spec.system_prompt"
            ),
            task_prompt=ContentIdentity.from_dict(
                data["task_prompt"], path="agent_spec.task_prompt"
            ),
            config=ContentIdentity.from_dict(data["config"], path="agent_spec.config"),
            feedback_policy=require_enum(data["feedback_policy"], "feedback_policy", FEEDBACK_POLICIES),
        )


@dataclass(frozen=True)
class ActionRequest(Contract):
    contract_type: ClassVar[str] = "action_request"

    session_id: str
    action_id: str
    action_name: str
    arguments: dict[str, JsonValue]
    client_sequence: int
    deadline_ms: int

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.action_id, "action_id")
        require_enum(self.action_name, "action_name", ACTION_NAMES)
        object.__setattr__(self, "arguments", _freeze_json_object(self.arguments, "arguments"))
        require_int(self.client_sequence, "client_sequence", minimum=1)
        require_int(self.deadline_ms, "deadline_ms", minimum=1)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "ActionRequest":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            action_id=require_str(data["action_id"], "action_id"),
            action_name=require_enum(data["action_name"], "action_name", ACTION_NAMES),
            arguments=data["arguments"],
            client_sequence=require_int(data["client_sequence"], "client_sequence", minimum=1),
            deadline_ms=require_int(data["deadline_ms"], "deadline_ms", minimum=1),
        )


@dataclass(frozen=True)
class BudgetDelta(Contract):
    contract_type: ClassVar[str] = "budget_delta"

    wall_clock_ms: int
    actions: int
    tests: int
    commands: int
    output_bytes: int
    provider_tokens: int

    def __post_init__(self) -> None:
        for path, value in (
            ("wall_clock_ms", self.wall_clock_ms),
            ("actions", self.actions),
            ("tests", self.tests),
            ("commands", self.commands),
            ("output_bytes", self.output_bytes),
            ("provider_tokens", self.provider_tokens),
        ):
            require_int(value, path, minimum=0)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "BudgetDelta":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(**{
            item.name: require_int(data[item.name], item.name, minimum=0)
            for item in fields(cls)
        })


@dataclass(frozen=True)
class ActionObservation(Contract):
    contract_type: ClassVar[str] = "action_observation"

    session_id: str
    action_id: str
    ok: bool
    error_code: str
    message: str
    data: dict[str, JsonValue]
    started_at_ms: int
    ended_at_ms: int
    budget_delta: BudgetDelta
    mutation_state: str

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.action_id, "action_id")
        require_bool(self.ok, "ok")
        require_enum(self.error_code, "error_code", ERROR_CODES)
        if self.ok and self.error_code != "ok":
            raise ContractError("error_code: successful observation must use 'ok'")
        if not self.ok and self.error_code == "ok":
            raise ContractError("error_code: failed observation cannot use 'ok'")
        require_str(self.message, "message", min_length=0)
        object.__setattr__(self, "data", _freeze_json_object(self.data, "data"))
        require_int(self.started_at_ms, "started_at_ms", minimum=0)
        require_int(self.ended_at_ms, "ended_at_ms", minimum=0)
        if self.ended_at_ms < self.started_at_ms:
            raise ContractError("ended_at_ms: must be >= started_at_ms")
        _require_instance(self.budget_delta, BudgetDelta, "budget_delta")
        require_enum(self.mutation_state, "mutation_state", MUTATION_STATES)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "ActionObservation":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            action_id=require_str(data["action_id"], "action_id"),
            ok=require_bool(data["ok"], "ok"),
            error_code=require_enum(data["error_code"], "error_code", ERROR_CODES),
            message=require_str(data["message"], "message", min_length=0),
            data=data["data"],
            started_at_ms=require_int(data["started_at_ms"], "started_at_ms", minimum=0),
            ended_at_ms=require_int(data["ended_at_ms"], "ended_at_ms", minimum=0),
            budget_delta=BudgetDelta.from_dict(data["budget_delta"], path="action_observation.budget_delta"),
            mutation_state=require_enum(data["mutation_state"], "mutation_state", MUTATION_STATES),
        )


@dataclass(frozen=True)
class EventRecord(Contract):
    contract_type: ClassVar[str] = "event_record"

    session_id: str
    sequence: int
    occurred_at_ms: int
    event_type: str
    public_payload: dict[str, JsonValue]
    previous_event_hash: str | None
    event_hash: str

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_int(self.sequence, "sequence", minimum=1)
        require_int(self.occurred_at_ms, "occurred_at_ms", minimum=0)
        require_enum(self.event_type, "event_type", EVENT_TYPES)
        object.__setattr__(
            self,
            "public_payload",
            _freeze_json_object(self.public_payload, "public_payload"),
        )
        _require_optional_hash(self.previous_event_hash, "previous_event_hash")
        require_str(self.event_hash, "event_hash", pattern=SHA256_PATTERN)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "EventRecord":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            sequence=require_int(data["sequence"], "sequence", minimum=1),
            occurred_at_ms=require_int(data["occurred_at_ms"], "occurred_at_ms", minimum=0),
            event_type=require_enum(data["event_type"], "event_type", EVENT_TYPES),
            public_payload=data["public_payload"],
            previous_event_hash=_require_optional_hash(data["previous_event_hash"], "previous_event_hash"),
            event_hash=require_str(data["event_hash"], "event_hash", pattern=SHA256_PATTERN),
        )


@dataclass(frozen=True)
class SessionSpec(Contract):
    contract_type: ClassVar[str] = "session_spec"

    session_id: str
    attempt_id: str
    workspace: ContentIdentity
    agent_task_view: ContentIdentity
    capability_policy: CapabilityPolicy
    budget_policy: BudgetPolicy
    deadline_ms: int
    adapter_config: ContentIdentity
    runtime: RuntimeProfile
    artifact_root_id: str
    resume_policy: str

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.attempt_id, "attempt_id")
        for value, path, identity_type in (
            (self.workspace, "workspace", "workspace"),
            (self.agent_task_view, "agent_task_view", "task_view"),
            (self.adapter_config, "adapter_config", "agent_config"),
        ):
            _require_instance(value, ContentIdentity, path)
            _require_identity_type(value, identity_type, path)
        _require_instance(self.capability_policy, CapabilityPolicy, "capability_policy")
        _require_instance(self.budget_policy, BudgetPolicy, "budget_policy")
        require_int(self.deadline_ms, "deadline_ms", minimum=1)
        _require_instance(self.runtime, RuntimeProfile, "runtime")
        require_str(self.artifact_root_id, "artifact_root_id")
        require_enum(self.resume_policy, "resume_policy", RESUME_POLICIES)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "SessionSpec":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            attempt_id=require_str(data["attempt_id"], "attempt_id"),
            workspace=ContentIdentity.from_dict(data["workspace"], path="session_spec.workspace"),
            agent_task_view=ContentIdentity.from_dict(
                data["agent_task_view"], path="session_spec.agent_task_view"
            ),
            capability_policy=CapabilityPolicy.from_dict(
                data["capability_policy"], path="session_spec.capability_policy"
            ),
            budget_policy=BudgetPolicy.from_dict(
                data["budget_policy"], path="session_spec.budget_policy"
            ),
            deadline_ms=require_int(data["deadline_ms"], "deadline_ms", minimum=1),
            adapter_config=ContentIdentity.from_dict(
                data["adapter_config"], path="session_spec.adapter_config"
            ),
            runtime=RuntimeProfile.from_dict(data["runtime"], path="session_spec.runtime"),
            artifact_root_id=require_str(data["artifact_root_id"], "artifact_root_id"),
            resume_policy=require_enum(data["resume_policy"], "resume_policy", RESUME_POLICIES),
        )


@dataclass(frozen=True)
class EvaluationSpec(Contract):
    contract_type: ClassVar[str] = "evaluation_spec"

    session_id: str
    attempt_id: str
    task: ContentIdentity
    source: ContentIdentity
    frozen_patch: ContentIdentity | None
    hidden_test_asset: ContentIdentity
    public_tests: tuple[TestSelector, ...]
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]
    runtime: RuntimeProfile
    timeout_ms: int
    evaluation: ContentIdentity
    scoring: ContentIdentity

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.attempt_id, "attempt_id")
        for value, path, identity_type in (
            (self.task, "task", "task"),
            (self.source, "source", "source"),
            (self.hidden_test_asset, "hidden_test_asset", "test"),
            (self.evaluation, "evaluation", "evaluation"),
            (self.scoring, "scoring", "scoring"),
        ):
            _require_instance(value, ContentIdentity, path)
            _require_identity_type(value, identity_type, path)
        if self.frozen_patch is not None:
            _require_instance(self.frozen_patch, ContentIdentity, "frozen_patch")
            _require_identity_type(self.frozen_patch, "patch", "frozen_patch")
        _validate_contract_values(self.public_tests, TestSelector, "public_tests")
        selector_ids: set[str] = set()
        for selector in self.public_tests:
            if selector.selector_id in selector_ids:
                raise ContractError(
                    "public_tests: duplicate selector_id "
                    f"{selector.selector_id!r}"
                )
            selector_ids.add(selector.selector_id)
        _validate_str_values(self.fail_to_pass, "fail_to_pass", allow_empty=False)
        _validate_str_values(self.pass_to_pass, "pass_to_pass", allow_empty=False)
        _validate_disjoint_scoring_groups(self.fail_to_pass, self.pass_to_pass)
        _require_instance(self.runtime, RuntimeProfile, "runtime")
        require_int(self.timeout_ms, "timeout_ms", minimum=1)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "EvaluationSpec":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            attempt_id=require_str(data["attempt_id"], "attempt_id"),
            task=ContentIdentity.from_dict(data["task"], path="evaluation_spec.task"),
            source=ContentIdentity.from_dict(data["source"], path="evaluation_spec.source"),
            frozen_patch=(
                None
                if data["frozen_patch"] is None
                else ContentIdentity.from_dict(
                    data["frozen_patch"], path="evaluation_spec.frozen_patch"
                )
            ),
            hidden_test_asset=ContentIdentity.from_dict(
                data["hidden_test_asset"], path="evaluation_spec.hidden_test_asset"
            ),
            public_tests=_parse_contract_tuple(data["public_tests"], TestSelector, "public_tests"),
            fail_to_pass=require_str_tuple(data["fail_to_pass"], "fail_to_pass", allow_empty=False),
            pass_to_pass=require_str_tuple(data["pass_to_pass"], "pass_to_pass", allow_empty=False),
            runtime=RuntimeProfile.from_dict(data["runtime"], path="evaluation_spec.runtime"),
            timeout_ms=require_int(data["timeout_ms"], "timeout_ms", minimum=1),
            evaluation=ContentIdentity.from_dict(
                data["evaluation"], path="evaluation_spec.evaluation"
            ),
            scoring=ContentIdentity.from_dict(data["scoring"], path="evaluation_spec.scoring"),
        )


@dataclass(frozen=True)
class SessionResult(Contract):
    contract_type: ClassVar[str] = "session_result"

    session_id: str
    attempt_id: str
    terminal_reason: str
    final_patch: ContentIdentity | None
    started_at_ms: int
    ended_at_ms: int

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.attempt_id, "attempt_id")
        require_enum(self.terminal_reason, "terminal_reason", TERMINAL_REASONS)
        if self.final_patch is not None:
            _require_instance(self.final_patch, ContentIdentity, "final_patch")
            _require_identity_type(self.final_patch, "patch", "final_patch")
        _validate_time_range(self.started_at_ms, self.ended_at_ms)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "SessionResult":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        patch = (
            None
            if data["final_patch"] is None
            else ContentIdentity.from_dict(data["final_patch"], path="session_result.final_patch")
        )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            attempt_id=require_str(data["attempt_id"], "attempt_id"),
            terminal_reason=require_enum(data["terminal_reason"], "terminal_reason", TERMINAL_REASONS),
            final_patch=patch,
            started_at_ms=require_int(data["started_at_ms"], "started_at_ms", minimum=0),
            ended_at_ms=require_int(data["ended_at_ms"], "ended_at_ms", minimum=0),
        )


@dataclass(frozen=True)
class TestExecutionSummary(Contract):
    contract_type: ClassVar[str] = "test_execution_summary"

    collected: int
    executed: int
    passed: int
    failed: int
    skipped: int

    def __post_init__(self) -> None:
        for path, value in (
            ("collected", self.collected),
            ("executed", self.executed),
            ("passed", self.passed),
            ("failed", self.failed),
            ("skipped", self.skipped),
        ):
            require_int(value, path, minimum=0)
        if self.executed != self.passed + self.failed:
            raise ContractError("executed: must equal passed + failed")
        if self.collected != self.executed + self.skipped:
            raise ContractError("collected: must equal executed + skipped")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "TestExecutionSummary":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(**{
            item.name: require_int(data[item.name], item.name, minimum=0)
            for item in fields(cls)
        })


@dataclass(frozen=True)
class EvaluationResultV06(Contract):
    contract_type: ClassVar[str] = "evaluation_result"

    session_id: str
    attempt_id: str
    attempt_validity: str
    agent_terminal: str | None
    evaluation_outcome: str
    invalid_reason: str | None
    patch: ContentIdentity | None
    fail_to_pass: TestExecutionSummary
    pass_to_pass: TestExecutionSummary
    duration_ms: int
    evaluation: ContentIdentity
    scoring: ContentIdentity

    def __post_init__(self) -> None:
        require_str(self.session_id, "session_id")
        require_str(self.attempt_id, "attempt_id")
        require_enum(self.attempt_validity, "attempt_validity", ATTEMPT_VALIDITIES)
        if self.agent_terminal is not None:
            require_enum(self.agent_terminal, "agent_terminal", AGENT_TERMINALS)
        require_enum(self.evaluation_outcome, "evaluation_outcome", EVALUATION_OUTCOMES)
        if self.attempt_validity == "infrastructure_invalid":
            if self.invalid_reason is None:
                raise ContractError("invalid_reason: required for infrastructure_invalid")
            require_str(self.invalid_reason, "invalid_reason")
        elif self.invalid_reason is not None:
            raise ContractError("invalid_reason: must be null for valid attempt")
        if self.patch is not None:
            _require_instance(self.patch, ContentIdentity, "patch")
            _require_identity_type(self.patch, "patch", "patch")
        _require_instance(self.fail_to_pass, TestExecutionSummary, "fail_to_pass")
        _require_instance(self.pass_to_pass, TestExecutionSummary, "pass_to_pass")
        require_int(self.duration_ms, "duration_ms", minimum=0)
        for value, path, identity_type in (
            (self.evaluation, "evaluation", "evaluation"),
            (self.scoring, "scoring", "scoring"),
        ):
            _require_instance(value, ContentIdentity, path)
            _require_identity_type(value, identity_type, path)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "EvaluationResultV06":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        patch = (
            None
            if data["patch"] is None
            else ContentIdentity.from_dict(data["patch"], path="evaluation_result.patch")
        )
        invalid_reason = data["invalid_reason"]
        if invalid_reason is not None:
            invalid_reason = require_str(invalid_reason, "invalid_reason")
        agent_terminal = data["agent_terminal"]
        if agent_terminal is not None:
            agent_terminal = require_enum(
                agent_terminal, "agent_terminal", AGENT_TERMINALS
            )
        return cls(
            session_id=require_str(data["session_id"], "session_id"),
            attempt_id=require_str(data["attempt_id"], "attempt_id"),
            attempt_validity=require_enum(
                data["attempt_validity"], "attempt_validity", ATTEMPT_VALIDITIES
            ),
            agent_terminal=agent_terminal,
            evaluation_outcome=require_enum(
                data["evaluation_outcome"], "evaluation_outcome", EVALUATION_OUTCOMES
            ),
            invalid_reason=invalid_reason,
            patch=patch,
            fail_to_pass=TestExecutionSummary.from_dict(
                data["fail_to_pass"], path="evaluation_result.fail_to_pass"
            ),
            pass_to_pass=TestExecutionSummary.from_dict(
                data["pass_to_pass"], path="evaluation_result.pass_to_pass"
            ),
            duration_ms=require_int(data["duration_ms"], "duration_ms", minimum=0),
            evaluation=ContentIdentity.from_dict(
                data["evaluation"], path="evaluation_result.evaluation"
            ),
            scoring=ContentIdentity.from_dict(
                data["scoring"], path="evaluation_result.scoring"
            ),
        )


@dataclass(frozen=True)
class IntegrityCheck(Contract):
    contract_type: ClassVar[str] = "integrity_check"

    check_id: str
    status: str
    message: str
    expected_hash: str | None
    actual_hash: str | None

    def __post_init__(self) -> None:
        require_str(self.check_id, "check_id")
        require_enum(self.status, "status", CHECK_STATUSES)
        require_str(self.message, "message")
        _require_optional_hash(self.expected_hash, "expected_hash")
        _require_optional_hash(self.actual_hash, "actual_hash")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "IntegrityCheck":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            check_id=require_str(data["check_id"], "check_id"),
            status=require_enum(data["status"], "status", CHECK_STATUSES),
            message=require_str(data["message"], "message"),
            expected_hash=_require_optional_hash(data["expected_hash"], "expected_hash"),
            actual_hash=_require_optional_hash(data["actual_hash"], "actual_hash"),
        )


@dataclass(frozen=True)
class IntegrityReport(Contract):
    contract_type: ClassVar[str] = "integrity_report"

    run_id: str
    status: str
    checks: tuple[IntegrityCheck, ...]

    def __post_init__(self) -> None:
        require_str(self.run_id, "run_id")
        require_enum(self.status, "status", INTEGRITY_STATUSES)
        _validate_contract_values(self.checks, IntegrityCheck, "checks")
        if not self.checks:
            raise ContractError("checks: must contain at least one value")
        failed = any(check.status == "failed" for check in self.checks)
        if self.status == "passed" and failed:
            raise ContractError("status: cannot be passed when a check failed")
        if self.status == "failed" and not failed:
            raise ContractError("status: cannot be failed when all checks passed")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "IntegrityReport":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            run_id=require_str(data["run_id"], "run_id"),
            status=require_enum(data["status"], "status", INTEGRITY_STATUSES),
            checks=_parse_contract_tuple(data["checks"], IntegrityCheck, "checks"),
        )


def _validate_str_values(
    values: object,
    path: str,
    *,
    allowed: tuple[str, ...] | None = None,
    allow_empty: bool = True,
) -> None:
    if not isinstance(values, tuple):
        raise ContractError(f"{path}: expected tuple")
    if not allow_empty and not values:
        raise ContractError(f"{path}: must contain at least one value")
    seen: set[str] = set()
    for index, value in enumerate(values):
        item_path = f"{path}[{index}]"
        text = require_enum(value, item_path, allowed) if allowed is not None else require_str(value, item_path)
        if text in seen:
            raise ContractError(f"{path}: duplicate value {text!r}")
        seen.add(text)


def _validate_contract_values(values: object, cls: type, path: str) -> None:
    if not isinstance(values, tuple):
        raise ContractError(f"{path}: expected tuple")
    seen: set[str] = set()
    for index, value in enumerate(values):
        _require_instance(value, cls, f"{path}[{index}]")
        content_hash = value.content_hash
        if content_hash in seen:
            raise ContractError(f"{path}: duplicate contract at index {index}")
        seen.add(content_hash)


def _validate_disjoint_scoring_groups(
    fail_to_pass: tuple[str, ...],
    pass_to_pass: tuple[str, ...],
) -> None:
    overlap = sorted(set(fail_to_pass) & set(pass_to_pass))
    if overlap:
        raise ContractError(
            "fail_to_pass and pass_to_pass must be disjoint; "
            f"overlap={overlap!r}"
        )


def _freeze_json_object(value: object, path: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    plain: dict[str, JsonValue] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ContractError(f"{path}: object keys must be strings")
        plain[key] = _wire_value(item)
    canonical_json(plain)
    return MappingProxyType({key: _freeze_json_value(item) for key, item in plain.items()})


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _require_optional_hash(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=SHA256_PATTERN)


def _validate_logical_workspace_target(value: object) -> str:
    target = require_str(value, "workspace_target")
    private_prefixes = (
        "/Users/",
        "/home/",
        "/private/",
        "/tmp/",
        "~/",
    )
    if target.startswith(private_prefixes) or ":\\" in target:
        raise ContractError("workspace_target: host path is not public")
    if target == "workspace":
        return target
    if not target.startswith("/"):
        raise ContractError("workspace_target: expected canonical logical path")
    parts = target.split("/")[1:]
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ContractError("workspace_target: expected canonical logical path")
    return target


def _validate_time_range(started_at_ms: object, ended_at_ms: object) -> None:
    start = require_int(started_at_ms, "started_at_ms", minimum=0)
    end = require_int(ended_at_ms, "ended_at_ms", minimum=0)
    if end < start:
        raise ContractError("ended_at_ms: must be >= started_at_ms")
