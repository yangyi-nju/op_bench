from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
import re
from typing import Any, ClassVar

from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import (
    AgentTaskView,
    AgentSpec,
    BudgetPolicy,
    CapabilityPolicy,
    ContentIdentity,
    Contract,
    FullTaskSpec,
    RuntimeProfile,
    SHA256_PATTERN,
    _contract_data,
    _parse_contract_tuple,
    _require_identity_type,
    _require_instance,
    _validate_contract_values,
)
from op_bench.runtime.task_view import (
    TaskViewPolicy,
    agent_task_view_identity,
    project_agent_task_view,
)
from op_bench.runtime.validation import (
    ContractError,
    require_int,
    require_str,
)


COHORT_PATTERN = r"cohort:v1:[0-9a-f]{64}"
ATTEMPT_PATTERN = r"attempt:v1:[0-9a-f]{64}"
UTC_SECONDS_PATTERN = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z"


@dataclass(frozen=True)
class ExpectedAttempt(Contract):
    contract_type: ClassVar[str] = "expected_attempt"

    attempt_id: str
    task: ContentIdentity
    task_view: ContentIdentity
    agent: ContentIdentity
    repeat: int
    effective_config_hash: str

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        _require_instance(self.task, ContentIdentity, "task")
        _require_identity_type(self.task, "task", "task")
        _require_instance(self.task_view, ContentIdentity, "task_view")
        _require_identity_type(self.task_view, "task_view", "task_view")
        _require_instance(self.agent, ContentIdentity, "agent")
        _require_identity_type(self.agent, "agent", "agent")
        require_int(self.repeat, "repeat", minimum=1)
        require_str(self.effective_config_hash, "effective_config_hash", pattern=SHA256_PATTERN)

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "ExpectedAttempt":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            attempt_id=require_str(data["attempt_id"], "attempt_id", pattern=ATTEMPT_PATTERN),
            task=ContentIdentity.from_dict(data["task"], path="expected_attempt.task"),
            task_view=ContentIdentity.from_dict(
                data["task_view"], path="expected_attempt.task_view"
            ),
            agent=ContentIdentity.from_dict(data["agent"], path="expected_attempt.agent"),
            repeat=require_int(data["repeat"], "repeat", minimum=1),
            effective_config_hash=require_str(
                data["effective_config_hash"],
                "effective_config_hash",
                pattern=SHA256_PATTERN,
            ),
        )


@dataclass(frozen=True)
class RunManifest(Contract):
    contract_type: ClassVar[str] = "run_manifest"

    platform_version: str
    action_protocol: str
    evaluation_protocol: str
    scoring_protocol: str
    dataset: ContentIdentity
    tasks: tuple[FullTaskSpec, ...]
    task_views: tuple[AgentTaskView, ...]
    agents: tuple[AgentSpec, ...]
    capability_policy: CapabilityPolicy
    budget_policy: BudgetPolicy
    runtime_profiles: tuple[RuntimeProfile, ...]
    retry_policy: ContentIdentity
    termination_policy: ContentIdentity
    scoring: ContentIdentity
    repeat_count: int
    created_at: str
    comparability_key: str
    cohort_id: str
    expected_attempts: tuple[ExpectedAttempt, ...]

    def __post_init__(self) -> None:
        for path, value in (
            ("platform_version", self.platform_version),
            ("action_protocol", self.action_protocol),
            ("evaluation_protocol", self.evaluation_protocol),
            ("scoring_protocol", self.scoring_protocol),
        ):
            require_str(value, path)
        _require_instance(self.dataset, ContentIdentity, "dataset")
        _require_identity_type(self.dataset, "dataset", "dataset")
        _validate_nonempty_contracts(
            self.tasks,
            FullTaskSpec,
            "tasks",
            identifier_key=lambda item: item.task.identifier,
        )
        _validate_nonempty_contracts(
            self.agents,
            AgentSpec,
            "agents",
            identifier_key=lambda item: item.agent.identifier,
        )
        _require_sorted(self.tasks, "tasks", lambda item: item.task.identifier)
        _validate_nonempty_contracts(
            self.task_views,
            AgentTaskView,
            "task_views",
            identifier_key=lambda item: item.task.identifier,
        )
        _require_sorted(self.task_views, "task_views", lambda item: item.task.identifier)
        _validate_task_views(
            self.tasks,
            self.task_views,
            self.capability_policy,
            self.budget_policy,
        )
        _require_sorted(self.agents, "agents", lambda item: item.agent.identifier)
        _require_instance(self.capability_policy, CapabilityPolicy, "capability_policy")
        _require_instance(self.budget_policy, BudgetPolicy, "budget_policy")
        _validate_nonempty_contracts(
            self.runtime_profiles,
            RuntimeProfile,
            "runtime_profiles",
            identifier_key=lambda item: item.profile_id,
        )
        _require_sorted(self.runtime_profiles, "runtime_profiles", lambda item: item.profile_id)
        runtime_hashes = {profile.content_hash for profile in self.runtime_profiles}
        for task in self.tasks:
            if task.runtime.content_hash not in runtime_hashes:
                raise ContractError(
                    f"runtime_profiles: missing runtime for task {task.task.identifier!r}"
                )
        for value, path, identity_type in (
            (self.retry_policy, "retry_policy", "policy"),
            (self.termination_policy, "termination_policy", "policy"),
            (self.scoring, "scoring", "scoring"),
        ):
            _require_instance(value, ContentIdentity, path)
            _require_identity_type(value, identity_type, path)
        require_int(self.repeat_count, "repeat_count", minimum=1)
        _validate_created_at(self.created_at)
        require_str(self.comparability_key, "comparability_key", pattern=SHA256_PATTERN)
        require_str(self.cohort_id, "cohort_id", pattern=COHORT_PATTERN)
        _validate_nonempty_contracts(self.expected_attempts, ExpectedAttempt, "expected_attempts")

        expected_key = comparability_key(self)
        if self.comparability_key != expected_key:
            raise ContractError("comparability_key: does not match manifest content")
        expected_cohort = cohort_id(expected_key)
        if self.cohort_id != expected_cohort:
            raise ContractError("cohort_id: does not match comparability key")
        expected_matrix = _expected_attempts_for(self)
        if self.expected_attempts != expected_matrix:
            raise ContractError("expected_attempts: does not match frozen matrix")

    @classmethod
    def from_dict(cls, value: object, *, path: str | None = None) -> "RunManifest":
        data = _contract_data(
            value,
            contract_type=cls.contract_type,
            field_names=tuple(item.name for item in fields(cls)),
            path=path,
        )
        return cls(
            platform_version=require_str(data["platform_version"], "platform_version"),
            action_protocol=require_str(data["action_protocol"], "action_protocol"),
            evaluation_protocol=require_str(data["evaluation_protocol"], "evaluation_protocol"),
            scoring_protocol=require_str(data["scoring_protocol"], "scoring_protocol"),
            dataset=ContentIdentity.from_dict(data["dataset"], path="run_manifest.dataset"),
            tasks=_parse_contract_tuple(data["tasks"], FullTaskSpec, "tasks"),
            task_views=_parse_contract_tuple(data["task_views"], AgentTaskView, "task_views"),
            agents=_parse_contract_tuple(data["agents"], AgentSpec, "agents"),
            capability_policy=CapabilityPolicy.from_dict(
                data["capability_policy"], path="run_manifest.capability_policy"
            ),
            budget_policy=BudgetPolicy.from_dict(
                data["budget_policy"], path="run_manifest.budget_policy"
            ),
            runtime_profiles=_parse_contract_tuple(
                data["runtime_profiles"], RuntimeProfile, "runtime_profiles"
            ),
            retry_policy=ContentIdentity.from_dict(
                data["retry_policy"], path="run_manifest.retry_policy"
            ),
            termination_policy=ContentIdentity.from_dict(
                data["termination_policy"], path="run_manifest.termination_policy"
            ),
            scoring=ContentIdentity.from_dict(data["scoring"], path="run_manifest.scoring"),
            repeat_count=require_int(data["repeat_count"], "repeat_count", minimum=1),
            created_at=require_str(data["created_at"], "created_at"),
            comparability_key=require_str(
                data["comparability_key"], "comparability_key", pattern=SHA256_PATTERN
            ),
            cohort_id=require_str(data["cohort_id"], "cohort_id", pattern=COHORT_PATTERN),
            expected_attempts=_parse_contract_tuple(
                data["expected_attempts"], ExpectedAttempt, "expected_attempts"
            ),
        )


def build_run_manifest(
    *,
    platform_version: str,
    action_protocol: str,
    evaluation_protocol: str,
    scoring_protocol: str,
    dataset: ContentIdentity,
    tasks: tuple[FullTaskSpec, ...],
    task_views: tuple[AgentTaskView, ...] | None = None,
    agents: tuple[AgentSpec, ...],
    capability_policy: CapabilityPolicy,
    budget_policy: BudgetPolicy,
    retry_policy: ContentIdentity,
    termination_policy: ContentIdentity,
    scoring: ContentIdentity,
    repeat_count: int,
    created_at: str,
) -> RunManifest:
    for path, value in (
        ("platform_version", platform_version),
        ("action_protocol", action_protocol),
        ("evaluation_protocol", evaluation_protocol),
        ("scoring_protocol", scoring_protocol),
    ):
        require_str(value, path)
    _require_instance(dataset, ContentIdentity, "dataset")
    _require_identity_type(dataset, "dataset", "dataset")
    _validate_nonempty_contracts(
        tasks,
        FullTaskSpec,
        "tasks",
        identifier_key=lambda item: item.task.identifier,
    )
    _validate_nonempty_contracts(
        agents,
        AgentSpec,
        "agents",
        identifier_key=lambda item: item.agent.identifier,
    )
    _require_instance(capability_policy, CapabilityPolicy, "capability_policy")
    _require_instance(budget_policy, BudgetPolicy, "budget_policy")
    for value, path, identity_type in (
        (retry_policy, "retry_policy", "policy"),
        (termination_policy, "termination_policy", "policy"),
        (scoring, "scoring", "scoring"),
    ):
        _require_instance(value, ContentIdentity, path)
        _require_identity_type(value, identity_type, path)
    require_int(repeat_count, "repeat_count", minimum=1)
    _validate_created_at(created_at)
    sorted_tasks = tuple(sorted(tasks, key=lambda item: item.task.identifier))
    if task_views is None:
        sorted_task_views = tuple(
            project_agent_task_view(task, capability_policy, budget_policy)
            for task in sorted_tasks
        )
    else:
        _validate_nonempty_contracts(
            task_views,
            AgentTaskView,
            "task_views",
            identifier_key=lambda item: item.task.identifier,
        )
        sorted_task_views = tuple(sorted(task_views, key=lambda item: item.task.identifier))
    _validate_task_views(
        sorted_tasks,
        sorted_task_views,
        capability_policy,
        budget_policy,
    )
    sorted_agents = tuple(sorted(agents, key=lambda item: item.agent.identifier))
    runtime_profiles = _unique_runtime_profiles(sorted_tasks)
    common: dict[str, Any] = {
        "platform_version": platform_version,
        "action_protocol": action_protocol,
        "evaluation_protocol": evaluation_protocol,
        "scoring_protocol": scoring_protocol,
        "dataset": dataset,
        "tasks": sorted_tasks,
        "task_views": sorted_task_views,
        "agents": sorted_agents,
        "capability_policy": capability_policy,
        "budget_policy": budget_policy,
        "runtime_profiles": runtime_profiles,
        "retry_policy": retry_policy,
        "termination_policy": termination_policy,
        "scoring": scoring,
        "repeat_count": repeat_count,
        "created_at": created_at,
    }
    key = _comparability_key_from_parts(**common)
    cohort = cohort_id(key)
    provisional = object.__new__(RunManifest)
    for name, value in common.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "comparability_key", key)
    object.__setattr__(provisional, "cohort_id", cohort)
    expected = _expected_attempts_for(provisional)
    return RunManifest(
        **common,
        comparability_key=key,
        cohort_id=cohort,
        expected_attempts=expected,
    )


def attempt_id(
    cohort_id: str,
    task: ContentIdentity,
    agent: ContentIdentity,
    repeat: int,
    effective_config_hash: str,
) -> str:
    require_str(cohort_id, "cohort_id", pattern=COHORT_PATTERN)
    _require_instance(task, ContentIdentity, "task")
    _require_identity_type(task, "task", "task")
    _require_instance(agent, ContentIdentity, "agent")
    _require_identity_type(agent, "agent", "agent")
    require_int(repeat, "repeat", minimum=1)
    require_str(effective_config_hash, "effective_config_hash", pattern=SHA256_PATTERN)
    digest = canonical_sha256(
        {
            "identity_version": "attempt-v1",
            "cohort_id": cohort_id,
            "task": task.to_dict(),
            "agent": agent.to_dict(),
            "repeat": repeat,
            "effective_config_hash": effective_config_hash,
        }
    )
    return f"attempt:v1:{digest.removeprefix('sha256:')}"


def comparability_key(manifest: RunManifest) -> str:
    if not isinstance(manifest, RunManifest):
        raise ContractError("manifest: expected RunManifest")
    return _comparability_key_from_parts(
        platform_version=manifest.platform_version,
        action_protocol=manifest.action_protocol,
        evaluation_protocol=manifest.evaluation_protocol,
        scoring_protocol=manifest.scoring_protocol,
        dataset=manifest.dataset,
        tasks=manifest.tasks,
        task_views=manifest.task_views,
        agents=manifest.agents,
        capability_policy=manifest.capability_policy,
        budget_policy=manifest.budget_policy,
        runtime_profiles=manifest.runtime_profiles,
        retry_policy=manifest.retry_policy,
        termination_policy=manifest.termination_policy,
        scoring=manifest.scoring,
        repeat_count=manifest.repeat_count,
        created_at=manifest.created_at,
    )


def _comparability_key_from_parts(
    *,
    platform_version: str,
    action_protocol: str,
    evaluation_protocol: str,
    scoring_protocol: str,
    dataset: ContentIdentity,
    tasks: tuple[FullTaskSpec, ...],
    task_views: tuple[AgentTaskView, ...],
    agents: tuple[AgentSpec, ...],
    capability_policy: CapabilityPolicy,
    budget_policy: BudgetPolicy,
    runtime_profiles: tuple[RuntimeProfile, ...],
    retry_policy: ContentIdentity,
    termination_policy: ContentIdentity,
    scoring: ContentIdentity,
    repeat_count: int,
    created_at: str,
) -> str:
    del created_at
    return canonical_sha256(
        {
            "identity_version": "comparability-v1",
            "platform_version": platform_version,
            "action_protocol": action_protocol,
            "evaluation_protocol": evaluation_protocol,
            "scoring_protocol": scoring_protocol,
            "dataset": dataset.to_dict(),
            "tasks": [task.to_dict() for task in tasks],
            "task_views": [task_view.to_dict() for task_view in task_views],
            "agents": [agent.to_dict() for agent in agents],
            "capability_policy": capability_policy.to_dict(),
            "budget_policy": budget_policy.to_dict(),
            "runtime_profiles": [profile.to_dict() for profile in runtime_profiles],
            "retry_policy": retry_policy.to_dict(),
            "termination_policy": termination_policy.to_dict(),
            "scoring": scoring.to_dict(),
            "repeat_count": repeat_count,
        }
    )


def cohort_id(comparability_key_value: str) -> str:
    require_str(comparability_key_value, "comparability_key", pattern=SHA256_PATTERN)
    return f"cohort:v1:{comparability_key_value.removeprefix('sha256:')}"


def _expected_attempts_for(manifest: RunManifest) -> tuple[ExpectedAttempt, ...]:
    attempts: list[ExpectedAttempt] = []
    views_by_task = {view.task.identifier: view for view in manifest.task_views}
    for task in manifest.tasks:
        task_view = views_by_task[task.task.identifier]
        task_view_identity = agent_task_view_identity(task_view)
        for agent in manifest.agents:
            effective_hash = canonical_sha256(
                {
                    "identity_version": "effective-attempt-config-v1",
                    "platform_version": manifest.platform_version,
                    "action_protocol": manifest.action_protocol,
                    "evaluation_protocol": manifest.evaluation_protocol,
                    "scoring_protocol": manifest.scoring_protocol,
                    "task": task.to_dict(),
                    "task_view": task_view.to_dict(),
                    "agent": agent.to_dict(),
                    "capability_policy": manifest.capability_policy.to_dict(),
                    "budget_policy": manifest.budget_policy.to_dict(),
                    "runtime": task.runtime.to_dict(),
                    "retry_policy": manifest.retry_policy.to_dict(),
                    "termination_policy": manifest.termination_policy.to_dict(),
                    "scoring": manifest.scoring.to_dict(),
                }
            )
            for repeat in range(1, manifest.repeat_count + 1):
                attempts.append(
                    ExpectedAttempt(
                        attempt_id=attempt_id(
                            manifest.cohort_id,
                            task.task,
                            agent.agent,
                            repeat,
                            effective_hash,
                        ),
                        task=task.task,
                        task_view=task_view_identity,
                        agent=agent.agent,
                        repeat=repeat,
                        effective_config_hash=effective_hash,
                    )
                )
    return tuple(attempts)


def _validate_task_views(
    tasks: tuple[FullTaskSpec, ...],
    task_views: tuple[AgentTaskView, ...],
    capability_policy: CapabilityPolicy,
    budget_policy: BudgetPolicy,
) -> None:
    if len(task_views) != len(tasks):
        raise ContractError("task_views: must contain exactly one view per task")
    views_by_task = {view.task.identifier: view for view in task_views}
    if len(views_by_task) != len(task_views):
        raise ContractError("task_views: duplicate task identity")
    for task in tasks:
        view = views_by_task.get(task.task.identifier)
        if view is None or view.task != task.task:
            raise ContractError(
                f"task_views: missing exact task identity for {task.task.identifier!r}"
            )
        try:
            expected = project_agent_task_view(
                task,
                capability_policy,
                budget_policy,
                policy=TaskViewPolicy(
                    termination_notes=view.termination_notes,
                    attachments=view.attachments,
                ),
            )
        except ContractError as exc:
            raise ContractError(f"task_views: invalid public projection: {exc}") from exc
        if view != expected:
            raise ContractError(
                f"task_views: view for {task.task.identifier!r} is not the canonical projection"
            )


def _unique_runtime_profiles(tasks: tuple[FullTaskSpec, ...]) -> tuple[RuntimeProfile, ...]:
    by_id: dict[str, RuntimeProfile] = {}
    for task in tasks:
        prior = by_id.get(task.runtime.profile_id)
        if prior is not None and prior != task.runtime:
            raise ContractError(
                f"runtime_profiles: conflicting profile_id {task.runtime.profile_id!r}"
            )
        by_id[task.runtime.profile_id] = task.runtime
    return tuple(by_id[key] for key in sorted(by_id))


def _validate_nonempty_contracts(
    values: object,
    cls: type,
    path: str,
    *,
    identifier_key=None,
) -> None:
    if not isinstance(values, tuple):
        raise ContractError(f"{path}: expected tuple")
    for index, value in enumerate(values):
        _require_instance(value, cls, f"{path}[{index}]")
    if identifier_key is not None:
        _require_unique_identifiers(values, path, identifier_key)
    _validate_contract_values(values, cls, path)
    if not values:
        raise ContractError(f"{path}: must contain at least one value")


def _require_unique_identifiers(values: tuple[Any, ...], path: str, key_fn) -> None:
    seen: set[str] = set()
    for value in values:
        identifier = key_fn(value)
        if identifier in seen:
            raise ContractError(f"{path}: duplicate identifier {identifier!r}")
        seen.add(identifier)


def _require_sorted(values: tuple[Any, ...], path: str, key_fn) -> None:
    identifiers = [key_fn(value) for value in values]
    if identifiers != sorted(identifiers):
        raise ContractError(f"{path}: values must be sorted by identifier")


def _validate_created_at(value: object) -> str:
    text = require_str(value, "created_at")
    if re.fullmatch(UTC_SECONDS_PATTERN, text) is None:
        raise ContractError("created_at: expected UTC RFC3339 seconds")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ContractError("created_at: expected UTC RFC3339 seconds") from exc
    return text
