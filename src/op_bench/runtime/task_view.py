from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re

from op_bench.runtime.contracts import (
    AgentTaskView,
    BudgetPolicy,
    CapabilityPolicy,
    ContentIdentity,
    Contract,
    FullTaskSpec,
)
from op_bench.runtime.validation import ContractError, require_str


_SENSITIVE_KEYS = frozenset(
    {
        "admission",
        "admission_evidence",
        "api_key",
        "authorization",
        "cookie",
        "credential",
        "credentials",
        "gold",
        "gold_patch",
        "hidden_test",
        "hidden_tests",
        "hidden_test_asset",
        "password",
        "private_output",
        "refresh_token",
        "secret",
        "access_token",
    }
)

_SENSITIVE_TEXT_PATTERNS = (
    re.compile(
        r"https?://github\.com/[^\s/]+/[^\s/]+/(?:pull|commit)/[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(r"https?://patch-diff\.githubusercontent\.com/[^\s]+", re.IGNORECASE),
    re.compile(
        r"https?://github\.com/[^\s/]+/[^\s/]+/issues/\d+#issuecomment-\d+",
        re.IGNORECASE,
    ),
    re.compile(
        r"https?://github\.com/[^\s/]+/[^\s/]+/issues/\d+\?[^\s]*focusedcommentid=\d+",
        re.IGNORECASE,
    ),
    re.compile(r"\bgold(?:[ _.-]?patch)?\b", re.IGNORECASE),
    re.compile(r"\bhidden[ _.-]?tests?(?:[ _.-]?asset)?\b", re.IGNORECASE),
    re.compile(r"\badmission(?:[ _.-]?evidence)?\b", re.IGNORECASE),
    re.compile(r"\bprivate[ _.-]?output\b", re.IGNORECASE),
    re.compile(r"\bauthorization\s*[:=]\s*(?:bearer|basic)\s+\S+", re.IGNORECASE),
    re.compile(
        r"\b(?:api[ _.-]?key|access[ _.-]?token|refresh[ _.-]?token|password|secret)"
        r"\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
    re.compile(r"\bfile:///[^\s\"']+", re.IGNORECASE),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{12,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
    re.compile(r"\b(?:sk|xox[baprs])-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"-----BEGIN (?:(?:RSA|EC|DSA|OPENSSH|PGP) )?PRIVATE KEY(?: BLOCK)?-----",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"(?:^|[\s:=\"'(<])/(?![/*])[^\s\"']+"),
    re.compile(
        r"(?:^|[\s:=\"'(<])[A-Za-z]:\\+[^\s\"']+",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class TaskViewPolicy:
    """Public-only values supplied independently from a full task specification."""

    termination_notes: str = "Finish by calling session_finish after testing the patch."
    attachments: tuple[ContentIdentity, ...] = ()

    def __post_init__(self) -> None:
        require_str(self.termination_notes, "termination_notes")
        if not isinstance(self.attachments, tuple):
            raise ContractError("attachments: expected tuple")
        for index, attachment in enumerate(self.attachments):
            if not isinstance(attachment, ContentIdentity) or attachment.identity_type != "attachment":
                raise ContractError(
                    f"attachments: expected attachment identity at index {index}"
                )
        assert_public_artifact_safe(
            {
                "termination_notes": self.termination_notes,
                "attachments": [attachment.to_dict() for attachment in self.attachments],
            }
        )


@dataclass(frozen=True)
class AgentLaunchInput:
    """The complete task-bearing input an Agent Adapter may receive."""

    task_view: AgentTaskView
    task_view_identity: ContentIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.task_view, AgentTaskView):
            raise ContractError("task_view: expected AgentTaskView")
        if not isinstance(self.task_view_identity, ContentIdentity):
            raise ContractError("task_view_identity: expected ContentIdentity")
        if self.task_view_identity.identity_type != "task_view":
            raise ContractError("task_view_identity: expected task_view identity")
        if self.task_view_identity != agent_task_view_identity(self.task_view):
            raise ContractError("task_view_identity: does not match task_view content")
        assert_public_artifact_safe(self.task_view.to_dict())


def project_agent_task_view(
    full_task: FullTaskSpec,
    capability_policy: CapabilityPolicy,
    budget_policy: BudgetPolicy,
    *,
    policy: TaskViewPolicy | None = None,
) -> AgentTaskView:
    """Project a FullTaskSpec through a fixed public-field whitelist."""

    if not isinstance(full_task, FullTaskSpec):
        raise ContractError("full_task: expected FullTaskSpec")
    if not isinstance(capability_policy, CapabilityPolicy):
        raise ContractError("capability_policy: expected CapabilityPolicy")
    if not isinstance(budget_policy, BudgetPolicy):
        raise ContractError("budget_policy: expected BudgetPolicy")
    selected_policy = policy if policy is not None else TaskViewPolicy()
    if not isinstance(selected_policy, TaskViewPolicy):
        raise ContractError("policy: expected TaskViewPolicy")

    runtime_hint = (
        f"tier={full_task.runtime.runtime_tier}; "
        f"platform={full_task.runtime.platform}; "
        f"gpu={'yes' if full_task.runtime.requires_gpu else 'no'}"
    )
    view = AgentTaskView(
        task=full_task.task,
        statement_title=full_task.statement_title,
        statement_body=full_task.statement_body,
        framework=full_task.framework,
        operator_name=full_task.operator_name,
        runtime_hint=runtime_hint,
        public_tests=full_task.public_tests,
        capability_policy=capability_policy,
        budget_policy=budget_policy,
        termination_notes=selected_policy.termination_notes,
        attachments=selected_policy.attachments,
    )
    assert_public_artifact_safe(view.to_dict())
    return view


def agent_task_view_identity(view: AgentTaskView) -> ContentIdentity:
    if not isinstance(view, AgentTaskView):
        raise ContractError("task_view: expected AgentTaskView")
    return ContentIdentity(
        identity_type="task_view",
        identifier=f"{view.task.identifier}:agent-task-view-v1",
        digest=view.content_hash,
        digest_kind="content_sha256",
    )


def assert_public_artifact_safe(value: object) -> None:
    """Reject values that would expose private evaluation or machine-local data."""

    _scan_public_value(value, path="$", seen=set())


def _scan_public_value(value: object, *, path: str, seen: set[int]) -> None:
    if isinstance(value, Contract):
        value = value.to_dict()

    if isinstance(value, str):
        for pattern in _SENSITIVE_TEXT_PATTERNS:
            match = pattern.search(value)
            if match is not None:
                raise ContractError(
                    f"public artifact {path}: sensitive value matched {pattern.pattern!r}"
                )
        return

    if value is None or isinstance(value, (bool, int)):
        return

    if isinstance(value, (bytes, bytearray, float)):
        raise ContractError(
            f"public artifact {path}: unsupported value type {type(value).__name__}"
        )

    marker = id(value)
    if marker in seen:
        raise ContractError(f"public artifact {path}: cyclic value")

    if isinstance(value, Mapping):
        seen.add(marker)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ContractError(f"public artifact {path}: non-string key")
                snake_key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key)
                normalized_key = snake_key.lower().replace("-", "_").replace(" ", "_")
                if normalized_key in _SENSITIVE_KEYS:
                    raise ContractError(f"public artifact {path}.{key}: sensitive key")
                _scan_public_value(item, path=f"{path}.{key}", seen=seen)
        finally:
            seen.remove(marker)
        return

    if isinstance(value, Sequence):
        seen.add(marker)
        try:
            for index, item in enumerate(value):
                _scan_public_value(item, path=f"{path}[{index}]", seen=seen)
        finally:
            seen.remove(marker)
        return

    raise ContractError(f"public artifact {path}: unsupported value type {type(value).__name__}")


__all__ = [
    "AgentLaunchInput",
    "TaskViewPolicy",
    "agent_task_view_identity",
    "assert_public_artifact_safe",
    "project_agent_task_view",
]
