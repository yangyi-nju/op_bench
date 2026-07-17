"""Versioned runtime contracts for the OpBench v0.6 evaluation platform."""

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.actions import (
    CanonicalActionService,
    CommandExecution,
    RegisteredTest,
)
from op_bench.runtime.adapters import (
    AdapterActionChannel,
    AdapterActionClient,
    AdapterContext,
)
from op_bench.runtime.artifacts import ArtifactReference, PublicArtifactStore
from op_bench.runtime.events import EventJournal, verify_action_pairing, verify_event_chain
from op_bench.runtime.session import (
    AttemptSession,
    SessionStateError,
    TerminationAttribution,
    termination_attribution,
)
from op_bench.runtime.resume import AttemptLedger, AttemptLedgerRecord, ResumeDecision
from op_bench.runtime.task_view import (
    AgentLaunchInput,
    TaskViewPolicy,
    agent_task_view_identity,
    assert_public_artifact_safe,
    project_agent_task_view,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.workspace import (
    AuthoritativeWorkspace,
    FrozenPatch,
    PatchArtifact,
    WorkspacePolicy,
    assert_patch_identity_handoff,
    build_patch_artifact,
)

__all__ = [
    "AgentLaunchInput",
    "AttemptLedger",
    "AttemptLedgerRecord",
    "AttemptSession",
    "ArtifactReference",
    "AdapterActionChannel",
    "AdapterActionClient",
    "AdapterContext",
    "AuthoritativeWorkspace",
    "CanonicalActionService",
    "CommandExecution",
    "ContractError",
    "EventJournal",
    "FrozenPatch",
    "PatchArtifact",
    "PublicArtifactStore",
    "RegisteredTest",
    "ResumeDecision",
    "SessionStateError",
    "TaskViewPolicy",
    "TerminationAttribution",
    "WorkspacePolicy",
    "agent_task_view_identity",
    "assert_patch_identity_handoff",
    "assert_public_artifact_safe",
    "build_patch_artifact",
    "canonical_json",
    "canonical_sha256",
    "project_agent_task_view",
    "termination_attribution",
    "verify_action_pairing",
    "verify_event_chain",
]
