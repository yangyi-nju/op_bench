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
from op_bench.runtime.evaluation import (
    AttemptEvaluationCoordinator,
    CompletedEvaluation,
    EvaluationInfrastructureError,
    FreshEvaluationBackend,
    FreshEvaluator,
    PrivateEvaluationEvidence,
    SelectorExecution,
    StrictPatchApplyError,
)
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitEvaluationBackend,
    LocalGitSource,
    git_archive_source_identity,
)
from op_bench.runtime.integrity import (
    persist_integrity_reports,
    selected_attempts_from_ledger,
    verify_run_artifacts,
)
from op_bench.runtime.session import (
    AttemptSession,
    SessionStateError,
    TerminationAttribution,
    termination_attribution,
)
from op_bench.runtime.summary import (
    SelectedAttempt,
    rebuild_results,
    rebuild_summary,
    result_record,
    write_rebuilt_outputs,
)
from op_bench.runtime.resume import (
    AttemptLedger,
    AttemptLedgerRecord,
    ResumeDecision,
    parse_attempt_ledger,
)
from op_bench.runtime.run_artifacts import (
    AttemptArtifactIndex,
    AttemptArtifactStore,
    EvaluationArtifactHashes,
    retry_directory_name,
)
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
    "AttemptArtifactIndex",
    "AttemptArtifactStore",
    "AttemptEvaluationCoordinator",
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
    "CompletedEvaluation",
    "EvaluationInfrastructureError",
    "EvaluationArtifactHashes",
    "EvaluationOnlyTestAsset",
    "EventJournal",
    "FreshEvaluationBackend",
    "FreshEvaluator",
    "FrozenPatch",
    "PatchArtifact",
    "persist_integrity_reports",
    "PrivateEvaluationEvidence",
    "PublicArtifactStore",
    "RegisteredTest",
    "ResumeDecision",
    "SessionStateError",
    "SelectorExecution",
    "SelectedAttempt",
    "StrictPatchApplyError",
    "TaskViewPolicy",
    "TerminationAttribution",
    "WorkspacePolicy",
    "LocalGitEvaluationBackend",
    "LocalGitSource",
    "agent_task_view_identity",
    "assert_patch_identity_handoff",
    "assert_public_artifact_safe",
    "build_patch_artifact",
    "canonical_json",
    "canonical_sha256",
    "git_archive_source_identity",
    "project_agent_task_view",
    "parse_attempt_ledger",
    "rebuild_results",
    "rebuild_summary",
    "retry_directory_name",
    "result_record",
    "selected_attempts_from_ledger",
    "termination_attribution",
    "verify_action_pairing",
    "verify_event_chain",
    "verify_run_artifacts",
    "write_rebuilt_outputs",
]
