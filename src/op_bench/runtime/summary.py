from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib

from op_bench.runtime.canonical import JsonValue, canonical_json
from op_bench.runtime.contracts import (
    EvaluationResultV06,
    SHA256_PATTERN,
)
from op_bench.runtime.manifest import ATTEMPT_PATTERN, ExpectedAttempt, RunManifest
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_int, require_str


@dataclass(frozen=True)
class SelectedAttempt:
    attempt_id: str
    retry_index: int
    evaluation_spec_hash: str
    evaluation_result: EvaluationResultV06

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(
            self.evaluation_spec_hash,
            "evaluation_spec_hash",
            pattern=SHA256_PATTERN,
        )
        if not isinstance(self.evaluation_result, EvaluationResultV06):
            raise ContractError(
                "evaluation_result: expected EvaluationResultV06"
            )
        if self.evaluation_result.attempt_id != self.attempt_id:
            raise ContractError(
                "attempt_id: does not match EvaluationResult"
            )


def result_record(
    manifest: RunManifest,
    expected_attempt: ExpectedAttempt,
    selected_attempt: SelectedAttempt,
) -> dict[str, JsonValue]:
    if not isinstance(manifest, RunManifest):
        raise ContractError("manifest: expected RunManifest")
    if not isinstance(expected_attempt, ExpectedAttempt):
        raise ContractError("expected_attempt: expected ExpectedAttempt")
    if not isinstance(selected_attempt, SelectedAttempt):
        raise ContractError("selected_attempt: expected SelectedAttempt")
    if selected_attempt.attempt_id != expected_attempt.attempt_id:
        raise ContractError("selected_attempt: attempt identity mismatch")
    result = selected_attempt.evaluation_result
    record: dict[str, JsonValue] = {
        "record_type": "v0.6_result",
        "schema_version": "v1",
        "attempt_id": expected_attempt.attempt_id,
        "session_id": result.session_id,
        "task": expected_attempt.task.to_dict(),
        "agent": expected_attempt.agent.to_dict(),
        "repeat": expected_attempt.repeat,
        "retry_index": selected_attempt.retry_index,
        "effective_config_hash": expected_attempt.effective_config_hash,
        "attempt_validity": result.attempt_validity,
        "agent_terminal": result.agent_terminal,
        "evaluation_outcome": result.evaluation_outcome,
        "invalid_reason": result.invalid_reason,
        "patch": None if result.patch is None else result.patch.to_dict(),
        "fail_to_pass": result.fail_to_pass.to_dict(),
        "pass_to_pass": result.pass_to_pass.to_dict(),
        "duration_ms": result.duration_ms,
        "evaluation": result.evaluation.to_dict(),
        "scoring": result.scoring.to_dict(),
        "evaluation_spec_hash": selected_attempt.evaluation_spec_hash,
        "evaluation_result_hash": result.content_hash,
    }
    assert_public_artifact_safe(record)
    return record


def rebuild_results(
    manifest: RunManifest,
    selected_attempts: Sequence[SelectedAttempt],
) -> bytes:
    selected = _selected_by_attempt(manifest, selected_attempts)
    lines = [
        canonical_json(result_record(manifest, expected, selected[expected.attempt_id]))
        for expected in manifest.expected_attempts
        if expected.attempt_id in selected
    ]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def rebuild_summary(
    manifest: RunManifest,
    selected_attempts: Sequence[SelectedAttempt],
) -> dict[str, JsonValue]:
    selected = _selected_by_attempt(manifest, selected_attempts)
    results_bytes = rebuild_results(manifest, tuple(selected.values()))
    expected_by_agent: dict[str, list[ExpectedAttempt]] = {
        agent.agent.identifier: [] for agent in manifest.agents
    }
    for expected in manifest.expected_attempts:
        expected_by_agent[expected.agent.identifier].append(expected)

    agents: dict[str, JsonValue] = {}
    all_selected: list[SelectedAttempt] = []
    for agent_id in sorted(expected_by_agent):
        expected = expected_by_agent[agent_id]
        observed = [
            selected[item.attempt_id]
            for item in expected
            if item.attempt_id in selected
        ]
        all_selected.extend(observed)
        agents[agent_id] = _aggregate(len(expected), observed)
    summary: dict[str, JsonValue] = {
        "summary_version": "opbench-v0.6-summary-v1",
        "manifest_hash": manifest.content_hash,
        "cohort_id": manifest.cohort_id,
        "comparability_key": manifest.comparability_key,
        "evaluation_protocol": manifest.evaluation_protocol,
        "evaluation": manifest.evaluation.to_dict(),
        "scoring_protocol": manifest.scoring_protocol,
        "scoring": manifest.scoring.to_dict(),
        "results_hash": "sha256:" + hashlib.sha256(results_bytes).hexdigest(),
        "expected_attempts": len(manifest.expected_attempts),
        "observed_attempts": len(selected),
        "missing_attempts": len(manifest.expected_attempts) - len(selected),
        "totals": _aggregate(len(manifest.expected_attempts), all_selected),
        "agents": agents,
    }
    assert_public_artifact_safe(summary)
    return summary


def write_rebuilt_outputs(
    store,
    selected_attempts: Sequence[SelectedAttempt],
) -> tuple[bytes, bytes]:
    from op_bench.runtime.run_artifacts import AttemptArtifactStore

    if not isinstance(store, AttemptArtifactStore):
        raise ContractError("store: expected AttemptArtifactStore")
    results_bytes = rebuild_results(store.manifest, selected_attempts)
    summary = rebuild_summary(store.manifest, selected_attempts)
    summary_bytes = (canonical_json(summary) + "\n").encode("utf-8")
    store.write_results_bytes(results_bytes)
    store.write_summary_bytes(summary_bytes)
    return results_bytes, summary_bytes


def _selected_by_attempt(
    manifest: RunManifest,
    selected_attempts: Sequence[SelectedAttempt],
) -> dict[str, SelectedAttempt]:
    if not isinstance(manifest, RunManifest):
        raise ContractError("manifest: expected RunManifest")
    if not isinstance(selected_attempts, Sequence):
        raise ContractError("selected_attempts: expected sequence")
    expected_ids = {item.attempt_id for item in manifest.expected_attempts}
    selected: dict[str, SelectedAttempt] = {}
    for index, item in enumerate(selected_attempts):
        if not isinstance(item, SelectedAttempt):
            raise ContractError(
                f"selected_attempts[{index}]: expected SelectedAttempt"
            )
        if item.attempt_id not in expected_ids:
            raise ContractError(
                f"selected_attempts[{index}]: unexpected attempt_id"
            )
        if item.attempt_id in selected:
            raise ContractError(
                f"selected_attempts: duplicate attempt_id {item.attempt_id!r}"
            )
        if item.evaluation_result.scoring != manifest.scoring:
            raise ContractError(
                f"selected_attempts[{index}]: scoring identity mismatch"
            )
        if item.evaluation_result.evaluation != manifest.evaluation:
            raise ContractError(
                f"selected_attempts[{index}]: evaluation identity mismatch"
            )
        selected[item.attempt_id] = item
    return selected


def _aggregate(
    expected_count: int,
    selected: Sequence[SelectedAttempt],
) -> dict[str, JsonValue]:
    results = [item.evaluation_result for item in selected]
    valid = sum(result.attempt_validity == "valid" for result in results)
    infrastructure_invalid = len(results) - valid
    resolved = sum(
        result.attempt_validity == "valid"
        and result.evaluation_outcome == "resolved"
        for result in results
    )
    outcomes = Counter(result.evaluation_outcome for result in results)
    terminals = Counter(
        "none" if result.agent_terminal is None else result.agent_terminal
        for result in results
    )
    return {
        "expected": expected_count,
        "observed": len(results),
        "valid": valid,
        "infrastructure_invalid": infrastructure_invalid,
        "resolved": resolved,
        "resolved_denominator": valid,
        "resolved_rate": {
            "numerator": resolved,
            "denominator": valid,
        },
        "retries": sum(item.retry_index - 1 for item in selected),
        "evaluation_outcomes": dict(sorted(outcomes.items())),
        "agent_terminals": dict(sorted(terminals.items())),
    }


__all__ = [
    "SelectedAttempt",
    "rebuild_results",
    "rebuild_summary",
    "result_record",
    "write_rebuilt_outputs",
]
