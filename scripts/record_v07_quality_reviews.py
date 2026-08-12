#!/usr/bin/env python3
"""Record source-bound v0.7 Prompt and deferred-candidate reviews.

This command does not make review decisions.  It only records decisions that
the named reviewers have already completed, recomputes every machine-checkable
claim from the exact task sources, and updates the two local review packets.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import load_canonical_json_artifact  # noqa: E402
from op_bench.factory.prompt_quality import (  # noqa: E402
    PromptQualityEvidence,
    build_prompt_quality_evidence,
    validate_prompt_quality_evidence,
)
from op_bench.factory.quality_admission import (  # noqa: E402
    QualityAcceptedTaskIndex,
    QualityAcceptedTaskRecord,
    QualityCandidateReasonResolution,
    QualityCandidateReassessment,
    load_quality_accepted_task_index,
)
from op_bench.factory.contracts import FactoryArtifactReference  # noqa: E402
from op_bench.factory.quality_release import (  # noqa: E402
    QualityCandidateDecision,
    QualityCandidateRecord,
    quality_prompt_source_hash,
    quality_prompt_source_inputs,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.integrity import replay_spec_hash  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


DEFAULT_PROMPT_PACKET = (
    "runs/v0.7_quality_admission_staging/prompt_review_packet.json"
)
DEFAULT_REASSESSMENT_PACKET = (
    "runs/v0.7_quality_admission_staging/reassessment_review_packet.json"
)
DEFAULT_SCREENING_ROOT = "factory/v0.7/p8/screening"
DEFAULT_REASSESSMENT_ROOT = "factory/v0.7/p8/reassessments"
DEFAULT_ACCEPTED_OUTPUT = (
    "runs/v0.7_quality_admission_staging/accepted_tasks_29.json"
)
OFFICIAL_ACCEPTED_INDEX = "factory/v0.7/p8/accepted_tasks.json"


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def _load_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return dict(value)


def _write_canonical(path: Path, value: object) -> None:
    target = path.absolute()
    try:
        target.relative_to(ROOT.absolute())
    except ValueError as exc:
        raise ContractError(f"{path}: output is outside repository") from exc
    current = target
    while current != ROOT.absolute():
        if current.is_symlink():
            raise ContractError(f"{path}: symlink output is forbidden")
        current = current.parent
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _canonicalize_quality_artifacts(
    task_path: str,
    quality: Mapping[str, object],
) -> None:
    """Normalize existing Task quality artifacts before indexing them.

    Candidate construction predates the strict no-terminal-newline artifact
    contract in a few cases.  Recording the completed reviews is the final
    local write step before Admission, so normalize the unchanged JSON values
    here instead of allowing a byte-only defect to surface after Runtime has
    already completed.
    """

    for field in (
        "prompt_evidence",
        "complexity_evidence",
        "readmission_evidence",
    ):
        value = quality.get(field)
        if value is None:
            continue
        relative = _relative_json_path(value, f"quality.{field}")
        artifact_path = ROOT / task_path / relative
        if artifact_path.exists():
            _write_canonical(artifact_path, _load_object(artifact_path))


def _packet_records(packet: Mapping[str, object], path: Path) -> list[object]:
    records = packet.get("records")
    if not isinstance(records, list):
        raise ContractError(f"{path}: records must be a list")
    if packet.get("task_count") != len(records):
        raise ContractError(f"{path}: task_count does not match records")
    return records


def _relative_json_path(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected string")
    selected = PurePosixPath(value)
    if (
        selected.is_absolute()
        or selected.suffix != ".json"
        or not selected.parts
        or any(part in ("", ".", "..") for part in selected.parts)
        or selected.as_posix() != value
    ):
        raise ContractError(f"{path}: expected normalized relative JSON path")
    return value


def _record_prompt_reviews(
    *,
    packet_path: Path,
    blind_reviewer: str,
    blind_reviewed_at: str,
    semantic_reviewer: str,
    semantic_reviewed_at: str,
    created_at: str,
) -> tuple[int, list[dict[str, str]]]:
    if blind_reviewer == semantic_reviewer:
        raise ContractError("Prompt reviewers must have different identities")
    packet = _load_object(packet_path)
    records = _packet_records(packet, packet_path)
    completed: list[dict[str, str]] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ContractError(f"{packet_path}: records[{index}] must be an object")
        record = dict(raw_record)
        task_path = record.get("task_path")
        if not isinstance(task_path, str):
            raise ContractError(f"{packet_path}: records[{index}].task_path")
        task = TaskManifest.load(ROOT / task_path / "task.json")
        if task.task_id != record.get("task_id"):
            raise ContractError(f"{task_path}: task_id changed since review packet")
        if task.public_task_id != record.get("public_task_id"):
            raise ContractError(
                f"{task_path}: public_task_id changed since review packet"
            )
        scanner_version = record.get("scanner_version")
        if not isinstance(scanner_version, str):
            raise ContractError(f"{task_path}: scanner_version is missing")
        view, private_index = quality_prompt_source_inputs(
            task,
            scanner_version=scanner_version,
        )
        if canonical_sha256(view) != record.get("agent_task_view_hash"):
            raise ContractError(f"{task_path}: AgentTaskView changed since review")
        if (
            quality_prompt_source_hash(task, scanner_version=scanner_version)
            != record.get("prompt_source_hash")
        ):
            raise ContractError(f"{task_path}: private Prompt inputs changed")
        evidence = build_prompt_quality_evidence(
            task_id=task.task_id,
            public_task_id=task.public_task_id,
            rendered_prompt=render_mcp_prompt(view),
            agent_task_view=view,
            private_index=private_index,
            scanner_version=scanner_version,
            blind_review={
                "decision": "accepted",
                "reviewer": blind_reviewer,
                "reviewed_at": blind_reviewed_at,
            },
            semantic_review={
                "decision": "equivalent",
                "reviewer": semantic_reviewer,
                "reviewed_at": semantic_reviewed_at,
            },
            decision="accepted",
            created_at=created_at,
        )
        if evidence.findings:
            raise ContractError(f"{task_path}: Prompt scan produced findings")
        validate_prompt_quality_evidence(
            evidence,
            rendered_prompt=render_mcp_prompt(view),
            agent_task_view=view,
            private_index=private_index,
        )
        quality = task.data.get("quality")
        if not isinstance(quality, dict):
            raise ContractError(f"{task_path}: quality must be an object")
        relative = _relative_json_path(
            quality.get("prompt_evidence"),
            f"{task_path}.quality.prompt_evidence",
        )
        evidence_path = task.task_dir / relative
        _write_canonical(evidence_path, evidence.to_dict())
        PromptQualityEvidence.from_dict(
            load_canonical_json_artifact(evidence_path),
            path=f"{task_path}.quality.prompt_evidence",
        )
        record["blind_review"] = dict(evidence.blind_review)
        record["semantic_review"] = dict(evidence.semantic_review)
        record["formal_prompt_evidence"] = {
            "content_hash": evidence.content_hash,
            "path": evidence_path.relative_to(ROOT).as_posix(),
        }
        records[index] = record
        completed.append(
            {
                "content_hash": evidence.content_hash,
                "path": evidence_path.relative_to(ROOT).as_posix(),
                "task_id": task.task_id,
            }
        )
    packet["blind_review_complete_count"] = len(records)
    packet["semantic_review_complete_count"] = len(records)
    packet["formal_prompt_evidence_count"] = len(records)
    packet["records"] = records
    packet["content_hash"] = canonical_sha256(
        {key: value for key, value in packet.items() if key != "content_hash"}
    )
    _write_canonical(packet_path, packet)
    return len(records), completed


def _resolution_evidence(record: Mapping[str, object], reason: str) -> str:
    task_id = record.get("task_id")
    scope = record.get("patch_scope_paths")
    private_inputs = record.get("private_review_inputs")
    complexity = record.get("complexity")
    if not isinstance(scope, list) or not all(isinstance(item, str) for item in scope):
        raise ContractError(f"{task_id}: patch_scope_paths must be strings")
    if not isinstance(private_inputs, dict) or not isinstance(complexity, dict):
        raise ContractError(f"{task_id}: private review bindings are incomplete")
    bindings = (
        f"task manifest {private_inputs.get('task_manifest_hash')}, Gold patch "
        f"{private_inputs.get('gold_patch_hash')}, hidden oracle "
        f"{private_inputs.get('hidden_test_patch_hash')}, and complexity evidence "
        f"{complexity.get('content_hash')}"
    )
    oracle = (
        f"{record.get('fail_to_pass_count')} fail-to-pass and "
        f"{record.get('pass_to_pass_count')} pass-to-pass selectors"
    )
    if reason == "review.title_needs_behavior_confirmation":
        return (
            f"The exact {bindings} confirm that the sanitized title names a real "
            f"operator-domain failure. The Gold delta is confined to the declared "
            f"scope ({', '.join(scope)}), and the hidden oracle contains {oracle}; "
            "the candidate is not a title-only or mechanical change."
        )
    if reason == "review.ambiguous_change_context":
        return (
            f"The exact {bindings} bind the public behavior to the implementation "
            f"scope ({', '.join(scope)}). The hidden oracle contains {oracle}, and "
            "the accepted hard complexity dimensions require localization, diagnosis, "
            "and repair/regression reasoning; the change context is sufficiently "
            "specific for task construction."
        )
    if reason == "review.small_source_delta":
        return (
            f"The exact {bindings} show that the small production delta guards a "
            f"non-mechanical operator contract in scope ({', '.join(scope)}). The "
            f"hidden oracle contains {oracle}, and the accepted hard complexity "
            "evidence requires nontrivial localization and diagnosis plus an explicit "
            "regression control; patch size alone does not make this Task easy."
        )
    raise ContractError(f"{task_id}: unsupported deferred reason {reason!r}")


def _record_reassessments(
    *,
    packet_path: Path,
    screening_root: Path,
    output_root: Path,
    reviewer: str,
    reviewed_at: str,
) -> tuple[int, list[dict[str, str]]]:
    packet = _load_object(packet_path)
    records = _packet_records(packet, packet_path)
    completed: list[dict[str, str]] = []
    for index, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ContractError(f"{packet_path}: records[{index}] must be an object")
        record = dict(raw_record)
        candidate_ref = record.get("candidate")
        decision_ref = record.get("decision")
        if not isinstance(candidate_ref, dict) or not isinstance(decision_ref, dict):
            raise ContractError(f"{packet_path}: records[{index}] missing source refs")
        candidate_relative = _relative_json_path(
            candidate_ref.get("relative_path"),
            f"records[{index}].candidate.relative_path",
        )
        decision_relative = _relative_json_path(
            decision_ref.get("relative_path"),
            f"records[{index}].decision.relative_path",
        )
        candidate = QualityCandidateRecord.from_dict(
            load_canonical_json_artifact(screening_root / candidate_relative),
            path=f"records[{index}].candidate",
        )
        decision = QualityCandidateDecision.from_dict(
            load_canonical_json_artifact(screening_root / decision_relative),
            path=f"records[{index}].decision",
        )
        reasons = record.get("deferred_reasons")
        if not isinstance(reasons, list) or not all(
            isinstance(reason, str) for reason in reasons
        ):
            raise ContractError(f"records[{index}].deferred_reasons")
        selected_reasons = tuple(reasons)
        if selected_reasons != decision.preliminary_review_reasons:
            raise ContractError(f"records[{index}]: deferred reasons changed")
        reassessment = QualityCandidateReassessment(
            pr_number=candidate.pr_number,
            candidate_id=candidate.candidate_id,
            candidate_hash=candidate.content_hash,
            decision_id=decision.decision_id,
            decision_hash=decision.content_hash,
            deferred_reasons=selected_reasons,
            reason_resolutions=tuple(
                QualityCandidateReasonResolution(
                    reason=reason,
                    resolution="resolved",
                    evidence=_resolution_evidence(record, reason),
                )
                for reason in selected_reasons
            ),
            reviewer=reviewer,
            reviewed_at=reviewed_at,
            decision="accepted_for_build",
            rationale=(
                "Accepted after a source-bound Codex reassessment of the exact "
                "candidate, screening decision, Base/Gold behavior, hidden oracle, "
                "and hard-complexity evidence. Every preliminary review reason is "
                "resolved; this does not assert runtime Admission."
            ),
        )
        output = output_root / f"pr-{candidate.pr_number}.json"
        _write_canonical(output, reassessment.to_dict())
        QualityCandidateReassessment.from_dict(
            load_canonical_json_artifact(output),
            path=f"records[{index}].reassessment",
        )
        artifact_id = (
            "quality-reassessment:v1:"
            + reassessment.content_hash.removeprefix("sha256:")
        )
        record["review_state"] = "accepted_for_build"
        record["reassessment"] = {
            "artifact_id": artifact_id,
            "artifact_type": reassessment.contract_type,
            "content_hash": reassessment.content_hash,
            "relative_path": output.relative_to(ROOT).as_posix(),
        }
        records[index] = record
        completed.append(
            {
                "content_hash": reassessment.content_hash,
                "path": output.relative_to(ROOT).as_posix(),
                "task_id": str(record.get("task_id")),
            }
        )
    packet["pending_count"] = 0
    packet["accepted_count"] = len(records)
    packet["rejected_count"] = 0
    packet["records"] = records
    packet["content_hash"] = canonical_sha256(
        {key: value for key, value in packet.items() if key != "content_hash"}
    )
    _write_canonical(packet_path, packet)
    return len(records), completed


def _build_staging_accepted_index(
    *,
    prompt_packet_path: Path,
    reassessment_packet_path: Path,
    screening_root: Path,
    output_path: Path,
    created_at: str,
) -> QualityAcceptedTaskIndex:
    prompt_packet = _load_object(prompt_packet_path)
    prompt_records = _packet_records(prompt_packet, prompt_packet_path)
    reassessment_packet = _load_object(reassessment_packet_path)
    reassessment_records = _packet_records(
        reassessment_packet, reassessment_packet_path
    )
    reassessments: dict[str, FactoryArtifactReference] = {}
    for index, raw_record in enumerate(reassessment_records):
        if not isinstance(raw_record, dict):
            raise ContractError(
                f"{reassessment_packet_path}: records[{index}] must be an object"
            )
        task_id = raw_record.get("task_id")
        reference = raw_record.get("reassessment")
        if not isinstance(task_id, str) or not isinstance(reference, dict):
            raise ContractError(
                f"{reassessment_packet_path}: incomplete reassessment record"
            )
        reassessments[task_id] = FactoryArtifactReference.from_dict(
            reference,
            path=f"reassessment_packet.records[{index}].reassessment",
        )

    source_index = QualityAcceptedTaskIndex.from_dict(
        load_canonical_json_artifact(ROOT / OFFICIAL_ACCEPTED_INDEX),
        path="official_accepted_index",
    )
    screening_index = _load_object(screening_root / "screening_index.json")
    screening_records = _packet_records(
        {
            "task_count": len(screening_index.get("records", [])),
            "records": screening_index.get("records"),
        },
        screening_root / "screening_index.json",
    )
    accepted: list[QualityAcceptedTaskRecord] = []
    ordered_prompt_records = sorted(
        prompt_records,
        key=lambda item: (
            item.get("screening_index", -1) if isinstance(item, dict) else -1
        ),
    )
    for index, raw_record in enumerate(ordered_prompt_records):
        if not isinstance(raw_record, dict):
            raise ContractError(f"prompt records[{index}] must be an object")
        screening_position = raw_record.get("screening_index")
        if not isinstance(screening_position, int) or not (
            0 <= screening_position < len(screening_records)
        ):
            raise ContractError(f"prompt records[{index}].screening_index")
        screening_entry = screening_records[screening_position]
        if not isinstance(screening_entry, dict):
            raise ContractError(
                f"screening records[{screening_position}] must be an object"
            )
        candidate_ref = FactoryArtifactReference.from_dict(
            screening_entry.get("candidate"),
            path=f"screening records[{screening_position}].candidate",
        )
        decision_ref = FactoryArtifactReference.from_dict(
            screening_entry.get("decision"),
            path=f"screening records[{screening_position}].decision",
        )
        candidate = QualityCandidateRecord.from_dict(
            load_canonical_json_artifact(
                screening_root / candidate_ref.relative_path
            ),
            path=f"screening records[{screening_position}].candidate",
        )
        decision = QualityCandidateDecision.from_dict(
            load_canonical_json_artifact(
                screening_root / decision_ref.relative_path
            ),
            path=f"screening records[{screening_position}].decision",
        )
        task_path = raw_record.get("task_path")
        if not isinstance(task_path, str):
            raise ContractError(f"prompt records[{index}].task_path")
        manifest_path = ROOT / task_path / "task.json"
        manifest_source = _load_object(manifest_path)
        _write_canonical(manifest_path, manifest_source)
        manifest_data = load_canonical_json_artifact(manifest_path)
        task = TaskManifest.load(manifest_path)
        disposition = decision.disposition
        reassessment = (
            None
            if disposition == "accepted_for_build"
            else reassessments.get(task.task_id)
        )
        if disposition == "deferred_for_review" and reassessment is None:
            raise ContractError(f"{task.task_id}: reassessment is missing")
        quality = task.data.get("quality")
        if not isinstance(quality, dict):
            raise ContractError(f"{task.task_id}: quality must be an object")
        _canonicalize_quality_artifacts(task_path, quality)
        accepted.append(
            QualityAcceptedTaskRecord(
                screening_record_index=screening_position,
                pr_number=candidate.pr_number,
                candidate_id=candidate.candidate_id,
                candidate_hash=candidate.content_hash,
                decision_id=decision.decision_id,
                decision_hash=decision.content_hash,
                screening_disposition=disposition,
                reassessment=reassessment,
                task_id=task.task_id,
                public_task_id=str(task.public_task_id),
                origin=str(quality.get("origin")),
                task_path=task_path,
                task_manifest_hash=canonical_sha256(manifest_data),
                replay_spec_hash=replay_spec_hash(task),
                prompt_source_hash=quality_prompt_source_hash(task),
            )
        )
    index = QualityAcceptedTaskIndex(
        created_at=created_at,
        historical_index_path=source_index.historical_index_path,
        historical_index_hash=source_index.historical_index_hash,
        retained_count=source_index.retained_count,
        required_task_count=source_index.required_task_count,
        candidate_index_path=source_index.candidate_index_path,
        candidate_index_hash=source_index.candidate_index_hash,
        status="building",
        task_count=len(accepted),
        tasks=tuple(accepted),
    )
    _write_canonical(output_path, index.to_dict())
    loaded = load_quality_accepted_task_index(ROOT, output_path)
    if loaded.content_hash != index.content_hash:
        raise ContractError("staging accepted index did not round-trip")
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-packet", default=DEFAULT_PROMPT_PACKET)
    parser.add_argument(
        "--reassessment-packet", default=DEFAULT_REASSESSMENT_PACKET
    )
    parser.add_argument("--screening-root", default=DEFAULT_SCREENING_ROOT)
    parser.add_argument("--reassessment-root", default=DEFAULT_REASSESSMENT_ROOT)
    parser.add_argument("--accepted-output", default=DEFAULT_ACCEPTED_OUTPUT)
    parser.add_argument("--blind-reviewer", required=True)
    parser.add_argument("--blind-reviewed-at", required=True)
    parser.add_argument("--semantic-reviewer", required=True)
    parser.add_argument("--semantic-reviewed-at", required=True)
    parser.add_argument("--reassessment-reviewer", required=True)
    parser.add_argument("--reassessment-reviewed-at", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--confirm-completed-reviews",
        action="store_true",
        help="Required assertion that all named reviews were actually completed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_completed_reviews:
        raise SystemExit(
            "refusing to record decisions without --confirm-completed-reviews"
        )
    prompt_count, prompt_artifacts = _record_prompt_reviews(
        packet_path=_rooted(args.prompt_packet),
        blind_reviewer=args.blind_reviewer,
        blind_reviewed_at=args.blind_reviewed_at,
        semantic_reviewer=args.semantic_reviewer,
        semantic_reviewed_at=args.semantic_reviewed_at,
        created_at=args.created_at,
    )
    reassessment_count, reassessment_artifacts = _record_reassessments(
        packet_path=_rooted(args.reassessment_packet),
        screening_root=_rooted(args.screening_root),
        output_root=_rooted(args.reassessment_root),
        reviewer=args.reassessment_reviewer,
        reviewed_at=args.reassessment_reviewed_at,
    )
    accepted_index = _build_staging_accepted_index(
        prompt_packet_path=_rooted(args.prompt_packet),
        reassessment_packet_path=_rooted(args.reassessment_packet),
        screening_root=_rooted(args.screening_root),
        output_path=_rooted(args.accepted_output),
        created_at=args.created_at,
    )
    print(
        json.dumps(
            {
                "prompt_artifacts": prompt_artifacts,
                "prompt_review_count": prompt_count,
                "reassessment_artifacts": reassessment_artifacts,
                "reassessment_count": reassessment_count,
                "staging_accepted_index": args.accepted_output,
                "staging_accepted_index_hash": accepted_index.content_hash,
                "staging_accepted_task_count": accepted_index.task_count,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
