#!/usr/bin/env python3
"""Synchronize verified v0.7 Task metadata from exact Admission outcomes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import (  # noqa: E402
    load_canonical_json_artifact,
    load_regular_file_bytes,
)
from op_bench.factory.quality_admission import (  # noqa: E402
    QualityAdmissionResultIndex,
    _promote_task_admission,
    _rebind_readmission,
    load_quality_accepted_task_index,
    load_quality_admission_result_index,
    quality_admission_bundle_hash,
    replay_spec_hash,
    validate_quality_admission_prompt,
)
from op_bench.factory.quality_release import (  # noqa: E402
    quality_bytes_hash,
    quality_prompt_source_hash,
    validate_quality_task,
)
from op_bench.registry import load_resolved_task  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from scripts.validate_task import validate_manifest  # noqa: E402


RUNTIME_FIELDS = (
    "preflight_status",
    "preflight_messages_hash",
    "admission_decision",
    "admission_verified",
    "baseline_status",
    "gold_status",
    "admission_evidence_hash",
    "admission_bundle_path",
    "admission_bundle_hash",
)


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--accepted-index",
        default="factory/v0.7/p8/accepted_tasks.json",
    )
    parser.add_argument(
        "--results",
        default="factory/v0.7/p8/admission_results.json",
    )
    parser.add_argument(
        "--confirm-reuse-runtime-evidence",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_reuse_runtime_evidence:
        raise SystemExit("--confirm-reuse-runtime-evidence is required")
    accepted_path = ROOT / args.accepted_index
    results_path = ROOT / args.results
    accepted = load_quality_accepted_task_index(
        ROOT, accepted_path, require_complete=True
    )
    results = QualityAdmissionResultIndex.from_dict(
        load_canonical_json_artifact(results_path)
    )
    if (
        accepted.task_count != 36
        or results.task_count != 36
        or results.verified_count != 36
        or not all(result.verified for result in results.results)
    ):
        raise ContractError("official quality Admission must be 36/36 verified")
    outcomes = {result.task_id: result for result in results.results}
    environment_path = ROOT / results.environment_registry_path
    source_path = ROOT / results.source_registry_path

    accepted_records = []
    rebound_results = []
    repaired_count = 0
    for authorized in accepted.tasks:
        old = outcomes[authorized.task_id]
        task = load_resolved_task(
            ROOT / authorized.task_path / "task.json",
            environment_registry_path=environment_path,
            source_registry_path=source_path,
        )
        metadata = task.data.get("metadata")
        metadata_verified = (
            isinstance(metadata, dict)
            and metadata.get("admission_status") == "verified"
            and metadata.get("curation_status") == "verified"
            and metadata.get("source_loading_verified") is True
        )
        if not metadata_verified:
            if not old.admission_verified or old.admission_decision != "verified":
                raise ContractError(f"{old.task_id}: verified Runtime outcome is required")
            if old.admission_bundle_path is None or old.admission_bundle_hash is None:
                raise ContractError(f"{old.task_id}: private Admission bundle is required")
            bundle_path = ROOT / old.admission_bundle_path
            if quality_admission_bundle_hash(bundle_path) != old.admission_bundle_hash:
                raise ContractError(f"{old.task_id}: private Admission bundle drift")
            stable_path = task.task_dir / "admission/evidence.json"
            if quality_bytes_hash(load_regular_file_bytes(stable_path)) != old.admission_evidence_hash:
                raise ContractError(f"{old.task_id}: stable Admission evidence drift")
            stable = _load_json(stable_path)
            verified_at = stable.get("created_at")
            if not isinstance(verified_at, str):
                raise ContractError(f"{old.task_id}: verified timestamp is missing")
            _rebind_readmission(task, stable_path)
            _promote_task_admission(task, verified_at=verified_at)
            task = load_resolved_task(
                task.task_json_path,
                environment_registry_path=environment_path,
                source_registry_path=source_path,
            )
            repaired_count += 1

        prompt = validate_quality_admission_prompt(
            task,
            expected_source_hash=authorized.prompt_source_hash,
        )
        formal_errors = validate_quality_task(
            ROOT,
            task,
            require_verified=True,
            environment_registry_path=environment_path,
            source_registry_path=source_path,
        )
        manifest_errors = validate_manifest(task.data)
        if formal_errors or manifest_errors:
            raise ContractError(
                f"{old.task_id}: post-repair validation failed "
                f"({len(formal_errors)} quality, {len(manifest_errors)} manifest)"
            )
        manifest = load_canonical_json_artifact(task.task_json_path)
        current_authorized = replace(
            authorized,
            task_manifest_hash=canonical_sha256(manifest),
            replay_spec_hash=replay_spec_hash(task),
            prompt_source_hash=quality_prompt_source_hash(task),
        )
        current_outcome = replace(
            old,
            task_manifest_hash=current_authorized.task_manifest_hash,
            replay_spec_hash=current_authorized.replay_spec_hash,
            prompt_evidence_hash=prompt.content_hash,
            final_quality_errors=(),
            verified=True,
        )
        for field in RUNTIME_FIELDS:
            if getattr(current_outcome, field) != getattr(old, field):
                raise ContractError(f"{old.task_id}: Runtime field changed {field}")
        accepted_records.append(current_authorized)
        rebound_results.append(current_outcome)

    final_accepted = replace(accepted, tasks=tuple(accepted_records))
    final_results = replace(
        results,
        accepted_index_hash=final_accepted.content_hash,
        verified_count=36,
        results=tuple(rebound_results),
    )
    previous_accepted = load_regular_file_bytes(accepted_path)
    previous_results = load_regular_file_bytes(results_path)
    try:
        accepted_path.write_bytes(canonical_json(final_accepted.to_dict()).encode("utf-8"))
        results_path.write_bytes(canonical_json(final_results.to_dict()).encode("utf-8"))
        loaded = load_quality_admission_result_index(
            ROOT,
            results_path,
            accepted_path,
            require_private_bundles=False,
        )
        if loaded.verified_count != 36:
            raise ContractError("post-repair result index is incomplete")
    except BaseException:
        accepted_path.write_bytes(previous_accepted)
        results_path.write_bytes(previous_results)
        raise
    print(
        canonical_json(
            {
                "accepted_index_hash": final_accepted.content_hash,
                "repaired_task_count": repaired_count,
                "result_index_hash": final_results.content_hash,
                "verified_count": final_results.verified_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
