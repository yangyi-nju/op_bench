#!/usr/bin/env python3
"""Rebind verified Runtime outcomes to an equivalent later quality review."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import load_canonical_json_artifact  # noqa: E402
from op_bench.factory.quality_admission import (  # noqa: E402
    QualityAdmissionResultIndex,
    _rebind_readmission,
    load_quality_accepted_task_index,
    load_quality_admission_result_index,
    validate_quality_admission_prompt,
)
from op_bench.factory.quality_release import validate_quality_task  # noqa: E402
from op_bench.registry import load_resolved_task  # noqa: E402
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


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


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"output is outside repository: {path}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-results", required=True)
    parser.add_argument("--target-accepted", required=True)
    parser.add_argument("--output-accepted", required=True)
    parser.add_argument("--output-results", required=True)
    parser.add_argument(
        "--confirm-equivalent-review",
        action="store_true",
        help="Assert the replacement review covers unchanged Prompt sources.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_equivalent_review:
        raise SystemExit("--confirm-equivalent-review is required")
    runtime_path = _rooted(args.runtime_results)
    target_path = _rooted(args.target_accepted)
    output_accepted = _rooted(args.output_accepted)
    output_results = _rooted(args.output_results)
    for output in (output_accepted, output_results):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")

    runtime = QualityAdmissionResultIndex.from_dict(
        load_canonical_json_artifact(runtime_path)
    )
    if not all(result.verified for result in runtime.results):
        raise ContractError("runtime shard must contain only verified outcomes")
    target = load_quality_accepted_task_index(
        ROOT, target_path, require_complete=False
    )
    targets = {record.task_id: record for record in target.tasks}
    environment_registry = ROOT / runtime.environment_registry_path
    source_registry = ROOT / runtime.source_registry_path
    accepted_records = []
    rebound_results = []
    for old in runtime.results:
        authorized = targets.get(old.task_id)
        if authorized is None:
            raise ContractError(f"{old.task_id}: target authorization missing")
        for field in (
            "screening_record_index",
            "pr_number",
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
        ):
            if getattr(old, field) != getattr(authorized, field):
                raise ContractError(f"{old.task_id}: identity field drift {field}")
        task = load_resolved_task(
            ROOT / authorized.task_path / "task.json",
            environment_registry_path=environment_registry,
            source_registry_path=source_registry,
        )
        prompt = validate_quality_admission_prompt(
            task,
            expected_source_hash=authorized.prompt_source_hash,
        )
        _rebind_readmission(task, task.task_dir / "admission/evidence.json")
        formal_errors = validate_quality_task(
            ROOT,
            task,
            require_verified=True,
            environment_registry_path=environment_registry,
            source_registry_path=source_registry,
        )
        if formal_errors:
            raise ContractError(
                f"{old.task_id}: formal quality failed: "
                + "; ".join(formal_errors)
            )
        rebound = replace(
            old,
            task_manifest_hash=authorized.task_manifest_hash,
            replay_spec_hash=authorized.replay_spec_hash,
            prompt_evidence_hash=prompt.content_hash,
            final_quality_errors=(),
            verified=True,
        )
        for field in RUNTIME_FIELDS:
            if getattr(rebound, field) != getattr(old, field):
                raise ContractError(f"{old.task_id}: Runtime field changed {field}")
        accepted_records.append(authorized)
        rebound_results.append(rebound)

    accepted_shard = replace(
        target,
        status="building",
        task_count=len(accepted_records),
        tasks=tuple(accepted_records),
    )
    result_shard = replace(
        runtime,
        accepted_index_path=_relative(output_accepted),
        accepted_index_hash=accepted_shard.content_hash,
        task_count=len(rebound_results),
        verified_count=len(rebound_results),
        results=tuple(rebound_results),
    )
    try:
        output_accepted.parent.mkdir(parents=True, exist_ok=True)
        output_results.parent.mkdir(parents=True, exist_ok=True)
        output_accepted.write_bytes(
            canonical_json(accepted_shard.to_dict()).encode("utf-8")
        )
        output_results.write_bytes(
            canonical_json(result_shard.to_dict()).encode("utf-8")
        )
        loaded = load_quality_admission_result_index(
            ROOT,
            output_results,
            output_accepted,
            require_private_bundles=True,
        )
    except BaseException:
        output_results.unlink(missing_ok=True)
        output_accepted.unlink(missing_ok=True)
        raise
    print(canonical_json({
        "accepted_index_hash": accepted_shard.content_hash,
        "result_index_hash": loaded.content_hash,
        "task_count": loaded.task_count,
        "verified_count": loaded.verified_count,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
