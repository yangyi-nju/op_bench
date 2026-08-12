#!/usr/bin/env python3
"""Assemble the final 29-task v0.7 quality Admission staging artifacts.

The input Runtime shards must be disjoint, fully verified, and cover every
Task in the target accepted index exactly once. Runtime outcomes are rebound
to the target's latest source-bound quality review while every Runtime field
and private Admission bundle remains unchanged.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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
    replay_spec_hash,
    validate_quality_admission_prompt,
)
from op_bench.factory.quality_release import (  # noqa: E402
    quality_prompt_source_hash,
    quality_bytes_hash,
    validate_quality_task,
)
from op_bench.registry import load_resolved_task  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


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


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _semantic_source(value: object) -> object:
    if not isinstance(value, dict):
        return value
    return {key: item for key, item in value.items() if key != "local_path"}


def _source_map(value: dict[str, object], *, label: str) -> dict[str, dict[str, object]]:
    raw_sources = value.get("sources")
    if not isinstance(raw_sources, list):
        raise ContractError(f"{label}: sources must be a list")
    selected: dict[str, dict[str, object]] = {}
    for index, raw_source in enumerate(raw_sources):
        if not isinstance(raw_source, dict) or not isinstance(raw_source.get("id"), str):
            raise ContractError(f"{label}: invalid sources[{index}]")
        source_id = str(raw_source["id"])
        if source_id in selected:
            raise ContractError(f"{label}: duplicate source id {source_id}")
        selected[source_id] = raw_source
    return selected


def _curated_source_registry(
    *,
    official_path: Path,
    staging_path: Path,
    output_path: Path,
    target_tasks: tuple[object, ...],
) -> dict[str, object]:
    official = _load_json_object(official_path)
    staging = _load_json_object(staging_path)
    if official.get("version") != staging.get("version"):
        raise ContractError("source registry: version mismatch")
    official_by_id = _source_map(official, label="official source registry")
    staging_by_id = _source_map(staging, label="staging source registry")
    if not set(official_by_id).issubset(staging_by_id):
        raise ContractError("staging source registry: official source missing")
    for source_id, official_source in official_by_id.items():
        if _semantic_source(official_source) != _semantic_source(staging_by_id[source_id]):
            raise ContractError(f"staging source registry: official source drift {source_id}")

    task_source_ids: list[str] = []
    task_source_paths: dict[str, Path] = {}
    for record in target_tasks:
        task_path = ROOT / str(record.task_path) / "task.json"
        task = TaskManifest.load(task_path)
        source_ref = task.data.get("source_ref")
        if not isinstance(source_ref, str):
            raise ContractError(f"{task.task_id}: source_ref is missing")
        source_snapshot_path = task.source_snapshot_path
        if source_snapshot_path is None:
            raise ContractError(f"{task.task_id}: source snapshot is missing")
        task_source_ids.append(source_ref)
        task_source_paths[source_ref] = source_snapshot_path.resolve()
    if len(task_source_ids) != len(set(task_source_ids)):
        raise ContractError("target accepted index: source refs must be unique")
    missing = sorted(set(task_source_ids) - set(staging_by_id))
    if missing:
        raise ContractError("staging source registry: target source missing")

    selected_ids = set(official_by_id) | set(task_source_ids)
    expected_count = len(official_by_id) + len(task_source_ids)
    if len(selected_ids) != expected_count:
        raise ContractError("target source ref unexpectedly overlaps the official registry")
    if expected_count != 72:
        raise ContractError("curated source registry: expected exact 43+29 composition")

    sources: list[dict[str, object]] = []
    for raw_source in staging.get("sources", []):
        assert isinstance(raw_source, dict)
        source_id = raw_source.get("id")
        if source_id not in selected_ids:
            continue
        source = dict(raw_source)
        if source_id in task_source_paths:
            absolute = task_source_paths[str(source_id)]
        else:
            official_local_path = official_by_id[str(source_id)].get("local_path")
            if not isinstance(official_local_path, str):
                raise ContractError(
                    f"official source registry: local_path missing for {source_id}"
                )
            selected_path = Path(official_local_path)
            absolute = (
                selected_path.resolve()
                if selected_path.is_absolute()
                else (official_path.parent / selected_path).resolve()
            )
        if not absolute.is_dir() or absolute.is_symlink():
            raise ContractError(f"staging source registry: unavailable source {source_id}")
        source["local_path"] = Path(
            os.path.relpath(absolute, output_path.parent.resolve())
        ).as_posix()
        sources.append(source)
    if len(sources) != expected_count:
        raise ContractError("curated source registry: incomplete source selection")
    return {"version": staging["version"], "sources": sources}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-accepted", required=True)
    parser.add_argument("--runtime-results", action="append", required=True)
    parser.add_argument("--official-source-registry", default="sources/registry.json")
    parser.add_argument("--staging-source-registry", required=True)
    parser.add_argument("--environment-registry", default="environments/registry.json")
    parser.add_argument("--output-source-registry", required=True)
    parser.add_argument("--output-accepted", required=True)
    parser.add_argument("--output-results", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--confirm-equivalent-review",
        action="store_true",
        help="Assert that current Prompt reviews cover unchanged Runtime tasks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.confirm_equivalent_review:
        raise SystemExit("--confirm-equivalent-review is required")

    target_path = _rooted(args.target_accepted)
    official_source_path = _rooted(args.official_source_registry)
    staging_source_path = _rooted(args.staging_source_registry)
    environment_path = _rooted(args.environment_registry)
    output_source_path = _rooted(args.output_source_registry)
    output_accepted_path = _rooted(args.output_accepted)
    output_results_path = _rooted(args.output_results)
    for output in (output_source_path, output_accepted_path, output_results_path):
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")

    target = load_quality_accepted_task_index(ROOT, target_path, require_complete=False)
    if target.task_count != 29 or len(target.tasks) != 29:
        raise ContractError("target accepted index: expected exactly 29 tasks")
    targets = {record.task_id: record for record in target.tasks}
    if len(targets) != 29:
        raise ContractError("target accepted index: duplicate task id")

    runtime_by_task: dict[str, object] = {}
    for raw_path in args.runtime_results:
        runtime_path = _rooted(raw_path)
        runtime = QualityAdmissionResultIndex.from_dict(
            load_canonical_json_artifact(runtime_path)
        )
        if runtime.task_count != runtime.verified_count or not all(
            result.verified for result in runtime.results
        ):
            raise ContractError(f"{runtime_path}: shard is not fully verified")
        for result in runtime.results:
            if result.task_id in runtime_by_task:
                raise ContractError(f"duplicate Runtime outcome: {result.task_id}")
            runtime_by_task[result.task_id] = result
    if set(runtime_by_task) != set(targets):
        missing = sorted(set(targets) - set(runtime_by_task))
        extra = sorted(set(runtime_by_task) - set(targets))
        raise ContractError(
            f"Runtime coverage mismatch: missing={len(missing)} extra={len(extra)}"
        )

    curated_registry = _curated_source_registry(
        official_path=official_source_path,
        staging_path=staging_source_path,
        output_path=output_source_path,
        target_tasks=target.tasks,
    )
    for output in (output_source_path, output_accepted_path, output_results_path):
        output.parent.mkdir(parents=True, exist_ok=True)
    output_source_path.write_bytes(canonical_json(curated_registry).encode("utf-8"))

    accepted_records = []
    rebound_results = []
    for original_authorized in target.tasks:
        authorized = original_authorized
        old = runtime_by_task[authorized.task_id]
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
            environment_registry_path=environment_path,
            source_registry_path=output_source_path,
        )
        prompt = validate_quality_admission_prompt(
            task,
            expected_source_hash=authorized.prompt_source_hash,
        )
        stable_path = task.task_dir / "admission/evidence.json"
        if quality_bytes_hash(load_regular_file_bytes(stable_path)) != old.admission_evidence_hash:
            raise ContractError(f"{old.task_id}: stable Admission evidence drift")
        _rebind_readmission(task, stable_path)
        admission = task.data.get("admission")
        if not isinstance(admission, dict):
            raise ContractError(f"{old.task_id}: admission metadata is missing")
        if old.admission_verified and admission.get("status") != "verified":
            stable = _load_json_object(stable_path)
            verified_at = stable.get("created_at")
            if not isinstance(verified_at, str):
                raise ContractError(f"{old.task_id}: verified_at is missing")
            _promote_task_admission(task, verified_at=verified_at)
            task = load_resolved_task(
                ROOT / authorized.task_path / "task.json",
                environment_registry_path=environment_path,
                source_registry_path=output_source_path,
            )
            manifest = load_canonical_json_artifact(task.task_json_path)
            authorized = replace(
                authorized,
                task_manifest_hash=canonical_sha256(manifest),
                replay_spec_hash=replay_spec_hash(task),
                prompt_source_hash=quality_prompt_source_hash(task),
            )
        formal_errors = validate_quality_task(
            ROOT,
            task,
            require_verified=True,
            environment_registry_path=environment_path,
            source_registry_path=output_source_path,
        )
        if formal_errors:
            raise ContractError(
                f"{old.task_id}: formal quality failed with {len(formal_errors)} errors"
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

    final_accepted = replace(target, tasks=tuple(accepted_records))
    output_accepted_path.write_bytes(
        canonical_json(final_accepted.to_dict()).encode("utf-8")
    )
    rebound_results.sort(key=lambda result: result.screening_record_index)
    results = QualityAdmissionResultIndex(
        created_at=args.created_at,
        accepted_index_path=_relative(output_accepted_path),
        accepted_index_hash=final_accepted.content_hash,
        environment_registry_path=_relative(environment_path),
        environment_registry_hash=quality_bytes_hash(
            load_regular_file_bytes(environment_path)
        ),
        source_registry_path=_relative(output_source_path),
        source_registry_hash=quality_bytes_hash(
            load_regular_file_bytes(output_source_path)
        ),
        task_count=29,
        verified_count=29,
        results=tuple(rebound_results),
    )
    output_results_path.write_bytes(canonical_json(results.to_dict()).encode("utf-8"))
    loaded = load_quality_admission_result_index(
        ROOT,
        output_results_path,
        output_accepted_path,
        require_private_bundles=True,
    )
    print(
        canonical_json(
            {
                "accepted_index_hash": final_accepted.content_hash,
                "result_index_hash": loaded.content_hash,
                "source_count": len(curated_registry["sources"]),
                "task_count": loaded.task_count,
                "verified_count": loaded.verified_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
