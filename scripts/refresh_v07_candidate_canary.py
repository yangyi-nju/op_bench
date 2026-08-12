#!/usr/bin/env python3
"""Refresh a pre-Admission v0.7 candidate canary against current Task bytes."""

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
from op_bench.factory.contracts import FactoryArtifactReference  # noqa: E402
from op_bench.factory.quality_admission import (  # noqa: E402
    QualityAcceptedTaskIndex,
    QualityCandidateReassessment,
    load_quality_accepted_task_index,
)
from op_bench.factory.quality_release import quality_prompt_source_hash  # noqa: E402
from op_bench.integrity import replay_spec_hash  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--created-at", required=True)
    return parser


def _current_reassessment(
    reference: FactoryArtifactReference | None,
) -> FactoryArtifactReference | None:
    if reference is None:
        return None
    path = ROOT / reference.relative_path
    reassessment = QualityCandidateReassessment.from_dict(
        load_canonical_json_artifact(path),
        path="candidate_reassessment",
    )
    return FactoryArtifactReference(
        artifact_type=reassessment.contract_type,
        artifact_id=(
            "quality-reassessment:v1:"
            + reassessment.content_hash.removeprefix("sha256:")
        ),
        content_hash=reassessment.content_hash,
        relative_path=reference.relative_path,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source_path = _rooted(args.source)
    output_path = _rooted(args.output)
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    source = QualityAcceptedTaskIndex.from_dict(
        load_canonical_json_artifact(source_path),
        path="candidate_canary",
    )
    refreshed = []
    for record in source.tasks:
        manifest_path = ROOT / record.task_path / "task.json"
        manifest = load_canonical_json_artifact(manifest_path)
        task = TaskManifest.load(manifest_path)
        if task.task_id != record.task_id:
            raise ContractError(f"{record.task_id}: Task identity changed")
        refreshed.append(
            replace(
                record,
                reassessment=_current_reassessment(record.reassessment),
                task_manifest_hash=canonical_sha256(manifest),
                replay_spec_hash=replay_spec_hash(task),
                prompt_source_hash=quality_prompt_source_hash(task),
            )
        )
    index = replace(
        source,
        created_at=args.created_at,
        status="building",
        task_count=len(refreshed),
        tasks=tuple(refreshed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json(index.to_dict()).encode("utf-8"))
    try:
        loaded = load_quality_accepted_task_index(
            ROOT, output_path, require_complete=False
        )
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    print(
        canonical_json(
            {
                "content_hash": loaded.content_hash,
                "task_count": loaded.task_count,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
