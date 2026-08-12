#!/usr/bin/env python3
"""Refresh exact source bindings before repeating v0.7 Codex reviews."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import load_canonical_json_artifact  # noqa: E402
from op_bench.factory.prompt_quality import scan_rendered_prompt  # noqa: E402
from op_bench.factory.quality_release import (  # noqa: E402
    quality_bytes_hash,
    quality_prompt_source_hash,
    quality_prompt_source_inputs,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


DEFAULT_PROMPT_PACKET = (
    "runs/v0.7_quality_admission_staging/prompt_review_packet.json"
)
DEFAULT_REASSESSMENT_PACKET = (
    "runs/v0.7_quality_admission_staging/reassessment_review_packet.json"
)


def _rooted(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else ROOT / selected


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _records(packet: dict[str, object], path: Path) -> list[object]:
    records = packet.get("records")
    if not isinstance(records, list):
        raise ContractError(f"{path}: records must be a list")
    return records


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value).encode("utf-8"))


def _private_payload(private_index: object) -> dict[str, object]:
    return {
        "changed_paths": list(private_index.changed_paths),
        "added_symbols": list(private_index.added_symbols),
        "distinctive_literals": list(private_index.distinctive_literals),
        "hidden_selectors": list(private_index.hidden_selectors),
        "internal_names": list(private_index.internal_names),
        "scanner_version": private_index.scanner_version,
    }


def refresh(prompt_path: Path, reassessment_path: Path) -> tuple[int, int]:
    prompt_packet = _load(prompt_path)
    task_bindings: dict[str, dict[str, str]] = {}
    verified_task_ids: set[str] = set()
    prompt_records = _records(prompt_packet, prompt_path)
    for position, value in enumerate(prompt_records):
        if not isinstance(value, dict):
            raise ContractError(f"{prompt_path}: records[{position}]")
        record = dict(value)
        task_path = record.get("task_path")
        scanner_version = record.get("scanner_version")
        if not isinstance(task_path, str) or not isinstance(
            scanner_version, str
        ):
            raise ContractError(f"{prompt_path}: records[{position}] binding")
        manifest_path = ROOT / task_path / "task.json"
        manifest = load_canonical_json_artifact(manifest_path)
        task = TaskManifest.load(manifest_path)
        if task.admission_status == "verified":
            verified_task_ids.add(task.task_id)
        view, private_index = quality_prompt_source_inputs(
            task, scanner_version=scanner_version
        )
        rendered = render_mcp_prompt(view)
        findings = scan_rendered_prompt(
            rendered,
            private_index,
        )
        quality = task.data.get("quality")
        artifacts = task.data.get("artifacts")
        if not isinstance(quality, dict) or not isinstance(artifacts, dict):
            raise ContractError(f"{task.task_id}: quality/artifacts")
        complexity_path = task.task_dir / str(quality["complexity_evidence"])
        complexity = load_canonical_json_artifact(complexity_path)
        gold_path = task.task_dir / str(artifacts["gold_patch"])
        hidden_path = task.task_dir / str(
            artifacts.get("hidden_test_patch", artifacts.get("test_patch"))
        )
        private_payload = _private_payload(private_index)
        bindings = {
            "task_manifest_hash": canonical_sha256(manifest),
            "gold_patch_hash": quality_bytes_hash(gold_path.read_bytes()),
            "hidden_test_patch_hash": quality_bytes_hash(hidden_path.read_bytes()),
            "complexity_hash": str(complexity.get("content_hash")),
        }
        task_bindings[task.task_id] = bindings
        record["agent_task_view"] = view
        record["agent_task_view_hash"] = canonical_sha256(view)
        record["rendered_prompt_hash"] = canonical_sha256(rendered)
        record["prompt_source_hash"] = quality_prompt_source_hash(
            task, scanner_version=scanner_version
        )
        record["private_answer_index_hash"] = canonical_sha256(private_payload)
        record["private_answer_index_counts"] = {
            key: len(value)
            for key, value in private_payload.items()
            if key != "scanner_version"
        }
        record["automated_findings"] = [
            finding.to_dict() for finding in findings
        ]
        record["automated_scan"] = "passed" if not findings else "failed"
        semantic = dict(record.get("semantic_review_inputs", {}))
        semantic.update(bindings)
        record["semantic_review_inputs"] = semantic
        prompt_records[position] = record
    prompt_packet["records"] = prompt_records
    prompt_packet["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in prompt_packet.items()
            if key != "content_hash"
        }
    )
    _write(prompt_path, prompt_packet)

    reassessment_packet = _load(reassessment_path)
    reassessment_records = _records(reassessment_packet, reassessment_path)
    for position, value in enumerate(reassessment_records):
        if not isinstance(value, dict):
            raise ContractError(f"{reassessment_path}: records[{position}]")
        record = dict(value)
        task_id = record.get("task_id")
        bindings = task_bindings.get(str(task_id))
        if bindings is None:
            raise ContractError(f"{task_id}: Prompt binding is unavailable")
        private_inputs = dict(record.get("private_review_inputs", {}))
        private_inputs.update(
            {
                "gold_patch_hash": bindings["gold_patch_hash"],
                "hidden_test_patch_hash": bindings["hidden_test_patch_hash"],
            }
        )
        if task_id not in verified_task_ids:
            private_inputs["task_manifest_hash"] = bindings[
                "task_manifest_hash"
            ]
        record["private_review_inputs"] = private_inputs
        complexity = dict(record.get("complexity", {}))
        complexity["content_hash"] = bindings["complexity_hash"]
        record["complexity"] = complexity
        reassessment_records[position] = record
    reassessment_packet["records"] = reassessment_records
    reassessment_packet["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in reassessment_packet.items()
            if key != "content_hash"
        }
    )
    _write(reassessment_path, reassessment_packet)
    return len(prompt_records), len(reassessment_records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-packet", default=DEFAULT_PROMPT_PACKET)
    parser.add_argument(
        "--reassessment-packet", default=DEFAULT_REASSESSMENT_PACKET
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts, reassessments = refresh(
        _rooted(args.prompt_packet), _rooted(args.reassessment_packet)
    )
    print(canonical_json({
        "prompt_count": prompts,
        "reassessment_count": reassessments,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
