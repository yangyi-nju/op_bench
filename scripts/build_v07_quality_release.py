#!/usr/bin/env python3
"""Build the final quality-first v0.7 50-Task release and derived slices."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.curation import summarize_verified_dataset  # noqa: E402
from op_bench.factory.artifacts import load_regular_file_bytes  # noqa: E402
from op_bench.factory.quality_admission import (  # noqa: E402
    load_quality_accepted_task_index,
    load_quality_admission_result_index,
)
from op_bench.factory.quality_release import (  # noqa: E402
    validate_historical_index,
    validate_quality_task,
)
from op_bench.factory.taxonomy import (  # noqa: E402
    TaskTaxonomyV2,
    derived_slices,
    parse_taxonomy_v2,
)
from op_bench.integrity import replay_spec_hash  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402
from scripts.validate_dataset import validate_dataset  # noqa: E402


DATASET_PATHS = {
    "cumulative": "datasets/pytorch_v0.7/dataset.json",
    "boundary": "datasets/pytorch_v0.7_boundary/dataset.json",
    "precision": "datasets/pytorch_v0.7_precision/dataset.json",
    "device": "datasets/pytorch_v0.7_device/dataset.json",
}
SUMMARY_PATHS = {
    role: path.replace("dataset.json", "summary.json")
    for role, path in DATASET_PATHS.items()
}
DATASET_IDS = {
    "cumulative": "pytorch_v0.7",
    "boundary": "pytorch_v0.7_boundary",
    "precision": "pytorch_v0.7_precision",
    "device": "pytorch_v0.7_device",
}
RELEASE_MANIFEST_PATH = "factory/v0.7/p9/release_manifest.json"
COVERAGE_PATH = "factory/v0.7/p9/coverage_matrix.json"
PRE_QUALITY_ARCHIVE_PATHS = (
    "datasets/pytorch_v0.7/dataset.json",
    "datasets/pytorch_v0.7/summary.json",
    "datasets/pytorch_v0.7_boundary/dataset.json",
    "datasets/pytorch_v0.7_boundary/summary.json",
    "datasets/pytorch_v0.7_precision/dataset.json",
    "datasets/pytorch_v0.7_precision/summary.json",
    "factory/v0.7/p4/release_request.json",
    "factory/v0.7/p4/release_manifest.json",
    "factory/v0.7/p4/validation_contract.json",
    "factory/v0.7/p4/boundary_freeze_request.json",
    "factory/v0.7/p4/boundary_freeze/freeze_manifest.json",
    "factory/v0.7/p4/boundary_freeze/dataset.json",
)


def _rooted(root: Path, value: Path | str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else root / selected


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"path is outside repository: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(load_regular_file_bytes(path))


def _encoded(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _with_content_hash(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _reference(root: Path, path: Path, *, artifact_type: str, artifact_id: str) -> dict[str, str]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "relative_path": _relative(root, path),
        "content_hash": _sha256_file(path),
    }


def _output_reference(
    *, path: str, content: bytes, artifact_type: str, artifact_id: str
) -> dict[str, str]:
    return {
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "relative_path": path,
        "content_hash": _sha256_bytes(content),
    }


def _quality_path(task: TaskManifest, field: str) -> Path:
    quality = task.data.get("quality")
    if not isinstance(quality, dict) or not isinstance(quality.get(field), str):
        raise ContractError(f"{task.task_id}: quality.{field} is missing")
    path = task.task_dir / str(quality[field])
    if not path.is_file() or path.is_symlink():
        raise ContractError(f"{task.task_id}: quality.{field} is unavailable")
    return path


def _counter(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _taxonomy_record(taxonomy: TaskTaxonomyV2) -> dict[str, object]:
    return {
        "contract_family": taxonomy.contract_family,
        "failure_type": taxonomy.failure_type,
        "devices": list(taxonomy.execution_context.devices),
        "modes": list(taxonomy.execution_context.modes),
        "phases": list(taxonomy.execution_context.phases),
        "distributed": taxonomy.execution_context.distributed,
        "trigger_tags": list(taxonomy.trigger_tags),
    }


def _task_release_record(
    root: Path,
    task_path: str,
    *,
    origin: str,
    environment_registry_path: Path,
    source_registry_path: Path,
    expected_public_task_id: str,
    expected_admission_hash: str | None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    task = TaskManifest.load(root / task_path / "task.json")
    if task.public_task_id != expected_public_task_id:
        raise ContractError(f"{task.task_id}: public Task identity drift")
    formal_errors = validate_quality_task(
        root,
        task,
        require_verified=True,
        environment_registry_path=environment_registry_path,
        source_registry_path=source_registry_path,
    )
    if formal_errors:
        raise ContractError(
            f"{task.task_id}: formal quality failed with {len(formal_errors)} errors"
        )
    if task.admission_status != "verified":
        raise ContractError(f"{task.task_id}: verified admission is required")

    taxonomy_value = task.data.get("taxonomy")
    taxonomy = parse_taxonomy_v2(taxonomy_value)
    slices = ("cumulative", *derived_slices(taxonomy))
    complexity_path = _quality_path(task, "complexity_evidence")
    complexity = _load_json(complexity_path)
    difficulty = complexity.get("difficulty")
    if difficulty not in ("medium", "hard"):
        raise ContractError(f"{task.task_id}: medium or hard difficulty is required")

    admission_path = task.task_dir / "admission/evidence.json"
    admission = _load_json(admission_path)
    admission_state = admission.get("admission")
    if (
        admission.get("task_id") != task.task_id
        or not isinstance(admission_state, dict)
        or admission_state.get("decision") != "verified"
        or admission_state.get("verified") is not True
    ):
        raise ContractError(f"{task.task_id}: stable Admission evidence is not verified")
    admission_hash = _sha256_file(admission_path)
    if expected_admission_hash is not None and admission_hash != expected_admission_hash:
        raise ContractError(f"{task.task_id}: Admission evidence hash drift")

    prompt_path = _quality_path(task, "prompt_evidence")
    readmission_path = _quality_path(task, "readmission_evidence")
    task_manifest = _load_json(task.task_json_path)
    task_entry = {
        "task_id": task.task_id,
        "task_path": task_path,
        "admission_status": "verified",
        "admission_evidence": _relative(root, admission_path),
        "runtime_tier": task.runtime_tier,
        "environment_status": "ready",
        "source_status": "ready",
        "replay_status": "verified",
    }
    request_record = {
        "task_id": task.task_id,
        "public_task_id": task.public_task_id,
        "task_path": task_path,
        "origin": origin,
        "difficulty": difficulty,
        "slices": list(slices),
        "task_manifest_hash": canonical_sha256(task_manifest),
        "replay_spec_hash": replay_spec_hash(task),
        "admission_evidence_hash": admission_hash,
        "prompt_evidence_hash": _sha256_file(prompt_path),
        "complexity_evidence_hash": _sha256_file(complexity_path),
        "readmission_evidence_hash": _sha256_file(readmission_path),
        "taxonomy_hash": canonical_sha256(taxonomy_value),
        "taxonomy": _taxonomy_record(taxonomy),
    }
    operator = task.data.get("operator")
    operator_data = operator if isinstance(operator, dict) else {}
    metadata = {
        "origin": origin,
        "difficulty": str(difficulty),
        "problem_dimension": str(operator_data.get("problem_dimension", "unclassified")),
        "problem_subclass": str(operator_data.get("problem_subclass", "unclassified")),
        "failure_contract": str(operator_data.get("failure_contract", "unclassified")),
        "contract_family": taxonomy.contract_family,
        "failure_type": taxonomy.failure_type,
        "devices": taxonomy.execution_context.devices,
        "modes": taxonomy.execution_context.modes,
        "phases": taxonomy.execution_context.phases,
        "distributed": taxonomy.execution_context.distributed,
        "trigger_tags": taxonomy.trigger_tags,
        "slices": slices,
    }
    return task_entry, request_record, metadata


def _summary(
    dataset: dict[str, object],
    dataset_bytes: bytes,
    task_metadata: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    selected_ids = [str(item["task_id"]) for item in dataset["tasks"]]  # type: ignore[index]
    selected = {task_id: task_metadata[task_id] for task_id in selected_ids}
    summary = summarize_verified_dataset(
        dataset,
        dataset_hash=_sha256_bytes(dataset_bytes),
        task_metadata=selected,
    )
    summary.update(
        {
            "origin": _counter(str(value["origin"]) for value in selected.values()),
            "difficulty": _counter(str(value["difficulty"]) for value in selected.values()),
            "contract_family": _counter(
                str(value["contract_family"]) for value in selected.values()
            ),
            "failure_type": _counter(
                str(value["failure_type"]) for value in selected.values()
            ),
            "devices": _counter(
                str(device)
                for value in selected.values()
                for device in value["devices"]  # type: ignore[union-attr]
            ),
            "modes": _counter(
                str(mode)
                for value in selected.values()
                for mode in value["modes"]  # type: ignore[union-attr]
            ),
            "phases": _counter(
                str(phase)
                for value in selected.values()
                for phase in value["phases"]  # type: ignore[union-attr]
            ),
        }
    )
    return summary


def _archive_pre_quality_release(root: Path) -> None:
    current_dataset = _load_json(root / DATASET_PATHS["cumulative"])
    archive_dataset = (
        root / "archives/v0.7-pre-quality/datasets/pytorch_v0.7/dataset.json"
    )
    current_count = len(current_dataset.get("tasks", []))
    if current_count == 50:
        archived = _load_json(archive_dataset)
        if len(archived.get("tasks", [])) != 25:
            raise ContractError("pre-quality archive: expected immutable 25-Task Dataset")
        for relative in PRE_QUALITY_ARCHIVE_PATHS:
            destination = root / "archives/v0.7-pre-quality" / relative
            if relative.startswith("datasets/"):
                if not destination.is_file() or destination.is_symlink():
                    raise ContractError(
                        f"pre-quality archive: missing historical {relative}"
                    )
                continue
            source = root / relative
            if not source.is_file() or source.is_symlink():
                raise ContractError(f"pre-quality archive: missing {relative}")
            content = load_regular_file_bytes(source)
            if destination.exists():
                if destination.is_symlink() or load_regular_file_bytes(destination) != content:
                    raise ContractError(f"pre-quality archive: collision for {relative}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return
    if current_count != 25:
        raise ContractError("pre-quality archive: expected current 25-Task historical Dataset")
    for relative in PRE_QUALITY_ARCHIVE_PATHS:
        source = root / relative
        if not source.is_file() or source.is_symlink():
            raise ContractError(f"pre-quality archive: missing {relative}")
        destination = root / "archives/v0.7-pre-quality" / relative
        content = load_regular_file_bytes(source)
        if destination.exists():
            if destination.is_symlink() or load_regular_file_bytes(destination) != content:
                raise ContractError(f"pre-quality archive: collision for {relative}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)


def build_release_outputs(
    *,
    root: Path,
    historical_index_path: Path,
    accepted_index_path: Path,
    admission_results_path: Path,
    request_path: Path,
    created_at: str,
) -> dict[str, bytes]:
    historical_errors = validate_historical_index(root, historical_index_path)
    if historical_errors:
        raise ContractError(
            f"historical index failed with {len(historical_errors)} errors"
        )
    historical = _load_json(historical_index_path)
    retained = [
        record
        for record in historical.get("records", [])
        if isinstance(record, dict) and record.get("disposition") == "retained"
    ]
    if len(retained) != 14 or historical.get("k") != 14:
        raise ContractError("historical index: expected exactly 14 retained Tasks")

    accepted = load_quality_accepted_task_index(
        root, accepted_index_path, require_complete=True
    )
    results = load_quality_admission_result_index(
        root,
        admission_results_path,
        accepted_index_path,
        require_private_bundles=False,
    )
    if (
        accepted.task_count != 36
        or results.task_count != 36
        or results.verified_count != 36
        or not all(result.verified for result in results.results)
    ):
        raise ContractError("quality Admission: expected exactly 36 verified Tasks")
    result_by_task = {result.task_id: result for result in results.results}

    environment_registry_path = root / "environments/registry.json"
    source_registry_path = root / "sources/registry.json"
    entries: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    metadata: dict[str, Mapping[str, object]] = {}
    for retained_record in retained:
        task_id = str(retained_record["task_id"])
        entry, record, task_metadata = _task_release_record(
            root,
            str(retained_record["task_path"]),
            origin="retained_historical",
            environment_registry_path=environment_registry_path,
            source_registry_path=source_registry_path,
            expected_public_task_id=str(retained_record["public_task_id"]),
            expected_admission_hash=None,
        )
        if entry["task_id"] != task_id:
            raise ContractError(f"{task_id}: historical Task identity drift")
        entries.append(entry)
        records.append(record)
        metadata[task_id] = task_metadata
    for accepted_record in accepted.tasks:
        outcome = result_by_task.get(accepted_record.task_id)
        if outcome is None or not outcome.verified:
            raise ContractError(f"{accepted_record.task_id}: verified outcome missing")
        entry, record, task_metadata = _task_release_record(
            root,
            accepted_record.task_path,
            origin=accepted_record.origin,
            environment_registry_path=environment_registry_path,
            source_registry_path=source_registry_path,
            expected_public_task_id=accepted_record.public_task_id,
            expected_admission_hash=outcome.admission_evidence_hash,
        )
        entries.append(entry)
        records.append(record)
        metadata[accepted_record.task_id] = task_metadata

    entries.sort(key=lambda item: str(item["task_id"]))
    records.sort(key=lambda item: str(item["task_id"]))
    task_ids = [str(entry["task_id"]) for entry in entries]
    public_ids = [str(record["public_task_id"]) for record in records]
    if len(entries) != 50 or len(set(task_ids)) != 50 or len(set(public_ids)) != 50:
        raise ContractError("final release: expected exactly 50 unique private/public Tasks")
    origin_counts = _counter(str(record["origin"]) for record in records)
    if (
        origin_counts.get("retained_historical") != 14
        or origin_counts.get("new", 0) + origin_counts.get("replacement", 0) != 36
        or set(origin_counts) - {"retained_historical", "new", "replacement"}
    ):
        raise ContractError("final release: expected exact 14 retained + 36 new/replacement")

    outputs: dict[str, bytes] = {}
    datasets: dict[str, dict[str, object]] = {}
    for role in ("cumulative", "boundary", "precision", "device"):
        selected = [
            entry
            for entry in entries
            if role == "cumulative" or role in metadata[str(entry["task_id"])]["slices"]  # type: ignore[operator]
        ]
        if not selected:
            raise ContractError(f"final release: derived {role} slice is empty")
        dataset = {
            "dataset_id": DATASET_IDS[role],
            "version": "v0.7",
            "status": "verified",
            "registries": {
                "environments": "environments/registry.json",
                "sources": "sources/registry.json",
            },
            "tasks": selected,
        }
        validation_errors = validate_dataset(
            dataset,
            (root / DATASET_PATHS[role]).parent,
            require_verified=True,
        )
        if validation_errors:
            raise ContractError(
                f"{role} Dataset failed with {len(validation_errors)} errors: "
                f"{validation_errors[0]}"
            )
        dataset_bytes = _encoded(dataset)
        summary = _summary(dataset, dataset_bytes, metadata)
        outputs[DATASET_PATHS[role]] = dataset_bytes
        outputs[SUMMARY_PATHS[role]] = _encoded(summary)
        datasets[role] = dataset

    coverage_records = [
        {
            "task_id": record["task_id"],
            "public_task_id": record["public_task_id"],
            "origin": record["origin"],
            "difficulty": record["difficulty"],
            "slices": record["slices"],
            **record["taxonomy"],  # type: ignore[arg-type]
        }
        for record in records
    ]
    all_devices = [device for record in coverage_records for device in record["devices"]]  # type: ignore[union-attr]
    all_modes = [mode for record in coverage_records for mode in record["modes"]]  # type: ignore[union-attr]
    all_phases = [phase for record in coverage_records for phase in record["phases"]]  # type: ignore[union-attr]
    gates = {
        "exact_total_50": len(coverage_records) == 50,
        "retained_14": sum(record["origin"] == "retained_historical" for record in coverage_records) == 14,
        "new_or_replacement_36": sum(record["origin"] != "retained_historical" for record in coverage_records) == 36,
        "cpu_present": "cpu" in all_devices,
        "cuda_present": "cuda" in all_devices,
        "compile_present": "compile" in all_modes,
        "backward_present": "backward" in all_phases,
        "no_easy_tasks": all(record["difficulty"] in ("medium", "hard") for record in coverage_records),
    }
    if not all(gates.values()):
        raise ContractError("coverage matrix: one or more hard gates failed")
    coverage = _with_content_hash(
        {
            "contract_type": "quality_coverage_matrix",
            "schema_version": "v1",
            "release_version": "v0.7",
            "created_at": created_at,
            "task_count": 50,
            "counts": {
                "origin": _counter(str(record["origin"]) for record in coverage_records),
                "difficulty": _counter(str(record["difficulty"]) for record in coverage_records),
                "contract_family": _counter(str(record["contract_family"]) for record in coverage_records),
                "failure_type": _counter(str(record["failure_type"]) for record in coverage_records),
                "devices": _counter(str(value) for value in all_devices),
                "modes": _counter(str(value) for value in all_modes),
                "phases": _counter(str(value) for value in all_phases),
                "derived_slices": {
                    role: len(datasets[role]["tasks"]) for role in ("boundary", "precision", "device")
                },
            },
            "hard_gates": gates,
            "records": coverage_records,
        }
    )
    coverage_bytes = _encoded(coverage)
    outputs[COVERAGE_PATH] = coverage_bytes

    request = _with_content_hash(
        {
            "contract_type": "quality_release_request",
            "schema_version": "v1",
            "release_version": "v0.7",
            "created_at": created_at,
            "composition": {
                "retained_historical": 14,
                "new_or_replacement": 36,
                "total": 50,
            },
            "inputs": {
                "historical_index": _reference(
                    root,
                    historical_index_path,
                    artifact_type="historical_readmission_index",
                    artifact_id="historical-readmission:v1",
                ),
                "accepted_index": _reference(
                    root,
                    accepted_index_path,
                    artifact_type="quality_accepted_task_index",
                    artifact_id="quality-accepted:v1",
                ),
                "admission_results": _reference(
                    root,
                    admission_results_path,
                    artifact_type="quality_admission_result_index",
                    artifact_id="quality-admission:v1",
                ),
            },
            "registries": {
                "environments": _reference(
                    root,
                    environment_registry_path,
                    artifact_type="environment_registry",
                    artifact_id="registry:environments:v1",
                ),
                "sources": _reference(
                    root,
                    source_registry_path,
                    artifact_type="source_registry",
                    artifact_id="registry:sources:v1",
                ),
            },
            "output_paths": {
                "release_manifest": RELEASE_MANIFEST_PATH,
                "coverage_matrix": COVERAGE_PATH,
                "datasets": {
                    role: {
                        "dataset": DATASET_PATHS[role],
                        "summary": SUMMARY_PATHS[role],
                    }
                    for role in DATASET_PATHS
                },
            },
            "records": records,
        }
    )
    request_bytes = _encoded(request)
    request_relative = _relative(root, request_path)
    outputs[request_relative] = request_bytes

    manifest_core = {
        "contract_type": "quality_release_manifest",
        "schema_version": "v1",
        "release_version": "v0.7",
        "created_at": created_at,
        "composition": request["composition"],
        "request": _output_reference(
            path=request_relative,
            content=request_bytes,
            artifact_type="quality_release_request",
            artifact_id="quality-release-request:v1",
        ),
        "coverage_matrix": _output_reference(
            path=COVERAGE_PATH,
            content=coverage_bytes,
            artifact_type="quality_coverage_matrix",
            artifact_id="quality-coverage:v1",
        ),
        "datasets": {
            role: {
                "task_count": len(datasets[role]["tasks"]),
                "dataset": _output_reference(
                    path=DATASET_PATHS[role],
                    content=outputs[DATASET_PATHS[role]],
                    artifact_type="dataset_manifest",
                    artifact_id=f"dataset:{DATASET_IDS[role]}",
                ),
                "summary": _output_reference(
                    path=SUMMARY_PATHS[role],
                    content=outputs[SUMMARY_PATHS[role]],
                    artifact_type="dataset_summary",
                    artifact_id=f"summary:{DATASET_IDS[role]}",
                ),
            }
            for role in DATASET_PATHS
        },
        "historical_archive": {
            "dataset": "archives/v0.7-pre-quality/datasets/pytorch_v0.7/dataset.json",
            "expected_task_count": 25,
        },
    }
    release_identity = canonical_sha256(manifest_core)
    manifest = _with_content_hash(
        {
            **manifest_core,
            "release_id": "quality-release:v1:" + release_identity.removeprefix("sha256:"),
        }
    )
    outputs[RELEASE_MANIFEST_PATH] = _encoded(manifest)
    return outputs


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--historical-index",
        type=Path,
        default=Path("factory/v0.7/p7/historical_readmission.json"),
    )
    parser.add_argument(
        "--accepted-index",
        type=Path,
        default=Path("factory/v0.7/p8/accepted_tasks.json"),
    )
    parser.add_argument(
        "--admission-results",
        type=Path,
        default=Path("factory/v0.7/p8/admission_results.json"),
    )
    parser.add_argument(
        "--request",
        type=Path,
        default=Path("factory/v0.7/p9/release_request.json"),
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    output_root = (args.output_root or root).resolve()
    historical_path = _rooted(root, args.historical_index)
    accepted_path = _rooted(root, args.accepted_index)
    results_path = _rooted(root, args.admission_results)
    request_path = _rooted(root, args.request)
    try:
        outputs = build_release_outputs(
            root=root,
            historical_index_path=historical_path,
            accepted_index_path=accepted_path,
            admission_results_path=results_path,
            request_path=request_path,
            created_at=args.created_at,
        )
        if args.verify_existing:
            mismatches = [
                relative
                for relative, content in outputs.items()
                if not (output_root / relative).is_file()
                or load_regular_file_bytes(output_root / relative) != content
            ]
            if mismatches:
                raise ContractError(
                    f"verify-existing: {len(mismatches)} generated artifacts differ"
                )
            status = "verified"
        else:
            if output_root == root:
                _archive_pre_quality_release(root)
            for relative, content in outputs.items():
                _write_atomic(output_root / relative, content)
            status = "written"
    except (ContractError, OSError, ValueError) as exc:
        print(f"quality release failed: {exc}", file=sys.stderr)
        return 1
    print(
        canonical_json(
            {
                "artifact_count": len(outputs),
                "release_manifest": RELEASE_MANIFEST_PATH,
                "status": status,
                "task_count": 50,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
