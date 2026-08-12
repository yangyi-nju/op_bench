#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.contracts import (  # noqa: E402
    DatasetFreezeEntry,
    DatasetFreezeManifest,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
)
from op_bench.factory.release import VerifiedReleaseEntry  # noqa: E402
from op_bench.factory.screening import (  # noqa: E402
    V07_BOUNDARY_SCREENING_V1,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.task import TaskManifest  # noqa: E402


BOUNDARY_DIRECTORIES = (
    "117065_index_copy_zero_dim",
    "118762_weight_norm_default_dim",
    "126461_cummin_rank_zero",
    "139751_triton_ygrid_mask",
    "143792_addmv_empty_matrix",
    "147352_storage_offset_overflow",
)
RESTORED_PRECISION_DIRECTORIES = (
    "129154_exp_decomp_numerics",
    "144073_vector_norm_scalar_overflow",
)
BOUNDARY_FAILURE_CONTRACTS = {
    "pytorch__117065__index_copy_zero_dim": "exception",
    "pytorch__118762__weight_norm_default_dim": "exception",
    "pytorch__126461__cummin_rank_zero": "exception",
    "pytorch__139751__triton_ygrid_mask": "wrong-result",
    "pytorch__143792__addmv_empty_matrix": "wrong-result",
    "pytorch__147352__storage_offset_overflow": "silent-acceptance",
}
FREEZE_CREATED_AT = "2026-07-27T06:00:00Z"
RELEASE_CREATED_AT = "2026-07-27T06:10:00Z"


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected JSON object")
    return value


def _reference(
    root: Path,
    *,
    artifact_type: str,
    artifact_id: str,
    relative_path: str,
    content_hash: str | None = None,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash=(
            content_hash
            if content_hash is not None
            else _file_hash(root / relative_path)
        ),
        relative_path=relative_path,
    )


def _task_directory(
    root: Path,
    dataset_directory: Path,
    relative_path: str,
) -> str:
    resolved = (dataset_directory / relative_path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError("Task path escapes repository root") from exc


def _freeze_entry(
    root: Path,
    directory: str,
) -> tuple[DatasetFreezeEntry, FactoryAdmissionRecord]:
    task_directory = f"tasks/pytorch/{directory}"
    task = TaskManifest.load(root / task_directory / "task.json")
    admission = FactoryAdmissionRecord.from_dict(
        _load_json(root / task_directory / "factory/admission.json")
    )
    evidence = _load_json(
        root / task_directory / "admission/evidence.json"
    )
    if task.task_id not in BOUNDARY_FAILURE_CONTRACTS:
        raise ContractError(f"Missing failure contract for {task.task_id}")
    if task.problem_dimension != "boundary" or not task.problem_subclass:
        raise ContractError(f"{task.task_id}: Boundary taxonomy is required")
    if not task.source_ref or not task.environment_ref:
        raise ContractError(f"{task.task_id}: Registry references are required")

    admission_reference = _reference(
        root,
        artifact_type="factory_admission",
        artifact_id=admission.admission_id,
        relative_path=f"{task_directory}/factory/admission.json",
        content_hash=admission.content_hash,
    )
    evidence_reference = _reference(
        root,
        artifact_type="admission_evidence",
        artifact_id=str(evidence["evidence_id"]),
        relative_path=f"{task_directory}/admission/evidence.json",
    )
    registry_hashes = {
        "sources/registry.json": _file_hash(root / "sources/registry.json"),
        "environments/registry.json": _file_hash(
            root / "environments/registry.json"
        ),
    }
    entry = DatasetFreezeEntry(
        candidate=admission.candidate,
        decision=admission.decision,
        admission=admission_reference,
        task=admission.task,
        admission_evidence=evidence_reference,
        source=_reference(
            root,
            artifact_type="source",
            artifact_id=task.source_ref,
            relative_path="sources/registry.json",
            content_hash=registry_hashes["sources/registry.json"],
        ),
        environment=_reference(
            root,
            artifact_type="environment",
            artifact_id=task.environment_ref,
            relative_path="environments/registry.json",
            content_hash=registry_hashes["environments/registry.json"],
        ),
        task_id=task.task_id,
        task_path=task_directory,
        admission_evidence_path=evidence_reference.relative_path,
        runtime_tier=task.runtime_tier,
        problem_dimension=task.problem_dimension,
        problem_subclass=task.problem_subclass,
        failure_contract=BOUNDARY_FAILURE_CONTRACTS[task.task_id],
        admission_state="verified",
    )
    return entry, admission


def build_boundary_freeze_request(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    base_dataset = _reference(
        root,
        artifact_type="dataset_manifest",
        artifact_id="dataset:pytorch_v0.5",
        relative_path="datasets/pytorch_v0.5/dataset.json",
    )
    factory_protocol = _reference(
        root,
        artifact_type="factory_protocol",
        artifact_id="factory-protocol:v1",
        relative_path="src/op_bench/factory/contracts.py",
    )
    screening_rule_set = _reference(
        root,
        artifact_type="screening_rule_set",
        artifact_id=V07_BOUNDARY_SCREENING_V1.rule_set_id,
        relative_path="src/op_bench/factory/screening.py",
    )
    exclusion_index_data = _load_json(
        root / "factory/v0.7/p3/screening/screening_index.json"
    )
    exclusion_index = _reference(
        root,
        artifact_type="exclusion_index",
        artifact_id=(
            f"{exclusion_index_data['contract_type']}:"
            f"{exclusion_index_data['schema_version']}"
        ),
        relative_path="factory/v0.7/p3/screening/screening_index.json",
    )

    selected = tuple(
        _freeze_entry(root, directory)
        for directory in BOUNDARY_DIRECTORIES
    )
    entries = tuple(item[0] for item in selected)
    admissions = tuple(item[1] for item in selected)
    reference_hashes: dict[str, str] = {}
    for reference in (
        base_dataset,
        factory_protocol,
        screening_rule_set,
        exclusion_index,
        *(reference for entry in entries for reference in entry.references()),
    ):
        existing = reference_hashes.setdefault(
            reference.relative_path,
            reference.content_hash,
        )
        if existing != reference.content_hash:
            raise ContractError(
                f"Reference path has conflicting hashes: "
                f"{reference.relative_path}"
            )

    return {
        "dataset_id": "pytorch_v0.7_boundary_factory",
        "dataset_version": "v0.7",
        "base_dataset": base_dataset.to_dict(),
        "factory_protocol": factory_protocol.to_dict(),
        "screening_rule_set": screening_rule_set.to_dict(),
        "exclusion_index": exclusion_index.to_dict(),
        "registries": {
            "environments": "environments/registry.json",
            "sources": "sources/registry.json",
        },
        "entries": [entry.to_dict() for entry in entries],
        "admissions": [admission.to_dict() for admission in admissions],
        "reference_hashes": {
            key: reference_hashes[key] for key in sorted(reference_hashes)
        },
        "created_at": FREEZE_CREATED_AT,
    }


def _release_entry(
    root: Path,
    task_directory: str,
    *,
    origin: str,
    slices: tuple[str, ...],
    failure_contract: str,
) -> VerifiedReleaseEntry:
    task = TaskManifest.load(root / task_directory / "task.json")
    evidence_relative = f"{task_directory}/admission/evidence.json"
    evidence = _load_json(root / evidence_relative)
    dimension = task.problem_dimension or "unclassified"
    subclass = task.problem_subclass or "unclassified"
    return VerifiedReleaseEntry(
        task=_reference(
            root,
            artifact_type="task_bundle",
            artifact_id=f"task:{task.task_id}",
            relative_path=f"{task_directory}/task.json",
        ),
        admission_evidence=_reference(
            root,
            artifact_type="admission_evidence",
            artifact_id=str(evidence["evidence_id"]),
            relative_path=evidence_relative,
        ),
        task_id=task.task_id,
        task_path=task_directory,
        admission_evidence_path=evidence_relative,
        runtime_tier=task.runtime_tier,
        problem_dimension=dimension,
        problem_subclass=subclass,
        failure_contract=failure_contract,
        origin=origin,
        slices=slices,
        admission_state="verified",
    )


def build_release_request(root: Path = ROOT) -> dict[str, object]:
    root = root.resolve()
    cumulative_path = "datasets/pytorch_v0.5/dataset.json"
    precision_path = "datasets/pytorch_v0.5_precision/dataset.json"
    boundary_path = "factory/v0.7/p4/boundary_freeze/dataset.json"
    freeze_path = "factory/v0.7/p4/boundary_freeze/freeze_manifest.json"
    cumulative = _load_json(root / cumulative_path)
    precision = _load_json(root / precision_path)
    freeze = DatasetFreezeManifest.from_dict(_load_json(root / freeze_path))
    precision_ids = {
        str(entry["task_id"])
        for entry in precision["tasks"]
        if isinstance(entry, dict)
    }

    entries: list[VerifiedReleaseEntry] = []
    for item in cumulative["tasks"]:
        if not isinstance(item, dict):
            raise ContractError("cumulative tasks: expected object")
        task_id = str(item["task_id"])
        task_directory = _task_directory(
            root,
            (root / cumulative_path).parent,
            str(item["task_path"]),
        )
        if task_id in precision_ids:
            entries.append(
                _release_entry(
                    root,
                    task_directory,
                    origin="inherited_precision",
                    slices=("cumulative", "precision"),
                    failure_contract="unclassified",
                )
            )
        else:
            entries.append(
                _release_entry(
                    root,
                    task_directory,
                    origin="inherited_cumulative",
                    slices=("cumulative",),
                    failure_contract="unclassified",
                )
            )
    entries.extend(
        _release_entry(
            root,
            f"tasks/pytorch/{directory}",
            origin="restored_precision",
            slices=("cumulative", "precision"),
            failure_contract="unclassified",
        )
        for directory in RESTORED_PRECISION_DIRECTORIES
    )
    entries.extend(
        _release_entry(
            root,
            entry.task_path,
            origin="factory_boundary",
            slices=("cumulative", "boundary"),
            failure_contract=entry.failure_contract,
        )
        for entry in freeze.entries
    )

    return {
        "schema_version": "v1",
        "release_version": "v0.7",
        "created_at": RELEASE_CREATED_AT,
        "inputs": [
            {
                "role": "cumulative_base",
                "dataset": _reference(
                    root,
                    artifact_type="dataset_manifest",
                    artifact_id="dataset:pytorch_v0.5",
                    relative_path=cumulative_path,
                ).to_dict(),
                "provenance": None,
            },
            {
                "role": "precision_base",
                "dataset": _reference(
                    root,
                    artifact_type="dataset_manifest",
                    artifact_id="dataset:pytorch_v0.5_precision",
                    relative_path=precision_path,
                ).to_dict(),
                "provenance": None,
            },
            {
                "role": "boundary_freeze",
                "dataset": _reference(
                    root,
                    artifact_type="dataset_manifest",
                    artifact_id=f"dataset:{freeze.dataset_id}",
                    relative_path=boundary_path,
                ).to_dict(),
                "provenance": _reference(
                    root,
                    artifact_type="dataset_freeze",
                    artifact_id=freeze.freeze_id,
                    relative_path=freeze_path,
                ).to_dict(),
            },
        ],
        "registries": {
            "environments": _reference(
                root,
                artifact_type="environment_registry",
                artifact_id="registry:environments:v1",
                relative_path="environments/registry.json",
            ).to_dict(),
            "sources": _reference(
                root,
                artifact_type="source_registry",
                artifact_id="registry:sources:v1",
                relative_path="sources/registry.json",
            ).to_dict(),
        },
        "entries": [entry.to_dict() for entry in entries],
        "dataset_ids": {
            "cumulative": "pytorch_v0.7",
            "boundary": "pytorch_v0.7_boundary",
            "precision": "pytorch_v0.7_precision",
        },
        "output_paths": {
            "release_manifest": "factory/v0.7/p4/release_manifest.json",
            "cumulative": {
                "dataset": "datasets/pytorch_v0.7/dataset.json",
                "summary": "datasets/pytorch_v0.7/summary.json",
            },
            "boundary": {
                "dataset": "datasets/pytorch_v0.7_boundary/dataset.json",
                "summary": "datasets/pytorch_v0.7_boundary/summary.json",
            },
            "precision": {
                "dataset": "datasets/pytorch_v0.7_precision/dataset.json",
                "summary": "datasets/pytorch_v0.7_precision/summary.json",
            },
        },
    }


def _write_request(path: Path, value: dict[str, object]) -> str:
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ContractError("Output exists with different bytes")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic OpBench v0.7 Dataset requests."
    )
    parser.add_argument(
        "kind",
        choices=("boundary-freeze", "release"),
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        value = (
            build_boundary_freeze_request(args.repo_root)
            if args.kind == "boundary-freeze"
            else build_release_request(args.repo_root)
        )
        status = _write_request(args.output, value)
    except (ContractError, KeyError, OSError, ValueError) as exc:
        print(f"[request_invalid] {exc}", file=sys.stderr)
        return 2
    print(canonical_json({"kind": args.kind, "status": status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
