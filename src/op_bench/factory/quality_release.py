"""Formal v0.7 quality validation and historical re-admission accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from op_bench.dataset import DatasetManifest
from op_bench.factory.artifacts import (
    load_canonical_json_artifact,
    load_factory_contract,
    load_regular_file_bytes,
)
from op_bench.factory.complexity import (
    ComplexityEvidence,
    build_complexity_evidence,
)
from op_bench.factory.contracts import FactoryArtifactReference
from op_bench.factory.prompt_quality import (
    PromptQualityEvidence,
    build_private_answer_index,
    build_prompt_quality_evidence,
    validate_prompt_quality_evidence,
)
from op_bench.factory.score_four_support import (
    load_score_four_support,
    validate_score_four_review_binding,
)
from op_bench.factory.taxonomy import parse_taxonomy_v2
from op_bench.integrity import REPLAY_SPEC_HASH_KIND, replay_spec_hash
from op_bench.registry import (
    EnvironmentRegistry,
    RegistryError,
    SourceRegistry,
    resolve_task_assets,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt
from op_bench.runtime.legacy import LegacyV05Defaults, full_task_spec_from_v05
from op_bench.runtime.task_view import project_agent_task_view
from op_bench.runtime.validation import ContractError
from op_bench.task import InvalidPublicTaskId, TaskManifest


_PUBLIC_TASK_ID = re.compile(r"opbench-v07-t[0-9]{4}")
_UTC_SECONDS = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_ORIGINS = ("retained_historical", "new", "replacement")
_DISPOSITIONS = ("retained", "deferred", "retired")
_QUALITY_FIELDS = (
    "prompt_evidence",
    "complexity_evidence",
    "readmission_evidence",
    "origin",
)
_HISTORICAL_TASK_IDS = (
    "pytorch__149693__lazylinear_init",
    "pytorch__147599__lazylinear_state_forward",
    "pytorch__160952__bilinear_lazy_check",
    "pytorch__162340__nn_arg_length",
    "pytorch__163961__dataloader_subset",
    "pytorch__168295__autograd_create_graph",
    "pytorch__161488__lbfgs_wolfe",
    "pytorch__150975__autograd_backward_inputs",
    "pytorch__124385__load_state_dict_prefix",
    "pytorch__143455__set_submodule",
    "pytorch__132835__njt_sdpa_autocast",
    "pytorch__132616__cuda_mem_get_info",
    "pytorch__144009__softmax_ilpreduce_size",
    "pytorch__140557__layer_norm_decomp_precision",
    "pytorch__139999__masked_mean_bool_upcast",
    "pytorch__129138__linear_add_bias_autocast",
    "pytorch__139372__histc_int8_cuda_bounds",
    "pytorch__129154__exp_decomp_numerics",
    "pytorch__144073__vector_norm_scalar_overflow",
    "pytorch__117065__index_copy_zero_dim",
    "pytorch__118762__weight_norm_default_dim",
    "pytorch__126461__cummin_rank_zero",
    "pytorch__139751__triton_ygrid_mask",
    "pytorch__143792__addmv_empty_matrix",
    "pytorch__147352__storage_offset_overflow",
)
_EXPECTED_HISTORICAL_DISPOSITIONS = {
    "pytorch__117065__index_copy_zero_dim": "retired",
    "pytorch__118762__weight_norm_default_dim": "retired",
    "pytorch__124385__load_state_dict_prefix": "retained",
    "pytorch__126461__cummin_rank_zero": "retired",
    "pytorch__129138__linear_add_bias_autocast": "retained",
    "pytorch__129154__exp_decomp_numerics": "retained",
    "pytorch__132616__cuda_mem_get_info": "retired",
    "pytorch__132835__njt_sdpa_autocast": "retained",
    "pytorch__139372__histc_int8_cuda_bounds": "retained",
    "pytorch__139751__triton_ygrid_mask": "retired",
    "pytorch__139999__masked_mean_bool_upcast": "retired",
    "pytorch__140557__layer_norm_decomp_precision": "retained",
    "pytorch__143455__set_submodule": "retained",
    "pytorch__143792__addmv_empty_matrix": "retired",
    "pytorch__144009__softmax_ilpreduce_size": "retained",
    "pytorch__144073__vector_norm_scalar_overflow": "retained",
    "pytorch__147352__storage_offset_overflow": "retained",
    "pytorch__147599__lazylinear_state_forward": "retired",
    "pytorch__149693__lazylinear_init": "deferred",
    "pytorch__150975__autograd_backward_inputs": "retired",
    "pytorch__160952__bilinear_lazy_check": "retained",
    "pytorch__161488__lbfgs_wolfe": "retained",
    "pytorch__162340__nn_arg_length": "retained",
    "pytorch__163961__dataloader_subset": "retained",
    "pytorch__168295__autograd_create_graph": "retired",
}


@dataclass(frozen=True)
class QualityTaskRecord:
    task_id: str
    public_task_id: str
    origin: str
    task_path: str
    taxonomy_hash: str
    prompt_evidence: FactoryArtifactReference
    complexity_evidence: FactoryArtifactReference
    admission_evidence: FactoryArtifactReference
    disposition: str

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ContractError("quality_task_record.task_id: expected string")
        if (
            not isinstance(self.public_task_id, str)
            or _PUBLIC_TASK_ID.fullmatch(self.public_task_id) is None
        ):
            raise ContractError(
                "quality_task_record.public_task_id: expected opaque v0.7 Task ID"
            )
        if self.origin not in _ORIGINS:
            raise ContractError("quality_task_record.origin: unsupported value")
        _safe_relative_path(self.task_path, "quality_task_record.task_path")
        _require_hash(self.taxonomy_hash, "quality_task_record.taxonomy_hash")
        for name in (
            "prompt_evidence",
            "complexity_evidence",
            "admission_evidence",
        ):
            if not isinstance(getattr(self, name), FactoryArtifactReference):
                raise ContractError(
                    f"quality_task_record.{name}: expected FactoryArtifactReference"
                )
        if self.disposition not in _DISPOSITIONS:
            raise ContractError("quality_task_record.disposition: unsupported value")

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "task_id",
            "public_task_id",
            "origin",
            "task_path",
            "taxonomy_hash",
            "prompt_evidence",
            "complexity_evidence",
            "admission_evidence",
            "disposition",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "public_task_id": self.public_task_id,
            "origin": self.origin,
            "task_path": self.task_path,
            "taxonomy_hash": self.taxonomy_hash,
            "prompt_evidence": self.prompt_evidence.to_dict(),
            "complexity_evidence": self.complexity_evidence.to_dict(),
            "admission_evidence": self.admission_evidence.to_dict(),
            "disposition": self.disposition,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_task_record",
    ) -> "QualityTaskRecord":
        data = _exact_mapping(value, path, cls.wire_fields())
        return cls(
            task_id=_string(data["task_id"], f"{path}.task_id"),
            public_task_id=_string(
                data["public_task_id"], f"{path}.public_task_id"
            ),
            origin=_string(data["origin"], f"{path}.origin"),
            task_path=_string(data["task_path"], f"{path}.task_path"),
            taxonomy_hash=_string(
                data["taxonomy_hash"], f"{path}.taxonomy_hash"
            ),
            prompt_evidence=FactoryArtifactReference.from_dict(
                data["prompt_evidence"],
                path=f"{path}.prompt_evidence",
            ),
            complexity_evidence=FactoryArtifactReference.from_dict(
                data["complexity_evidence"],
                path=f"{path}.complexity_evidence",
            ),
            admission_evidence=FactoryArtifactReference.from_dict(
                data["admission_evidence"],
                path=f"{path}.admission_evidence",
            ),
            disposition=_string(
                data["disposition"], f"{path}.disposition"
            ),
        )


@dataclass(frozen=True)
class _AuditArtifact:
    relative_path: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class _HistoricalAudit:
    dataset_id: str
    dataset_hash: str
    created_at: str
    records: tuple[QualityTaskRecord, ...]
    artifacts: tuple[_AuditArtifact, ...]


def validate_quality_task(
    root: Path,
    task: TaskManifest,
    *,
    require_verified: bool,
) -> tuple[str, ...]:
    """Return deterministic formal-quality errors for one Task."""

    errors: list[str] = []
    if not isinstance(root, Path):
        return ("root: expected Path",)
    if not isinstance(task, TaskManifest):
        return ("task: expected TaskManifest",)

    taxonomy_hash: str | None = None
    taxonomy_value = task.data.get("taxonomy")
    if taxonomy_value is None:
        if require_verified:
            errors.append("taxonomy: required for formal v0.7")
    else:
        try:
            parse_taxonomy_v2(taxonomy_value)
            taxonomy_hash = canonical_sha256(taxonomy_value)
        except ContractError as exc:
            errors.append(str(exc))

    public_task_id: str | None = None
    try:
        public_task_id = task.public_task_id
    except InvalidPublicTaskId as exc:
        errors.append(str(exc))
    if public_task_id is None and require_verified:
        errors.append("agent_visible.public_task_id: required")

    metadata = task.data.get("metadata")
    difficulty = (
        metadata.get("difficulty") if isinstance(metadata, Mapping) else None
    )
    if difficulty == "easy":
        errors.append("metadata.difficulty: easy is forbidden")
    elif require_verified and difficulty not in ("medium", "hard"):
        errors.append("metadata.difficulty: expected medium or hard")
    if require_verified:
        if task.admission_status != "verified":
            errors.append("admission.status: verified required")
        admission_task = task
        try:
            admission_task = resolve_task_assets(
                task,
                environment_registry=EnvironmentRegistry.load(
                    root / "environments/registry.json"
                ),
                source_registry=SourceRegistry.load(
                    root / "sources/registry.json"
                ),
            )
        except RegistryError as exc:
            errors.append(f"admission: registry truth unavailable: {exc}")
        _, admission_errors = _admission_truth(admission_task)
        errors.extend(admission_errors)
        patch_scope = task.data.get("patch_scope")
        if (
            not isinstance(patch_scope, Mapping)
            or patch_scope.get("mode", "enforced") != "enforced"
            or not task.patch_scope_paths
        ):
            errors.append("patch_scope: private enforced scope required")

    quality = task.data.get("quality")
    if not isinstance(quality, Mapping):
        if require_verified:
            errors.extend(
                f"quality.{name}: required"
                for name in _QUALITY_FIELDS[:3]
            )
            errors.append("quality.origin: required")
        return tuple(errors)

    unknown = sorted(set(quality) - set(_QUALITY_FIELDS))
    if unknown:
        errors.append(f"quality: unknown fields {unknown}")
    origin = quality.get("origin")
    if origin is None:
        if require_verified:
            errors.append("quality.origin: required")
    elif origin not in _ORIGINS:
        errors.append("quality.origin: unsupported value")

    selected_paths: dict[str, Path] = {}
    for field in _QUALITY_FIELDS[:3]:
        value = quality.get(field)
        if value is None:
            if require_verified:
                errors.append(f"quality.{field}: required")
            continue
        try:
            selected_paths[field] = _task_relative_file(
                task.task_dir,
                value,
                f"quality.{field}",
            )
        except ContractError as exc:
            errors.append(str(exc))

    prompt: PromptQualityEvidence | None = None
    prompt_path = selected_paths.get("prompt_evidence")
    if prompt_path is not None:
        try:
            loaded = load_factory_contract(prompt_path)
            if not isinstance(loaded, PromptQualityEvidence):
                raise ContractError(
                    "quality.prompt_evidence: expected prompt_quality contract"
                )
            prompt = loaded
        except ContractError as exc:
            errors.append(_prefixed("quality.prompt_evidence", exc))

    complexity: ComplexityEvidence | None = None
    complexity_path = selected_paths.get("complexity_evidence")
    if complexity_path is not None:
        try:
            loaded = load_factory_contract(complexity_path)
            if not isinstance(loaded, ComplexityEvidence):
                raise ContractError(
                    "quality.complexity_evidence: expected complexity_evidence contract"
                )
            complexity = loaded
        except ContractError as exc:
            errors.append(_prefixed("quality.complexity_evidence", exc))

    view: dict[str, object] | None = None
    private_index = None
    if prompt is not None and public_task_id is not None:
        try:
            view = _quality_agent_task_view(task)
            private_index = _private_answer_index(task)
            validate_prompt_quality_evidence(
                prompt,
                rendered_prompt=render_mcp_prompt(view),
                agent_task_view=view,
                private_index=private_index,
            )
        except (ContractError, OSError, UnicodeDecodeError) as exc:
            errors.append(_prefixed("quality.prompt_evidence", exc))
        if prompt.task_id != task.task_id:
            errors.append("quality.prompt_evidence: task_id mismatch")
        if prompt.public_task_id != public_task_id:
            errors.append("quality.prompt_evidence: public_task_id mismatch")
        if require_verified and prompt.decision != "accepted":
            errors.append("quality.prompt_evidence: accepted decision required")

    if complexity is not None:
        if complexity.task_id != task.task_id:
            errors.append("quality.complexity_evidence: task_id mismatch")
        if difficulty is not None and complexity.difficulty != difficulty:
            errors.append(
                "quality.complexity_evidence: difficulty does not match metadata"
            )
        if require_verified and complexity.decision != "accepted":
            errors.append(
                "quality.complexity_evidence: accepted decision required"
            )

    readmission_path = selected_paths.get("readmission_evidence")
    if readmission_path is not None:
        try:
            readmission = load_canonical_json_artifact(readmission_path)
            _validate_readmission(
                readmission,
                task=task,
                public_task_id=public_task_id,
                origin=origin,
                taxonomy_hash=taxonomy_hash,
                prompt=prompt,
                prompt_relative=quality.get("prompt_evidence"),
                complexity=complexity,
                complexity_relative=quality.get("complexity_evidence"),
                require_verified=require_verified,
            )
        except ContractError as exc:
            errors.append(_prefixed("quality.readmission_evidence", exc))

    return tuple(_ordered_unique(errors))


def validate_historical_index(
    root: Path,
    index_path: Path,
) -> tuple[str, ...]:
    """Revalidate the formal historical index and every referenced Task gate."""

    errors: list[str] = []
    try:
        encoded = load_regular_file_bytes(index_path)
        value = json.loads(encoded.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (f"historical_index: {exc}",)
    if not isinstance(value, Mapping):
        return ("historical_index: expected object",)
    required = {
        "contract_type",
        "schema_version",
        "dataset_id",
        "dataset_hash",
        "created_at",
        "task_count",
        "k",
        "required_candidate_count",
        "records",
        "content_hash",
    }
    if set(value) != required:
        return ("historical_index: unexpected contract fields",)
    try:
        if encoded != canonical_json(value).encode("utf-8"):
            errors.append("historical_index: expected canonical JSON bytes")
    except ContractError as exc:
        errors.append(f"historical_index: {exc}")
    if value["contract_type"] != "historical_readmission_index":
        errors.append("historical_index.contract_type: mismatch")
    if value["schema_version"] != "v1":
        errors.append("historical_index.schema_version: mismatch")
    if (
        not isinstance(value["created_at"], str)
        or _UTC_SECONDS.fullmatch(value["created_at"]) is None
    ):
        errors.append("historical_index.created_at: invalid")
    if value["content_hash"] != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_hash"}
    ):
        errors.append("historical_index.content_hash: payload hash mismatch")

    dataset_path = root / "datasets/pytorch_v0.7/dataset.json"
    try:
        dataset_payload = json.loads(
            load_regular_file_bytes(dataset_path).decode("utf-8")
        )
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (*errors, f"historical_index.dataset: {exc}")
    if not isinstance(dataset_payload, Mapping):
        return (*errors, "historical_index.dataset: expected object")
    if value["dataset_id"] != dataset_payload.get("dataset_id"):
        errors.append("historical_index.dataset_id: mismatch")
    if value["dataset_hash"] != canonical_sha256(dataset_payload):
        errors.append("historical_index.dataset_hash: mismatch")

    if (
        isinstance(value["created_at"], str)
        and _UTC_SECONDS.fullmatch(value["created_at"]) is not None
    ):
        try:
            rebuilt = _build_historical_audit(
                root,
                dataset_path,
                root / "factory/v0.7/p7/reviews",
                value["created_at"],
                public_task_ids_path=None,
            )
            rebuilt_index = _historical_index_payload(rebuilt)
            if encoded != canonical_json(rebuilt_index).encode("utf-8"):
                errors.append(
                    "historical_index: bytes differ from exact review rebuild"
                )
            for artifact in rebuilt.artifacts:
                artifact_path = root / PurePosixPath(artifact.relative_path)
                expected_bytes = canonical_json(artifact.payload).encode("utf-8")
                try:
                    actual_bytes = load_regular_file_bytes(artifact_path)
                except ContractError as exc:
                    errors.append(
                        f"{artifact.relative_path}: rebuild artifact unavailable: {exc}"
                    )
                    continue
                if actual_bytes != expected_bytes:
                    errors.append(
                        f"{artifact.relative_path}: bytes differ from exact "
                        "review rebuild"
                    )
        except ContractError as exc:
            errors.append(f"historical_index.review_rebuild: {exc}")

    records_value = value["records"]
    records: list[QualityTaskRecord] = []
    if not isinstance(records_value, list):
        errors.append("historical_index.records: expected array")
        return tuple(_ordered_unique(errors))
    for index, record_value in enumerate(records_value):
        try:
            records.append(
                QualityTaskRecord.from_dict(
                    record_value,
                    path=f"historical_index.records[{index}]",
                )
            )
        except ContractError as exc:
            errors.append(str(exc))
    if len(records) != 25 or value["task_count"] != 25:
        errors.append("historical_index: expected exactly 25 records")
    if len(records) == 25:
        try:
            _validate_historical_records(tuple(records))
        except ContractError as exc:
            errors.append(str(exc))

    retained = sum(record.disposition == "retained" for record in records)
    deferred = sum(record.disposition == "deferred" for record in records)
    retired = sum(record.disposition == "retired" for record in records)
    if (retained, deferred, retired) != (14, 1, 10):
        errors.append(
            "historical_index.disposition: expected retained=14, "
            "deferred=1, retired=10"
        )
    if value["k"] != retained:
        errors.append("historical_index.k: retained count mismatch")
    if value["required_candidate_count"] != 3 * (50 - retained):
        errors.append("historical_index.required_candidate_count: mismatch")
    actual_dispositions = {
        record.task_id: record.disposition for record in records
    }
    if actual_dispositions != _EXPECTED_HISTORICAL_DISPOSITIONS:
        errors.append("historical_index.disposition: approved mapping mismatch")

    for record in records:
        task_path = root / PurePosixPath(record.task_path) / "task.json"
        try:
            task = TaskManifest.load(task_path)
        except (ContractError, OSError, ValueError) as exc:
            errors.append(f"{record.task_id}: Task load failed: {exc}")
            continue
        if task.task_id != record.task_id:
            errors.append(f"{record.task_id}: task_id mismatch")
        try:
            if task.public_task_id != record.public_task_id:
                errors.append(f"{record.task_id}: public_task_id mismatch")
        except InvalidPublicTaskId as exc:
            errors.append(f"{record.task_id}: {exc}")
        taxonomy = task.data.get("taxonomy")
        try:
            if canonical_sha256(taxonomy) != record.taxonomy_hash:
                errors.append(f"{record.task_id}: taxonomy_hash mismatch")
        except ContractError as exc:
            errors.append(f"{record.task_id}: taxonomy invalid: {exc}")

        expected_paths = {
            "prompt_evidence": f"{record.task_path}/quality/prompt.json",
            "complexity_evidence": (
                f"{record.task_path}/quality/complexity.json"
            ),
            "admission_evidence": (
                f"{record.task_path}/quality/readmission.json"
            ),
        }
        for field, expected_path in expected_paths.items():
            reference = getattr(record, field)
            if reference.relative_path != expected_path:
                errors.append(
                    f"{record.task_id}.{field}.relative_path: mismatch"
                )
                continue
            try:
                artifact = load_canonical_json_artifact(
                    root / PurePosixPath(reference.relative_path)
                )
            except ContractError as exc:
                errors.append(f"{record.task_id}.{field}: {exc}")
                continue
            if artifact.get("content_hash") != reference.content_hash:
                errors.append(
                    f"{record.task_id}.{field}.content_hash: mismatch"
                )
            if artifact.get("task_id") != record.task_id:
                errors.append(f"{record.task_id}.{field}.task_id: mismatch")
            if (
                field == "admission_evidence"
                and artifact.get("disposition") != record.disposition
            ):
                errors.append(
                    f"{record.task_id}.disposition: readmission mismatch"
                )
        if record.disposition == "retained":
            errors.extend(
                f"{record.task_id}: {error}"
                for error in validate_quality_task(
                    root,
                    task,
                    require_verified=True,
                )
            )
    return tuple(_ordered_unique(errors))


def build_historical_dispositions(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    created_at: str,
    *,
    public_task_ids_path: Path | None = None,
) -> tuple[QualityTaskRecord, ...]:
    """Build one deterministic disposition for every historical Dataset Task."""

    return _build_historical_audit(
        root,
        dataset_path,
        review_root,
        created_at,
        public_task_ids_path=public_task_ids_path,
    ).records


def write_historical_dispositions(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    output_path: Path,
    created_at: str,
    *,
    public_task_ids_path: Path | None = None,
) -> tuple[QualityTaskRecord, ...]:
    """Build and write the canonical task evidence tree and global index."""

    _assert_no_symlink_ancestors(output_path, "output path")
    audit = _build_historical_audit(
        root,
        dataset_path,
        review_root,
        created_at,
        public_task_ids_path=public_task_ids_path,
    )
    output_root = output_path.parent
    official_output = (
        output_path.resolve()
        == (root / "factory/v0.7/p7/historical_readmission.json").resolve()
    )
    artifact_root = root if official_output else output_root
    for artifact in audit.artifacts:
        _write_canonical(
            artifact_root / PurePosixPath(artifact.relative_path),
            artifact.payload,
        )
    _write_canonical(output_path, _historical_index_payload(audit))
    return audit.records


def _historical_index_payload(audit: _HistoricalAudit) -> dict[str, object]:
    retained = sum(record.disposition == "retained" for record in audit.records)
    payload: dict[str, object] = {
        "contract_type": "historical_readmission_index",
        "schema_version": "v1",
        "dataset_id": audit.dataset_id,
        "dataset_hash": audit.dataset_hash,
        "created_at": audit.created_at,
        "task_count": len(audit.records),
        "k": retained,
        "required_candidate_count": 3 * (50 - retained),
        "records": [record.to_dict() for record in audit.records],
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _build_historical_audit(
    root: Path,
    dataset_path: Path,
    review_root: Path,
    created_at: str,
    *,
    public_task_ids_path: Path | None,
) -> _HistoricalAudit:
    if not isinstance(root, Path) or not root.is_dir():
        raise ContractError("root: expected repository directory")
    if not isinstance(dataset_path, Path):
        raise ContractError("dataset_path: expected Path")
    if not isinstance(review_root, Path):
        raise ContractError("review_root: expected Path")
    if not isinstance(created_at, str) or _UTC_SECONDS.fullmatch(created_at) is None:
        raise ContractError("created_at: expected UTC RFC3339 seconds")
    _assert_no_symlink_ancestors(review_root, "review root")

    dataset_bytes = load_regular_file_bytes(dataset_path)
    try:
        dataset_payload = json.loads(dataset_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("dataset: invalid JSON") from exc
    if not isinstance(dataset_payload, Mapping):
        raise ContractError("dataset: expected JSON object")
    entries = dataset_payload.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ContractError("dataset.tasks: expected non-empty array")
    task_ids = [
        entry.get("task_id") if isinstance(entry, Mapping) else None
        for entry in entries
    ]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ContractError("dataset.tasks: every entry requires task_id")
    if (
        len(task_ids) != len(_HISTORICAL_TASK_IDS)
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) != set(_HISTORICAL_TASK_IDS)
    ):
        raise ContractError(
            "dataset.tasks: expected exact historical 25 frozen identities"
        )
    public_id_mapping = _load_public_task_id_mapping(
        public_task_ids_path
        or root / "factory/v0.7/p6/public_task_ids.json"
    )
    if set(public_id_mapping) != set(task_ids):
        raise ContractError(
            "public Task ID mapping: identities do not match historical Dataset"
        )
    score_four_support = load_score_four_support(
        root / "factory/v0.7/p7/pilot_factual_evidence.json",
        root / "factory/v0.7/p7/second_complexity_review.json",
    )

    loaded_tasks = DatasetManifest.load(dataset_path).load_tasks()
    by_id = {task.task_id: task for task in loaded_tasks}
    if set(by_id) != set(task_ids):
        raise ContractError("dataset.tasks: loaded Task identities do not match")

    records: list[QualityTaskRecord] = []
    artifacts: list[_AuditArtifact] = []
    accepted_fingerprints: set[str] = set()
    official_review_dispositions: dict[str, str] = {}
    for task_id, public_task_id in public_id_mapping.items():
        original = by_id[task_id]
        data = dict(original.data)
        agent_visible = dict(
            data.get("agent_visible")
            if isinstance(data.get("agent_visible"), Mapping)
            else {}
        )
        agent_visible["public_task_id"] = public_task_id
        data["agent_visible"] = agent_visible
        task = TaskManifest(task_dir=original.task_dir, data=data)
        admission_task = task
        registry_error: str | None = None
        try:
            admission_task = resolve_task_assets(
                task,
                environment_registry=EnvironmentRegistry.load(
                    root / "environments/registry.json"
                ),
                source_registry=SourceRegistry.load(
                    root / "sources/registry.json"
                ),
            )
        except RegistryError as exc:
            registry_error = str(exc)
        task_path = _repo_relative(root, task.task_dir, "task_path")
        artifact_prefix = f"{task_path}/quality"

        errors: list[str] = []
        retirement = False
        taxonomy_value = task.data.get("taxonomy")
        if taxonomy_value is None:
            taxonomy_hash = canonical_sha256(
                {"task_id": task_id, "taxonomy": "missing"}
            )
            errors.append("taxonomy: missing")
        else:
            try:
                taxonomy_hash = _json_hash(taxonomy_value)
                parse_taxonomy_v2(taxonomy_value)
            except ContractError as exc:
                errors.append(str(exc))

        view: dict[str, object] | None = None
        private_index = None
        try:
            view = _quality_agent_task_view(task)
            private_index = _private_answer_index(task)
        except (ContractError, OSError, UnicodeDecodeError) as exc:
            errors.append(f"agent_task_view: {exc}")

        prompt_review_path = _review_path(
            review_root,
            task_id,
            "prompt",
        )
        prompt_review, prompt_review_hash, prompt_review_error = _review_json(
            prompt_review_path,
            "prompt review",
        )
        (
            prompt_review,
            prompt_source_evidence,
            requested_disposition,
            combined_prompt_error,
        ) = (
            _review_section(
                prompt_review,
                kind="prompt",
                task_id=task_id,
                public_task_id=public_task_id,
            )
        )
        if combined_prompt_error is not None:
            prompt_review_error = combined_prompt_error
        if requested_disposition is not None:
            official_review_dispositions[task_id] = requested_disposition
        prompt_evidence: PromptQualityEvidence | None = None
        prompt_errors: list[str] = []
        if prompt_review_error is not None:
            prompt_errors.append(prompt_review_error)
        elif view is None or private_index is None:
            prompt_errors.append("agent_task_view unavailable")
        else:
            try:
                prompt_evidence = _prompt_evidence_from_review(
                    prompt_review,
                    task=task,
                    public_task_id=public_task_id,
                    view=view,
                    private_index=private_index,
                    created_at=created_at,
                )
                if prompt_evidence.decision == "rejected":
                    retirement = True
                elif prompt_evidence.decision != "accepted":
                    prompt_errors.append(
                        f"decision is {prompt_evidence.decision}"
                    )
            except ContractError as exc:
                prompt_errors.append(str(exc))
        errors.extend(f"prompt: {error}" for error in prompt_errors)
        prompt_payload = (
            prompt_evidence.to_dict()
            if prompt_evidence is not None
            else _audit_payload(
                "historical_prompt_review",
                task_id=task_id,
                public_task_id=public_task_id,
                review_hash=prompt_review_hash,
                errors=prompt_errors,
                created_at=created_at,
            )
        )
        prompt_ref = FactoryArtifactReference(
            artifact_type=str(prompt_payload["contract_type"]),
            artifact_id=task_id,
            content_hash=str(prompt_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/prompt.json",
        )
        artifacts.append(_AuditArtifact(prompt_ref.relative_path, prompt_payload))

        complexity_review_path = _review_path(
            review_root,
            task_id,
            "complexity",
        )
        (
            complexity_review,
            complexity_review_hash,
            complexity_review_error,
        ) = _review_json(complexity_review_path, "complexity review")
        (
            complexity_review,
            complexity_source_evidence,
            complexity_disposition,
            combined_complexity_error,
        ) = _review_section(
            complexity_review,
            kind="complexity",
            task_id=task_id,
            public_task_id=public_task_id,
        )
        if combined_complexity_error is not None:
            complexity_review_error = combined_complexity_error
        if (
            requested_disposition is not None
            and complexity_disposition != requested_disposition
        ):
            complexity_review_error = (
                "combined review disposition differs between sections"
            )
        if prompt_source_evidence != complexity_source_evidence:
            complexity_review_error = (
                "combined review source_evidence differs between sections"
            )
        complexity: ComplexityEvidence | None = None
        complexity_errors: list[str] = []
        if complexity_review_error is not None:
            complexity_errors.append(complexity_review_error)
        else:
            try:
                complexity = _complexity_from_review(
                    complexity_review,
                    task_id=task_id,
                    created_at=created_at,
                )
                if complexity.decision == "rejected":
                    retirement = True
                elif complexity.decision != "accepted":
                    complexity_errors.append(
                        f"decision is {complexity.decision}"
                    )
                difficulty = (
                    task.data.get("metadata", {}).get("difficulty")
                    if isinstance(task.data.get("metadata"), Mapping)
                    else None
                )
                if (
                    complexity.decision == "accepted"
                    and complexity.difficulty != difficulty
                ):
                    complexity_errors.append(
                        "difficulty does not match Task metadata"
                    )
                if (
                    complexity.total == 4
                    and requested_disposition == "retained"
                ):
                    validate_score_four_review_binding(
                        score_four_support,
                        public_task_id=public_task_id,
                        prompt_review=prompt_review,
                        complexity_review=complexity_review,
                        source_evidence=complexity_source_evidence,
                    )
            except ContractError as exc:
                complexity_errors.append(str(exc))
        errors.extend(
            f"complexity: {error}" for error in complexity_errors
        )
        complexity_payload = (
            complexity.to_dict()
            if complexity is not None
            else _audit_payload(
                "historical_complexity_review",
                task_id=task_id,
                public_task_id=public_task_id,
                review_hash=complexity_review_hash,
                errors=complexity_errors,
                created_at=created_at,
            )
        )
        complexity_ref = FactoryArtifactReference(
            artifact_type=str(complexity_payload["contract_type"]),
            artifact_id=task_id,
            content_hash=str(complexity_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/complexity.json",
        )
        artifacts.append(
            _AuditArtifact(complexity_ref.relative_path, complexity_payload)
        )

        if task.admission_status != "verified":
            errors.append("admission.status: verified required")
        if registry_error is not None:
            errors.append(
                f"admission: registry truth unavailable: {registry_error}"
            )
        patch_scope = task.data.get("patch_scope")
        if (
            not isinstance(patch_scope, Mapping)
            or patch_scope.get("mode", "enforced") != "enforced"
            or not task.patch_scope_paths
        ):
            errors.append("patch_scope: enforced required")
        admission_hash, admission_errors = _admission_truth(admission_task)
        errors.extend(admission_errors)
        if (
            admission_task.metadata_source_loading_verified is not True
            and admission_task.runtime_tier != "cpu_source_snapshot_fuller"
        ):
            errors.append("runtime: source loading truth is missing")
        if (
            not retirement
            and not errors
            and complexity is not None
            and complexity.decision == "accepted"
        ):
            if complexity.duplicate_fingerprint in accepted_fingerprints:
                errors.append(
                    "complexity: duplicate fingerprint already retained"
                )
                retirement = True
            else:
                accepted_fingerprints.add(
                    complexity.duplicate_fingerprint
                )

        if requested_disposition == "deferred" and retirement:
            if (
                complexity is not None
                and complexity.hard_rejections
                == ("standard_admission_failure",)
            ):
                retirement = False
                errors.append(
                    "complexity: standard admission failure requires repaired "
                    "private evidence and fresh Admission"
                )
            else:
                errors.append(
                    "review: deferred cannot override a non-repairable rejection"
                )
        if requested_disposition == "retired" and not retirement:
            errors.append("review: retired disposition lacks a formal rejection")
        if requested_disposition == "deferred" and not errors:
            errors.append("review: deferred disposition lacks an unresolved gate")

        disposition = (
            "retired"
            if retirement
            else "deferred"
            if errors
            else "retained"
        )
        readmission_payload: dict[str, object] = {
            "contract_type": "quality_readmission",
            "schema_version": "v1",
            "task_id": task_id,
            "public_task_id": public_task_id,
            "origin": "retained_historical",
            "disposition": disposition,
            "taxonomy_hash": taxonomy_hash,
            "prompt_evidence": FactoryArtifactReference(
                artifact_type=prompt_ref.artifact_type,
                artifact_id=prompt_ref.artifact_id,
                content_hash=prompt_ref.content_hash,
                relative_path="quality/prompt.json",
            ).to_dict(),
            "complexity_evidence": FactoryArtifactReference(
                artifact_type=complexity_ref.artifact_type,
                artifact_id=complexity_ref.artifact_id,
                content_hash=complexity_ref.content_hash,
                relative_path="quality/complexity.json",
            ).to_dict(),
            "admission_evidence_hash": admission_hash,
            "review_input_hashes": {
                "prompt": prompt_review_hash,
                "complexity": complexity_review_hash,
            },
            "errors": list(_ordered_unique(errors)),
            "created_at": created_at,
        }
        readmission_payload["content_hash"] = canonical_sha256(
            readmission_payload
        )
        readmission_ref = FactoryArtifactReference(
            artifact_type="quality_readmission",
            artifact_id=task_id,
            content_hash=str(readmission_payload["content_hash"]),
            relative_path=f"{artifact_prefix}/readmission.json",
        )
        artifacts.append(
            _AuditArtifact(
                readmission_ref.relative_path,
                readmission_payload,
            )
        )
        records.append(
            QualityTaskRecord(
                task_id=task_id,
                public_task_id=public_task_id,
                origin="retained_historical",
                task_path=task_path,
                taxonomy_hash=taxonomy_hash,
                prompt_evidence=prompt_ref,
                complexity_evidence=complexity_ref,
                admission_evidence=readmission_ref,
                disposition=disposition,
            )
        )

    result = _HistoricalAudit(
        dataset_id=_string(
            dataset_payload.get("dataset_id"),
            "dataset.dataset_id",
        ),
        dataset_hash=canonical_sha256(dataset_payload),
        created_at=created_at,
        records=tuple(records),
        artifacts=tuple(artifacts),
    )
    _validate_historical_records(result.records)
    if len(official_review_dispositions) == len(_HISTORICAL_TASK_IDS):
        actual = {
            record.task_id: record.disposition for record in result.records
        }
        if official_review_dispositions != _EXPECTED_HISTORICAL_DISPOSITIONS:
            raise ContractError(
                "combined reviews: expected exact approved historical dispositions"
            )
        if actual != _EXPECTED_HISTORICAL_DISPOSITIONS:
            raise ContractError(
                "review-derived dispositions: expected retained=14, "
                "deferred=1, retired=10"
            )
    return result


def _quality_agent_task_view(task: TaskManifest) -> dict[str, object]:
    spec = full_task_spec_from_v05(task)
    defaults = LegacyV05Defaults.standard()
    capability = replace(
        defaults.capability_policy,
        policy_id="opbench-v0.7-repository-root-v1",
        writable_paths=(".",),
        registered_tests=tuple(
            sorted(selector.selector_id for selector in spec.public_tests)
        ),
    )
    return project_agent_task_view(
        spec,
        capability,
        defaults.budget_policy,
    ).to_dict()


def _private_answer_index(task: TaskManifest):
    artifacts = task.data.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ContractError("artifacts: expected object")
    gold_path = _task_relative_file(
        task.task_dir,
        artifacts.get("gold_patch"),
        "artifacts.gold_patch",
    )
    hidden_value = artifacts.get(
        "hidden_test_patch",
        artifacts.get("test_patch"),
    )
    hidden_path = _task_relative_file(
        task.task_dir,
        hidden_value,
        "artifacts.hidden_test_patch",
    )
    spec = full_task_spec_from_v05(task)
    return build_private_answer_index(
        gold_patch=load_regular_file_bytes(gold_path).decode("utf-8"),
        hidden_test_patch=load_regular_file_bytes(hidden_path).decode("utf-8"),
        patch_scope=tuple(task.patch_scope_paths),
        hidden_selectors=tuple(
            selector.selector_id for selector in spec.hidden_tests
        ),
    )


def _validate_readmission(
    value: Mapping[str, object],
    *,
    task: TaskManifest,
    public_task_id: str | None,
    origin: object,
    taxonomy_hash: str | None,
    prompt: PromptQualityEvidence | None,
    prompt_relative: object,
    complexity: ComplexityEvidence | None,
    complexity_relative: object,
    require_verified: bool,
) -> None:
    required = (
        "contract_type",
        "schema_version",
        "task_id",
        "public_task_id",
        "origin",
        "disposition",
        "taxonomy_hash",
        "prompt_evidence",
        "complexity_evidence",
        "admission_evidence_hash",
        "created_at",
        "content_hash",
    )
    missing = sorted(set(required) - set(value))
    if missing:
        raise ContractError(f"missing fields {missing}")
    if value["contract_type"] != "quality_readmission":
        raise ContractError("contract_type: expected 'quality_readmission'")
    if value["schema_version"] != "v1":
        raise ContractError("schema_version: expected 'v1'")
    stored_hash = value["content_hash"]
    _require_hash(stored_hash, "content_hash")
    if stored_hash != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_hash"}
    ):
        raise ContractError("content_hash: payload hash mismatch")
    if value["task_id"] != task.task_id:
        raise ContractError("task_id mismatch")
    if value["public_task_id"] != public_task_id:
        raise ContractError("public_task_id mismatch")
    if value["origin"] != origin:
        raise ContractError("origin mismatch")
    if value["taxonomy_hash"] != taxonomy_hash:
        raise ContractError("taxonomy_hash mismatch")
    if value["disposition"] not in _DISPOSITIONS:
        raise ContractError("disposition: unsupported value")
    if require_verified and value["disposition"] != "retained":
        raise ContractError("disposition: retained required")
    if prompt is not None:
        _validate_embedded_reference(
            value["prompt_evidence"],
            artifact_type=prompt.contract_type,
            artifact_id=task.task_id,
            content_hash=prompt.content_hash,
            relative_path=prompt_relative,
            path="prompt_evidence",
        )
    if complexity is not None:
        _validate_embedded_reference(
            value["complexity_evidence"],
            artifact_type=complexity.contract_type,
            artifact_id=task.task_id,
            content_hash=complexity.content_hash,
            relative_path=complexity_relative,
            path="complexity_evidence",
        )
    admission = task.data.get("admission")
    if not isinstance(admission, Mapping):
        raise ContractError("Task admission evidence is missing")
    admission_path = _task_relative_file(
        task.task_dir,
        admission.get("evidence"),
        "admission.evidence",
    )
    admission_bytes = load_regular_file_bytes(admission_path)
    try:
        json.loads(admission_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("admission evidence: invalid JSON") from exc
    expected_admission_hash = _bytes_hash(admission_bytes)
    if value["admission_evidence_hash"] != expected_admission_hash:
        raise ContractError("admission_evidence_hash mismatch")


def _validate_embedded_reference(
    value: object,
    *,
    artifact_type: str,
    artifact_id: str,
    content_hash: str,
    relative_path: object,
    path: str,
) -> None:
    reference = FactoryArtifactReference.from_dict(value, path=path)
    if reference.artifact_type != artifact_type:
        raise ContractError(f"{path}: artifact_type mismatch")
    if reference.artifact_id != artifact_id:
        raise ContractError(f"{path}: artifact_id mismatch")
    if reference.content_hash != content_hash:
        raise ContractError(f"{path}: content_hash mismatch")
    if reference.relative_path != relative_path:
        raise ContractError(f"{path}: relative_path mismatch")


def _prompt_evidence_from_review(
    review: object,
    *,
    task: TaskManifest,
    public_task_id: str,
    view: Mapping[str, object],
    private_index,
    created_at: str,
) -> PromptQualityEvidence:
    if not isinstance(review, Mapping):
        raise ContractError("prompt review: expected object")
    if review.get("contract_type") == PromptQualityEvidence.contract_type:
        evidence = PromptQualityEvidence.from_dict(review)
        validate_prompt_quality_evidence(
            evidence,
            rendered_prompt=render_mcp_prompt(view),
            agent_task_view=view,
            private_index=private_index,
        )
        if evidence.task_id != task.task_id:
            raise ContractError("prompt review: task_id mismatch")
        if evidence.public_task_id != public_task_id:
            raise ContractError("prompt review: public_task_id mismatch")
        return evidence
    required = ("blind_review", "semantic_review", "decision")
    missing = sorted(set(required) - set(review))
    if missing:
        raise ContractError(f"prompt review: missing fields {missing}")
    return build_prompt_quality_evidence(
        task_id=task.task_id,
        public_task_id=public_task_id,
        rendered_prompt=render_mcp_prompt(view),
        agent_task_view=view,
        private_index=private_index,
        scanner_version=str(
            review.get("scanner_version", "prompt-overlap-v1")
        ),
        blind_review=_mapping(review["blind_review"], "blind_review"),
        semantic_review=_mapping(
            review["semantic_review"],
            "semantic_review",
        ),
        decision=_string(review["decision"], "decision"),
        created_at=str(review.get("created_at", created_at)),
    )


def _complexity_from_review(
    review: object,
    *,
    task_id: str,
    created_at: str,
) -> ComplexityEvidence:
    if not isinstance(review, Mapping):
        raise ContractError("complexity review: expected object")
    if review.get("contract_type") == ComplexityEvidence.contract_type:
        evidence = ComplexityEvidence.from_dict(review)
        if evidence.task_id != task_id:
            raise ContractError("complexity review: task_id mismatch")
        return evidence
    required = (
        "localization",
        "diagnosis",
        "repair_regression",
        "dimension_evidence",
        "hard_rejections",
        "risk_signals",
        "duplicate_fingerprint",
        "duplicate_decision",
        "blind_pilot",
        "second_review",
        "reviewer",
    )
    missing = sorted(set(required) - set(review))
    if missing:
        raise ContractError(f"complexity review: missing fields {missing}")
    return build_complexity_evidence(
        task_id=task_id,
        localization=review["localization"],  # type: ignore[arg-type]
        diagnosis=review["diagnosis"],  # type: ignore[arg-type]
        repair_regression=review["repair_regression"],  # type: ignore[arg-type]
        dimension_evidence=_mapping(
            review["dimension_evidence"],
            "dimension_evidence",
        ),
        hard_rejections=tuple(
            _list(review["hard_rejections"], "hard_rejections")
        ),
        risk_signals=tuple(
            _list(review["risk_signals"], "risk_signals")
        ),
        duplicate_fingerprint=_string(
            review["duplicate_fingerprint"],
            "duplicate_fingerprint",
        ),
        duplicate_decision=_string(
            review["duplicate_decision"],
            "duplicate_decision",
        ),
        blind_pilot=(
            None
            if review["blind_pilot"] is None
            else _mapping(review["blind_pilot"], "blind_pilot")
        ),
        second_review=review["second_review"],  # type: ignore[arg-type]
        reviewer=_string(review["reviewer"], "reviewer"),
        reviewed_at=str(review.get("reviewed_at", created_at)),
    )


def _admission_truth(task: TaskManifest) -> tuple[str, tuple[str, ...]]:
    admission = task.data.get("admission")
    if not isinstance(admission, Mapping):
        return (
            canonical_sha256(
                {"task_id": task.task_id, "admission": "missing"}
            ),
            ("admission: evidence is missing",),
        )
    try:
        path = _task_relative_file(
            task.task_dir,
            admission.get("evidence"),
            "admission.evidence",
        )
        admission_bytes = load_regular_file_bytes(path)
        payload = json.loads(admission_bytes.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            canonical_sha256(
                {"task_id": task.task_id, "admission": "invalid"}
            ),
            (f"admission: {exc}",),
        )
    if not isinstance(payload, Mapping):
        return (
            _bytes_hash(admission_bytes),
            ("admission: evidence must be an object",),
        )
    errors: list[str] = []
    _expect_admission_value(errors, payload, "schema_version", "v1")
    _expect_admission_value(errors, payload, "task_id", task.task_id)

    hash_kind = payload.get("task_manifest_hash_kind")
    if hash_kind is None:
        expected_manifest_hash = _bytes_hash(task.task_json_path.read_bytes())
    elif hash_kind == REPLAY_SPEC_HASH_KIND:
        expected_manifest_hash = replay_spec_hash(task)
    else:
        expected_manifest_hash = ""
        errors.append("admission: task_manifest_hash_kind is unsupported")
    if expected_manifest_hash:
        _expect_admission_value(
            errors,
            payload,
            "task_manifest_hash",
            expected_manifest_hash,
        )

    created_at = payload.get("created_at")
    if (
        not isinstance(created_at, str)
        or _UTC_SECONDS.fullmatch(created_at) is None
    ):
        errors.append("admission: created_at must be UTC RFC3339 seconds")
    else:
        expected_evidence_id = (
            f"{task.task_id}:"
            f"{expected_manifest_hash.removeprefix('sha256:')[:12]}:"
            f"{created_at}"
        )
        _expect_admission_value(
            errors,
            payload,
            "evidence_id",
            expected_evidence_id,
        )

    _validate_admission_identity(
        errors,
        payload.get("source"),
        "source",
        {
            "id": task.source_ref,
            "repo_url": task.repo_url,
            "base_commit": task.base_commit,
            "snapshot_hash": task.source_snapshot_hash,
            "snapshot_method": task.source_snapshot_method,
        },
    )
    environment = task.data.get("environment")
    environment_backend = (
        environment.get("backend", "local")
        if isinstance(environment, Mapping)
        else None
    )
    _validate_admission_identity(
        errors,
        payload.get("environment"),
        "environment",
        {
            "id": task.environment_ref,
            "runtime_tier": task.runtime_tier,
            "backend": environment_backend,
            "image": task.environment_image,
            "image_digest": task.environment_image_digest,
            "digest_kind": task.environment_digest_kind,
            "platform": task.environment_platform,
        },
    )

    decision = payload.get("admission")
    if not isinstance(decision, Mapping):
        errors.append("admission: admission decision must be an object")
    else:
        for field, expected in (
            ("decision", "verified"),
            ("verified", True),
            ("failure_classification", None),
        ):
            if decision.get(field) != expected:
                errors.append(f"admission: admission.{field} mismatch")

    _validate_admission_execution(
        errors,
        payload.get("baseline"),
        task=task,
        phase="baseline",
    )
    _validate_admission_execution(
        errors,
        payload.get("gold"),
        task=task,
        phase="gold",
    )
    return _bytes_hash(admission_bytes), tuple(_ordered_unique(errors))


def _expect_admission_value(
    errors: list[str],
    payload: Mapping[str, object],
    field: str,
    expected: object,
) -> None:
    if payload.get(field) != expected:
        errors.append(f"admission: {field} mismatch")


def _validate_admission_identity(
    errors: list[str],
    value: object,
    label: str,
    expected: Mapping[str, object],
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"admission: {label} must be an object")
        return
    for field, expected_value in expected.items():
        if value.get(field) != expected_value:
            errors.append(f"admission: {label}.{field} mismatch")


def _validate_admission_execution(
    errors: list[str],
    value: object,
    *,
    task: TaskManifest,
    phase: str,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"admission: {phase} must be an object")
        return
    fail_total = len(task.fail_to_pass_tests)
    pass_total = len(task.pass_to_pass_tests)
    expected = {
        "task_id": task.task_id,
        "mode": phase,
        "status": (
            "baseline_reproduced" if phase == "baseline" else "resolved"
        ),
        "fail_to_pass_total": fail_total,
        "pass_to_pass_total": pass_total,
        "fail_to_pass_passed": 0 if phase == "baseline" else fail_total,
        "pass_to_pass_passed": pass_total,
    }
    if fail_total <= 0 or pass_total <= 0:
        errors.append(
            "admission: Task selectors require positive F2P and P2P counts"
        )
    for field, expected_value in expected.items():
        actual = value.get(field)
        if (
            field.endswith(("_total", "_passed"))
            and (isinstance(actual, bool) or not isinstance(actual, int))
        ):
            errors.append(f"admission: {phase}.{field} must be an integer")
        elif actual != expected_value:
            errors.append(f"admission: {phase}.{field} mismatch")
    duration = value.get("duration_sec")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration < 0
    ):
        errors.append(
            f"admission: {phase}.duration_sec must be non-negative"
        )


def _review_path(root: Path, task_id: str, kind: str) -> Path:
    candidates = (
        root / f"{task_id}.json",
        root / task_id / f"{kind}.json",
        root / kind / f"{task_id}.json",
        root / f"{task_id}.{kind}.json",
    )
    selected = next(
        (path for path in candidates if path.exists()),
        candidates[0],
    )
    _assert_no_symlink_ancestors(selected, f"{kind} review path")
    return selected


def _review_section(
    value: object | None,
    *,
    kind: str,
    task_id: str,
    public_task_id: str,
) -> tuple[object | None, object | None, str | None, str | None]:
    if not isinstance(value, Mapping) or value.get(
        "contract_type"
    ) != "historical_task_quality_review":
        return value, None, None, None
    required = {
        "contract_type",
        "schema_version",
        "task_id",
        "public_task_id",
        "prompt",
        "complexity",
        "disposition",
        "source_evidence",
        "readmission_note",
        "content_hash",
    }
    if set(value) != required:
        return None, None, None, "combined review: unexpected contract fields"
    if value.get("schema_version") != "v1":
        return None, None, None, "combined review: unsupported schema_version"
    if value.get("task_id") != task_id:
        return None, None, None, "combined review: task_id mismatch"
    if value.get("public_task_id") != public_task_id:
        return None, None, None, "combined review: public_task_id mismatch"
    disposition = value.get("disposition")
    if disposition not in _DISPOSITIONS:
        return None, None, None, "combined review: unsupported disposition"
    stored_hash = value.get("content_hash")
    if not isinstance(stored_hash, str) or stored_hash != canonical_sha256(
        {key: item for key, item in value.items() if key != "content_hash"}
    ):
        return None, None, None, "combined review: content_hash mismatch"
    section = value.get(kind)
    if not isinstance(section, Mapping):
        return (
            None,
            None,
            None,
            f"combined review: {kind} must be an object",
        )
    source_evidence = value.get("source_evidence")
    if not isinstance(source_evidence, Mapping):
        return (
            None,
            None,
            None,
            "combined review: source_evidence must be an object",
        )
    return section, source_evidence, str(disposition), None


def _load_public_task_id_mapping(path: Path) -> dict[str, str]:
    try:
        encoded = load_regular_file_bytes(path)
        value = json.loads(encoded.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"public Task ID mapping: invalid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ContractError("public Task ID mapping: expected object")
    if set(value) != {"contract_type", "schema_version", "tasks"}:
        raise ContractError("public Task ID mapping: unexpected contract fields")
    if (
        value["contract_type"] != "public_task_id_mapping"
        or value["schema_version"] != "v1"
    ):
        raise ContractError("public Task ID mapping: unsupported contract")
    entries = value["tasks"]
    if not isinstance(entries, list) or len(entries) != 25:
        raise ContractError("public Task ID mapping: expected exactly 25 entries")

    expected_task_ids = tuple(sorted(_HISTORICAL_TASK_IDS))
    expected_public_ids = tuple(
        f"opbench-v07-t{index:04d}" for index in range(1, 26)
    )
    pairs: list[tuple[str, str]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping) or set(entry) != {
            "task_id",
            "public_task_id",
        }:
            raise ContractError(
                f"public Task ID mapping: tasks[{index}] has invalid fields"
            )
        task_id = entry["task_id"]
        public_task_id = entry["public_task_id"]
        if not isinstance(task_id, str) or not isinstance(public_task_id, str):
            raise ContractError(
                f"public Task ID mapping: tasks[{index}] requires string IDs"
            )
        pairs.append((task_id, public_task_id))
    if tuple(task_id for task_id, _ in pairs) != expected_task_ids:
        raise ContractError(
            "public Task ID mapping: canonical IDs must be in frozen lexical order"
        )
    if tuple(public_task_id for _, public_task_id in pairs) != expected_public_ids:
        raise ContractError(
            "public Task ID mapping: opaque IDs must match frozen lexical positions"
        )
    return dict(pairs)


def _review_json(
    path: Path,
    label: str,
) -> tuple[object | None, str, str | None]:
    if not path.exists():
        return (
            None,
            canonical_sha256({"status": "missing", "label": label}),
            f"{label} is missing",
        )
    try:
        content = load_regular_file_bytes(path)
        value = json.loads(content.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            None,
            canonical_sha256({"status": "invalid", "label": label}),
            f"{label} is invalid: {exc}",
        )
    return value, _json_hash(value), None


def _audit_payload(
    contract_type: str,
    *,
    task_id: str,
    public_task_id: str,
    review_hash: str,
    errors: list[str],
    created_at: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": contract_type,
        "schema_version": "v1",
        "task_id": task_id,
        "public_task_id": public_task_id,
        "review_hash": review_hash,
        "status": "deferred",
        "errors": list(_ordered_unique(errors)),
        "created_at": created_at,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _task_relative_file(
    task_root: Path,
    value: object,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}: expected non-empty task-relative path")
    if "\\" in value:
        raise ContractError(f"{label}: expected normalized relative path")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != value
    ):
        raise ContractError(f"{label}: path escapes task root")
    root = task_root.resolve()
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label}: symlinks are not allowed")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{label}: artifact file is unavailable") from exc
    if root != resolved and root not in resolved.parents:
        raise ContractError(f"{label}: path escapes task root")
    if not resolved.is_file():
        raise ContractError(f"{label}: expected regular file")
    return resolved


def _repo_relative(root: Path, path: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ContractError(f"{label}: path is outside repository") from exc
    return _safe_relative_path(relative, label)


def _safe_relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != text
    ):
        raise ContractError(f"{label}: expected safe relative path")
    return text


def _exact_mapping(
    value: object,
    path: str,
    fields: tuple[str, ...],
) -> Mapping[str, object]:
    data = _mapping(value, path)
    missing = sorted(set(fields) - set(data))
    unknown = sorted(set(data) - set(fields))
    if missing:
        raise ContractError(f"{path}: missing fields {missing}")
    if unknown:
        raise ContractError(f"{path}: unknown fields {unknown}")
    return data


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    return value


def _list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{path}: expected array")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path}: expected non-empty string")
    return value


def _require_hash(value: object, path: str) -> str:
    text = _string(value, path)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text) is None:
        raise ContractError(f"{path}: expected sha256 digest")
    return text


def _bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _json_hash(value: object) -> str:
    try:
        return canonical_sha256(value)
    except ContractError:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return _bytes_hash(encoded)


def _prefixed(path: str, exc: BaseException) -> str:
    text = str(exc)
    return text if text.startswith(f"{path}:") else f"{path}: {text}"


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _validate_historical_records(
    records: tuple[QualityTaskRecord, ...],
) -> None:
    task_ids = tuple(record.task_id for record in records)
    public_task_ids = tuple(record.public_task_id for record in records)
    expected_public_ids = tuple(
        f"opbench-v07-t{index:04d}"
        for index in range(1, len(_HISTORICAL_TASK_IDS) + 1)
    )
    if task_ids != tuple(sorted(_HISTORICAL_TASK_IDS)):
        raise ContractError(
            "records: expected exact historical 25 frozen identities"
        )
    if (
        public_task_ids != expected_public_ids
        or len(set(public_task_ids)) != len(public_task_ids)
    ):
        raise ContractError(
            "records: expected unique deterministic public Task identities"
        )


def _assert_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = path.absolute()
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ContractError(f"{label}: symlink ancestor is forbidden")


def _write_canonical(path: Path, value: object) -> None:
    _assert_no_symlink_ancestors(path, "output path")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json(value).encode("utf-8")
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ContractError("output path: expected regular file")
        if path.read_bytes() == content:
            return
    path.write_bytes(content)


__all__ = [
    "QualityTaskRecord",
    "build_historical_dispositions",
    "validate_quality_task",
    "write_historical_dispositions",
]
