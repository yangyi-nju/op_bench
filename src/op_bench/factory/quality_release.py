"""Formal v0.7 quality validation and historical re-admission accounting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, ClassVar

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
from op_bench.factory.taxonomy import (
    CONTRACT_FAMILIES,
    DEVICES,
    MODES,
    PHASES,
    TRIGGER_TAGS,
    ExecutionContext,
    parse_taxonomy_v2,
)
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
QUALITY_CANDIDATE_STATUSES = (
    "accepted_for_build",
    "deferred_for_review",
    "hard_rejected",
)
HARD_CANDIDATE_REJECTION_REASONS = (
    "change.documentation_cleanup_refactor_only",
    "duplicate.exact_provenance",
    "runtime.hardware_outside_v07_scope",
    "runtime.unsupported_cpu_cuda",
    "source.missing_immutable_commits",
    "source.unavailable",
    "test.no_behavioral_evidence",
)
_QUALITY_CHANGE_TYPES = (
    "ADDED",
    "COPIED",
    "DELETED",
    "MODIFIED",
    "RENAMED",
)
_QUALITY_CHANGE_KINDS = (
    "bugfix",
    "cleanup",
    "documentation",
    "feature",
    "refactor",
)
_CAPTURE_SET_FIELDS = (
    "contract_type",
    "schema_version",
    "repository",
    "captured_at",
    "acquisition_receipt_set_hash",
    "acquisition",
    "candidates",
    "content_hash",
)
_CAPTURE_FIELDS = (
    "repository",
    "pr_number",
    "pr_url",
    "base_commit",
    "merge_commit",
    "base_ref_name",
    "head_ref_name",
    "acquisition_receipt_hash",
    "merged_at",
    "title",
    "description",
    "linked_issues",
    "changed_files",
    "changed_file_count",
    "behavioral_test_evidence",
    "change_kind",
    "source_available",
    "runtime_supported",
    "required_hardware",
    "execution_hints",
    "proposed_contract_families",
    "proposed_trigger_tags",
    "preliminary_review_reasons",
)
_ACQUISITION_FIELDS = (
    "connector_first",
    "connector_queries",
    "bulk_method",
    "merge_commit_rule",
    "base_commit_rule",
    "changed_files_rule",
)
_ACQUISITION_RECEIPT_FIELDS = (
    "contract_type",
    "schema_version",
    "repository",
    "pr_number",
    "pr_url",
    "merged_at",
    "merge_commit",
    "base_commit",
    "base_ref_name",
    "head_ref_name",
    "files_total_count",
    "files_captured_node_count",
    "files_has_next_page",
    "files_pagination_complete",
    "changed_files_hash",
    "capture_method",
    "captured_at",
    "content_hash",
)
_ACQUISITION_RECEIPT_SET_FIELDS = (
    "contract_type",
    "schema_version",
    "repository",
    "captured_at",
    "capture_method",
    "receipts",
    "content_hash",
)
_BACKPORT_REF = re.compile(
    r"(?:^|[-_/])(?:cherry(?:-pick)?|backport|cp)(?:[-_/]|$)|release",
    re.IGNORECASE,
)
_BACKPORT_TEXT = re.compile(
    r"cherry[- ]?pick|backport|release/[0-9]",
    re.IGNORECASE,
)
_REVERSAL_TITLE = re.compile(
    r"\b(?:revert(?:ed|ing)?|back\s+out|rollback)\b",
    re.IGNORECASE,
)
_DISTRIBUTED_ONLY_EVIDENCE = re.compile(
    r"\bddp\b|DistributedDataParallel|\bTorchElastic\b|"
    r"torch(?:/|\.)distributed(?:/|\.)elastic",
    re.IGNORECASE,
)
_DISTRIBUTED_TEST_PATH = re.compile(
    r"(?:^|/)test/distributed/",
    re.IGNORECASE,
)
_DISTRIBUTED_COLLECTIVE_EVIDENCE = re.compile(
    r"reduce_scatter|all_gather|\bcollectives?\b|\bBucketMode\b",
    re.IGNORECASE,
)
_ROCM_ONLY_EVIDENCE = re.compile(
    r"\brocm\b|\bhip(?:\s+error|launch|kernel|runtime)\b|hipLaunchKernel",
    re.IGNORECASE,
)
_ROCM_TITLE_SCOPE = re.compile(
    r"\brocm\b|\bhip\b|\bamd\b",
    re.IGNORECASE,
)
_NVIDIA_EVIDENCE = re.compile(
    r"\bnvidia\b|\bcudnn\b|\bcublas\b|\bnvcc\b|\bnvrtc\b",
    re.IGNORECASE,
)
_GRADIENT_EVIDENCE = re.compile(
    r"autograd|backward(?![- ]compatib)|\bgradient\b|"
    r"\bgrad(?:check)?\b|jvp|vjp",
    re.IGNORECASE,
)
_TEXT_ONLY_CHANGE_TITLE = re.compile(
    r"\b(?:grammar|spelling|typo)\b.*\b(?:message|docstring|docs?)\b|"
    r"\b(?:message|docstring|docs?)\b.*\b(?:grammar|spelling|typo)\b",
    re.IGNORECASE,
)
_LOW_SIGNAL_REVIEW_EVIDENCE = re.compile(
    r"\bDeprecationWarning\b|\bunit test failure\b|"
    r"\brunnable repro\b|\brepro scripts?\b|"
    r"\bimports_for_benchmark_kernel\b|\bIndentationError\b|"
    r"torch/_dynamo/repro/",
    re.IGNORECASE,
)
_FBCODE_TITLE_EVIDENCE = re.compile(r"\bFBCODE\b", re.IGNORECASE)
# Replaced after the official capture/receipt tree is regenerated. The
# validator always compares the root-fixed composite against this code-pinned
# value, independent of the candidate index's physical location.
_OFFICIAL_QUALITY_ACQUISITION_ROOT = (
    "sha256:24fee4b07edc634f130681555335246eca7fb0d13e068088ab359cb79bd606e3"
)


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
class QualityChangedFile:
    path: str
    additions: int
    deletions: int
    change_type: str
    is_test: bool

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("path", "additions", "deletions", "change_type", "is_test")

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "changed_file.path")
        _nonnegative_int(self.additions, "changed_file.additions")
        _nonnegative_int(self.deletions, "changed_file.deletions")
        if self.change_type not in _QUALITY_CHANGE_TYPES:
            raise ContractError("changed_file.change_type: unsupported value")
        if not isinstance(self.is_test, bool):
            raise ContractError("changed_file.is_test: expected boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "additions": self.additions,
            "deletions": self.deletions,
            "change_type": self.change_type,
            "is_test": self.is_test,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "changed_file",
    ) -> "QualityChangedFile":
        data = _exact_mapping(value, path, cls.wire_fields())
        return cls(
            path=_string(data["path"], f"{path}.path"),
            additions=_nonnegative_int(
                data["additions"], f"{path}.additions"
            ),
            deletions=_nonnegative_int(
                data["deletions"], f"{path}.deletions"
            ),
            change_type=_enum(
                data["change_type"],
                f"{path}.change_type",
                _QUALITY_CHANGE_TYPES,
            ),
            is_test=_boolean(data["is_test"], f"{path}.is_test"),
        )


@dataclass(frozen=True)
class QualityLinkedIssue:
    number: int
    url: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("number", "url")

    def __post_init__(self) -> None:
        _positive_int(self.number, "linked_issue.number")
        expected = f"https://github.com/pytorch/pytorch/issues/{self.number}"
        if self.url != expected:
            raise ContractError(
                f"linked_issue.url: expected {expected!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {"number": self.number, "url": self.url}

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "linked_issue",
    ) -> "QualityLinkedIssue":
        data = _exact_mapping(value, path, cls.wire_fields())
        return cls(
            number=_positive_int(data["number"], f"{path}.number"),
            url=_string(data["url"], f"{path}.url"),
        )


@dataclass(frozen=True)
class QualityCandidateAcquisitionReceipt:
    contract_type: ClassVar[str] = "quality_candidate_acquisition_receipt"
    schema_version: ClassVar[str] = "v1"

    repository: str
    pr_number: int
    pr_url: str
    merged_at: str
    merge_commit: str | None
    base_commit: str | None
    base_ref_name: str
    head_ref_name: str
    files_total_count: int
    files_captured_node_count: int
    files_has_next_page: bool
    files_pagination_complete: bool
    changed_files_hash: str
    capture_method: str
    captured_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return _ACQUISITION_RECEIPT_FIELDS

    def __post_init__(self) -> None:
        if self.repository != "pytorch/pytorch":
            raise ContractError(
                "quality_candidate_acquisition_receipt.repository: mismatch"
            )
        _positive_int(
            self.pr_number,
            "quality_candidate_acquisition_receipt.pr_number",
        )
        expected_url = (
            f"https://github.com/{self.repository}/pull/{self.pr_number}"
        )
        if self.pr_url != expected_url:
            raise ContractError(
                "quality_candidate_acquisition_receipt.pr_url: mismatch"
            )
        _timestamp(
            self.merged_at,
            "quality_candidate_acquisition_receipt.merged_at",
        )
        _optional_commit(
            self.merge_commit,
            "quality_candidate_acquisition_receipt.merge_commit",
        )
        _optional_commit(
            self.base_commit,
            "quality_candidate_acquisition_receipt.base_commit",
        )
        _string(
            self.base_ref_name,
            "quality_candidate_acquisition_receipt.base_ref_name",
        )
        _string(
            self.head_ref_name,
            "quality_candidate_acquisition_receipt.head_ref_name",
        )
        _positive_int(
            self.files_total_count,
            "quality_candidate_acquisition_receipt.files_total_count",
        )
        _positive_int(
            self.files_captured_node_count,
            "quality_candidate_acquisition_receipt."
            "files_captured_node_count",
        )
        _boolean(
            self.files_has_next_page,
            "quality_candidate_acquisition_receipt.files_has_next_page",
        )
        _boolean(
            self.files_pagination_complete,
            "quality_candidate_acquisition_receipt."
            "files_pagination_complete",
        )
        if (
            self.files_has_next_page
            or not self.files_pagination_complete
            or self.files_total_count != self.files_captured_node_count
        ):
            raise ContractError(
                "quality_candidate_acquisition_receipt.files: "
                "incomplete GraphQL connection"
            )
        _require_hash(
            self.changed_files_hash,
            "quality_candidate_acquisition_receipt.changed_files_hash",
        )
        if (
            self.capture_method
            != "authenticated_read_only_gh_api_graphql"
        ):
            raise ContractError(
                "quality_candidate_acquisition_receipt.capture_method: "
                "unsupported value"
            )
        _timestamp(
            self.captured_at,
            "quality_candidate_acquisition_receipt.captured_at",
        )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "repository": self.repository,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "merged_at": self.merged_at,
            "merge_commit": self.merge_commit,
            "base_commit": self.base_commit,
            "base_ref_name": self.base_ref_name,
            "head_ref_name": self.head_ref_name,
            "files_total_count": self.files_total_count,
            "files_captured_node_count": self.files_captured_node_count,
            "files_has_next_page": self.files_has_next_page,
            "files_pagination_complete": self.files_pagination_complete,
            "changed_files_hash": self.changed_files_hash,
            "capture_method": self.capture_method,
            "captured_at": self.captured_at,
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate_acquisition_receipt",
    ) -> "QualityCandidateAcquisitionReceipt":
        data = _exact_mapping(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(f"{path}.contract_type: mismatch")
        if data["schema_version"] != cls.schema_version:
            raise ContractError(f"{path}.schema_version: mismatch")
        receipt = cls(
            repository=_string(
                data["repository"], f"{path}.repository"
            ),
            pr_number=_positive_int(
                data["pr_number"], f"{path}.pr_number"
            ),
            pr_url=_string(data["pr_url"], f"{path}.pr_url"),
            merged_at=_timestamp(
                data["merged_at"], f"{path}.merged_at"
            ),
            merge_commit=_optional_commit(
                data["merge_commit"], f"{path}.merge_commit"
            ),
            base_commit=_optional_commit(
                data["base_commit"], f"{path}.base_commit"
            ),
            base_ref_name=_string(
                data["base_ref_name"], f"{path}.base_ref_name"
            ),
            head_ref_name=_string(
                data["head_ref_name"], f"{path}.head_ref_name"
            ),
            files_total_count=_positive_int(
                data["files_total_count"],
                f"{path}.files_total_count",
            ),
            files_captured_node_count=_positive_int(
                data["files_captured_node_count"],
                f"{path}.files_captured_node_count",
            ),
            files_has_next_page=_boolean(
                data["files_has_next_page"],
                f"{path}.files_has_next_page",
            ),
            files_pagination_complete=_boolean(
                data["files_pagination_complete"],
                f"{path}.files_pagination_complete",
            ),
            changed_files_hash=_require_hash(
                data["changed_files_hash"],
                f"{path}.changed_files_hash",
            ),
            capture_method=_string(
                data["capture_method"], f"{path}.capture_method"
            ),
            captured_at=_timestamp(
                data["captured_at"], f"{path}.captured_at"
            ),
        )
        if data["content_hash"] != receipt.content_hash:
            raise ContractError(f"{path}.content_hash: payload mismatch")
        return receipt


@dataclass(frozen=True)
class QualityCandidateExclusion:
    repository: str
    pr_number: int
    kept_pr_number: int
    base_commit: str | None
    merge_commit: str | None
    reason: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "repository",
            "pr_number",
            "kept_pr_number",
            "base_commit",
            "merge_commit",
            "reason",
        )

    def __post_init__(self) -> None:
        if self.repository != "pytorch/pytorch":
            raise ContractError("quality_candidate_exclusion.repository: mismatch")
        _positive_int(
            self.pr_number, "quality_candidate_exclusion.pr_number"
        )
        _positive_int(
            self.kept_pr_number,
            "quality_candidate_exclusion.kept_pr_number",
        )
        if self.kept_pr_number >= self.pr_number:
            raise ContractError(
                "quality_candidate_exclusion.kept_pr_number: "
                "lowest PR must be retained"
            )
        _optional_commit(
            self.base_commit, "quality_candidate_exclusion.base_commit"
        )
        _optional_commit(
            self.merge_commit, "quality_candidate_exclusion.merge_commit"
        )
        if self.reason != "duplicate.exact_provenance":
            raise ContractError(
                "quality_candidate_exclusion.reason: unsupported value"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "pr_number": self.pr_number,
            "kept_pr_number": self.kept_pr_number,
            "base_commit": self.base_commit,
            "merge_commit": self.merge_commit,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate_exclusion",
    ) -> "QualityCandidateExclusion":
        data = _exact_mapping(value, path, cls.wire_fields())
        return cls(
            repository=_string(
                data["repository"], f"{path}.repository"
            ),
            pr_number=_positive_int(
                data["pr_number"], f"{path}.pr_number"
            ),
            kept_pr_number=_positive_int(
                data["kept_pr_number"], f"{path}.kept_pr_number"
            ),
            base_commit=_optional_commit(
                data["base_commit"], f"{path}.base_commit"
            ),
            merge_commit=_optional_commit(
                data["merge_commit"], f"{path}.merge_commit"
            ),
            reason=_string(data["reason"], f"{path}.reason"),
        )


@dataclass(frozen=True)
class QualityCandidateRecord:
    contract_type: ClassVar[str] = "quality_candidate"
    schema_version: ClassVar[str] = "v1"

    candidate_id: str
    repository: str
    pr_number: int
    pr_url: str
    base_commit: str | None
    merge_commit: str | None
    base_ref_name: str
    head_ref_name: str
    acquisition_receipt_hash: str
    merged_at: str
    title: str
    description: str
    linked_issues: tuple[QualityLinkedIssue, ...]
    changed_files: tuple[QualityChangedFile, ...]
    changed_file_count: int
    behavioral_test_evidence: bool
    source_available: bool
    runtime_supported: bool
    required_hardware: tuple[str, ...]
    execution_hints: ExecutionContext
    proposed_contract_families: tuple[str, ...]
    proposed_trigger_tags: tuple[str, ...]
    candidate_status: str
    created_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "candidate_id",
            "repository",
            "pr_number",
            "pr_url",
            "base_commit",
            "merge_commit",
            "base_ref_name",
            "head_ref_name",
            "acquisition_receipt_hash",
            "merged_at",
            "title",
            "description",
            "linked_issues",
            "changed_files",
            "changed_file_count",
            "behavioral_test_evidence",
            "source_available",
            "runtime_supported",
            "required_hardware",
            "execution_hints",
            "proposed_contract_families",
            "proposed_trigger_tags",
            "candidate_status",
            "created_at",
            "content_hash",
        )

    @classmethod
    def candidate_id_for(
        cls,
        *,
        repository: str,
        pr_number: int,
        base_commit: str | None,
        merge_commit: str | None,
    ) -> str:
        digest = canonical_sha256(
            {
                "repository": repository,
                "pr_number": pr_number,
                "base_commit": base_commit,
                "merge_commit": merge_commit,
            }
        )
        return "quality-candidate:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        if re.fullmatch(
            r"quality-candidate:v1:[0-9a-f]{64}", self.candidate_id
        ) is None:
            raise ContractError("quality_candidate.candidate_id: invalid")
        if self.repository != "pytorch/pytorch":
            raise ContractError(
                "quality_candidate.repository: expected 'pytorch/pytorch'"
            )
        _positive_int(self.pr_number, "quality_candidate.pr_number")
        expected_url = (
            f"https://github.com/{self.repository}/pull/{self.pr_number}"
        )
        if self.pr_url != expected_url:
            raise ContractError(
                f"quality_candidate.pr_url: expected {expected_url!r}"
            )
        _optional_commit(
            self.base_commit, "quality_candidate.base_commit"
        )
        _optional_commit(
            self.merge_commit, "quality_candidate.merge_commit"
        )
        if (
            self.base_commit is not None
            and self.merge_commit is not None
            and self.base_commit == self.merge_commit
        ):
            raise ContractError(
                "quality_candidate: base_commit must be the landed "
                "commit's distinct first parent"
            )
        if self.base_ref_name != "main":
            raise ContractError(
                "quality_candidate.base_ref_name: expected 'main'"
            )
        _string(
            self.head_ref_name, "quality_candidate.head_ref_name"
        )
        _require_hash(
            self.acquisition_receipt_hash,
            "quality_candidate.acquisition_receipt_hash",
        )
        _timestamp(self.merged_at, "quality_candidate.merged_at")
        _text(self.title, "quality_candidate.title")
        _text(self.description, "quality_candidate.description")
        if not isinstance(self.linked_issues, tuple):
            raise ContractError(
                "quality_candidate.linked_issues: expected tuple"
            )
        issue_numbers: list[int] = []
        for index, issue in enumerate(self.linked_issues):
            if not isinstance(issue, QualityLinkedIssue):
                raise ContractError(
                    f"quality_candidate.linked_issues[{index}]: "
                    "expected QualityLinkedIssue"
                )
            issue_numbers.append(issue.number)
        if issue_numbers != sorted(set(issue_numbers)):
            raise ContractError(
                "quality_candidate.linked_issues: expected unique number order"
            )
        if not isinstance(self.changed_files, tuple) or not self.changed_files:
            raise ContractError(
                "quality_candidate.changed_files: expected non-empty tuple"
            )
        changed_paths: list[str] = []
        for index, changed_file in enumerate(self.changed_files):
            if not isinstance(changed_file, QualityChangedFile):
                raise ContractError(
                    f"quality_candidate.changed_files[{index}]: "
                    "expected QualityChangedFile"
                )
            changed_paths.append(changed_file.path)
        if changed_paths != sorted(set(changed_paths)):
            raise ContractError(
                "quality_candidate.changed_files: expected unique path order"
            )
        _positive_int(
            self.changed_file_count,
            "quality_candidate.changed_file_count",
        )
        if self.changed_file_count != len(self.changed_files):
            raise ContractError(
                "quality_candidate.changed_file_count: incomplete "
                "changed-file capture"
            )
        if not isinstance(self.behavioral_test_evidence, bool):
            raise ContractError(
                "quality_candidate.behavioral_test_evidence: expected boolean"
            )
        if self.behavioral_test_evidence != any(
            item.is_test for item in self.changed_files
        ):
            raise ContractError(
                "quality_candidate.behavioral_test_evidence: must match "
                "changed test files"
            )
        for value, path in (
            (self.source_available, "quality_candidate.source_available"),
            (self.runtime_supported, "quality_candidate.runtime_supported"),
        ):
            _boolean(value, path)
        _canonical_strings(
            self.required_hardware,
            "quality_candidate.required_hardware",
            allow_empty=False,
        )
        _validate_candidate_execution_context(
            self.execution_hints,
            "quality_candidate.execution_hints",
        )
        _registry_tuple(
            self.proposed_contract_families,
            "quality_candidate.proposed_contract_families",
            CONTRACT_FAMILIES,
            allow_empty=False,
        )
        _registry_tuple(
            self.proposed_trigger_tags,
            "quality_candidate.proposed_trigger_tags",
            TRIGGER_TAGS,
            allow_empty=True,
        )
        if self.candidate_status not in QUALITY_CANDIDATE_STATUSES:
            raise ContractError(
                "quality_candidate.candidate_status: unsupported value"
            )
        _timestamp(self.created_at, "quality_candidate.created_at")
        expected_id = self.candidate_id_for(
            repository=self.repository,
            pr_number=self.pr_number,
            base_commit=self.base_commit,
            merge_commit=self.merge_commit,
        )
        if self.candidate_id != expected_id:
            raise ContractError(
                f"quality_candidate.candidate_id: expected {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "repository": self.repository,
            "pr_number": self.pr_number,
            "pr_url": self.pr_url,
            "base_commit": self.base_commit,
            "merge_commit": self.merge_commit,
            "base_ref_name": self.base_ref_name,
            "head_ref_name": self.head_ref_name,
            "acquisition_receipt_hash": self.acquisition_receipt_hash,
            "merged_at": self.merged_at,
            "title": self.title,
            "description": self.description,
            "linked_issues": [item.to_dict() for item in self.linked_issues],
            "changed_files": [item.to_dict() for item in self.changed_files],
            "changed_file_count": self.changed_file_count,
            "behavioral_test_evidence": self.behavioral_test_evidence,
            "source_available": self.source_available,
            "runtime_supported": self.runtime_supported,
            "required_hardware": list(self.required_hardware),
            "execution_hints": _execution_context_dict(
                self.execution_hints
            ),
            "proposed_contract_families": list(
                self.proposed_contract_families
            ),
            "proposed_trigger_tags": list(self.proposed_trigger_tags),
            "candidate_status": self.candidate_status,
            "created_at": self.created_at,
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate",
    ) -> "QualityCandidateRecord":
        data = _exact_mapping(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        record = cls(
            candidate_id=_string(
                data["candidate_id"], f"{path}.candidate_id"
            ),
            repository=_string(data["repository"], f"{path}.repository"),
            pr_number=_positive_int(
                data["pr_number"], f"{path}.pr_number"
            ),
            pr_url=_string(data["pr_url"], f"{path}.pr_url"),
            base_commit=_optional_commit(
                data["base_commit"], f"{path}.base_commit"
            ),
            merge_commit=_optional_commit(
                data["merge_commit"], f"{path}.merge_commit"
            ),
            base_ref_name=_string(
                data["base_ref_name"], f"{path}.base_ref_name"
            ),
            head_ref_name=_string(
                data["head_ref_name"], f"{path}.head_ref_name"
            ),
            acquisition_receipt_hash=_require_hash(
                data["acquisition_receipt_hash"],
                f"{path}.acquisition_receipt_hash",
            ),
            merged_at=_timestamp(
                data["merged_at"], f"{path}.merged_at"
            ),
            title=_text(data["title"], f"{path}.title"),
            description=_text(
                data["description"], f"{path}.description"
            ),
            linked_issues=tuple(
                QualityLinkedIssue.from_dict(
                    item,
                    path=f"{path}.linked_issues[{index}]",
                )
                for index, item in enumerate(
                    _list(data["linked_issues"], f"{path}.linked_issues")
                )
            ),
            changed_files=tuple(
                QualityChangedFile.from_dict(
                    item,
                    path=f"{path}.changed_files[{index}]",
                )
                for index, item in enumerate(
                    _list(data["changed_files"], f"{path}.changed_files")
                )
            ),
            changed_file_count=_positive_int(
                data["changed_file_count"],
                f"{path}.changed_file_count",
            ),
            behavioral_test_evidence=_boolean(
                data["behavioral_test_evidence"],
                f"{path}.behavioral_test_evidence",
            ),
            source_available=_boolean(
                data["source_available"], f"{path}.source_available"
            ),
            runtime_supported=_boolean(
                data["runtime_supported"], f"{path}.runtime_supported"
            ),
            required_hardware=_canonical_strings(
                data["required_hardware"],
                f"{path}.required_hardware",
                allow_empty=False,
            ),
            execution_hints=_parse_candidate_execution_context(
                data["execution_hints"], f"{path}.execution_hints"
            ),
            proposed_contract_families=_registry_tuple(
                data["proposed_contract_families"],
                f"{path}.proposed_contract_families",
                CONTRACT_FAMILIES,
                allow_empty=False,
            ),
            proposed_trigger_tags=_registry_tuple(
                data["proposed_trigger_tags"],
                f"{path}.proposed_trigger_tags",
                TRIGGER_TAGS,
                allow_empty=True,
            ),
            candidate_status=_enum(
                data["candidate_status"],
                f"{path}.candidate_status",
                QUALITY_CANDIDATE_STATUSES,
            ),
            created_at=_timestamp(
                data["created_at"], f"{path}.created_at"
            ),
        )
        stored_hash = _require_hash(
            data["content_hash"], f"{path}.content_hash"
        )
        if stored_hash != record.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {record.content_hash!r}"
            )
        return record


@dataclass(frozen=True)
class QualityCandidateDecision:
    contract_type: ClassVar[str] = "quality_candidate_decision"
    schema_version: ClassVar[str] = "v1"

    decision_id: str
    candidate_id: str
    candidate_hash: str
    disposition: str
    hard_rejection_reasons: tuple[str, ...]
    preliminary_review_reasons: tuple[str, ...]
    created_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "decision_id",
            "candidate_id",
            "candidate_hash",
            "disposition",
            "hard_rejection_reasons",
            "preliminary_review_reasons",
            "created_at",
            "content_hash",
        )

    @classmethod
    def decision_id_for(
        cls,
        *,
        candidate_id: str,
        candidate_hash: str,
        disposition: str,
        hard_rejection_reasons: tuple[str, ...],
        preliminary_review_reasons: tuple[str, ...],
    ) -> str:
        digest = canonical_sha256(
            {
                "candidate_id": candidate_id,
                "candidate_hash": candidate_hash,
                "disposition": disposition,
                "hard_rejection_reasons": list(hard_rejection_reasons),
                "preliminary_review_reasons": list(
                    preliminary_review_reasons
                ),
            }
        )
        return "quality-decision:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        if re.fullmatch(
            r"quality-decision:v1:[0-9a-f]{64}", self.decision_id
        ) is None:
            raise ContractError("quality_candidate_decision.decision_id: invalid")
        if re.fullmatch(
            r"quality-candidate:v1:[0-9a-f]{64}", self.candidate_id
        ) is None:
            raise ContractError(
                "quality_candidate_decision.candidate_id: invalid"
            )
        _require_hash(
            self.candidate_hash,
            "quality_candidate_decision.candidate_hash",
        )
        if self.disposition not in QUALITY_CANDIDATE_STATUSES:
            raise ContractError(
                "quality_candidate_decision.disposition: unsupported value"
            )
        _registry_tuple(
            self.hard_rejection_reasons,
            "quality_candidate_decision.hard_rejection_reasons",
            HARD_CANDIDATE_REJECTION_REASONS,
            allow_empty=True,
        )
        _canonical_strings(
            self.preliminary_review_reasons,
            "quality_candidate_decision.preliminary_review_reasons",
            allow_empty=True,
        )
        if bool(self.hard_rejection_reasons) != (
            self.disposition == "hard_rejected"
        ):
            raise ContractError(
                "quality_candidate_decision: hard reasons must exactly "
                "match hard_rejected disposition"
            )
        if (
            self.disposition == "accepted_for_build"
            and self.preliminary_review_reasons
        ):
            raise ContractError(
                "quality_candidate_decision: accepted candidate cannot "
                "carry preliminary review reasons"
            )
        _timestamp(
            self.created_at, "quality_candidate_decision.created_at"
        )
        expected_id = self.decision_id_for(
            candidate_id=self.candidate_id,
            candidate_hash=self.candidate_hash,
            disposition=self.disposition,
            hard_rejection_reasons=self.hard_rejection_reasons,
            preliminary_review_reasons=self.preliminary_review_reasons,
        )
        if self.decision_id != expected_id:
            raise ContractError(
                f"quality_candidate_decision.decision_id: "
                f"expected {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "decision_id": self.decision_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "disposition": self.disposition,
            "hard_rejection_reasons": list(self.hard_rejection_reasons),
            "preliminary_review_reasons": list(
                self.preliminary_review_reasons
            ),
            "created_at": self.created_at,
        }
        if include_hash:
            payload["content_hash"] = canonical_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "quality_candidate_decision",
    ) -> "QualityCandidateDecision":
        data = _exact_mapping(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        hard_reasons = _registry_tuple(
            data["hard_rejection_reasons"],
            f"{path}.hard_rejection_reasons",
            HARD_CANDIDATE_REJECTION_REASONS,
            allow_empty=True,
        )
        review_reasons = _canonical_strings(
            data["preliminary_review_reasons"],
            f"{path}.preliminary_review_reasons",
            allow_empty=True,
        )
        decision = cls(
            decision_id=_string(
                data["decision_id"], f"{path}.decision_id"
            ),
            candidate_id=_string(
                data["candidate_id"], f"{path}.candidate_id"
            ),
            candidate_hash=_require_hash(
                data["candidate_hash"], f"{path}.candidate_hash"
            ),
            disposition=_enum(
                data["disposition"],
                f"{path}.disposition",
                QUALITY_CANDIDATE_STATUSES,
            ),
            hard_rejection_reasons=hard_reasons,
            preliminary_review_reasons=review_reasons,
            created_at=_timestamp(
                data["created_at"], f"{path}.created_at"
            ),
        )
        stored_hash = _require_hash(
            data["content_hash"], f"{path}.content_hash"
        )
        if stored_hash != decision.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {decision.content_hash!r}"
            )
        return decision


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


@dataclass(frozen=True)
class _CapturedQualityCandidate:
    repository: str
    pr_number: int
    pr_url: str
    base_commit: str | None
    merge_commit: str | None
    base_ref_name: str
    head_ref_name: str
    acquisition_receipt_hash: str
    merged_at: str
    title: str
    description: str
    linked_issues: tuple[QualityLinkedIssue, ...]
    changed_files: tuple[QualityChangedFile, ...]
    changed_file_count: int
    behavioral_test_evidence: bool
    change_kind: str
    source_available: bool
    runtime_supported: bool
    required_hardware: tuple[str, ...]
    execution_hints: ExecutionContext
    proposed_contract_families: tuple[str, ...]
    proposed_trigger_tags: tuple[str, ...]
    preliminary_review_reasons: tuple[str, ...]


@dataclass(frozen=True)
class _QualityCandidateFunnel:
    capture_set: Mapping[str, object]
    receipt_set: Mapping[str, object]
    records: tuple[QualityCandidateRecord, ...]
    decisions: tuple[QualityCandidateDecision, ...]
    exclusions: tuple[QualityCandidateExclusion, ...]
    index: Mapping[str, object]


def load_quality_candidate_captures(
    path: Path,
    receipts_path: Path,
) -> tuple[QualityCandidateRecord, ...]:
    """Load canonical private captures as unscreened typed candidate facts."""

    capture_set, captures = _load_quality_candidate_capture_set(
        path, receipts_path
    )
    created_at = _timestamp(
        capture_set["captured_at"], "capture_set.captured_at"
    )
    records: list[QualityCandidateRecord] = []
    for capture in captures:
        record, _ = _screen_quality_capture(capture, created_at)
        records.append(record)
    return tuple(records)


def write_quality_candidate_funnel(
    root: Path,
    captures_path: Path,
    receipts_path: Path,
    historical_index_path: Path,
    output_dir: Path,
    created_at: str,
) -> Mapping[str, object]:
    """Screen private candidate captures into canonical per-candidate facts."""

    if not isinstance(root, Path) or not root.is_dir():
        raise ContractError("root: expected repository directory")
    if not isinstance(output_dir, Path):
        raise ContractError("output_dir: expected Path")
    _assert_no_symlink_ancestors(output_dir, "output directory")
    funnel = _build_quality_candidate_funnel(
        captures_path,
        receipts_path,
        historical_index_path,
        created_at,
        root=root,
    )
    expected_candidate_names: set[str] = set()
    expected_decision_names: set[str] = set()
    for candidate, decision in zip(funnel.records, funnel.decisions):
        candidate_name = f"pr-{candidate.pr_number}.json"
        decision_name = f"pr-{candidate.pr_number}.json"
        expected_candidate_names.add(candidate_name)
        expected_decision_names.add(decision_name)
        _write_canonical(
            output_dir / "candidates" / candidate_name,
            candidate.to_dict(),
        )
        _write_canonical(
            output_dir / "decisions" / decision_name,
            decision.to_dict(),
        )
    for directory, expected in (
        (output_dir / "candidates", expected_candidate_names),
        (output_dir / "decisions", expected_decision_names),
    ):
        if directory.exists():
            actual = {
                item.name
                for item in directory.iterdir()
                if item.is_file() and item.suffix == ".json"
            }
            stale = sorted(actual - expected)
            if stale:
                raise ContractError(
                    f"output directory: stale candidate artifacts {stale}"
                )
    _write_canonical(output_dir / "screening_index.json", funnel.index)
    return funnel.index


def validate_candidate_index(
    root: Path,
    index_path: Path,
    *,
    require_minimum: bool = True,
) -> tuple[str, ...]:
    """Validate a canonical preliminary candidate index and its full tree."""

    errors: list[str] = []
    try:
        encoded = load_regular_file_bytes(index_path)
        value = json.loads(encoded.decode("utf-8"))
    except (ContractError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (f"candidate_index: {exc}",)
    if not isinstance(value, Mapping):
        return ("candidate_index: expected object",)
    index_fields = (
        "contract_type",
        "schema_version",
        "repository",
        "created_at",
        "historical_k",
        "required_candidate_count",
        "capture_set_hash",
        "receipt_set_hash",
        "historical_index_hash",
        "acquisition_root_hash",
        "capture_count",
        "candidate_count",
        "exclusion_count",
        "eligible_candidate_count",
        "disposition_counts",
        "proposed_contract_families",
        "proposed_trigger_tags",
        "execution_context_summary",
        "provenance",
        "records",
        "exclusions",
        "content_hash",
    )
    try:
        _exact_mapping(value, "candidate_index", index_fields)
        if encoded != canonical_json(value).encode("utf-8"):
            errors.append("candidate_index: expected canonical JSON bytes")
        if value["contract_type"] != "quality_candidate_screening_index":
            errors.append("candidate_index.contract_type: mismatch")
        if value["schema_version"] != "v1":
            errors.append("candidate_index.schema_version: mismatch")
        if value["repository"] != "pytorch/pytorch":
            errors.append("candidate_index.repository: mismatch")
        _timestamp(value["created_at"], "candidate_index.created_at")
        _require_hash(
            value["capture_set_hash"], "candidate_index.capture_set_hash"
        )
        receipt_set_hash = _require_hash(
            value["receipt_set_hash"],
            "candidate_index.receipt_set_hash",
        )
        historical_index_hash = _require_hash(
            value["historical_index_hash"],
            "candidate_index.historical_index_hash",
        )
        acquisition_root_hash = _require_hash(
            value["acquisition_root_hash"],
            "candidate_index.acquisition_root_hash",
        )
        if acquisition_root_hash != _quality_acquisition_root_hash(
            value["capture_set_hash"],
            receipt_set_hash,
            historical_index_hash,
        ):
            errors.append(
                "candidate_index.acquisition_root_hash: mismatch"
            )
        if value["content_hash"] != canonical_sha256(
            {
                key: item
                for key, item in value.items()
                if key != "content_hash"
            }
        ):
            errors.append("candidate_index.content_hash: payload hash mismatch")
    except ContractError as exc:
        errors.append(str(exc))

    records_value = value.get("records")
    if not isinstance(records_value, list):
        return (*_ordered_unique(errors), "candidate_index.records: expected array")
    exclusions_value = value.get("exclusions")
    if not isinstance(exclusions_value, list):
        return (
            *_ordered_unique(errors),
            "candidate_index.exclusions: expected array",
        )
    exclusions: list[QualityCandidateExclusion] = []
    try:
        exclusions = [
            QualityCandidateExclusion.from_dict(
                item,
                path=f"candidate_index.exclusions[{index}]",
            )
            for index, item in enumerate(exclusions_value)
        ]
        excluded_prs = [item.pr_number for item in exclusions]
        if excluded_prs != sorted(set(excluded_prs)):
            raise ContractError(
                "candidate_index.exclusions: "
                "expected unique ascending PR order"
            )
    except ContractError as exc:
        errors.append(str(exc))
    candidates: list[QualityCandidateRecord] = []
    decisions: list[QualityCandidateDecision] = []
    seen_prs: set[int] = set()
    seen_ids: set[str] = set()
    seen_provenance: set[tuple[object, ...]] = set()
    for index, entry_value in enumerate(records_value):
        path = f"candidate_index.records[{index}]"
        try:
            entry = _exact_mapping(
                entry_value,
                path,
                ("pr_number", "candidate", "decision", "disposition"),
            )
            pr_number = _positive_int(
                entry["pr_number"], f"{path}.pr_number"
            )
            candidate_reference = FactoryArtifactReference.from_dict(
                entry["candidate"], path=f"{path}.candidate"
            )
            decision_reference = FactoryArtifactReference.from_dict(
                entry["decision"], path=f"{path}.decision"
            )
            if candidate_reference.relative_path != (
                f"candidates/pr-{pr_number}.json"
            ):
                raise ContractError(
                    f"{path}.candidate.relative_path: mismatch"
                )
            if decision_reference.relative_path != (
                f"decisions/pr-{pr_number}.json"
            ):
                raise ContractError(
                    f"{path}.decision.relative_path: mismatch"
                )
            candidate_value = load_factory_contract(
                index_path.parent
                / PurePosixPath(candidate_reference.relative_path)
            )
            decision_value = load_factory_contract(
                index_path.parent
                / PurePosixPath(decision_reference.relative_path)
            )
            if not isinstance(candidate_value, QualityCandidateRecord):
                raise ContractError(f"{path}.candidate: wrong contract")
            if not isinstance(decision_value, QualityCandidateDecision):
                raise ContractError(f"{path}.decision: wrong contract")
            if candidate_reference.artifact_type != candidate_value.contract_type:
                raise ContractError(
                    f"{path}.candidate.artifact_type: mismatch"
                )
            if candidate_reference.artifact_id != candidate_value.candidate_id:
                raise ContractError(f"{path}.candidate.artifact_id: mismatch")
            if candidate_reference.content_hash != candidate_value.content_hash:
                raise ContractError(f"{path}.candidate.content_hash: mismatch")
            if decision_reference.artifact_type != decision_value.contract_type:
                raise ContractError(
                    f"{path}.decision.artifact_type: mismatch"
                )
            if decision_reference.artifact_id != decision_value.decision_id:
                raise ContractError(f"{path}.decision.artifact_id: mismatch")
            if decision_reference.content_hash != decision_value.content_hash:
                raise ContractError(f"{path}.decision.content_hash: mismatch")
            if candidate_value.pr_number != pr_number:
                raise ContractError(f"{path}.pr_number: candidate mismatch")
            if decision_value.candidate_id != candidate_value.candidate_id:
                raise ContractError(f"{path}.decision.candidate_id: mismatch")
            if decision_value.candidate_hash != candidate_value.content_hash:
                raise ContractError(f"{path}.decision.candidate_hash: mismatch")
            if (
                entry["disposition"]
                != candidate_value.candidate_status
                or entry["disposition"] != decision_value.disposition
            ):
                raise ContractError(f"{path}.disposition: mismatch")
            provenance = (
                candidate_value.repository,
                candidate_value.base_commit,
                candidate_value.merge_commit,
            )
            if pr_number in seen_prs:
                raise ContractError(f"{path}.pr_number: duplicate PR")
            if candidate_value.candidate_id in seen_ids:
                raise ContractError(f"{path}.candidate_id: duplicate")
            if provenance in seen_provenance:
                raise ContractError(f"{path}: exact provenance duplicate")
            seen_prs.add(pr_number)
            seen_ids.add(candidate_value.candidate_id)
            seen_provenance.add(provenance)
            candidates.append(candidate_value)
            decisions.append(decision_value)
        except (ContractError, OSError) as exc:
            errors.append(str(exc))

    if len(candidates) == len(records_value):
        try:
            if set(seen_prs) & {
                item.pr_number for item in exclusions
            }:
                raise ContractError(
                    "candidate_index.exclusions: candidate PR overlap"
                )
            _validate_candidate_index_aggregates(
                value,
                tuple(candidates),
                tuple(decisions),
                tuple(exclusions),
                require_minimum=require_minimum,
            )
        except ContractError as exc:
            errors.append(str(exc))

    captures_path = root / "factory/v0.7/p8/captures.json"
    receipts_path = (
        root / "factory/v0.7/p8/acquisition_receipts.json"
    )
    historical_path = (
        root / "factory/v0.7/p7/historical_readmission.json"
    )
    try:
        capture_root = load_canonical_json_artifact(captures_path)
        receipt_root = load_canonical_json_artifact(receipts_path)
        historical_root = load_canonical_json_artifact(historical_path)
        official_composite = _quality_acquisition_root_hash(
            _require_hash(
                capture_root.get("content_hash"),
                "official_capture.content_hash",
            ),
            _require_hash(
                receipt_root.get("content_hash"),
                "official_receipts.content_hash",
            ),
            _require_hash(
                historical_root.get("content_hash"),
                "official_historical.content_hash",
            ),
        )
        if official_composite != _OFFICIAL_QUALITY_ACQUISITION_ROOT:
            errors.append(
                "candidate_index: pinned official acquisition root mismatch"
            )
        rebuilt = _build_quality_candidate_funnel(
            captures_path,
            receipts_path,
            historical_path,
            _timestamp(value["created_at"], "candidate_index.created_at"),
            root=root,
        )
        if encoded != canonical_json(rebuilt.index).encode("utf-8"):
            errors.append(
                "candidate_index: bytes differ from exact official rebuild"
            )
        for candidate, decision in zip(
            rebuilt.records, rebuilt.decisions
        ):
            expected = (
                (
                    index_path.parent
                    / "candidates"
                    / f"pr-{candidate.pr_number}.json",
                    candidate.to_dict(),
                ),
                (
                    index_path.parent
                    / "decisions"
                    / f"pr-{candidate.pr_number}.json",
                    decision.to_dict(),
                ),
            )
            for artifact_path, payload in expected:
                if load_regular_file_bytes(
                    artifact_path
                ) != canonical_json(payload).encode("utf-8"):
                    errors.append(
                        f"{artifact_path}: bytes differ from "
                        "exact official rebuild"
                    )
    except (ContractError, OSError) as exc:
        errors.append(f"candidate_index.capture_rebuild: {exc}")
    return tuple(_ordered_unique(errors))


def _build_quality_candidate_funnel(
    captures_path: Path,
    receipts_path: Path,
    historical_index_path: Path,
    created_at: str,
    *,
    root: Path,
) -> _QualityCandidateFunnel:
    _timestamp(created_at, "created_at")
    receipt_set, _ = _load_quality_acquisition_receipt_set(
        receipts_path
    )
    capture_set, captures = _load_quality_candidate_capture_set(
        captures_path,
        receipts_path,
    )
    historical_errors = validate_historical_index(
        root, historical_index_path
    )
    if historical_errors:
        raise ContractError(
            "historical_index: full validation failed: "
            + "; ".join(historical_errors)
        )
    historical = load_canonical_json_artifact(historical_index_path)
    historical_records = _list(
        historical.get("records"), "historical_index.records"
    )
    k = sum(
        _mapping(
            record, f"historical_index.records[{index}]"
        ).get("disposition")
        == "retained"
        for index, record in enumerate(historical_records)
    )
    if historical.get("k") != k:
        raise ContractError(
            "historical_index.k: retained-record mismatch"
        )
    if historical.get("required_candidate_count") != 3 * (50 - k):
        raise ContractError(
            "historical_index.required_candidate_count: mismatch"
        )
    historical_prs: set[int] = set()
    for index, record in enumerate(historical_records):
        record_data = _mapping(
            record, f"historical_index.records[{index}]"
        )
        task_id = _string(
            record_data.get("task_id"),
            f"historical_index.records[{index}].task_id",
        )
        parts = task_id.split("__", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            raise ContractError(
                f"historical_index.records[{index}].task_id: "
                "missing PR provenance"
            )
        historical_prs.add(int(parts[1]))

    records: list[QualityCandidateRecord] = []
    decisions: list[QualityCandidateDecision] = []
    exclusions: list[QualityCandidateExclusion] = []
    provenance_keepers: dict[
        tuple[str, str | None, str | None], int
    ] = {}
    for capture in captures:
        if capture.pr_number in historical_prs:
            raise ContractError(
                f"captures: PR {capture.pr_number} is historical"
            )
        provenance = (
            capture.repository,
            capture.base_commit,
            capture.merge_commit,
        )
        kept_pr = provenance_keepers.get(provenance)
        if kept_pr is not None:
            exclusions.append(
                QualityCandidateExclusion(
                    repository=capture.repository,
                    pr_number=capture.pr_number,
                    kept_pr_number=kept_pr,
                    base_commit=capture.base_commit,
                    merge_commit=capture.merge_commit,
                    reason="duplicate.exact_provenance",
                )
            )
            continue
        provenance_keepers[provenance] = capture.pr_number
        record, decision = _screen_quality_capture(capture, created_at)
        records.append(record)
        decisions.append(decision)
    pairs = sorted(
        zip(records, decisions),
        key=lambda pair: (pair[0].repository, pair[0].pr_number),
    )
    records_tuple = tuple(pair[0] for pair in pairs)
    decisions_tuple = tuple(pair[1] for pair in pairs)
    index = _quality_candidate_index_payload(
        capture_set,
        records_tuple,
        decisions_tuple,
        tuple(exclusions),
        historical_k=k,
        historical_index_hash=_require_hash(
            historical.get("content_hash"),
            "historical_index.content_hash",
        ),
        created_at=created_at,
    )
    return _QualityCandidateFunnel(
        capture_set=capture_set,
        receipt_set=receipt_set,
        records=records_tuple,
        decisions=decisions_tuple,
        exclusions=tuple(exclusions),
        index=index,
    )


def _load_quality_candidate_capture_set(
    path: Path,
    receipts_path: Path,
) -> tuple[Mapping[str, object], tuple[_CapturedQualityCandidate, ...]]:
    value = load_canonical_json_artifact(path)
    data = _exact_mapping(
        value, "capture_set", _CAPTURE_SET_FIELDS
    )
    if data["contract_type"] != "quality_candidate_capture_set":
        raise ContractError("capture_set.contract_type: mismatch")
    if data["schema_version"] != "v1":
        raise ContractError("capture_set.schema_version: mismatch")
    if data["repository"] != "pytorch/pytorch":
        raise ContractError("capture_set.repository: mismatch")
    _timestamp(data["captured_at"], "capture_set.captured_at")
    receipt_set_hash = _require_hash(
        data["acquisition_receipt_set_hash"],
        "capture_set.acquisition_receipt_set_hash",
    )
    acquisition = _exact_mapping(
        data["acquisition"],
        "capture_set.acquisition",
        _ACQUISITION_FIELDS,
    )
    if acquisition["connector_first"] is not True:
        raise ContractError(
            "capture_set.acquisition.connector_first: required"
        )
    queries = _canonical_strings(
        acquisition["connector_queries"],
        "capture_set.acquisition.connector_queries",
        allow_empty=False,
        preserve_order=True,
    )
    if len(queries) < 2:
        raise ContractError(
            "capture_set.acquisition.connector_queries: "
            "multiple search cohorts required"
        )
    if (
        acquisition["bulk_method"]
        != "authenticated_read_only_gh_api_graphql"
    ):
        raise ContractError(
            "capture_set.acquisition.bulk_method: unsupported value"
        )
    for field in (
        "merge_commit_rule",
        "base_commit_rule",
        "changed_files_rule",
    ):
        _string(acquisition[field], f"capture_set.acquisition.{field}")
    if data["content_hash"] != canonical_sha256(
        {
            key: item
            for key, item in data.items()
            if key != "content_hash"
        }
    ):
        raise ContractError("capture_set.content_hash: payload mismatch")
    candidates_value = _list(data["candidates"], "capture_set.candidates")
    if not candidates_value:
        raise ContractError("capture_set.candidates: expected non-empty array")
    captures = tuple(
        _parse_quality_capture(
            item, path=f"capture_set.candidates[{index}]"
        )
        for index, item in enumerate(candidates_value)
    )
    pr_numbers = [item.pr_number for item in captures]
    if len(set(pr_numbers)) != len(pr_numbers):
        raise ContractError("capture_set.candidates: duplicate PR")
    if pr_numbers != sorted(pr_numbers):
        raise ContractError(
            "capture_set.candidates: expected ascending PR number order"
        )
    loaded_receipt_set, receipts = _load_quality_acquisition_receipt_set(
        receipts_path
    )
    if receipt_set_hash != loaded_receipt_set["content_hash"]:
        raise ContractError(
            "capture_set.acquisition_receipt_set_hash: "
            "receipt root mismatch"
        )
    if data["captured_at"] != loaded_receipt_set["captured_at"]:
        raise ContractError(
            "capture_set.captured_at: receipt timestamp mismatch"
        )
    receipts_by_pr = {item.pr_number: item for item in receipts}
    if set(pr_numbers) != set(receipts_by_pr):
        raise ContractError(
            "capture_set.candidates: receipt PR coverage mismatch"
        )
    for capture in captures:
        _validate_capture_receipt(
            capture,
            receipts_by_pr[capture.pr_number],
        )
    return data, captures


def _changed_files_metadata_hash(
    changed_files: Sequence[QualityChangedFile],
) -> str:
    return canonical_sha256(
        [
            {
                "path": item.path,
                "additions": item.additions,
                "deletions": item.deletions,
                "change_type": item.change_type,
            }
            for item in changed_files
        ]
    )


def _load_quality_acquisition_receipt_set(
    path: Path,
) -> tuple[
    Mapping[str, object],
    tuple[QualityCandidateAcquisitionReceipt, ...],
]:
    value = load_canonical_json_artifact(path)
    data = _exact_mapping(
        value,
        "receipt_set",
        _ACQUISITION_RECEIPT_SET_FIELDS,
    )
    if (
        data["contract_type"]
        != "quality_candidate_acquisition_receipt_set"
    ):
        raise ContractError("receipt_set.contract_type: mismatch")
    if data["schema_version"] != "v1":
        raise ContractError("receipt_set.schema_version: mismatch")
    if data["repository"] != "pytorch/pytorch":
        raise ContractError("receipt_set.repository: mismatch")
    captured_at = _timestamp(
        data["captured_at"], "receipt_set.captured_at"
    )
    if (
        data["capture_method"]
        != "authenticated_read_only_gh_api_graphql"
    ):
        raise ContractError("receipt_set.capture_method: mismatch")
    if data["content_hash"] != canonical_sha256(
        {
            key: item
            for key, item in data.items()
            if key != "content_hash"
        }
    ):
        raise ContractError("receipt_set.content_hash: payload mismatch")
    receipts = tuple(
        QualityCandidateAcquisitionReceipt.from_dict(
            item,
            path=f"receipt_set.receipts[{index}]",
        )
        for index, item in enumerate(
            _list(data["receipts"], "receipt_set.receipts")
        )
    )
    if not receipts:
        raise ContractError("receipt_set.receipts: expected non-empty array")
    pr_numbers = [item.pr_number for item in receipts]
    if pr_numbers != sorted(set(pr_numbers)):
        raise ContractError(
            "receipt_set.receipts: duplicate or non-ascending PR number"
        )
    if any(item.captured_at != captured_at for item in receipts):
        raise ContractError(
            "receipt_set.receipts: captured_at mismatch"
        )
    return data, receipts


def _validate_capture_receipt(
    capture: _CapturedQualityCandidate,
    receipt: QualityCandidateAcquisitionReceipt,
) -> None:
    pairs = (
        ("repository", capture.repository, receipt.repository),
        ("pr_number", capture.pr_number, receipt.pr_number),
        ("pr_url", capture.pr_url, receipt.pr_url),
        ("merged_at", capture.merged_at, receipt.merged_at),
        ("merge_commit", capture.merge_commit, receipt.merge_commit),
        ("base_commit", capture.base_commit, receipt.base_commit),
        ("base_ref_name", capture.base_ref_name, receipt.base_ref_name),
        ("head_ref_name", capture.head_ref_name, receipt.head_ref_name),
    )
    for field, captured, acquired in pairs:
        if captured != acquired:
            raise ContractError(
                f"capture_set.candidate.{field}: receipt mismatch"
            )
    if capture.acquisition_receipt_hash != receipt.content_hash:
        raise ContractError(
            "capture_set.candidate.acquisition_receipt_hash: mismatch"
        )
    if (
        capture.changed_file_count != receipt.files_total_count
        or capture.changed_file_count
        != receipt.files_captured_node_count
    ):
        raise ContractError(
            "capture_set.candidate.changed_file_count: receipt mismatch"
        )
    if (
        _changed_files_metadata_hash(capture.changed_files)
        != receipt.changed_files_hash
    ):
        raise ContractError(
            "capture_set.candidate.changed_files: receipt digest mismatch"
        )


def _parse_quality_capture(
    value: object,
    *,
    path: str,
) -> _CapturedQualityCandidate:
    data = _exact_mapping(value, path, _CAPTURE_FIELDS)
    repository = _string(data["repository"], f"{path}.repository")
    if repository != "pytorch/pytorch":
        raise ContractError(f"{path}.repository: mismatch")
    pr_number = _positive_int(data["pr_number"], f"{path}.pr_number")
    pr_url = _string(data["pr_url"], f"{path}.pr_url")
    if pr_url != f"https://github.com/{repository}/pull/{pr_number}":
        raise ContractError(f"{path}.pr_url: mismatch")
    base_commit = _optional_commit(
        data["base_commit"], f"{path}.base_commit"
    )
    merge_commit = _optional_commit(
        data["merge_commit"], f"{path}.merge_commit"
    )
    if (
        base_commit is not None
        and merge_commit is not None
        and base_commit == merge_commit
    ):
        raise ContractError(
            f"{path}.base_commit: expected landed commit first parent"
        )
    base_ref_name = _string(
        data["base_ref_name"], f"{path}.base_ref_name"
    )
    if base_ref_name != "main":
        raise ContractError(
            f"{path}.base_ref_name: expected primary 'main' PR"
        )
    head_ref_name = _string(
        data["head_ref_name"], f"{path}.head_ref_name"
    )
    if _BACKPORT_REF.search(head_ref_name):
        raise ContractError(
            f"{path}.head_ref_name: backport/cherry-pick ref denied"
        )
    acquisition_receipt_hash = _require_hash(
        data["acquisition_receipt_hash"],
        f"{path}.acquisition_receipt_hash",
    )
    title = _text(data["title"], f"{path}.title")
    description = _text(
        data["description"], f"{path}.description"
    )
    if _BACKPORT_TEXT.search(title + "\n" + description):
        raise ContractError(f"{path}: backport/cherry-pick text denied")
    if _REVERSAL_TITLE.search(title):
        raise ContractError(
            f"{path}.title: expected primary forward fix, not reversal"
        )
    linked_issues = tuple(
        QualityLinkedIssue.from_dict(
            item, path=f"{path}.linked_issues[{index}]"
        )
        for index, item in enumerate(
            _list(data["linked_issues"], f"{path}.linked_issues")
        )
    )
    changed_files = tuple(
        QualityChangedFile.from_dict(
            item, path=f"{path}.changed_files[{index}]"
        )
        for index, item in enumerate(
            _list(data["changed_files"], f"{path}.changed_files")
        )
    )
    if not changed_files:
        raise ContractError(f"{path}.changed_files: expected non-empty array")
    changed_file_count = _positive_int(
        data["changed_file_count"], f"{path}.changed_file_count"
    )
    if changed_file_count != len(changed_files):
        raise ContractError(
            f"{path}.changed_file_count: incomplete changed-file capture"
        )
    evidence = _boolean(
        data["behavioral_test_evidence"],
        f"{path}.behavioral_test_evidence",
    )
    if evidence != any(item.is_test for item in changed_files):
        raise ContractError(
            f"{path}.behavioral_test_evidence: "
            "must match changed test files"
        )
    execution_hints = _parse_candidate_execution_context(
        data["execution_hints"], f"{path}.execution_hints"
    )
    proposed_families = _registry_tuple(
        data["proposed_contract_families"],
        f"{path}.proposed_contract_families",
        CONTRACT_FAMILIES,
        allow_empty=False,
    )
    behavioral_hint_evidence = "\n".join(
        (
            title,
            description,
            *(
                item.path
                for item in changed_files
                if item.is_test
            ),
        )
    )
    if (
        "backward" in execution_hints.phases
        and _GRADIENT_EVIDENCE.search(behavioral_hint_evidence) is None
    ):
        raise ContractError(
            f"{path}.execution_hints.phases: backward lacks "
            "behavioral evidence"
        )
    if (
        "gradient" in proposed_families
        and _GRADIENT_EVIDENCE.search(behavioral_hint_evidence) is None
    ):
        raise ContractError(
            f"{path}.proposed_contract_families: gradient lacks "
            "behavioral evidence"
        )
    return _CapturedQualityCandidate(
        repository=repository,
        pr_number=pr_number,
        pr_url=pr_url,
        base_commit=base_commit,
        merge_commit=merge_commit,
        base_ref_name=base_ref_name,
        head_ref_name=head_ref_name,
        acquisition_receipt_hash=acquisition_receipt_hash,
        merged_at=_timestamp(
            data["merged_at"], f"{path}.merged_at"
        ),
        title=title,
        description=description,
        linked_issues=linked_issues,
        changed_files=changed_files,
        changed_file_count=changed_file_count,
        behavioral_test_evidence=evidence,
        change_kind=_enum(
            data["change_kind"], f"{path}.change_kind", _QUALITY_CHANGE_KINDS
        ),
        source_available=_boolean(
            data["source_available"], f"{path}.source_available"
        ),
        runtime_supported=_boolean(
            data["runtime_supported"], f"{path}.runtime_supported"
        ),
        required_hardware=_canonical_strings(
            data["required_hardware"],
            f"{path}.required_hardware",
            allow_empty=False,
        ),
        execution_hints=execution_hints,
        proposed_contract_families=proposed_families,
        proposed_trigger_tags=_registry_tuple(
            data["proposed_trigger_tags"],
            f"{path}.proposed_trigger_tags",
            TRIGGER_TAGS,
            allow_empty=True,
        ),
        preliminary_review_reasons=_canonical_strings(
            data["preliminary_review_reasons"],
            f"{path}.preliminary_review_reasons",
            allow_empty=True,
        ),
    )


def _screen_quality_capture(
    capture: _CapturedQualityCandidate,
    created_at: str,
) -> tuple[QualityCandidateRecord, QualityCandidateDecision]:
    hard_reason_set: set[str] = set()
    if capture.base_commit is None or capture.merge_commit is None:
        hard_reason_set.add("source.missing_immutable_commits")
    if not capture.source_available:
        hard_reason_set.add("source.unavailable")
    if not capture.runtime_supported:
        hard_reason_set.add("runtime.unsupported_cpu_cuda")
    if capture.change_kind in ("cleanup", "documentation", "refactor"):
        hard_reason_set.add("change.documentation_cleanup_refactor_only")
    if not capture.behavioral_test_evidence:
        hard_reason_set.add("test.no_behavioral_evidence")
    if (
        not set(capture.required_hardware) <= {"cpu", "cuda"}
        or capture.execution_hints.distributed
    ):
        hard_reason_set.add("runtime.hardware_outside_v07_scope")
    hardware_evidence = "\n".join(
        (
            capture.title,
            capture.description,
            capture.base_ref_name,
            capture.head_ref_name,
            *(item.path for item in capture.changed_files),
        )
    )
    behavioral_paths = "\n".join(
        item.path for item in capture.changed_files if item.is_test
    )
    distributed_scope = (
        _DISTRIBUTED_ONLY_EVIDENCE.search(hardware_evidence) is not None
        or (
            _DISTRIBUTED_TEST_PATH.search(behavioral_paths) is not None
            and _DISTRIBUTED_COLLECTIVE_EVIDENCE.search(
                hardware_evidence
            )
            is not None
        )
    )
    if distributed_scope:
        hard_reason_set.add("runtime.hardware_outside_v07_scope")
    execution_hints = (
        replace(capture.execution_hints, distributed=True)
        if distributed_scope
        else capture.execution_hints
    )
    if _ROCM_TITLE_SCOPE.search(capture.title):
        hard_reason_set.add("runtime.hardware_outside_v07_scope")
    if (
        _ROCM_ONLY_EVIDENCE.search(hardware_evidence) is not None
        and _NVIDIA_EVIDENCE.search(hardware_evidence) is None
    ):
        hard_reason_set.add("runtime.hardware_outside_v07_scope")
    if _TEXT_ONLY_CHANGE_TITLE.search(capture.title):
        hard_reason_set.add(
            "change.documentation_cleanup_refactor_only"
        )
    review_reason_set = set(capture.preliminary_review_reasons)
    if _LOW_SIGNAL_REVIEW_EVIDENCE.search(hardware_evidence):
        review_reason_set.add("review.ambiguous_change_context")
    if _FBCODE_TITLE_EVIDENCE.search(capture.title):
        review_reason_set.add("review.ambiguous_change_context")
    review_reasons = tuple(sorted(review_reason_set))
    hard_reasons = tuple(
        reason
        for reason in HARD_CANDIDATE_REJECTION_REASONS
        if reason in hard_reason_set
    )
    if hard_reasons:
        status = "hard_rejected"
    elif review_reasons:
        status = "deferred_for_review"
    else:
        status = "accepted_for_build"
    candidate_id = QualityCandidateRecord.candidate_id_for(
        repository=capture.repository,
        pr_number=capture.pr_number,
        base_commit=capture.base_commit,
        merge_commit=capture.merge_commit,
    )
    record = QualityCandidateRecord(
        candidate_id=candidate_id,
        repository=capture.repository,
        pr_number=capture.pr_number,
        pr_url=capture.pr_url,
        base_commit=capture.base_commit,
        merge_commit=capture.merge_commit,
        base_ref_name=capture.base_ref_name,
        head_ref_name=capture.head_ref_name,
        acquisition_receipt_hash=capture.acquisition_receipt_hash,
        merged_at=capture.merged_at,
        title=capture.title,
        description=capture.description,
        linked_issues=capture.linked_issues,
        changed_files=capture.changed_files,
        changed_file_count=capture.changed_file_count,
        behavioral_test_evidence=capture.behavioral_test_evidence,
        source_available=capture.source_available,
        runtime_supported=capture.runtime_supported,
        required_hardware=capture.required_hardware,
        execution_hints=execution_hints,
        proposed_contract_families=capture.proposed_contract_families,
        proposed_trigger_tags=capture.proposed_trigger_tags,
        candidate_status=status,
        created_at=created_at,
    )
    decision_id = QualityCandidateDecision.decision_id_for(
        candidate_id=record.candidate_id,
        candidate_hash=record.content_hash,
        disposition=status,
        hard_rejection_reasons=hard_reasons,
        preliminary_review_reasons=review_reasons,
    )
    decision = QualityCandidateDecision(
        decision_id=decision_id,
        candidate_id=record.candidate_id,
        candidate_hash=record.content_hash,
        disposition=status,
        hard_rejection_reasons=hard_reasons,
        preliminary_review_reasons=review_reasons,
        created_at=created_at,
    )
    return record, decision


def _quality_candidate_index_payload(
    capture_set: Mapping[str, object],
    records: tuple[QualityCandidateRecord, ...],
    decisions: tuple[QualityCandidateDecision, ...],
    exclusions: tuple[QualityCandidateExclusion, ...],
    *,
    historical_k: int,
    historical_index_hash: str,
    created_at: str,
) -> Mapping[str, object]:
    if len(records) != len(decisions):
        raise ContractError("candidate funnel: record/decision count mismatch")
    disposition_counts = {
        status: sum(item.disposition == status for item in decisions)
        for status in QUALITY_CANDIDATE_STATUSES
    }
    proposed_families = tuple(
        family
        for family in CONTRACT_FAMILIES
        if any(family in record.proposed_contract_families for record in records)
    )
    proposed_triggers = tuple(
        trigger
        for trigger in TRIGGER_TAGS
        if any(trigger in record.proposed_trigger_tags for record in records)
    )
    device_counts = {
        value: sum(
            value in record.execution_hints.devices for record in records
        )
        for value in DEVICES
    }
    mode_counts = {
        value: sum(
            value in record.execution_hints.modes for record in records
        )
        for value in MODES
    }
    phase_counts = {
        value: sum(
            value in record.execution_hints.phases for record in records
        )
        for value in PHASES
    }
    acquisition = _mapping(
        capture_set["acquisition"], "capture_set.acquisition"
    )
    capture_set_hash = _require_hash(
        capture_set["content_hash"], "capture_set.content_hash"
    )
    receipt_set_hash = _require_hash(
        capture_set["acquisition_receipt_set_hash"],
        "capture_set.acquisition_receipt_set_hash",
    )
    acquisition_root_hash = _quality_acquisition_root_hash(
        capture_set_hash,
        receipt_set_hash,
        historical_index_hash,
    )
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_screening_index",
        "schema_version": "v1",
        "repository": "pytorch/pytorch",
        "created_at": created_at,
        "historical_k": historical_k,
        "required_candidate_count": 3 * (50 - historical_k),
        "capture_set_hash": capture_set_hash,
        "receipt_set_hash": receipt_set_hash,
        "historical_index_hash": historical_index_hash,
        "acquisition_root_hash": acquisition_root_hash,
        "capture_count": len(
            _list(capture_set["candidates"], "capture_set.candidates")
        ),
        "candidate_count": len(records),
        "exclusion_count": len(exclusions),
        "eligible_candidate_count": sum(
            decision.disposition != "hard_rejected"
            for decision in decisions
        ),
        "disposition_counts": disposition_counts,
        "proposed_contract_families": list(proposed_families),
        "proposed_trigger_tags": list(proposed_triggers),
        "execution_context_summary": {
            "devices": [
                value for value in DEVICES if device_counts[value] > 0
            ],
            "modes": [
                value for value in MODES if mode_counts[value] > 0
            ],
            "phases": [
                value for value in PHASES if phase_counts[value] > 0
            ],
            "candidate_counts": {
                **device_counts,
                **mode_counts,
                **phase_counts,
            },
        },
        "provenance": {
            field: acquisition[field] for field in _ACQUISITION_FIELDS
        },
        "records": [
            {
                "pr_number": record.pr_number,
                "candidate": FactoryArtifactReference(
                    artifact_type=record.contract_type,
                    artifact_id=record.candidate_id,
                    content_hash=record.content_hash,
                    relative_path=f"candidates/pr-{record.pr_number}.json",
                ).to_dict(),
                "decision": FactoryArtifactReference(
                    artifact_type=decision.contract_type,
                    artifact_id=decision.decision_id,
                    content_hash=decision.content_hash,
                    relative_path=f"decisions/pr-{record.pr_number}.json",
                ).to_dict(),
                "disposition": decision.disposition,
            }
            for record, decision in zip(records, decisions)
        ],
        "exclusions": [item.to_dict() for item in exclusions],
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _quality_acquisition_root_hash(
    capture_set_hash: str,
    receipt_set_hash: str,
    historical_index_hash: str,
) -> str:
    return canonical_sha256(
        {
            "capture_set_hash": _require_hash(
                capture_set_hash, "capture_set_hash"
            ),
            "receipt_set_hash": _require_hash(
                receipt_set_hash, "receipt_set_hash"
            ),
            "historical_index_hash": _require_hash(
                historical_index_hash, "historical_index_hash"
            ),
        }
    )


def _validate_candidate_index_aggregates(
    value: Mapping[str, object],
    records: tuple[QualityCandidateRecord, ...],
    decisions: tuple[QualityCandidateDecision, ...],
    exclusions: tuple[QualityCandidateExclusion, ...],
    *,
    require_minimum: bool,
) -> None:
    historical_k = _nonnegative_int(
        value.get("historical_k"), "candidate_index.historical_k"
    )
    if historical_k > 25:
        raise ContractError(
            "candidate_index.historical_k: expected at most 25"
        )
    required = 3 * (50 - historical_k)
    if value.get("required_candidate_count") != required:
        raise ContractError(
            "candidate_index.required_candidate_count: mismatch"
        )
    if value.get("capture_count") != len(records) + len(exclusions):
        raise ContractError("candidate_index.capture_count: mismatch")
    if value.get("candidate_count") != len(records):
        raise ContractError("candidate_index.candidate_count: mismatch")
    if value.get("exclusion_count") != len(exclusions):
        raise ContractError("candidate_index.exclusion_count: mismatch")
    if value.get("exclusions") != [
        item.to_dict() for item in exclusions
    ]:
        raise ContractError("candidate_index.exclusions: mismatch")
    eligible = sum(
        decision.disposition != "hard_rejected" for decision in decisions
    )
    if value.get("eligible_candidate_count") != eligible:
        raise ContractError(
            "candidate_index.eligible_candidate_count: mismatch"
        )
    if require_minimum and len(records) < required:
        raise ContractError(
            "candidate_index.candidate_count: below required "
            f"minimum {required}"
        )
    counts = {
        status: sum(item.disposition == status for item in decisions)
        for status in QUALITY_CANDIDATE_STATUSES
    }
    if value.get("disposition_counts") != counts:
        raise ContractError(
            "candidate_index.disposition_counts: mismatch"
        )
    expected_families = [
        family
        for family in CONTRACT_FAMILIES
        if any(family in item.proposed_contract_families for item in records)
    ]
    if value.get("proposed_contract_families") != expected_families:
        raise ContractError(
            "candidate_index.proposed_contract_families: mismatch"
        )
    expected_triggers = [
        trigger
        for trigger in TRIGGER_TAGS
        if any(trigger in item.proposed_trigger_tags for item in records)
    ]
    if value.get("proposed_trigger_tags") != expected_triggers:
        raise ContractError(
            "candidate_index.proposed_trigger_tags: mismatch"
        )
    context = value.get("execution_context_summary")
    expected_context = {
        "devices": [
            item
            for item in DEVICES
            if any(item in record.execution_hints.devices for record in records)
        ],
        "modes": [
            item
            for item in MODES
            if any(item in record.execution_hints.modes for record in records)
        ],
        "phases": [
            item
            for item in PHASES
            if any(item in record.execution_hints.phases for record in records)
        ],
        "candidate_counts": {
            **{
                item: sum(
                    item in record.execution_hints.devices for record in records
                )
                for item in DEVICES
            },
            **{
                item: sum(
                    item in record.execution_hints.modes for record in records
                )
                for item in MODES
            },
            **{
                item: sum(
                    item in record.execution_hints.phases for record in records
                )
                for item in PHASES
            },
        },
    }
    if context != expected_context:
        raise ContractError(
            "candidate_index.execution_context_summary: mismatch"
        )


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


def _text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{path}: expected boolean")
    return value


def _nonnegative_int(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{path}: expected non-negative integer")
    return value


def _positive_int(value: object, path: str) -> int:
    result = _nonnegative_int(value, path)
    if result < 1:
        raise ContractError(f"{path}: expected positive integer")
    return result


def _enum(
    value: object,
    path: str,
    allowed: Sequence[str],
) -> str:
    text = _string(value, path)
    if text not in allowed:
        raise ContractError(f"{path}: unsupported value")
    return text


def _optional_commit(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ContractError(f"{path}: expected 40-character lowercase Git SHA")
    return value


def _timestamp(value: object, path: str) -> str:
    if not isinstance(value, str) or _UTC_SECONDS.fullmatch(value) is None:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds") from exc
    return value


def _canonical_strings(
    value: object,
    path: str,
    *,
    allow_empty: bool,
    preserve_order: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ContractError(f"{path}: expected array")
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if not allow_empty and not result:
        raise ContractError(f"{path}: expected non-empty array")
    if len(set(result)) != len(result):
        raise ContractError(f"{path}: duplicate value")
    if not preserve_order and result != tuple(sorted(result)):
        raise ContractError(f"{path}: expected lexical order")
    return result


def _registry_tuple(
    value: object,
    path: str,
    registry: Sequence[str],
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    result = _canonical_strings(
        value,
        path,
        allow_empty=allow_empty,
        preserve_order=True,
    )
    if not set(result) <= set(registry):
        raise ContractError(f"{path}: unsupported value")
    expected = tuple(item for item in registry if item in result)
    if result != expected:
        raise ContractError(f"{path}: expected registry order")
    return result


def _parse_candidate_execution_context(
    value: object,
    path: str,
) -> ExecutionContext:
    data = _exact_mapping(
        value,
        path,
        ("devices", "modes", "phases", "distributed"),
    )
    return ExecutionContext(
        devices=_registry_tuple(
            data["devices"],
            f"{path}.devices",
            DEVICES,
            allow_empty=False,
        ),
        modes=_registry_tuple(
            data["modes"],
            f"{path}.modes",
            MODES,
            allow_empty=False,
        ),
        phases=_registry_tuple(
            data["phases"],
            f"{path}.phases",
            PHASES,
            allow_empty=False,
        ),
        distributed=_boolean(
            data["distributed"], f"{path}.distributed"
        ),
    )


def _validate_candidate_execution_context(
    value: object,
    path: str,
) -> None:
    if not isinstance(value, ExecutionContext):
        raise ContractError(f"{path}: expected ExecutionContext")
    _registry_tuple(
        value.devices, f"{path}.devices", DEVICES, allow_empty=False
    )
    _registry_tuple(
        value.modes, f"{path}.modes", MODES, allow_empty=False
    )
    _registry_tuple(
        value.phases, f"{path}.phases", PHASES, allow_empty=False
    )
    _boolean(value.distributed, f"{path}.distributed")


def _execution_context_dict(value: ExecutionContext) -> dict[str, object]:
    return {
        "devices": list(value.devices),
        "modes": list(value.modes),
        "phases": list(value.phases),
        "distributed": value.distributed,
    }


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
    "HARD_CANDIDATE_REJECTION_REASONS",
    "QUALITY_CANDIDATE_STATUSES",
    "QualityCandidateAcquisitionReceipt",
    "QualityCandidateDecision",
    "QualityCandidateExclusion",
    "QualityCandidateRecord",
    "QualityChangedFile",
    "QualityLinkedIssue",
    "QualityTaskRecord",
    "build_historical_dispositions",
    "load_quality_candidate_captures",
    "validate_candidate_index",
    "validate_quality_task",
    "write_quality_candidate_funnel",
    "write_historical_dispositions",
]
