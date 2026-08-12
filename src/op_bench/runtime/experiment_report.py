from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import EventRecord, IntegrityReport
from op_bench.runtime.integrity import load_run_manifest_artifact, verify_run_artifacts
from op_bench.runtime.mcp import McpAdapterTrace
from op_bench.runtime.resume import parse_attempt_ledger
from op_bench.runtime.task_view import assert_public_artifact_safe, project_agent_task_view
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
)
from op_bench.runtime.workspace import _patch_paths_from_bytes


_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_REPORT_FILES = (
    "experiment_index.json",
    "experiment_summary.json",
    "experiment_report.md",
)


@dataclass(frozen=True)
class McpExperimentCohortBinding:
    """Content identities that make one Agent cohort comparable."""

    run_manifest_digest: str
    runtime_profile_digest: str
    capability_policy_digest: str
    budget_policy_digest: str
    task_view_digests: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("run_manifest_digest", self.run_manifest_digest),
            ("runtime_profile_digest", self.runtime_profile_digest),
            ("capability_policy_digest", self.capability_policy_digest),
            ("budget_policy_digest", self.budget_policy_digest),
        ):
            require_str(value, name, pattern=r"sha256:[0-9a-f]{64}")
        if not isinstance(self.task_view_digests, tuple) or not self.task_view_digests:
            raise ContractError("task_view_digests: expected non-empty tuple")
        task_ids: list[str] = []
        for task_id, digest in self.task_view_digests:
            task_ids.append(require_str(task_id, "task_view_digests.task_id"))
            require_str(
                digest,
                "task_view_digests.digest",
                pattern=r"sha256:[0-9a-f]{64}",
            )
        if tuple(sorted(set(task_ids))) != tuple(sorted(task_ids)):
            raise ContractError("task_view_digests: duplicate task ID")

    def to_dict(self) -> dict[str, object]:
        return {
            "run_manifest_digest": self.run_manifest_digest,
            "runtime_profile_digest": self.runtime_profile_digest,
            "capability_policy_digest": self.capability_policy_digest,
            "budget_policy_digest": self.budget_policy_digest,
            "task_view_digests": [
                {"task_id": task_id, "digest": digest}
                for task_id, digest in self.task_view_digests
            ],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "cohort.binding",
    ) -> "McpExperimentCohortBinding":
        data = require_exact_fields(
            value,
            path,
            (
                "run_manifest_digest",
                "runtime_profile_digest",
                "capability_policy_digest",
                "budget_policy_digest",
                "task_view_digests",
            ),
        )
        task_views: list[tuple[str, str]] = []
        for index, item in enumerate(
            require_list(data["task_view_digests"], f"{path}.task_view_digests")
        ):
            selected = require_exact_fields(
                item,
                f"{path}.task_view_digests[{index}]",
                ("task_id", "digest"),
            )
            task_views.append(
                (
                    require_str(
                        selected["task_id"],
                        f"{path}.task_view_digests[{index}].task_id",
                    ),
                    require_str(
                        selected["digest"],
                        f"{path}.task_view_digests[{index}].digest",
                        pattern=r"sha256:[0-9a-f]{64}",
                    ),
                )
            )
        return cls(
            run_manifest_digest=require_str(
                data["run_manifest_digest"],
                f"{path}.run_manifest_digest",
            ),
            runtime_profile_digest=require_str(
                data["runtime_profile_digest"],
                f"{path}.runtime_profile_digest",
            ),
            capability_policy_digest=require_str(
                data["capability_policy_digest"],
                f"{path}.capability_policy_digest",
            ),
            budget_policy_digest=require_str(
                data["budget_policy_digest"],
                f"{path}.budget_policy_digest",
            ),
            task_view_digests=tuple(task_views),
        )


@dataclass(frozen=True)
class McpExperimentFrozenConfig:
    """Global identities shared by every cohort in a frozen experiment."""

    release_digest: str
    adapter_id: str
    model_id: str
    codex_cli_version: str
    agent_spec_digest: str
    system_prompt_digest: str
    task_prompt_digest: str
    prompt_renderer_digest: str
    action_protocol: str
    evaluation_protocol: str
    scoring_protocol: str
    evaluation_digest: str
    retry_policy_digest: str
    termination_policy_digest: str
    scoring_digest: str

    def __post_init__(self) -> None:
        require_str(self.adapter_id, "adapter_id")
        require_str(self.model_id, "model_id")
        require_str(self.codex_cli_version, "codex_cli_version")
        require_str(self.action_protocol, "action_protocol")
        require_str(self.evaluation_protocol, "evaluation_protocol")
        require_str(self.scoring_protocol, "scoring_protocol")
        for name in (
            "release_digest",
            "agent_spec_digest",
            "system_prompt_digest",
            "task_prompt_digest",
            "prompt_renderer_digest",
            "evaluation_digest",
            "retry_policy_digest",
            "termination_policy_digest",
            "scoring_digest",
        ):
            require_str(
                getattr(self, name),
                name,
                pattern=r"sha256:[0-9a-f]{64}",
            )

    def to_dict(self) -> dict[str, str]:
        return {
            name: getattr(self, name)
            for name in (
                "release_digest",
                "adapter_id",
                "model_id",
                "codex_cli_version",
                "agent_spec_digest",
                "system_prompt_digest",
                "task_prompt_digest",
                "prompt_renderer_digest",
                "action_protocol",
                "evaluation_protocol",
                "scoring_protocol",
                "evaluation_digest",
                "retry_policy_digest",
                "termination_policy_digest",
                "scoring_digest",
            )
        }

    @classmethod
    def from_dict(cls, value: object) -> "McpExperimentFrozenConfig":
        fields = (
            "release_digest",
            "adapter_id",
            "model_id",
            "codex_cli_version",
            "agent_spec_digest",
            "system_prompt_digest",
            "task_prompt_digest",
            "prompt_renderer_digest",
            "action_protocol",
            "evaluation_protocol",
            "scoring_protocol",
            "evaluation_digest",
            "retry_policy_digest",
            "termination_policy_digest",
            "scoring_digest",
        )
        data = require_exact_fields(value, "frozen_config", fields)
        return cls(**{name: require_str(data[name], f"frozen_config.{name}") for name in fields})


@dataclass(frozen=True)
class McpExperimentCohortContract:
    profile_id: str
    task_repeats: tuple[tuple[str, tuple[int, ...]], ...]
    binding: McpExperimentCohortBinding | None = None

    def __post_init__(self) -> None:
        require_str(self.profile_id, "profile_id")
        if not isinstance(self.task_repeats, tuple) or not self.task_repeats:
            raise ContractError("task_repeats: expected non-empty tuple")
        task_ids: list[str] = []
        for task_id, repeats in self.task_repeats:
            selected_task = require_str(task_id, "task_id")
            if not isinstance(repeats, tuple) or not repeats:
                raise ContractError("task repeats: expected non-empty tuple")
            if tuple(sorted(set(repeats))) != repeats or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in repeats
            ):
                raise ContractError("task repeats: expected sorted positive unique integers")
            task_ids.append(selected_task)
        if tuple(sorted(set(task_ids))) != tuple(sorted(task_ids)):
            raise ContractError("task_repeats: duplicate task ID")
        if self.binding is not None:
            if not isinstance(self.binding, McpExperimentCohortBinding):
                raise ContractError("binding: expected MCP experiment cohort binding")
            if {task_id for task_id, _ in self.binding.task_view_digests} != set(task_ids):
                raise ContractError("binding: TaskView identities must match cohort Tasks")

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task_id for task_id, _ in self.task_repeats)

    @property
    def expected_attempts(self) -> frozenset[tuple[str, int]]:
        return frozenset(
            (task_id, repeat)
            for task_id, repeats in self.task_repeats
            for repeat in repeats
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "profile_id": self.profile_id,
            "task_repeats": [
                {
                    "task_id": task_id,
                    "repeats": list(repeats),
                }
                for task_id, repeats in self.task_repeats
            ],
        }
        if self.binding is not None:
            payload["binding"] = self.binding.to_dict()
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "cohort",
    ) -> "McpExperimentCohortContract":
        if not isinstance(value, Mapping):
            raise ContractError(f"{path}: expected object")
        fields = (
            ("profile_id", "task_repeats", "binding")
            if "binding" in value
            else ("profile_id", "task_repeats")
        )
        data = require_exact_fields(value, path, fields)
        task_repeats: list[tuple[str, tuple[int, ...]]] = []
        for index, item in enumerate(
            require_list(data["task_repeats"], f"{path}.task_repeats")
        ):
            selected = require_exact_fields(
                item,
                f"{path}.task_repeats[{index}]",
                ("task_id", "repeats"),
            )
            repeats = require_list(
                selected["repeats"],
                f"{path}.task_repeats[{index}].repeats",
            )
            task_repeats.append(
                (
                    require_str(
                        selected["task_id"],
                        f"{path}.task_repeats[{index}].task_id",
                    ),
                    tuple(
                        require_int(
                            repeat,
                            f"{path}.task_repeats[{index}].repeats[{repeat_index}]",
                            minimum=1,
                        )
                        for repeat_index, repeat in enumerate(repeats)
                    ),
                )
            )
        return cls(
            profile_id=require_str(data["profile_id"], f"{path}.profile_id"),
            task_repeats=tuple(task_repeats),
            binding=(
                McpExperimentCohortBinding.from_dict(
                    data["binding"], path=f"{path}.binding"
                )
                if "binding" in data
                else None
            ),
        )


@dataclass(frozen=True)
class McpExperimentContract:
    dataset_identifier: str
    dataset_digest: str
    platform_version: str
    cohorts: tuple[McpExperimentCohortContract, ...]
    frozen_config: McpExperimentFrozenConfig | None = None

    def __post_init__(self) -> None:
        require_str(self.dataset_identifier, "dataset_identifier")
        require_str(
            self.dataset_digest,
            "dataset_digest",
            pattern=r"sha256:[0-9a-f]{64}",
        )
        require_str(self.platform_version, "platform_version")
        if not isinstance(self.cohorts, tuple) or not self.cohorts:
            raise ContractError("cohorts: expected non-empty cohort contracts")
        if not all(isinstance(item, McpExperimentCohortContract) for item in self.cohorts):
            raise ContractError("cohorts: expected MCP cohort contracts")
        all_tasks = [task_id for cohort in self.cohorts for task_id in cohort.task_ids]
        if len(all_tasks) != len(set(all_tasks)):
            raise ContractError("cohorts: task IDs must be partitioned exactly once")
        bound = tuple(item.binding is not None for item in self.cohorts)
        if self.frozen_config is None:
            if any(bound):
                raise ContractError("frozen_config: required for bound cohorts")
        else:
            if not isinstance(self.frozen_config, McpExperimentFrozenConfig):
                raise ContractError("frozen_config: expected frozen MCP configuration")
            if not all(bound):
                raise ContractError("cohorts: every frozen cohort requires a binding")

    @property
    def expected_attempt_count(self) -> int:
        return sum(len(item.expected_attempts) for item in self.cohorts)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "v2" if self.frozen_config is not None else "v1",
            "dataset_identifier": self.dataset_identifier,
            "dataset_digest": self.dataset_digest,
            "platform_version": self.platform_version,
            "cohorts": [cohort.to_dict() for cohort in self.cohorts],
        }
        if self.frozen_config is not None:
            payload["frozen_config"] = self.frozen_config.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: object) -> "McpExperimentContract":
        if not isinstance(value, Mapping):
            raise ContractError("mcp_experiment_contract: expected object")
        schema_version = value.get("schema_version")
        if schema_version not in {"v1", "v2"}:
            raise ContractError("schema_version: expected 'v1' or 'v2'")
        fields = (
            (
                "schema_version",
                "dataset_identifier",
                "dataset_digest",
                "platform_version",
                "cohorts",
                "frozen_config",
            )
            if schema_version == "v2"
            else (
                "schema_version",
                "dataset_identifier",
                "dataset_digest",
                "platform_version",
                "cohorts",
            )
        )
        data = require_exact_fields(
            value,
            "mcp_experiment_contract",
            fields,
        )
        return cls(
            dataset_identifier=require_str(
                data["dataset_identifier"],
                "dataset_identifier",
            ),
            dataset_digest=require_str(
                data["dataset_digest"],
                "dataset_digest",
            ),
            platform_version=require_str(
                data["platform_version"],
                "platform_version",
            ),
            cohorts=tuple(
                McpExperimentCohortContract.from_dict(
                    item,
                    path=f"cohorts[{index}]",
                )
                for index, item in enumerate(
                    require_list(data["cohorts"], "cohorts")
                )
            ),
            frozen_config=(
                McpExperimentFrozenConfig.from_dict(data["frozen_config"])
                if schema_version == "v2"
                else None
            ),
        )


FORMAL_MCP_EXPERIMENT_CONTRACT = McpExperimentContract(
    dataset_identifier="pytorch_v0.5",
    dataset_digest="sha256:ff9d0c2999d1175a45165b387e0731dcaa211a190d994b176441ce81a0382abc",
    platform_version="opbench-v0.6.0",
    cohorts=(
        McpExperimentCohortContract(
            profile_id="remote-cpu-pytorch-2.6-py311-v1",
            task_repeats=tuple(
                (task_id, (1, 2, 3))
                for task_id in (
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
                    "pytorch__140557__layer_norm_decomp_precision",
                    "pytorch__139999__masked_mean_bool_upcast",
                )
            ),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cpu-compile-pytorch-2.6-py311-v1",
            task_repeats=(("pytorch__129138__linear_add_bias_autocast", (1, 2, 3)),),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cuda-overlay-pytorch-2.6-cu124-v1",
            task_repeats=(
                ("pytorch__132835__njt_sdpa_autocast", (1, 2, 3)),
                ("pytorch__132616__cuda_mem_get_info", (1, 2, 3)),
            ),
        ),
        McpExperimentCohortContract(
            profile_id="remote-cuda-kernel-pytorch-2.6-cu124-v1",
            task_repeats=(
                ("pytorch__144009__softmax_ilpreduce_size", (1, 2, 3)),
                ("pytorch__139372__histc_int8_cuda_bounds", (1, 2, 3)),
            ),
        ),
    ),
)


class _RunReader:
    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ContractError("run_root: expected Path")
        if root.is_symlink() or not root.is_dir():
            raise ContractError("run_root: expected real directory")
        self.root = root
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        try:
            self.descriptor = os.open(root, flags)
        except OSError as exc:
            raise ContractError("run_root: expected real directory") from exc
        metadata = os.fstat(self.descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(self.descriptor)
            raise ContractError("run_root: expected real directory")
        self.identity = metadata.st_dev, metadata.st_ino

    def close(self) -> None:
        descriptor = getattr(self, "descriptor", None)
        if descriptor is not None:
            os.close(descriptor)
            self.descriptor = None

    def read(self, *components: str, optional: bool = False) -> bytes | None:
        if self.descriptor is None:
            raise ContractError("run reader is closed")
        if not components:
            raise ContractError("artifact path: expected components")
        for component in components:
            if (
                not isinstance(component, str)
                or not component
                or component in {".", ".."}
                or "/" in component
                or "\\" in component
            ):
                raise ContractError("artifact path: invalid component")
        root_metadata = os.fstat(self.descriptor)
        if (root_metadata.st_dev, root_metadata.st_ino) != self.identity:
            raise ContractError("run_root: directory binding changed")
        opened: list[int] = []
        current = self.descriptor
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(
            os,
            "O_NOFOLLOW",
            0,
        )
        try:
            for component in components[:-1]:
                try:
                    selected = os.open(component, directory_flags, dir_fd=current)
                except OSError as exc:
                    if optional and isinstance(exc, FileNotFoundError):
                        return None
                    raise ContractError("run artifact directory is missing or invalid") from exc
                opened.append(selected)
                current = selected
            try:
                descriptor = os.open(
                    components[-1],
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current,
                )
            except OSError as exc:
                if optional and isinstance(exc, FileNotFoundError):
                    return None
                raise ContractError("run artifact is missing or invalid") from exc
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError("run artifact is not a regular file")
                if metadata.st_size > _MAX_ARTIFACT_BYTES:
                    raise ContractError("run artifact exceeds size limit")
                chunks: list[bytes] = []
                remaining = _MAX_ARTIFACT_BYTES + 1
                while remaining:
                    chunk = os.read(descriptor, min(65_536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                if len(raw) > _MAX_ARTIFACT_BYTES:
                    raise ContractError("run artifact exceeds size limit")
                return raw
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(opened):
                os.close(descriptor)


def _canonical_object(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ContractError(f"{label}: invalid JSON") from None
    if not isinstance(value, dict):
        raise ContractError(f"{label}: expected object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise ContractError(f"{label}: expected canonical JSON")
    return value


def load_mcp_experiment_contract(path: Path | str) -> McpExperimentContract:
    contract_path = Path(path)
    if contract_path.is_symlink():
        raise ContractError("contract_path: symlink is denied")
    try:
        descriptor = os.open(
            contract_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ContractError("contract_path: cannot open regular file") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("contract_path: expected regular file")
        if metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise ContractError("contract_path: file exceeds size limit")
        chunks: list[bytes] = []
        remaining = _MAX_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_ARTIFACT_BYTES:
            raise ContractError("contract_path: file exceeds size limit")
    finally:
        os.close(descriptor)
    return McpExperimentContract.from_dict(
        _canonical_object(raw, "MCP experiment contract")
    )


def load_public_task_id_aliases(path: Path | str) -> dict[str, str]:
    """Load the frozen canonical-to-public Task identity projection."""

    mapping_path = Path(path)
    if mapping_path.is_symlink():
        raise ContractError("public_task_id_mapping: symlink is denied")
    try:
        descriptor = os.open(
            mapping_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise ContractError(
            "public_task_id_mapping: cannot open regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("public_task_id_mapping: expected regular file")
        if metadata.st_size > _MAX_ARTIFACT_BYTES:
            raise ContractError("public_task_id_mapping: file exceeds size limit")
        chunks: list[bytes] = []
        remaining = _MAX_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65_536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > _MAX_ARTIFACT_BYTES:
            raise ContractError("public_task_id_mapping: file exceeds size limit")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise ContractError("public_task_id_mapping: invalid JSON") from None
    if not isinstance(value, dict):
        raise ContractError("public_task_id_mapping: expected object")
    if raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise ContractError("public_task_id_mapping: expected canonical JSON")
    data = require_exact_fields(
        value,
        "public_task_id_mapping",
        ("contract_type", "schema_version", "tasks"),
    )
    if (
        data["contract_type"] != "public_task_id_mapping"
        or data["schema_version"] != "v1"
    ):
        raise ContractError("public_task_id_mapping: unsupported contract")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(
        require_list(data["tasks"], "public_task_id_mapping.tasks")
    ):
        entry = require_exact_fields(
            item,
            f"public_task_id_mapping.tasks[{index}]",
            ("public_task_id", "task_id"),
        )
        pairs.append(
            (
                require_str(
                    entry["task_id"],
                    f"public_task_id_mapping.tasks[{index}].task_id",
                ),
                require_str(
                    entry["public_task_id"],
                    f"public_task_id_mapping.tasks[{index}].public_task_id",
                ),
            )
        )
    if not pairs:
        raise ContractError("public_task_id_mapping.tasks: expected non-empty array")
    canonical_ids = tuple(task_id for task_id, _ in pairs)
    public_ids = tuple(public_task_id for _, public_task_id in pairs)
    if canonical_ids != tuple(sorted(set(canonical_ids))):
        raise ContractError(
            "public_task_id_mapping.tasks: canonical IDs must be sorted and unique"
        )
    if len(public_ids) != len(set(public_ids)):
        raise ContractError(
            "public_task_id_mapping.tasks: public IDs must be unique"
        )
    return dict(pairs)


def _canonical_lines(raw: bytes, label: str) -> list[dict[str, object]]:
    if raw and not raw.endswith(b"\n"):
        raise ContractError(f"{label}: missing final newline")
    result: list[dict[str, object]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ContractError(f"{label} line {index}: invalid JSON") from None
        if not isinstance(value, dict):
            raise ContractError(f"{label} line {index}: expected object")
        if line != canonical_json(value).encode("utf-8"):
            raise ContractError(f"{label} line {index}: expected canonical JSON")
        result.append(value)
    return result


def _event_records(raw: bytes) -> tuple[EventRecord, ...]:
    return tuple(
        EventRecord.from_dict(value, path=f"events[{index}]")
        for index, value in enumerate(_canonical_lines(raw, "events.jsonl"))
    )


def _distribution(values: Sequence[int]) -> dict[str, object]:
    selected = tuple(values)
    if not selected:
        return {
            "count": 0,
            "sum": 0,
            "min": None,
            "max": None,
            "mean": {"numerator": 0, "denominator": 0},
        }
    return {
        "count": len(selected),
        "sum": sum(selected),
        "min": min(selected),
        "max": max(selected),
        "mean": {"numerator": sum(selected), "denominator": len(selected)},
    }


def _quality_category_summary(
    attempt_rows: Sequence[Mapping[str, object]],
    task_metadata: Mapping[str, Mapping[str, object]],
    memberships: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for category in sorted(memberships):
        task_ids = frozenset(memberships[category])
        rows = [row for row in attempt_rows if row.get("task_id") in task_ids]
        outcomes = Counter(str(row["evaluation_outcome"]) for row in rows)
        resolved = outcomes["resolved"]
        eligible = len(task_ids) >= 3
        result[category] = {
            "task_count": len(task_ids),
            "attempt_count": len(rows),
            "outcomes": dict(sorted(outcomes.items())),
            "standalone_score_eligible": eligible,
            "resolved_rate": (
                {"numerator": resolved, "denominator": len(rows)}
                if eligible
                else None
            ),
        }
    return result


def add_quality_experiment_metadata(
    index: dict[str, object],
    summary: dict[str, object],
    task_metadata: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    """Join public v0.7 taxonomy metadata and publish aggregate-only views."""

    raw_attempts = index.get("attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ContractError("quality report: expected non-empty Attempt rows")
    attempts: list[Mapping[str, object]] = []
    task_ids: set[str] = set()
    for row in raw_attempts:
        if not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str):
            raise ContractError("quality report: invalid Attempt row")
        attempts.append(row)
        task_ids.add(str(row["task_id"]))
    if set(task_metadata) != task_ids:
        raise ContractError("quality report: Task metadata partition mismatch")

    public_tasks: list[dict[str, object]] = []
    origin_groups: dict[str, list[str]] = {}
    difficulty_groups: dict[str, list[str]] = {}
    family_groups: dict[str, list[str]] = {}
    failure_groups: dict[str, list[str]] = {}
    device_groups: dict[str, list[str]] = {}
    mode_groups: dict[str, list[str]] = {}
    phase_groups: dict[str, list[str]] = {}
    slice_groups: dict[str, list[str]] = {}

    def add(groups: dict[str, list[str]], category: str, task_id: str) -> None:
        groups.setdefault(category, []).append(task_id)

    for task_id in sorted(task_metadata):
        metadata = task_metadata[task_id]
        origin = require_str(metadata.get("origin"), f"task_metadata[{task_id}].origin")
        origin_group = "retained_historical" if origin == "retained_historical" else "new_or_replacement"
        difficulty = require_str(
            metadata.get("difficulty"), f"task_metadata[{task_id}].difficulty"
        )
        taxonomy = metadata.get("taxonomy")
        if not isinstance(taxonomy, Mapping):
            raise ContractError("quality report: invalid taxonomy metadata")
        family = require_str(
            taxonomy.get("contract_family"),
            f"task_metadata[{task_id}].taxonomy.contract_family",
        )
        failure_type = require_str(
            taxonomy.get("failure_type"),
            f"task_metadata[{task_id}].taxonomy.failure_type",
        )
        dimensions: dict[str, tuple[str, ...]] = {}
        for name in ("devices", "modes", "phases"):
            raw_values = taxonomy.get(name)
            if not isinstance(raw_values, list) or not raw_values:
                raise ContractError(f"quality report: invalid {name} metadata")
            values = tuple(require_str(value, f"task_metadata[{task_id}].{name}") for value in raw_values)
            dimensions[name] = values
        raw_slices = metadata.get("slices")
        if not isinstance(raw_slices, list) or "cumulative" not in raw_slices:
            raise ContractError("quality report: invalid derived Slice metadata")
        slices = tuple(
            require_str(value, f"task_metadata[{task_id}].slices")
            for value in raw_slices
            if value != "cumulative"
        )
        public_tasks.append(
            {
                "task_id": task_id,
                "origin": origin_group,
                "difficulty": difficulty,
                "contract_family": family,
                "failure_type": failure_type,
                "devices": list(dimensions["devices"]),
                "modes": list(dimensions["modes"]),
                "phases": list(dimensions["phases"]),
                "derived_slices": list(slices),
            }
        )
        add(origin_groups, origin_group, task_id)
        add(difficulty_groups, difficulty, task_id)
        add(family_groups, family, task_id)
        add(failure_groups, failure_type, task_id)
        for value in dimensions["devices"]:
            add(device_groups, value, task_id)
        for value in dimensions["modes"]:
            add(mode_groups, value, task_id)
        for value in dimensions["phases"]:
            add(phase_groups, value, task_id)
        for value in slices:
            add(slice_groups, value, task_id)

    if set(slice_groups) != {"boundary", "device", "precision"}:
        raise ContractError("quality report: final derived Slice coverage mismatch")
    outcomes_by_task: dict[str, Counter[str]] = {task_id: Counter() for task_id in task_ids}
    for row in attempts:
        outcomes_by_task[str(row["task_id"])][str(row["evaluation_outcome"])] += 1
    all_resolved = sorted(
        task_id
        for task_id, task_outcomes in outcomes_by_task.items()
        if task_outcomes["resolved"] == sum(task_outcomes.values())
    )
    zero_resolved = sorted(
        task_id for task_id, task_outcomes in outcomes_by_task.items() if not task_outcomes["resolved"]
    )
    totals = summary.get("totals")
    attribution = summary.get("attribution")
    if not isinstance(totals, dict) or not isinstance(attribution, dict):
        raise ContractError("quality report: base summary is invalid")
    total_attempts = require_int(totals.get("attempts"), "totals.attempts", minimum=1)
    resolved_attempts = sum(
        1 for row in attempts if row.get("evaluation_outcome") == "resolved"
    )
    infrastructure_retries = sum(
        int(attribution.get(name, 0)) for name in ("provider", "mcp", "runtime")
    )
    totals["accepted_invalid"] = 0
    quality = {
        "task_count": len(task_ids),
        "attempt_count": total_attempts,
        "origin": _quality_category_summary(attempts, task_metadata, origin_groups),
        "difficulty": _quality_category_summary(attempts, task_metadata, difficulty_groups),
        "contract_family": _quality_category_summary(attempts, task_metadata, family_groups),
        "failure_type": _quality_category_summary(attempts, task_metadata, failure_groups),
        "devices": _quality_category_summary(attempts, task_metadata, device_groups),
        "modes": _quality_category_summary(attempts, task_metadata, mode_groups),
        "phases": _quality_category_summary(attempts, task_metadata, phase_groups),
        "derived_slices": _quality_category_summary(attempts, task_metadata, slice_groups),
        "failure_attribution": {
            "task": 0,
            "agent": total_attempts - resolved_attempts,
            "evaluator": 0,
            "runtime": int(attribution.get("runtime", 0)),
            "infrastructure_retries": infrastructure_retries,
        },
        "ceiling_floor_observations": {
            "all_resolved_task_ids": all_resolved,
            "zero_resolved_task_ids": zero_resolved,
            "interpretation": "descriptive_current_configuration_only",
        },
    }
    quality_index = dict(index)
    quality_summary = dict(summary)
    quality_index["quality_tasks"] = public_tasks
    quality_summary["quality"] = quality
    for field in (
        "origin",
        "difficulty",
        "contract_family",
        "failure_type",
        "devices",
        "modes",
        "phases",
        "derived_slices",
        "failure_attribution",
        "ceiling_floor_observations",
    ):
        quality_summary[field] = quality[field]
    assert_public_artifact_safe(quality_index)
    assert_public_artifact_safe(quality_summary)
    return quality_index, quality_summary


def build_mcp_experiment_report(
    run_roots: Sequence[Path],
    *,
    expected_adapter_id: str,
    expected_model_id: str,
    expected_codex_cli_version: str,
    experiment_contract: McpExperimentContract,
    task_id_aliases: Mapping[str, str] | None = None,
    task_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(run_roots, (str, bytes)) or not isinstance(run_roots, Sequence):
        raise ContractError("run_roots: expected a sequence of Paths")
    roots = tuple(run_roots)
    adapter_id = require_str(expected_adapter_id, "expected_adapter_id")
    model_id = require_str(expected_model_id, "expected_model_id")
    cli_version = require_str(
        expected_codex_cli_version,
        "expected_codex_cli_version",
    )
    if adapter_id != "codex_mcp_canonical":
        raise ContractError("expected_adapter_id: expected codex_mcp_canonical")
    if not isinstance(experiment_contract, McpExperimentContract):
        raise ContractError("experiment_contract: expected McpExperimentContract")
    frozen_config = experiment_contract.frozen_config
    if frozen_config is not None and (
        frozen_config.adapter_id != adapter_id
        or frozen_config.model_id != model_id
        or frozen_config.codex_cli_version != cli_version
    ):
        raise ContractError("frozen Agent configuration does not match report inputs")
    if len(roots) != len(experiment_contract.cohorts):
        raise ContractError("run_roots: count does not match cohort contract")
    if task_id_aliases is None:
        aliases: dict[str, str] | None = None
    else:
        if not isinstance(task_id_aliases, Mapping):
            raise ContractError("task_id_aliases: expected mapping")
        aliases = {}
        for canonical_id, public_id in task_id_aliases.items():
            canonical = require_str(canonical_id, "task_id_aliases key")
            public = require_str(public_id, f"task_id_aliases[{canonical!r}]")
            aliases[canonical] = public
        if len(aliases) != len(task_id_aliases):
            raise ContractError("task_id_aliases: duplicate canonical identity")
        if len(set(aliases.values())) != len(aliases):
            raise ContractError("task_id_aliases: duplicate public identity")

    def report_task_id(canonical_id: str) -> str:
        if aliases is None:
            return canonical_id
        try:
            return aliases[canonical_id]
        except KeyError as exc:
            raise ContractError(
                f"task_id_aliases: missing canonical identity {canonical_id!r}"
            ) from exc

    cohort_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    seen_attempts: set[str] = set()
    seen_comparability: set[str] = set()
    selected_counts: list[int] = []
    outcomes: Counter[str] = Counter()
    terminals: Counter[str] = Counter()
    action_calls: Counter[str] = Counter()
    action_errors: Counter[str] = Counter()
    mcp_terminals: Counter[str] = Counter()
    budgets = Counter()
    coverage = Counter()
    durations: list[int] = []
    patch_sizes: list[int] = []
    changed_files: list[int] = []
    retry_counts: list[int] = []
    trace_complete = 0
    provider_attribution = 0
    mcp_attribution = 0
    runtime_attribution = 0
    mcp_initialize = 0
    mcp_list = 0
    mcp_calls = 0
    mcp_protocol_errors = 0
    matched_contracts: set[int] = set()

    for root in roots:
        if not isinstance(root, Path):
            raise ContractError("run_roots: every root must be a Path")
        manifest = load_run_manifest_artifact(root)
        if (
            manifest.dataset.identifier != experiment_contract.dataset_identifier
            or manifest.dataset.digest != experiment_contract.dataset_digest
        ):
            raise ContractError("dataset identity does not match experiment contract")
        if manifest.platform_version != experiment_contract.platform_version:
            raise ContractError("platform identity does not match experiment contract")
        profile_ids = tuple(profile.profile_id for profile in manifest.runtime_profiles)
        if len(profile_ids) != 1:
            raise ContractError("each experiment root must use exactly one Runtime Profile")
        task_ids = frozenset(
            report_task_id(task.task.identifier) for task in manifest.tasks
        )
        candidates = [
            (index, cohort)
            for index, cohort in enumerate(experiment_contract.cohorts)
            if cohort.profile_id == profile_ids[0]
            and frozenset(cohort.task_ids) == task_ids
        ]
        if len(candidates) != 1:
            raise ContractError("cohort profile/task partition does not match contract")
        contract_index, cohort_contract = candidates[0]
        if contract_index in matched_contracts:
            raise ContractError("duplicate experiment cohort contract")
        matched_contracts.add(contract_index)
        observed_attempts = frozenset(
            (report_task_id(item.task.identifier), item.repeat)
            for item in manifest.expected_attempts
        )
        if observed_attempts != cohort_contract.expected_attempts:
            raise ContractError("cohort repeat matrix does not match contract")
        if cohort_contract.binding is not None:
            binding = cohort_contract.binding
            observed_views = tuple(
                project_agent_task_view(
                    task,
                    manifest.capability_policy,
                    manifest.budget_policy,
                )
                for task in manifest.tasks
            )
            observed_task_views = {
                report_task_id(view.task.identifier): view.content_hash
                for view in observed_views
            }
            if (
                canonical_sha256(manifest.to_dict()) != binding.run_manifest_digest
                or canonical_sha256(manifest.runtime_profiles[0].to_dict())
                != binding.runtime_profile_digest
                or canonical_sha256(manifest.capability_policy.to_dict())
                != binding.capability_policy_digest
                or canonical_sha256(manifest.budget_policy.to_dict())
                != binding.budget_policy_digest
                or observed_task_views != dict(binding.task_view_digests)
            ):
                raise ContractError("cohort frozen binding does not match run artifacts")
        if frozen_config is not None:
            if (
                manifest.action_protocol != frozen_config.action_protocol
                or manifest.evaluation_protocol != frozen_config.evaluation_protocol
                or manifest.scoring_protocol != frozen_config.scoring_protocol
                or manifest.evaluation.digest != frozen_config.evaluation_digest
                or manifest.retry_policy.digest != frozen_config.retry_policy_digest
                or manifest.termination_policy.digest
                != frozen_config.termination_policy_digest
                or manifest.scoring.digest != frozen_config.scoring_digest
                or len(manifest.agents) != 1
                or canonical_sha256(manifest.agents[0].to_dict())
                != frozen_config.agent_spec_digest
                or manifest.agents[0].system_prompt.digest
                != frozen_config.system_prompt_digest
                or manifest.agents[0].task_prompt.digest
                != frozen_config.task_prompt_digest
            ):
                raise ContractError("global frozen configuration does not match run artifacts")
        fresh_integrity = verify_run_artifacts(root, manifest)
        if fresh_integrity.status != "passed" or any(
            check.status != "passed" for check in fresh_integrity.checks
        ):
            raise ContractError("Integrity verification failed")
        if manifest.comparability_key in seen_comparability:
            raise ContractError("duplicate Comparability Key")
        seen_comparability.add(manifest.comparability_key)
        for agent in manifest.agents:
            if agent.adapter.identifier != adapter_id:
                raise ContractError("adapter identity mismatch")
            if agent.model.identifier != model_id:
                raise ContractError("model identity mismatch")

        reader = _RunReader(root)
        try:
            persisted_integrity = IntegrityReport.from_dict(
                _canonical_object(
                    reader.read("integrity.json"),
                    "integrity.json",
                )
            )
            if persisted_integrity != fresh_integrity:
                raise ContractError("persisted Integrity report does not match verification")
            results = _canonical_lines(
                reader.read("results.jsonl"),
                "results.jsonl",
            )
            ledger_records = parse_attempt_ledger(reader.read("attempts.jsonl"))
            histories: dict[str, list[object]] = {}
            for record in ledger_records:
                histories.setdefault(record.attempt_id, []).append(record)
            expected_by_id = {
                item.attempt_id: item for item in manifest.expected_attempts
            }
            tasks = {item.task.identifier: item for item in manifest.tasks}
            result_by_id: dict[str, dict[str, object]] = {}
            for result in results:
                attempt_id = require_str(result.get("attempt_id"), "result.attempt_id")
                if attempt_id in result_by_id:
                    raise ContractError("duplicate selected result Attempt ID")
                result_by_id[attempt_id] = result
            if set(result_by_id) != set(expected_by_id):
                raise ContractError("run root has missing or blocked selected Attempts")

            selected_counts.append(len(results))
            cohort_rows.append(
                {
                    "cohort_id": manifest.cohort_id,
                    "comparability_key": manifest.comparability_key,
                    "runtime_profile_ids": [
                        profile.profile_id for profile in manifest.runtime_profiles
                    ],
                    "selected_attempts": len(results),
                }
            )
            for attempt_id in sorted(result_by_id):
                if attempt_id in seen_attempts:
                    raise ContractError("duplicate Attempt ID across run roots")
                seen_attempts.add(attempt_id)
                expected = expected_by_id[attempt_id]
                result = result_by_id[attempt_id]
                history = histories.get(attempt_id, [])
                valid = [item for item in history if item.attempt_validity == "valid"]
                if len(valid) != 1:
                    raise ContractError(
                        "selected Attempt is missing or infrastructure-invalid"
                    )
                selected = valid[0]
                if result.get("retry_index") != selected.retry_index:
                    raise ContractError("selected retry attribution mismatch")
                if result.get("attempt_validity") != "valid":
                    raise ContractError("selected Attempt is infrastructure-invalid")
                if result.get("evaluation_result_hash") != selected.evaluation_result_hash:
                    raise ContractError("selected Evaluation result binding mismatch")

                retry_name = f"retry-{selected.retry_index:04d}"
                prefix = (
                    "attempts",
                    attempt_id,
                    "retries",
                    retry_name,
                )
                trace = McpAdapterTrace.from_dict(
                    _canonical_object(
                        reader.read(*prefix, "adapter_trace.json"),
                        "adapter_trace.json",
                    )
                )
                if trace.adapter_id != adapter_id:
                    raise ContractError("adapter trace identity mismatch")
                if trace.model_id != model_id:
                    raise ContractError("model trace identity mismatch")
                if trace.codex_cli_version != cli_version:
                    raise ContractError("Codex CLI version identity mismatch")
                events = _event_records(reader.read(*prefix, "events.jsonl"))
                requests = [
                    record
                    for record in events
                    if record.event_type == "action_requested"
                ]
                observations = [
                    record
                    for record in events
                    if record.event_type == "action_observed"
                ]
                if trace.tools_call_count != len(requests):
                    raise ContractError("MCP trace pairing is incomplete")
                request_ids = {
                    record.to_dict()["public_payload"]["action_id"]
                    for record in requests
                }
                observation_ids = {
                    record.to_dict()["public_payload"]["action_id"]
                    for record in observations
                }
                if request_ids != observation_ids:
                    raise ContractError("Action request/observation pairing is incomplete")
                trace_complete += 1
                mcp_initialize += trace.initialize_count
                mcp_list += trace.tools_list_count
                mcp_calls += trace.tools_call_count
                mcp_protocol_errors += trace.protocol_error_count
                mcp_terminals[trace.server_terminal_status] += 1

                action_names: set[str] = set()
                for record in requests:
                    payload = record.to_dict()["public_payload"]
                    name = require_str(payload.get("action_name"), "action_name")
                    action_calls[name] += 1
                    action_names.add(name)
                for record in observations:
                    payload = record.to_dict()["public_payload"]
                    if payload.get("ok") is not True:
                        action_errors[
                            require_str(payload.get("error_code"), "error_code")
                        ] += 1
                    delta = payload.get("budget_delta")
                    if not isinstance(delta, dict):
                        raise ContractError("Action Budget delta is missing")
                    for name in (
                        "wall_clock_ms",
                        "actions",
                        "tests",
                        "commands",
                        "output_bytes",
                        "provider_tokens",
                    ):
                        value = delta.get(name)
                        if isinstance(value, bool) or not isinstance(value, int):
                            raise ContractError("Action Budget delta is invalid")
                        budgets[name] += value
                categories = {
                    "read": {"workspace_list", "workspace_search", "workspace_read"},
                    "edit": {"workspace_write", "workspace_apply_patch"},
                    "test": {"test_run"},
                    "diff": {"vcs_diff"},
                    "finish": {"session_finish"},
                }
                for category, names in categories.items():
                    if action_names & names:
                        coverage[category] += 1

                patch_bytes = reader.read(*prefix, "final.patch")
                paths = _patch_paths_from_bytes(patch_bytes)
                patch_size = len(patch_bytes)
                changed_count = len(paths)
                patch_sizes.append(patch_size)
                changed_files.append(changed_count)
                retry_count = selected.retry_index - 1
                retry_counts.append(retry_count)
                outcome = require_str(result.get("evaluation_outcome"), "evaluation_outcome")
                terminal = require_str(result.get("agent_terminal"), "agent_terminal")
                duration = result.get("duration_ms")
                if isinstance(duration, bool) or not isinstance(duration, int) or duration < 0:
                    raise ContractError("duration_ms: expected non-negative integer")
                outcomes[outcome] += 1
                terminals[terminal] += 1
                durations.append(duration)

                for prior in history:
                    if prior.retry_index >= selected.retry_index:
                        continue
                    if prior.attempt_validity != "infrastructure_invalid":
                        raise ContractError("retry history has invalid attribution")
                    if prior.session_result.terminal_reason == "provider_error":
                        provider_attribution += 1
                        continue
                    prior_trace_raw = reader.read(
                        "attempts",
                        attempt_id,
                        "retries",
                        f"retry-{prior.retry_index:04d}",
                        "adapter_trace.json",
                        optional=True,
                    )
                    prior_trace = (
                        None
                        if prior_trace_raw is None
                        else McpAdapterTrace.from_dict(
                            _canonical_object(prior_trace_raw, "adapter_trace.json")
                        )
                    )
                    if (
                        prior.session_result.terminal_reason == "runtime_error"
                        and prior_trace is not None
                        and prior_trace.server_terminal_status == "protocol_failed"
                    ):
                        mcp_attribution += 1
                    else:
                        runtime_attribution += 1

                task = tasks[expected.task.identifier]
                attempt_rows.append(
                    {
                        "cohort_id": manifest.cohort_id,
                        "comparability_key": manifest.comparability_key,
                        "runtime_profile_id": task.runtime.profile_id,
                        "task_id": report_task_id(expected.task.identifier),
                        "repeat": expected.repeat,
                        "attempt_id": attempt_id,
                        "retry_index": selected.retry_index,
                        "evaluation_result_hash": selected.evaluation_result_hash,
                        "terminal_reason": selected.session_result.terminal_reason,
                        "agent_terminal": terminal,
                        "evaluation_outcome": outcome,
                        "duration_ms": duration,
                        "patch_size_bytes": patch_size,
                        "changed_file_count": changed_count,
                        "action_count": len(requests),
                        "action_error_count": sum(
                            record.to_dict()["public_payload"].get("ok") is not True
                            for record in observations
                        ),
                        "trace_complete": True,
                    }
                )
        finally:
            reader.close()

    expected_counts = tuple(
        sorted(len(item.expected_attempts) for item in experiment_contract.cohorts)
    )
    if tuple(sorted(selected_counts)) != expected_counts:
        raise ContractError("cohort selected counts do not match contract")
    if matched_contracts != set(range(len(experiment_contract.cohorts))):
        raise ContractError("experiment cohort contract is incomplete")
    attempt_rows.sort(
        key=lambda item: (
            item["cohort_id"],
            item["task_id"],
            item["repeat"],
            item["attempt_id"],
        )
    )
    cohort_rows.sort(key=lambda item: item["cohort_id"])
    total = len(attempt_rows)
    if total != experiment_contract.expected_attempt_count:
        raise ContractError("cohort report has the wrong selected Attempt count")
    identities = {
        "adapter_id": adapter_id,
        "model_id": model_id,
        "codex_cli_version": cli_version,
        "dataset_identifier": experiment_contract.dataset_identifier,
        "dataset_digest": experiment_contract.dataset_digest,
        "platform_version": experiment_contract.platform_version,
    }
    index: dict[str, object] = {
        "report_type": "mcp_experiment_index",
        "schema_version": "v1",
        **identities,
        "cohorts": cohort_rows,
        "attempts": attempt_rows,
    }
    summary: dict[str, object] = {
        "report_type": "mcp_experiment_summary",
        "schema_version": "v1",
        **identities,
        "totals": {
            "cohorts": len(cohort_rows),
            "attempts": total,
            "trace_complete": trace_complete,
            "retries": sum(retry_counts),
            "actions": sum(action_calls.values()),
            "action_errors": sum(action_errors.values()),
        },
        "evaluation_outcomes": dict(sorted(outcomes.items())),
        "agent_terminals": dict(sorted(terminals.items())),
        "action_calls": dict(sorted(action_calls.items())),
        "action_errors": dict(sorted(action_errors.items())),
        "budget_totals": {
            key: budgets[key]
            for key in (
                "wall_clock_ms",
                "actions",
                "tests",
                "commands",
                "output_bytes",
                "provider_tokens",
            )
        },
        "action_coverage": {
            key: {"numerator": coverage[key], "denominator": total}
            for key in ("read", "edit", "test", "diff", "finish")
        },
        "duration_ms": _distribution(durations),
        "patch_size_bytes": _distribution(patch_sizes),
        "changed_file_count": _distribution(changed_files),
        "retry_count": _distribution(retry_counts),
        "attribution": {
            "provider": provider_attribution,
            "mcp": mcp_attribution,
            "runtime": runtime_attribution,
            "agent": total,
        },
        "mcp": {
            "initialize_count": mcp_initialize,
            "tools_list_count": mcp_list,
            "tools_call_count": mcp_calls,
            "protocol_error_count": mcp_protocol_errors,
            "server_terminals": dict(sorted(mcp_terminals.items())),
        },
    }
    assert_public_artifact_safe(index)
    assert_public_artifact_safe(summary)
    if task_metadata is not None:
        return add_quality_experiment_metadata(index, summary, task_metadata)
    return index, summary


def render_mcp_experiment_markdown(
    index: dict[str, object],
    summary: dict[str, object],
) -> str:
    totals = summary["totals"]
    lines = [
        (
            f"# OpBench {index['dataset_identifier']} MCP Validation "
            f"({index['platform_version']})"
        ),
        "",
        f"- Adapter: `{summary['adapter_id']}`",
        f"- Model: `{summary['model_id']}`",
        f"- Codex CLI: `{summary['codex_cli_version']}`",
        f"- Cohorts: {totals['cohorts']}",
        f"- Selected Attempts: {totals['attempts']}",
        f"- Complete MCP traces: {totals['trace_complete']}",
        f"- Retries: {totals['retries']}",
        "",
        "## Evaluation outcomes",
        "",
    ]
    for name, count in summary["evaluation_outcomes"].items():
        lines.append(f"- `{name}`: {count}")
    lines.extend(("", "## Cohorts", ""))
    for cohort in index["cohorts"]:
        profiles = ", ".join(f"`{item}`" for item in cohort["runtime_profile_ids"])
        lines.append(
            f"- `{cohort['cohort_id']}`: {cohort['selected_attempts']} Attempts; "
            f"profiles {profiles}"
        )
    quality = summary.get("quality")
    if isinstance(quality, dict):
        lines.extend(("", "## Quality-axis coverage", ""))
        for axis in (
            "origin",
            "difficulty",
            "contract_family",
            "failure_type",
            "devices",
            "modes",
            "phases",
            "derived_slices",
        ):
            categories = quality.get(axis)
            if not isinstance(categories, dict):
                continue
            rendered = ", ".join(
                f"`{name}` {value['task_count']} Tasks/{value['attempt_count']} Attempts"
                for name, value in categories.items()
            )
            lines.append(f"- {axis}: {rendered}")
        attribution = quality.get("failure_attribution")
        if isinstance(attribution, dict):
            lines.extend(("", "## Failure attribution", ""))
            for name, count in attribution.items():
                lines.append(f"- `{name}`: {count}")
            lines.append(
                "- Agent counts describe selected valid outcomes; Runtime and "
                "infrastructure-retry counts describe prior invalid retry history "
                "and are not additive with the Agent denominator."
            )
        observations = quality.get("ceiling_floor_observations")
        if isinstance(observations, dict):
            all_resolved = observations.get("all_resolved_task_ids", [])
            zero_resolved = observations.get("zero_resolved_task_ids", [])
            lines.extend(
                (
                    "",
                    "## Ceiling/floor observations",
                    "",
                    f"- All observed Attempts resolved: {len(all_resolved)} Tasks",
                    f"- No observed Attempt resolved: {len(zero_resolved)} Tasks",
                    "- These are descriptive observations for the frozen current configuration, not leaderboard claims.",
                )
            )
    return "\n".join(lines) + "\n"


def write_mcp_experiment_report(
    output_dir: Path,
    index: dict[str, object],
    summary: dict[str, object],
) -> tuple[Path, Path, Path]:
    if not isinstance(output_dir, Path):
        raise ContractError("output_dir: expected Path")
    if output_dir.is_symlink():
        raise ContractError("output_dir: symlink is denied")
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ContractError("output_dir: expected real directory")
    payloads = {
        "experiment_index.json": (canonical_json(index) + "\n").encode("utf-8"),
        "experiment_summary.json": (canonical_json(summary) + "\n").encode("utf-8"),
        "experiment_report.md": render_mcp_experiment_markdown(index, summary).encode(
            "utf-8"
        ),
    }
    for filename, encoded in payloads.items():
        path = output_dir / filename
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != encoded:
                raise ContractError("output_dir contains a nonmatching report")
    for filename, encoded in payloads.items():
        path = output_dir / filename
        if path.exists():
            continue
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(descriptor, view):]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return tuple(output_dir / filename for filename in _REPORT_FILES)


__all__ = [
    "FORMAL_MCP_EXPERIMENT_CONTRACT",
    "McpExperimentCohortBinding",
    "McpExperimentCohortContract",
    "McpExperimentContract",
    "McpExperimentFrozenConfig",
    "add_quality_experiment_metadata",
    "build_mcp_experiment_report",
    "load_mcp_experiment_contract",
    "load_public_task_id_aliases",
    "render_mcp_experiment_markdown",
    "write_mcp_experiment_report",
]
