from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re
from typing import ClassVar

from op_bench.runtime.canonical import JsonValue, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
    require_str_tuple,
)


SCHEMA_VERSION = "v1"
CONTRACT_TYPE = "matched_runtime_compatibility"
SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
GIT_COMMIT_PATTERN = r"[0-9a-f]{40}"
IDENTIFIER_PATTERN = r"[A-Za-z0-9][A-Za-z0-9._+-]*"
EVIDENCE_ID_PATTERN = r"compatibility:v1:[0-9a-f]{64}"
UTC_SECONDS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
MAX_SUMMARY_LENGTH = 500

MATCH_STRATEGIES = (
    "matched_wheel",
    "source_built_wheel",
    "full_source_build",
)
COMPATIBILITY_STATUSES = ("compatible", "incompatible", "unavailable")
CHECK_STATUSES = ("passed", "failed", "unavailable")
REQUIRED_CHECKS = (
    "source_identity",
    "runtime_identity",
    "target_module_provenance",
    "target_import",
    "selector_collection",
    "minimal_operation",
)
ARTIFACT_KINDS = (
    "official_wheel",
    "source_built_wheel",
    "full_source_build",
)
ARTIFACT_DIGEST_KINDS = ("wheel_sha256", "build_artifact_sha256")
SNAPSHOT_DIGEST_KINDS = ("git_archive_sha256", "tree_sha256")
SOURCE_LOADING_MODES = ("python_overlay", "inplace_build", "full_source_build")
FAILURE_CODES = (
    "artifact_not_found",
    "artifact_digest_mismatch",
    "source_identity_mismatch",
    "python_abi_mismatch",
    "cuda_runtime_mismatch",
    "compute_capability_unsupported",
    "target_module_not_from_snapshot",
    "target_import_failed",
    "selector_not_collected",
    "minimal_operation_failed",
    "build_failed",
    "runtime_unavailable",
    "cleanup_failed",
)


def compatibility_content_hash(payload: dict[str, JsonValue]) -> str:
    """Hash a compatibility artifact without its self-referential hash."""

    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "content_hash"}
    )


def _identity_hash(payload: dict[str, JsonValue]) -> str:
    return canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in {"evidence_id", "content_hash"}
        }
    )


def _validate_relative_path(value: object, path: str) -> str:
    text = require_str(value, path)
    if "\\" in text:
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != text
    ):
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    return text


def _validate_optional_str(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path)


def _validate_optional_hash(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=SHA256_PATTERN)


def _validate_utc_seconds(value: object, path: str) -> str:
    text = require_str(value, path)
    if UTC_SECONDS_PATTERN.fullmatch(text) is None:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds") from exc
    return text


def _validate_sorted_unique(values: tuple[str, ...], path: str) -> None:
    if tuple(sorted(set(values))) != values:
        raise ContractError(f"{path}: expected sorted unique values")


@dataclass(frozen=True)
class SourceIdentity:
    source_id: str
    commit: str
    snapshot_digest: str
    snapshot_digest_kind: str
    target_module_path: str
    target_module_sha256: str
    runtime_path_suffix: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "source_id",
            "commit",
            "snapshot_digest",
            "snapshot_digest_kind",
            "target_module_path",
            "target_module_sha256",
            "runtime_path_suffix",
        )

    def __post_init__(self) -> None:
        require_str(self.source_id, "source.source_id", pattern=IDENTIFIER_PATTERN)
        require_str(self.commit, "source.commit", pattern=GIT_COMMIT_PATTERN)
        require_str(
            self.snapshot_digest,
            "source.snapshot_digest",
            pattern=SHA256_PATTERN,
        )
        require_enum(
            self.snapshot_digest_kind,
            "source.snapshot_digest_kind",
            SNAPSHOT_DIGEST_KINDS,
        )
        _validate_relative_path(
            self.target_module_path,
            "source.target_module_path",
        )
        require_str(
            self.target_module_sha256,
            "source.target_module_sha256",
            pattern=SHA256_PATTERN,
        )
        _validate_relative_path(
            self.runtime_path_suffix,
            "source.runtime_path_suffix",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "source_id": self.source_id,
            "commit": self.commit,
            "snapshot_digest": self.snapshot_digest,
            "snapshot_digest_kind": self.snapshot_digest_kind,
            "target_module_path": self.target_module_path,
            "target_module_sha256": self.target_module_sha256,
            "runtime_path_suffix": self.runtime_path_suffix,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "source") -> "SourceIdentity":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            source_id=require_str(data["source_id"], f"{path}.source_id"),
            commit=require_str(
                data["commit"],
                f"{path}.commit",
                pattern=GIT_COMMIT_PATTERN,
            ),
            snapshot_digest=require_str(
                data["snapshot_digest"],
                f"{path}.snapshot_digest",
                pattern=SHA256_PATTERN,
            ),
            snapshot_digest_kind=require_enum(
                data["snapshot_digest_kind"],
                f"{path}.snapshot_digest_kind",
                SNAPSHOT_DIGEST_KINDS,
            ),
            target_module_path=_validate_relative_path(
                data["target_module_path"],
                f"{path}.target_module_path",
            ),
            target_module_sha256=require_str(
                data["target_module_sha256"],
                f"{path}.target_module_sha256",
                pattern=SHA256_PATTERN,
            ),
            runtime_path_suffix=_validate_relative_path(
                data["runtime_path_suffix"],
                f"{path}.runtime_path_suffix",
            ),
        )


@dataclass(frozen=True)
class RuntimeIdentity:
    environment_id: str
    artifact_kind: str
    artifact_id: str
    artifact_digest: str
    artifact_digest_kind: str
    torch_version: str
    python_implementation: str
    python_abi: str
    platform: str
    cuda_build: str | None
    cuda_runtime: str | None
    device_name: str | None
    compute_capability: str | None
    source_loading_mode: str
    target_module_path_suffix: str
    target_module_sha256: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "environment_id",
            "artifact_kind",
            "artifact_id",
            "artifact_digest",
            "artifact_digest_kind",
            "torch_version",
            "python_implementation",
            "python_abi",
            "platform",
            "cuda_build",
            "cuda_runtime",
            "device_name",
            "compute_capability",
            "source_loading_mode",
            "target_module_path_suffix",
            "target_module_sha256",
        )

    def __post_init__(self) -> None:
        require_str(
            self.environment_id,
            "runtime.environment_id",
            pattern=IDENTIFIER_PATTERN,
        )
        require_enum(self.artifact_kind, "runtime.artifact_kind", ARTIFACT_KINDS)
        if len(require_str(self.artifact_id, "runtime.artifact_id")) > 300:
            raise ContractError("runtime.artifact_id: must contain at most 300 characters")
        require_str(
            self.artifact_digest,
            "runtime.artifact_digest",
            pattern=SHA256_PATTERN,
        )
        require_enum(
            self.artifact_digest_kind,
            "runtime.artifact_digest_kind",
            ARTIFACT_DIGEST_KINDS,
        )
        for value, path in (
            (self.torch_version, "runtime.torch_version"),
            (self.python_implementation, "runtime.python_implementation"),
            (self.python_abi, "runtime.python_abi"),
            (self.platform, "runtime.platform"),
        ):
            require_str(value, path)
        cuda_values = (
            self.cuda_build,
            self.cuda_runtime,
            self.device_name,
            self.compute_capability,
        )
        if any(value is not None for value in cuda_values) and not all(
            value is not None for value in cuda_values
        ):
            raise ContractError(
                "runtime CUDA identity: build, runtime, device and compute capability "
                "must all be present or all be null"
            )
        for value, path in (
            (self.cuda_build, "runtime.cuda_build"),
            (self.cuda_runtime, "runtime.cuda_runtime"),
            (self.device_name, "runtime.device_name"),
            (self.compute_capability, "runtime.compute_capability"),
        ):
            _validate_optional_str(value, path)
        require_enum(
            self.source_loading_mode,
            "runtime.source_loading_mode",
            SOURCE_LOADING_MODES,
        )
        _validate_relative_path(
            self.target_module_path_suffix,
            "runtime.target_module_path_suffix",
        )
        require_str(
            self.target_module_sha256,
            "runtime.target_module_sha256",
            pattern=SHA256_PATTERN,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "environment_id": self.environment_id,
            "artifact_kind": self.artifact_kind,
            "artifact_id": self.artifact_id,
            "artifact_digest": self.artifact_digest,
            "artifact_digest_kind": self.artifact_digest_kind,
            "torch_version": self.torch_version,
            "python_implementation": self.python_implementation,
            "python_abi": self.python_abi,
            "platform": self.platform,
            "cuda_build": self.cuda_build,
            "cuda_runtime": self.cuda_runtime,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "source_loading_mode": self.source_loading_mode,
            "target_module_path_suffix": self.target_module_path_suffix,
            "target_module_sha256": self.target_module_sha256,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "runtime") -> "RuntimeIdentity":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            environment_id=require_str(
                data["environment_id"],
                f"{path}.environment_id",
            ),
            artifact_kind=require_enum(
                data["artifact_kind"],
                f"{path}.artifact_kind",
                ARTIFACT_KINDS,
            ),
            artifact_id=require_str(data["artifact_id"], f"{path}.artifact_id"),
            artifact_digest=require_str(
                data["artifact_digest"],
                f"{path}.artifact_digest",
                pattern=SHA256_PATTERN,
            ),
            artifact_digest_kind=require_enum(
                data["artifact_digest_kind"],
                f"{path}.artifact_digest_kind",
                ARTIFACT_DIGEST_KINDS,
            ),
            torch_version=require_str(
                data["torch_version"],
                f"{path}.torch_version",
            ),
            python_implementation=require_str(
                data["python_implementation"],
                f"{path}.python_implementation",
            ),
            python_abi=require_str(data["python_abi"], f"{path}.python_abi"),
            platform=require_str(data["platform"], f"{path}.platform"),
            cuda_build=_validate_optional_str(
                data["cuda_build"],
                f"{path}.cuda_build",
            ),
            cuda_runtime=_validate_optional_str(
                data["cuda_runtime"],
                f"{path}.cuda_runtime",
            ),
            device_name=_validate_optional_str(
                data["device_name"],
                f"{path}.device_name",
            ),
            compute_capability=_validate_optional_str(
                data["compute_capability"],
                f"{path}.compute_capability",
            ),
            source_loading_mode=require_enum(
                data["source_loading_mode"],
                f"{path}.source_loading_mode",
                SOURCE_LOADING_MODES,
            ),
            target_module_path_suffix=_validate_relative_path(
                data["target_module_path_suffix"],
                f"{path}.target_module_path_suffix",
            ),
            target_module_sha256=require_str(
                data["target_module_sha256"],
                f"{path}.target_module_sha256",
                pattern=SHA256_PATTERN,
            ),
        )


@dataclass(frozen=True)
class BuildIdentity:
    flags: tuple[str, ...]
    gpu_arches: tuple[str, ...]
    ccache_key: str | None
    artifact_digest: str | None
    toolchain: tuple[str, ...]

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "flags",
            "gpu_arches",
            "ccache_key",
            "artifact_digest",
            "toolchain",
        )

    def __post_init__(self) -> None:
        for values, path in (
            (self.flags, "build.flags"),
            (self.gpu_arches, "build.gpu_arches"),
            (self.toolchain, "build.toolchain"),
        ):
            if not isinstance(values, tuple):
                raise ContractError(f"{path}: expected tuple")
            for index, value in enumerate(values):
                require_str(value, f"{path}[{index}]")
            _validate_sorted_unique(values, path)
        if self.ccache_key is not None:
            require_str(
                self.ccache_key,
                "build.ccache_key",
                pattern=IDENTIFIER_PATTERN,
            )
        _validate_optional_hash(self.artifact_digest, "build.artifact_digest")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "flags": list(self.flags),
            "gpu_arches": list(self.gpu_arches),
            "ccache_key": self.ccache_key,
            "artifact_digest": self.artifact_digest,
            "toolchain": list(self.toolchain),
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "build") -> "BuildIdentity":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            flags=require_str_tuple(data["flags"], f"{path}.flags"),
            gpu_arches=require_str_tuple(
                data["gpu_arches"],
                f"{path}.gpu_arches",
            ),
            ccache_key=_validate_optional_str(
                data["ccache_key"],
                f"{path}.ccache_key",
            ),
            artifact_digest=_validate_optional_hash(
                data["artifact_digest"],
                f"{path}.artifact_digest",
            ),
            toolchain=require_str_tuple(
                data["toolchain"],
                f"{path}.toolchain",
            ),
        )


@dataclass(frozen=True)
class CompatibilityCheck:
    name: str
    command_digest: str
    exit_code: int | None
    status: str
    summary: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("name", "command_digest", "exit_code", "status", "summary")

    def __post_init__(self) -> None:
        require_enum(self.name, "check.name", REQUIRED_CHECKS)
        require_str(
            self.command_digest,
            "check.command_digest",
            pattern=SHA256_PATTERN,
        )
        require_enum(self.status, "check.status", CHECK_STATUSES)
        summary = require_str(self.summary, "check.summary")
        if len(summary) > MAX_SUMMARY_LENGTH:
            raise ContractError(
                f"check.summary: must contain at most {MAX_SUMMARY_LENGTH} characters"
            )
        if self.status == "passed":
            if self.exit_code != 0:
                raise ContractError("check.exit_code: passed check must exit 0")
        elif self.status == "failed":
            require_int(self.exit_code, "check.exit_code")
            if self.exit_code == 0:
                raise ContractError("check.exit_code: failed check must be non-zero")
        elif self.exit_code is not None:
            raise ContractError(
                "check.exit_code: unavailable check must have a null exit code"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "command_digest": self.command_digest,
            "exit_code": self.exit_code,
            "status": self.status,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "check",
    ) -> "CompatibilityCheck":
        data = require_exact_fields(value, path, cls.wire_fields())
        exit_code = data["exit_code"]
        if exit_code is not None:
            exit_code = require_int(exit_code, f"{path}.exit_code")
        return cls(
            name=require_enum(
                data["name"],
                f"{path}.name",
                REQUIRED_CHECKS,
            ),
            command_digest=require_str(
                data["command_digest"],
                f"{path}.command_digest",
                pattern=SHA256_PATTERN,
            ),
            exit_code=exit_code,
            status=require_enum(
                data["status"],
                f"{path}.status",
                CHECK_STATUSES,
            ),
            summary=require_str(data["summary"], f"{path}.summary"),
        )


@dataclass(frozen=True)
class CompatibilityFailure:
    code: str
    check: str
    summary: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("code", "check", "summary")

    def __post_init__(self) -> None:
        require_enum(self.code, "failure.code", FAILURE_CODES)
        require_enum(self.check, "failure.check", REQUIRED_CHECKS)
        summary = require_str(self.summary, "failure.summary")
        if len(summary) > MAX_SUMMARY_LENGTH:
            raise ContractError(
                f"failure.summary: must contain at most {MAX_SUMMARY_LENGTH} characters"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "check": self.check,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "failure",
    ) -> "CompatibilityFailure":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            code=require_enum(data["code"], f"{path}.code", FAILURE_CODES),
            check=require_enum(
                data["check"],
                f"{path}.check",
                REQUIRED_CHECKS,
            ),
            summary=require_str(data["summary"], f"{path}.summary"),
        )


@dataclass(frozen=True)
class CompatibilityEvidence:
    task_id: str
    strategy: str
    status: str
    source: SourceIdentity
    runtime: RuntimeIdentity
    build: BuildIdentity
    checks: tuple[CompatibilityCheck, ...]
    failure: CompatibilityFailure | None
    created_at: str

    contract_type: ClassVar[str] = CONTRACT_TYPE
    schema_version: ClassVar[str] = SCHEMA_VERSION

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "evidence_id",
            "task_id",
            "strategy",
            "status",
            "source",
            "runtime",
            "build",
            "checks",
            "failure",
            "created_at",
            "content_hash",
        )

    def __post_init__(self) -> None:
        require_str(self.task_id, "task_id", pattern=IDENTIFIER_PATTERN)
        require_enum(self.strategy, "strategy", MATCH_STRATEGIES)
        require_enum(self.status, "status", COMPATIBILITY_STATUSES)
        if not isinstance(self.source, SourceIdentity):
            raise ContractError("source: expected SourceIdentity")
        if not isinstance(self.runtime, RuntimeIdentity):
            raise ContractError("runtime: expected RuntimeIdentity")
        if not isinstance(self.build, BuildIdentity):
            raise ContractError("build: expected BuildIdentity")
        if not isinstance(self.checks, tuple):
            raise ContractError("checks: expected tuple")
        if tuple(check.name for check in self.checks) != REQUIRED_CHECKS:
            raise ContractError(
                "checks: expected each required check exactly once in canonical order"
            )
        if self.strategy == "matched_wheel":
            if self.runtime.artifact_kind != "official_wheel":
                raise ContractError(
                    "matched_wheel: runtime artifact must be an official wheel"
                )
            if any(
                (
                    self.build.flags,
                    self.build.gpu_arches,
                    self.build.ccache_key,
                    self.build.artifact_digest,
                    self.build.toolchain,
                )
            ):
                raise ContractError(
                    "matched_wheel: build-only identity fields must be empty"
                )
        else:
            expected_kind = (
                "source_built_wheel"
                if self.strategy == "source_built_wheel"
                else "full_source_build"
            )
            if self.runtime.artifact_kind != expected_kind:
                raise ContractError(
                    f"build: runtime artifact kind must be {expected_kind!r}"
                )
            if (
                not self.build.flags
                or not self.build.ccache_key
                or not self.build.artifact_digest
                or not self.build.toolchain
            ):
                raise ContractError(
                    "build: source build requires flags, ccache key, artifact "
                    "digest and toolchain"
                )
            if self.runtime.artifact_digest != self.build.artifact_digest:
                raise ContractError(
                    "build: runtime and build artifact digests must match"
                )
        if self.status == "compatible":
            if self.failure is not None:
                raise ContractError("failure: compatible evidence must not fail")
            if any(check.status != "passed" for check in self.checks):
                raise ContractError("checks: compatible evidence requires all checks passed")
            if (
                self.source.target_module_sha256
                != self.runtime.target_module_sha256
                or self.source.runtime_path_suffix
                != self.runtime.target_module_path_suffix
            ):
                raise ContractError(
                    "target module: runtime bytes and path must match the snapshot"
                )
        else:
            if not isinstance(self.failure, CompatibilityFailure):
                raise ContractError(
                    "failure: incompatible or unavailable evidence requires a failure"
                )
            expected_status = (
                "failed" if self.status == "incompatible" else "unavailable"
            )
            matching = [
                check
                for check in self.checks
                if check.name == self.failure.check
                and check.status == expected_status
            ]
            if not matching:
                raise ContractError(
                    f"failure: must reference a {expected_status} check"
                )
        _validate_utc_seconds(self.created_at, "created_at")

    @property
    def evidence_id(self) -> str:
        digest = _identity_hash(self.to_dict(include_hash=False, include_id=False))
        return "compatibility:v1:" + digest.removeprefix("sha256:")

    @property
    def content_hash(self) -> str:
        return compatibility_content_hash(self.to_dict(include_hash=False))

    def to_dict(
        self,
        *,
        include_hash: bool = True,
        include_id: bool = True,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "strategy": self.strategy,
            "status": self.status,
            "source": self.source.to_dict(),
            "runtime": self.runtime.to_dict(),
            "build": self.build.to_dict(),
            "checks": [check.to_dict() for check in self.checks],
            "failure": None if self.failure is None else self.failure.to_dict(),
            "created_at": self.created_at,
        }
        if include_id:
            payload["evidence_id"] = self.evidence_id
        if include_hash:
            payload["content_hash"] = compatibility_content_hash(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "matched_runtime_compatibility",
    ) -> "CompatibilityEvidence":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        checks = require_list(data["checks"], f"{path}.checks")
        failure = data["failure"]
        evidence = cls(
            task_id=require_str(data["task_id"], f"{path}.task_id"),
            strategy=require_enum(
                data["strategy"],
                f"{path}.strategy",
                MATCH_STRATEGIES,
            ),
            status=require_enum(
                data["status"],
                f"{path}.status",
                COMPATIBILITY_STATUSES,
            ),
            source=SourceIdentity.from_dict(
                data["source"],
                path=f"{path}.source",
            ),
            runtime=RuntimeIdentity.from_dict(
                data["runtime"],
                path=f"{path}.runtime",
            ),
            build=BuildIdentity.from_dict(
                data["build"],
                path=f"{path}.build",
            ),
            checks=tuple(
                CompatibilityCheck.from_dict(
                    item,
                    path=f"{path}.checks[{index}]",
                )
                for index, item in enumerate(checks)
            ),
            failure=(
                None
                if failure is None
                else CompatibilityFailure.from_dict(
                    failure,
                    path=f"{path}.failure",
                )
            ),
            created_at=_validate_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
        )
        stored_id = require_str(
            data["evidence_id"],
            f"{path}.evidence_id",
            pattern=EVIDENCE_ID_PATTERN,
        )
        if stored_id != evidence.evidence_id:
            raise ContractError(
                f"{path}.evidence_id: expected {evidence.evidence_id!r}"
            )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=SHA256_PATTERN,
        )
        if stored_hash != evidence.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {evidence.content_hash!r}"
            )
        return evidence
