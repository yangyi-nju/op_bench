from __future__ import annotations

from dataclasses import dataclass, fields
import fcntl
import json
import os
from pathlib import Path
import stat
import threading
import uuid
from typing import Callable

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import SCHEMA_VERSION, SHA256_PATTERN
from op_bench.runtime.manifest import ATTEMPT_PATTERN
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
)


RESOURCE_TYPES = ("workspace", "process", "container", "remote_workspace")
RESOURCE_TRANSITIONS = (
    "declared",
    "created",
    "released",
    "cleanup_failed",
    "create_failed",
)
CLEANUP_STATUSES = ("released", "cleanup_failed", "create_failed")
RESOURCE_ID_PATTERN = r"resource:v1:[0-9a-f]{64}"


def runtime_resource_id(
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
    resource_type: str,
    ordinal: int,
) -> str:
    attempt = require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
    retry = require_int(retry_index, "retry_index", minimum=1)
    profile = require_str(
        runtime_profile_hash,
        "runtime_profile_hash",
        pattern=SHA256_PATTERN,
    )
    kind = require_enum(resource_type, "resource_type", RESOURCE_TYPES)
    index = require_int(ordinal, "ordinal", minimum=1)
    digest = canonical_sha256(
        {
            "identity_type": "runtime_resource",
            "schema_version": SCHEMA_VERSION,
            "attempt_id": attempt,
            "retry_index": retry,
            "runtime_profile_hash": profile,
            "resource_type": kind,
            "ordinal": index,
        }
    ).removeprefix("sha256:")
    return f"resource:v1:{digest}"


def runtime_raw_handle_hash(raw_handle: str) -> str:
    handle = require_str(raw_handle, "raw_handle")
    return canonical_sha256({"raw_handle": handle})


@dataclass(frozen=True)
class RuntimeResourceRecord:
    sequence: int
    previous_hash: str | None
    record_hash: str
    attempt_id: str
    retry_index: int
    runtime_profile_hash: str
    resource_id: str
    resource_type: str
    ordinal: int
    transition: str
    raw_handle_hash: str | None
    recorded_at_ms: int

    def __post_init__(self) -> None:
        require_int(self.sequence, "sequence", minimum=1)
        if self.previous_hash is not None:
            require_str(self.previous_hash, "previous_hash", pattern=SHA256_PATTERN)
        require_str(self.record_hash, "record_hash", pattern=SHA256_PATTERN)
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(
            self.runtime_profile_hash,
            "runtime_profile_hash",
            pattern=SHA256_PATTERN,
        )
        require_str(self.resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        require_enum(self.resource_type, "resource_type", RESOURCE_TYPES)
        require_int(self.ordinal, "ordinal", minimum=1)
        require_enum(self.transition, "transition", RESOURCE_TRANSITIONS)
        if self.raw_handle_hash is not None:
            require_str(
                self.raw_handle_hash,
                "raw_handle_hash",
                pattern=SHA256_PATTERN,
            )
        require_int(self.recorded_at_ms, "recorded_at_ms", minimum=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": "runtime_resource_record",
            "schema_version": SCHEMA_VERSION,
            "sequence": self.sequence,
            "previous_hash": self.previous_hash,
            "record_hash": self.record_hash,
            "attempt_id": self.attempt_id,
            "retry_index": self.retry_index,
            "runtime_profile_hash": self.runtime_profile_hash,
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "ordinal": self.ordinal,
            "transition": self.transition,
            "raw_handle_hash": self.raw_handle_hash,
            "recorded_at_ms": self.recorded_at_ms,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "runtime_resource_record",
    ) -> "RuntimeResourceRecord":
        data = require_exact_fields(
            value,
            path,
            (
                "record_type",
                "schema_version",
                *(item.name for item in fields(cls)),
            ),
        )
        if require_str(data["record_type"], f"{path}.record_type") != "runtime_resource_record":
            raise ContractError(f"{path}.record_type: expected 'runtime_resource_record'")
        if require_str(data["schema_version"], f"{path}.schema_version") != SCHEMA_VERSION:
            raise ContractError(f"{path}.schema_version: expected {SCHEMA_VERSION!r}")
        raw_hash = data["raw_handle_hash"]
        return cls(
            sequence=require_int(data["sequence"], "sequence", minimum=1),
            previous_hash=_optional_hash(data["previous_hash"], "previous_hash"),
            record_hash=require_str(data["record_hash"], "record_hash", pattern=SHA256_PATTERN),
            attempt_id=require_str(data["attempt_id"], "attempt_id", pattern=ATTEMPT_PATTERN),
            retry_index=require_int(data["retry_index"], "retry_index", minimum=1),
            runtime_profile_hash=require_str(
                data["runtime_profile_hash"],
                "runtime_profile_hash",
                pattern=SHA256_PATTERN,
            ),
            resource_id=require_str(data["resource_id"], "resource_id", pattern=RESOURCE_ID_PATTERN),
            resource_type=require_enum(data["resource_type"], "resource_type", RESOURCE_TYPES),
            ordinal=require_int(data["ordinal"], "ordinal", minimum=1),
            transition=require_enum(data["transition"], "transition", RESOURCE_TRANSITIONS),
            raw_handle_hash=(
                None
                if raw_hash is None
                else require_str(raw_hash, "raw_handle_hash", pattern=SHA256_PATTERN)
            ),
            recorded_at_ms=require_int(data["recorded_at_ms"], "recorded_at_ms", minimum=0),
        )


def runtime_resource_record_hash(record: RuntimeResourceRecord) -> str:
    if not isinstance(record, RuntimeResourceRecord):
        raise ContractError("record: expected RuntimeResourceRecord")
    payload = record.to_dict()
    del payload["record_hash"]
    return canonical_sha256(payload)


def parse_runtime_resource_ledger(
    raw: bytes,
    *,
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
) -> tuple[RuntimeResourceRecord, ...]:
    if not isinstance(raw, bytes):
        raise ContractError("runtime_resources.jsonl: expected bytes")
    attempt = require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
    retry = require_int(retry_index, "retry_index", minimum=1)
    profile = require_str(
        runtime_profile_hash,
        "runtime_profile_hash",
        pattern=SHA256_PATTERN,
    )
    if raw and not raw.endswith(b"\n"):
        raise ContractError("runtime_resources.jsonl: missing final newline")
    records: list[RuntimeResourceRecord] = []
    for line_number, encoded in enumerate(raw.splitlines(), start=1):
        try:
            value = json.loads(encoded.decode("utf-8"))
            if canonical_json(value).encode("utf-8") != encoded:
                raise ContractError("expected canonical JSON")
            record = RuntimeResourceRecord.from_dict(
                value,
                path=f"runtime_resources.jsonl line {line_number}",
            )
            assert_public_artifact_safe(record.to_dict())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"runtime_resources.jsonl line {line_number}: invalid JSON"
            ) from exc
        records.append(record)
    _verify_runtime_resource_records(tuple(records), attempt, retry, profile)
    return tuple(records)


@dataclass(frozen=True)
class RuntimeResourceHandle:
    resource_id: str
    resource_type: str
    ordinal: int
    raw_handle: str
    raw_handle_hash: str

    def __post_init__(self) -> None:
        require_str(self.resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        require_enum(self.resource_type, "resource_type", RESOURCE_TYPES)
        require_int(self.ordinal, "ordinal", minimum=1)
        require_str(self.raw_handle, "raw_handle")
        require_str(self.raw_handle_hash, "raw_handle_hash", pattern=SHA256_PATTERN)
        if self.raw_handle_hash != runtime_raw_handle_hash(self.raw_handle):
            raise ContractError("raw_handle_hash: does not match raw_handle")

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "ordinal": self.ordinal,
            "raw_handle": self.raw_handle,
            "raw_handle_hash": self.raw_handle_hash,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str) -> "RuntimeResourceHandle":
        data = require_exact_fields(
            value,
            path,
            ("resource_id", "resource_type", "ordinal", "raw_handle", "raw_handle_hash"),
        )
        return cls(
            resource_id=require_str(data["resource_id"], "resource_id", pattern=RESOURCE_ID_PATTERN),
            resource_type=require_enum(data["resource_type"], "resource_type", RESOURCE_TYPES),
            ordinal=require_int(data["ordinal"], "ordinal", minimum=1),
            raw_handle=require_str(data["raw_handle"], "raw_handle"),
            raw_handle_hash=require_str(
                data["raw_handle_hash"],
                "raw_handle_hash",
                pattern=SHA256_PATTERN,
            ),
        )


@dataclass(frozen=True)
class RuntimeLease:
    attempt_id: str
    retry_index: int
    runtime_profile_hash: str
    handles: tuple[RuntimeResourceHandle, ...]

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(self.runtime_profile_hash, "runtime_profile_hash", pattern=SHA256_PATTERN)
        if not isinstance(self.handles, tuple):
            raise ContractError("handles: expected tuple")
        for index, handle in enumerate(self.handles):
            if not isinstance(handle, RuntimeResourceHandle):
                raise ContractError(f"handles[{index}]: expected RuntimeResourceHandle")


def parse_runtime_lease_store(
    raw: bytes,
    *,
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
) -> tuple[RuntimeResourceHandle, ...]:
    if not isinstance(raw, bytes):
        raise ContractError("private_runtime_resources.json: expected bytes")
    attempt = require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
    retry = require_int(retry_index, "retry_index", minimum=1)
    profile = require_str(
        runtime_profile_hash,
        "runtime_profile_hash",
        pattern=SHA256_PATTERN,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("private_runtime_resources.json: invalid JSON") from exc
    if (canonical_json(value) + "\n").encode("utf-8") != raw:
        raise ContractError("private_runtime_resources.json: expected canonical JSON")
    data = require_exact_fields(
        value,
        "private_runtime_resources",
        (
            "store_type",
            "schema_version",
            "attempt_id",
            "retry_index",
            "runtime_profile_hash",
            "handles",
        ),
    )
    if data["store_type"] != "runtime_lease_store":
        raise ContractError("store_type: expected 'runtime_lease_store'")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
    if (
        data["attempt_id"] != attempt
        or data["retry_index"] != retry
        or data["runtime_profile_hash"] != profile
    ):
        raise ContractError("private lease store identity mismatch")
    raw_handles = require_list(data["handles"], "handles")
    handles = tuple(
        RuntimeResourceHandle.from_dict(item, path=f"handles[{index}]")
        for index, item in enumerate(raw_handles)
    )
    if tuple(sorted(handles, key=lambda item: item.resource_id)) != handles:
        raise ContractError("handles: expected sorted resource_id order")
    ids = [item.resource_id for item in handles]
    hashes = [item.raw_handle_hash for item in handles]
    if len(ids) != len(set(ids)):
        raise ContractError("handles: duplicate resource_id")
    if len(hashes) != len(set(hashes)):
        raise ContractError("handles: duplicate raw handle")
    for handle in handles:
        expected = runtime_resource_id(
            attempt,
            retry,
            profile,
            handle.resource_type,
            handle.ordinal,
        )
        if handle.resource_id != expected:
            raise ContractError("resource_id does not belong to lease store")
    return handles


@dataclass(frozen=True)
class RuntimeCleanupEntry:
    resource_id: str
    resource_type: str
    status: str
    error_code: str | None

    def __post_init__(self) -> None:
        require_str(self.resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        require_enum(self.resource_type, "resource_type", RESOURCE_TYPES)
        require_enum(self.status, "status", CLEANUP_STATUSES)
        if self.status == "released":
            if self.error_code is not None:
                raise ContractError("error_code: released resource cannot have an error")
        elif self.error_code is None:
            raise ContractError("error_code: failed resource requires an error code")
        else:
            require_str(self.error_code, "error_code")

    def to_dict(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type,
            "status": self.status,
            "error_code": self.error_code,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str) -> "RuntimeCleanupEntry":
        data = require_exact_fields(
            value,
            path,
            ("resource_id", "resource_type", "status", "error_code"),
        )
        error = data["error_code"]
        return cls(
            resource_id=require_str(data["resource_id"], "resource_id", pattern=RESOURCE_ID_PATTERN),
            resource_type=require_enum(data["resource_type"], "resource_type", RESOURCE_TYPES),
            status=require_enum(data["status"], "status", CLEANUP_STATUSES),
            error_code=None if error is None else require_str(error, "error_code"),
        )


@dataclass(frozen=True)
class RuntimeCleanupReport:
    attempt_id: str
    retry_index: int
    runtime_profile_hash: str
    entries: tuple[RuntimeCleanupEntry, ...]
    all_released: bool

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(self.runtime_profile_hash, "runtime_profile_hash", pattern=SHA256_PATTERN)
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ContractError("entries: expected non-empty tuple")
        identifiers: list[str] = []
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, RuntimeCleanupEntry):
                raise ContractError(f"entries[{index}]: expected RuntimeCleanupEntry")
            if entry.resource_id in identifiers:
                raise ContractError(f"entries: duplicate resource_id {entry.resource_id!r}")
            identifiers.append(entry.resource_id)
        if identifiers != sorted(identifiers):
            raise ContractError("entries: expected sorted resource_id order")
        if not isinstance(self.all_released, bool):
            raise ContractError("all_released: expected boolean")
        expected = all(entry.status in {"released", "create_failed"} for entry in self.entries)
        if self.all_released != expected:
            raise ContractError("all_released: does not match cleanup entries")
        assert_public_artifact_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "report_type": "runtime_cleanup_report",
            "schema_version": SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "retry_index": self.retry_index,
            "runtime_profile_hash": self.runtime_profile_hash,
            "entries": [entry.to_dict() for entry in self.entries],
            "all_released": self.all_released,
        }

    @classmethod
    def from_dict(cls, value: object) -> "RuntimeCleanupReport":
        data = require_exact_fields(
            value,
            "runtime_cleanup_report",
            (
                "report_type",
                "schema_version",
                "attempt_id",
                "retry_index",
                "runtime_profile_hash",
                "entries",
                "all_released",
            ),
        )
        if data["report_type"] != "runtime_cleanup_report":
            raise ContractError("report_type: expected 'runtime_cleanup_report'")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
        raw_entries = require_list(data["entries"], "entries")
        return cls(
            attempt_id=require_str(data["attempt_id"], "attempt_id", pattern=ATTEMPT_PATTERN),
            retry_index=require_int(data["retry_index"], "retry_index", minimum=1),
            runtime_profile_hash=require_str(
                data["runtime_profile_hash"],
                "runtime_profile_hash",
                pattern=SHA256_PATTERN,
            ),
            entries=tuple(
                RuntimeCleanupEntry.from_dict(item, path=f"entries[{index}]")
                for index, item in enumerate(raw_entries)
            ),
            all_released=data["all_released"],
        )


def verify_runtime_resource_evidence(
    records: tuple[RuntimeResourceRecord, ...],
    handles: tuple[RuntimeResourceHandle, ...],
    cleanup_report: RuntimeCleanupReport,
) -> None:
    verify_runtime_resource_ownership(records, handles)
    verify_runtime_cleanup(records, cleanup_report)


def verify_runtime_resource_ownership(
    records: tuple[RuntimeResourceRecord, ...],
    handles: tuple[RuntimeResourceHandle, ...],
) -> None:
    if not isinstance(records, tuple) or not records:
        raise ContractError("runtime resource records: expected non-empty tuple")
    if not isinstance(handles, tuple):
        raise ContractError("runtime resource handles: expected tuple")
    first = records[0]
    _verify_runtime_resource_records(
        records,
        first.attempt_id,
        first.retry_index,
        first.runtime_profile_hash,
    )
    handle_by_id = {handle.resource_id: handle for handle in handles}
    if len(handle_by_id) != len(handles):
        raise ContractError("private lease store has duplicate resource_id")
    histories: dict[str, list[RuntimeResourceRecord]] = {}
    for record in records:
        histories.setdefault(record.resource_id, []).append(record)
    created_ids = {
        resource_id
        for resource_id, history in histories.items()
        if any(record.transition == "created" for record in history)
    }
    if set(handle_by_id) != created_ids:
        raise ContractError("private lease handles differ from created resources")
    for resource_id in created_ids:
        created = next(
            record
            for record in histories[resource_id]
            if record.transition == "created"
        )
        handle = handle_by_id[resource_id]
        if (
            handle.resource_type != created.resource_type
            or handle.ordinal != created.ordinal
            or handle.raw_handle_hash != created.raw_handle_hash
        ):
            raise ContractError("private lease handle hash binding mismatch")


def verify_runtime_cleanup(
    records: tuple[RuntimeResourceRecord, ...],
    cleanup_report: RuntimeCleanupReport,
) -> None:
    if not isinstance(records, tuple) or not records:
        raise ContractError("runtime resource records: expected non-empty tuple")
    if not isinstance(cleanup_report, RuntimeCleanupReport):
        raise ContractError("cleanup_report: expected RuntimeCleanupReport")
    first = records[0]
    _verify_runtime_resource_records(
        records,
        first.attempt_id,
        first.retry_index,
        first.runtime_profile_hash,
    )
    if (
        cleanup_report.attempt_id != first.attempt_id
        or cleanup_report.retry_index != first.retry_index
        or cleanup_report.runtime_profile_hash != first.runtime_profile_hash
    ):
        raise ContractError("cleanup report identity mismatch")
    histories: dict[str, list[RuntimeResourceRecord]] = {}
    for record in records:
        histories.setdefault(record.resource_id, []).append(record)
    final_by_id = {
        resource_id: history[-1]
        for resource_id, history in histories.items()
    }
    active = [
        resource_id
        for resource_id, final in final_by_id.items()
        if final.transition in {"declared", "created"}
    ]
    if active:
        raise ContractError(f"active resource has no final transition: {active[0]}")
    report_by_id = {entry.resource_id: entry for entry in cleanup_report.entries}
    if set(report_by_id) != set(final_by_id):
        raise ContractError("cleanup report differs from resource ledger")
    for resource_id, final in final_by_id.items():
        entry = report_by_id[resource_id]
        if entry.resource_type != final.resource_type or entry.status != final.transition:
            raise ContractError("cleanup report differs from resource ledger")
    if any(final.transition == "cleanup_failed" for final in final_by_id.values()):
        raise ContractError("runtime resource cleanup_failed")
    if not cleanup_report.all_released:
        raise ContractError("runtime cleanup report is not all_released")


class AttemptResourceLedger:
    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        retry_index: int,
        runtime_profile_hash: str,
        clock_ms: Callable[[], int],
    ) -> None:
        if not isinstance(path, Path):
            raise ContractError("ledger_path: expected Path")
        if path.is_symlink():
            raise ContractError("ledger_path: symlink is denied")
        self.attempt_id = require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        self.retry_index = require_int(retry_index, "retry_index", minimum=1)
        self.runtime_profile_hash = require_str(
            runtime_profile_hash,
            "runtime_profile_hash",
            pattern=SHA256_PATTERN,
        )
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        self._clock_ms = clock_ms
        self.path = path
        self._lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ContractError("ledger_path parent: symlink is denied")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            self._parent_fd = os.open(path.parent, directory_flags)
        except OSError as exc:
            raise ContractError("ledger_path parent: expected real directory") from exc
        self._filename = path.name
        flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            try:
                self._file_fd = os.open(
                    self._filename,
                    flags,
                    0o600,
                    dir_fd=self._parent_fd,
                )
            except OSError as exc:
                raise ContractError("ledger_path: invalid file") from exc
            if not stat.S_ISREG(os.fstat(self._file_fd).st_mode):
                raise ContractError("ledger_path: expected regular file")
            os.fsync(self._parent_fd)
            self._load_locked()
        except Exception:
            if hasattr(self, "_file_fd"):
                os.close(self._file_fd)
            os.close(self._parent_fd)
            raise

    @property
    def records(self) -> tuple[RuntimeResourceRecord, ...]:
        with self._lock:
            self._ensure_open()
            return self._load_locked()

    def declare(self, resource_type: str, ordinal: int) -> RuntimeResourceRecord:
        kind = require_enum(resource_type, "resource_type", RESOURCE_TYPES)
        index = require_int(ordinal, "ordinal", minimum=1)
        resource_id = runtime_resource_id(
            self.attempt_id,
            self.retry_index,
            self.runtime_profile_hash,
            kind,
            index,
        )
        with self._lock:
            records = self._load_locked()
            if any(record.resource_id == resource_id for record in records):
                raise ContractError("resource_id: already declared")
            return self._append_locked(records, resource_id, kind, index, "declared", None)

    def verify(self) -> tuple[RuntimeResourceRecord, ...]:
        return self.records

    def created(self, resource_id: str, raw_handle_hash: str) -> RuntimeResourceRecord:
        return self._transition(resource_id, "created", raw_handle_hash)

    def create_failed(self, resource_id: str) -> RuntimeResourceRecord:
        return self._transition(resource_id, "create_failed", None)

    def released(self, resource_id: str) -> RuntimeResourceRecord:
        return self._transition(resource_id, "released", None)

    def cleanup_failed(self, resource_id: str) -> RuntimeResourceRecord:
        return self._transition(resource_id, "cleanup_failed", None)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.close(self._file_fd)
            os.close(self._parent_fd)
            self._closed = True

    def _transition(
        self,
        resource_id: str,
        transition: str,
        raw_handle_hash: str | None,
    ) -> RuntimeResourceRecord:
        identifier = require_str(resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        if raw_handle_hash is not None:
            raw_handle_hash = require_str(
                raw_handle_hash,
                "raw_handle_hash",
                pattern=SHA256_PATTERN,
            )
        with self._lock:
            records = self._load_locked()
            history = [record for record in records if record.resource_id == identifier]
            if not history:
                raise ContractError("resource_id: unknown resource_id")
            last = history[-1]
            if last.transition in {"released", "cleanup_failed", "create_failed"}:
                raise ContractError("resource_id: terminal transition already exists")
            if transition in {"released", "cleanup_failed"} and last.transition != "created":
                raise ContractError(f"resource transition: expected created before {transition}")
            if transition in {"created", "create_failed"} and last.transition != "declared":
                raise ContractError(
                    f"resource transition: expected declared before {transition}"
                )
            if transition == "created" and raw_handle_hash is None:
                raise ContractError("raw_handle_hash: created resource requires a hash")
            if transition != "created" and raw_handle_hash is not None:
                raise ContractError("raw_handle_hash: only created transition accepts a hash")
            effective_hash = (
                raw_handle_hash
                if transition == "created"
                else (last.raw_handle_hash if transition in {"released", "cleanup_failed"} else None)
            )
            return self._append_locked(
                records,
                identifier,
                last.resource_type,
                last.ordinal,
                transition,
                effective_hash,
            )

    def _append_locked(
        self,
        records: tuple[RuntimeResourceRecord, ...],
        resource_id: str,
        resource_type: str,
        ordinal: int,
        transition: str,
        raw_handle_hash: str | None,
    ) -> RuntimeResourceRecord:
        sequence = len(records) + 1
        previous_hash = records[-1].record_hash if records else None
        draft = RuntimeResourceRecord(
            sequence=sequence,
            previous_hash=previous_hash,
            record_hash="sha256:" + "0" * 64,
            attempt_id=self.attempt_id,
            retry_index=self.retry_index,
            runtime_profile_hash=self.runtime_profile_hash,
            resource_id=resource_id,
            resource_type=resource_type,
            ordinal=ordinal,
            transition=transition,
            raw_handle_hash=raw_handle_hash,
            recorded_at_ms=require_int(self._clock_ms(), "clock_ms", minimum=0),
        )
        record = RuntimeResourceRecord(
            **{
                **{item.name: getattr(draft, item.name) for item in fields(draft)},
                "record_hash": runtime_resource_record_hash(draft),
            }
        )
        encoded = (canonical_json(record.to_dict()) + "\n").encode("utf-8")
        fcntl.flock(self._parent_fd, fcntl.LOCK_EX)
        try:
            self._assert_bound_locked()
            fcntl.flock(self._file_fd, fcntl.LOCK_EX)
            try:
                raw = self._read_all()
                current = self._decode(raw)
                if current != records:
                    raise ContractError("runtime resource ledger changed concurrently")
                original_size = len(raw)
                try:
                    _write_all(self._file_fd, encoded)
                    os.fsync(self._file_fd)
                    self._assert_bound_locked()
                    os.fsync(self._parent_fd)
                except Exception as exc:  # noqa: BLE001 - reconcile durable state.
                    try:
                        os.ftruncate(self._file_fd, original_size)
                        os.fsync(self._file_fd)
                    except Exception:  # noqa: BLE001 - reconciliation decides state.
                        pass
                    reconciled = self._reconcile_failed_append(records, record)
                    if reconciled == "committed":
                        return record
                    if reconciled == "rolled_back":
                        if isinstance(exc, ContractError):
                            raise
                        raise ContractError("runtime resource ledger append failed") from exc
                    self._poisoned = True
                    raise ContractError(
                        "runtime resource ledger append has uncertain state and is poisoned"
                    ) from exc
            finally:
                fcntl.flock(self._file_fd, fcntl.LOCK_UN)
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)
        return record

    def _reconcile_failed_append(
        self,
        existing: tuple[RuntimeResourceRecord, ...],
        appended: RuntimeResourceRecord,
    ) -> str:
        try:
            durable = self._decode(self._read_all())
        except Exception:  # noqa: BLE001 - malformed tail is uncertain.
            return "uncertain"
        if durable == existing:
            state = "rolled_back"
        elif durable == (*existing, appended):
            state = "committed"
        else:
            return "uncertain"
        try:
            self._assert_bound_locked()
            os.fsync(self._file_fd)
            os.fsync(self._parent_fd)
        except Exception:  # noqa: BLE001 - durability remains uncertain.
            return "uncertain"
        return state

    def _load_locked(self) -> tuple[RuntimeResourceRecord, ...]:
        self._ensure_open()
        fcntl.flock(self._parent_fd, fcntl.LOCK_SH)
        try:
            self._assert_bound_locked()
            fcntl.flock(self._file_fd, fcntl.LOCK_SH)
            try:
                return self._decode(self._read_all())
            finally:
                fcntl.flock(self._file_fd, fcntl.LOCK_UN)
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    def _read_all(self) -> bytes:
        os.lseek(self._file_fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(self._file_fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _decode(self, raw: bytes) -> tuple[RuntimeResourceRecord, ...]:
        if raw and not raw.endswith(b"\n"):
            raise ContractError("runtime_resources.jsonl: missing final newline")
        records: list[RuntimeResourceRecord] = []
        for line_number, encoded in enumerate(raw.splitlines(), start=1):
            try:
                text = encoded.decode("utf-8")
                value = json.loads(text)
                if canonical_json(value).encode("utf-8") != encoded:
                    raise ContractError("expected canonical JSON")
                record = RuntimeResourceRecord.from_dict(
                    value,
                    path=f"runtime_resources.jsonl line {line_number}",
                )
                assert_public_artifact_safe(record.to_dict())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"runtime_resources.jsonl line {line_number}: invalid JSON"
                ) from exc
            records.append(record)
        self._verify_records(tuple(records))
        return tuple(records)

    def _verify_records(self, records: tuple[RuntimeResourceRecord, ...]) -> None:
        _verify_runtime_resource_records(
            records,
            self.attempt_id,
            self.retry_index,
            self.runtime_profile_hash,
        )

    def _assert_bound_locked(self) -> None:
        try:
            path_stat = os.stat(
                self._filename,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ContractError("runtime resource durable history is unavailable") from exc
        file_stat = os.fstat(self._file_fd)
        if not stat.S_ISREG(path_stat.st_mode) or (
            path_stat.st_dev,
            path_stat.st_ino,
        ) != (file_stat.st_dev, file_stat.st_ino):
            raise ContractError("ledger_path: symlink or replacement is denied")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractError("runtime resource ledger is closed")
        if self._poisoned:
            raise ContractError("runtime resource ledger is poisoned")


class RuntimeLeaseStore:
    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        retry_index: int,
        runtime_profile_hash: str,
    ) -> None:
        if not isinstance(path, Path):
            raise ContractError("lease_store_path: expected Path")
        if path.is_symlink():
            raise ContractError("lease_store_path: symlink is denied")
        self.attempt_id = require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        self.retry_index = require_int(retry_index, "retry_index", minimum=1)
        self.runtime_profile_hash = require_str(
            runtime_profile_hash,
            "runtime_profile_hash",
            pattern=SHA256_PATTERN,
        )
        self.path = path
        self._lock = threading.RLock()
        self._closed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ContractError("lease_store_path parent: symlink is denied")
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._parent_fd = os.open(path.parent, flags)
        except OSError as exc:
            raise ContractError("lease_store_path parent: expected real directory") from exc
        self._filename = path.name
        self._load_handles()
        try:
            os.stat(
                self._filename,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            self._write_handles(())

    @property
    def active_handles(self) -> tuple[RuntimeResourceHandle, ...]:
        with self._lock:
            self._ensure_open()
            return self._load_handles()

    def put_exact(
        self,
        resource_id: str,
        resource_type: str,
        ordinal: int,
        raw_handle: str,
    ) -> RuntimeResourceHandle:
        identifier = require_str(resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        kind = require_enum(resource_type, "resource_type", RESOURCE_TYPES)
        index = require_int(ordinal, "ordinal", minimum=1)
        expected = runtime_resource_id(
            self.attempt_id,
            self.retry_index,
            self.runtime_profile_hash,
            kind,
            index,
        )
        if identifier != expected:
            alternate = any(
                identifier
                == runtime_resource_id(
                    self.attempt_id,
                    self.retry_index,
                    self.runtime_profile_hash,
                    candidate,
                    index,
                )
                for candidate in RESOURCE_TYPES
            )
            if alternate:
                raise ContractError("resource_type mismatch for resource_id")
            raise ContractError("resource_id does not belong to lease store")
        handle = RuntimeResourceHandle(
            resource_id=identifier,
            resource_type=kind,
            ordinal=index,
            raw_handle=require_str(raw_handle, "raw_handle"),
            raw_handle_hash=runtime_raw_handle_hash(raw_handle),
        )
        with self._lock:
            existing = self._load_handles()
            by_id = {item.resource_id: item for item in existing}
            prior = by_id.get(identifier)
            if prior is not None:
                if prior == handle:
                    return prior
                raise ContractError("resource_id: conflicting private handle")
            if any(item.raw_handle_hash == handle.raw_handle_hash for item in existing):
                raise ContractError("raw handle is already owned by another resource")
            updated = tuple(sorted((*existing, handle), key=lambda item: item.resource_id))
            self._write_handles(updated)
            return handle

    def get_exact(self, resource_id: str) -> RuntimeResourceHandle:
        identifier = require_str(resource_id, "resource_id", pattern=RESOURCE_ID_PATTERN)
        with self._lock:
            for handle in self._load_handles():
                if handle.resource_id == identifier:
                    return handle
        raise ContractError("resource_id: unknown resource_id")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.close(self._parent_fd)
            self._closed = True

    def _load_handles(self) -> tuple[RuntimeResourceHandle, ...]:
        self._ensure_open()
        fcntl.flock(self._parent_fd, fcntl.LOCK_SH)
        try:
            flags = os.O_RDONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            if hasattr(os, "O_NONBLOCK"):
                flags |= os.O_NONBLOCK
            try:
                descriptor = os.open(self._filename, flags, dir_fd=self._parent_fd)
            except FileNotFoundError:
                return ()
            except OSError as exc:
                raise ContractError("lease_store_path: symlink or invalid file is denied") from exc
            try:
                if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                    raise ContractError("lease_store_path: expected regular file")
                raw = _read_descriptor(descriptor)
            finally:
                os.close(descriptor)
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("private_runtime_resources.json: invalid JSON") from exc
        if (canonical_json(value) + "\n").encode("utf-8") != raw:
            raise ContractError("private_runtime_resources.json: expected canonical JSON")
        data = require_exact_fields(
            value,
            "private_runtime_resources",
            (
                "store_type",
                "schema_version",
                "attempt_id",
                "retry_index",
                "runtime_profile_hash",
                "handles",
            ),
        )
        if data["store_type"] != "runtime_lease_store":
            raise ContractError("store_type: expected 'runtime_lease_store'")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
        if (
            data["attempt_id"] != self.attempt_id
            or data["retry_index"] != self.retry_index
            or data["runtime_profile_hash"] != self.runtime_profile_hash
        ):
            raise ContractError("private lease store identity mismatch")
        raw_handles = require_list(data["handles"], "handles")
        handles = tuple(
            RuntimeResourceHandle.from_dict(item, path=f"handles[{index}]")
            for index, item in enumerate(raw_handles)
        )
        if tuple(sorted(handles, key=lambda item: item.resource_id)) != handles:
            raise ContractError("handles: expected sorted resource_id order")
        ids = [item.resource_id for item in handles]
        hashes = [item.raw_handle_hash for item in handles]
        if len(ids) != len(set(ids)):
            raise ContractError("handles: duplicate resource_id")
        if len(hashes) != len(set(hashes)):
            raise ContractError("handles: duplicate raw handle")
        for handle in handles:
            expected = runtime_resource_id(
                self.attempt_id,
                self.retry_index,
                self.runtime_profile_hash,
                handle.resource_type,
                handle.ordinal,
            )
            if handle.resource_id != expected:
                raise ContractError("resource_id does not belong to lease store")
        return handles

    def _write_handles(self, handles: tuple[RuntimeResourceHandle, ...]) -> None:
        temporary = f".{self._filename}.{uuid.uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fcntl.flock(self._parent_fd, fcntl.LOCK_EX)
        descriptor: int | None = None
        try:
            existing = self._load_handles_without_lock()
            merged = {item.resource_id: item for item in existing}
            raw_hashes = {
                item.raw_handle_hash: item.resource_id for item in existing
            }
            for handle in handles:
                prior = merged.get(handle.resource_id)
                if prior is not None and prior != handle:
                    raise ContractError("resource_id: conflicting private handle")
                owner = raw_hashes.get(handle.raw_handle_hash)
                if owner is not None and owner != handle.resource_id:
                    raise ContractError(
                        "raw handle is already owned by another resource"
                    )
                merged[handle.resource_id] = handle
                raw_hashes[handle.raw_handle_hash] = handle.resource_id
            updated = tuple(sorted(merged.values(), key=lambda item: item.resource_id))
            payload = {
                "store_type": "runtime_lease_store",
                "schema_version": SCHEMA_VERSION,
                "attempt_id": self.attempt_id,
                "retry_index": self.retry_index,
                "runtime_profile_hash": self.runtime_profile_hash,
                "handles": [handle.to_dict() for handle in updated],
            }
            encoded = (canonical_json(payload) + "\n").encode("utf-8")
            descriptor = os.open(temporary, flags, 0o600, dir_fd=self._parent_fd)
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.rename(
                temporary,
                self._filename,
                src_dir_fd=self._parent_fd,
                dst_dir_fd=self._parent_fd,
            )
            os.fsync(self._parent_fd)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                os.unlink(temporary, dir_fd=self._parent_fd)
            except FileNotFoundError:
                pass
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    def _load_handles_without_lock(self) -> tuple[RuntimeResourceHandle, ...]:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(self._filename, flags, dir_fd=self._parent_fd)
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise ContractError("lease_store_path: symlink or invalid file is denied") from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ContractError("lease_store_path: expected regular file")
            raw = _read_descriptor(descriptor)
        finally:
            os.close(descriptor)
        return parse_runtime_lease_store(
            raw,
            attempt_id=self.attempt_id,
            retry_index=self.retry_index,
            runtime_profile_hash=self.runtime_profile_hash,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractError("runtime lease store is closed")


def _verify_runtime_resource_records(
    records: tuple[RuntimeResourceRecord, ...],
    attempt_id: str,
    retry_index: int,
    runtime_profile_hash: str,
) -> None:
    previous: str | None = None
    state: dict[str, str] = {}
    handle_hashes: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        if record.sequence != index:
            raise ContractError("runtime resource sequence is not continuous")
        if record.previous_hash != previous:
            raise ContractError("runtime resource previous_hash mismatch")
        if record.record_hash != runtime_resource_record_hash(record):
            raise ContractError("runtime resource record_hash mismatch")
        if (
            record.attempt_id != attempt_id
            or record.retry_index != retry_index
            or record.runtime_profile_hash != runtime_profile_hash
        ):
            raise ContractError("runtime resource record does not match ledger identity")
        expected_id = runtime_resource_id(
            record.attempt_id,
            record.retry_index,
            record.runtime_profile_hash,
            record.resource_type,
            record.ordinal,
        )
        if record.resource_id != expected_id:
            raise ContractError("resource_id does not match resource identity")
        prior = state.get(record.resource_id)
        if record.transition == "declared":
            if prior is not None:
                raise ContractError("resource_id: already declared")
            if record.raw_handle_hash is not None:
                raise ContractError("declared resource cannot have raw_handle_hash")
        elif record.transition in {"created", "create_failed"}:
            if prior != "declared":
                if prior in {"released", "cleanup_failed", "create_failed"}:
                    raise ContractError("resource_id: terminal transition already exists")
                raise ContractError(
                    f"resource transition: expected declared before {record.transition}"
                )
            if record.transition == "created":
                if record.raw_handle_hash is None:
                    raise ContractError("created resource requires raw_handle_hash")
                handle_hashes[record.resource_id] = record.raw_handle_hash
            elif record.raw_handle_hash is not None:
                raise ContractError("create_failed resource cannot have raw_handle_hash")
        else:
            if prior != "created":
                if prior in {"released", "cleanup_failed", "create_failed"}:
                    raise ContractError("resource_id: terminal transition already exists")
                raise ContractError(
                    f"resource transition: expected created before {record.transition}"
                )
            if record.raw_handle_hash != handle_hashes[record.resource_id]:
                raise ContractError("raw_handle_hash changed across resource transitions")
        state[record.resource_id] = record.transition
        previous = record.record_hash


def _optional_hash(value: object, path: str) -> str | None:
    if value is None:
        return None
    return require_str(value, path, pattern=SHA256_PATTERN)


def _read_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("runtime resource write made no progress")
        offset += written


__all__ = [
    "AttemptResourceLedger",
    "RuntimeCleanupEntry",
    "RuntimeCleanupReport",
    "RuntimeLease",
    "RuntimeLeaseStore",
    "RuntimeResourceHandle",
    "RuntimeResourceRecord",
    "runtime_raw_handle_hash",
    "runtime_resource_id",
    "runtime_resource_record_hash",
    "parse_runtime_lease_store",
    "parse_runtime_resource_ledger",
    "verify_runtime_resource_evidence",
    "verify_runtime_resource_ownership",
    "verify_runtime_cleanup",
]
