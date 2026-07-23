from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
import os
from pathlib import Path
import stat
import threading

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import (
    ATTEMPT_VALIDITIES,
    RESUME_POLICIES,
    SCHEMA_VERSION,
    SHA256_PATTERN,
    EvaluationResultV06,
    SessionResult,
)
from op_bench.runtime.manifest import ATTEMPT_PATTERN
from op_bench.runtime.evaluation import validate_session_evaluation_binding
from op_bench.runtime.session import termination_attribution
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_int,
    require_str,
)


_DECISIONS = ("run", "skip", "blocked")


@dataclass(frozen=True)
class AttemptLedgerRecord:
    attempt_id: str
    session_id: str
    retry_index: int
    attempt_validity: str
    session_result: SessionResult
    session_result_hash: str
    evaluation_result: EvaluationResultV06
    evaluation_result_hash: str
    evaluation_spec_hash: str
    recorded_at_ms: int

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        require_str(self.session_id, "session_id")
        require_int(self.retry_index, "retry_index", minimum=1)
        require_enum(self.attempt_validity, "attempt_validity", ATTEMPT_VALIDITIES)
        if not isinstance(self.session_result, SessionResult):
            raise ContractError("session_result: expected SessionResult")
        if self.session_result.attempt_id != self.attempt_id:
            raise ContractError("attempt_id: does not match SessionResult")
        if self.session_result.session_id != self.session_id:
            raise ContractError("session_id: does not match SessionResult")
        require_str(
            self.session_result_hash,
            "session_result_hash",
            pattern=SHA256_PATTERN,
        )
        if self.session_result_hash != self.session_result.content_hash:
            raise ContractError("session_result_hash: does not match SessionResult")
        if not isinstance(self.evaluation_result, EvaluationResultV06):
            raise ContractError("evaluation_result: expected EvaluationResultV06")
        if self.evaluation_result.attempt_id != self.attempt_id:
            raise ContractError("attempt_id: does not match EvaluationResult")
        if self.evaluation_result.session_id != self.session_id:
            raise ContractError("session_id: does not match EvaluationResult")
        require_str(
            self.evaluation_result_hash,
            "evaluation_result_hash",
            pattern=SHA256_PATTERN,
        )
        if self.evaluation_result_hash != self.evaluation_result.content_hash:
            raise ContractError(
                "evaluation_result_hash: does not match EvaluationResult"
            )
        require_str(
            self.evaluation_spec_hash,
            "evaluation_spec_hash",
            pattern=SHA256_PATTERN,
        )
        require_int(self.recorded_at_ms, "recorded_at_ms", minimum=0)
        if self.attempt_validity != self.evaluation_result.attempt_validity:
            raise ContractError(
                "attempt_validity: does not match EvaluationResult"
            )
        attribution = termination_attribution(self.session_result.terminal_reason)
        if self.evaluation_result.agent_terminal != attribution.agent_terminal:
            raise ContractError("agent_terminal: does not match SessionResult")
        if self.evaluation_result.patch != self.session_result.final_patch:
            raise ContractError("patch: SessionResult and EvaluationResult differ")
        validate_session_evaluation_binding(
            self.session_result,
            self.evaluation_result,
        )
        assert_public_artifact_safe(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "record_type": "attempt_ledger_record",
            "schema_version": SCHEMA_VERSION,
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "retry_index": self.retry_index,
            "attempt_validity": self.attempt_validity,
            "session_result": self.session_result.to_dict(),
            "session_result_hash": self.session_result_hash,
            "evaluation_result": self.evaluation_result.to_dict(),
            "evaluation_result_hash": self.evaluation_result_hash,
            "evaluation_spec_hash": self.evaluation_spec_hash,
            "recorded_at_ms": self.recorded_at_ms,
        }

    @classmethod
    def from_dict(cls, value: object) -> "AttemptLedgerRecord":
        data = require_exact_fields(
            value,
            "attempt_ledger_record",
            (
                "record_type",
                "schema_version",
                "attempt_id",
                "session_id",
                "retry_index",
                "attempt_validity",
                "session_result",
                "session_result_hash",
                "evaluation_result",
                "evaluation_result_hash",
                "evaluation_spec_hash",
                "recorded_at_ms",
            ),
        )
        if data["record_type"] != "attempt_ledger_record":
            raise ContractError("record_type: expected 'attempt_ledger_record'")
        if data["schema_version"] != SCHEMA_VERSION:
            raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
        return cls(
            attempt_id=require_str(data["attempt_id"], "attempt_id"),
            session_id=require_str(data["session_id"], "session_id"),
            retry_index=require_int(data["retry_index"], "retry_index", minimum=1),
            attempt_validity=require_enum(
                data["attempt_validity"],
                "attempt_validity",
                ATTEMPT_VALIDITIES,
            ),
            session_result=SessionResult.from_dict(
                data["session_result"], path="attempt_ledger_record.session_result"
            ),
            session_result_hash=require_str(
                data["session_result_hash"], "session_result_hash"
            ),
            evaluation_result=EvaluationResultV06.from_dict(
                data["evaluation_result"],
                path="attempt_ledger_record.evaluation_result",
            ),
            evaluation_result_hash=require_str(
                data["evaluation_result_hash"], "evaluation_result_hash"
            ),
            evaluation_spec_hash=require_str(
                data["evaluation_spec_hash"], "evaluation_spec_hash"
            ),
            recorded_at_ms=require_int(
                data["recorded_at_ms"], "recorded_at_ms", minimum=0
            ),
        )


@dataclass(frozen=True)
class ResumeDecision:
    action: str
    retry_index: int
    reason: str

    def __post_init__(self) -> None:
        require_enum(self.action, "action", _DECISIONS)
        require_int(self.retry_index, "retry_index", minimum=1)
        require_str(self.reason, "reason")


class AttemptLedger:
    """Strict append-only retry audit and resume decision authority."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise ContractError("ledger path: expected Path")
        if path.is_symlink():
            raise ContractError("ledger path: symlink is denied")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.parent.is_symlink():
            raise ContractError("ledger path parent: symlink is denied")
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            self._parent_fd = os.open(path.parent, directory_flags)
        except OSError as exc:
            raise ContractError("ledger path parent: expected real directory") from exc
        self._filename = path.name
        self._file_fd: int | None = None
        file_flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            file_flags |= os.O_NONBLOCK
        fcntl.flock(self._parent_fd, fcntl.LOCK_EX)
        try:
            try:
                descriptor = os.open(
                    self._filename,
                    file_flags,
                    0o600,
                    dir_fd=self._parent_fd,
                )
            except OSError as exc:
                raise ContractError(
                    "ledger path: symlink or invalid file is denied"
                ) from exc
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                os.close(descriptor)
                raise ContractError("ledger path: expected regular file")
            self._file_fd = descriptor
            os.fsync(self._parent_fd)
        except Exception:
            if self._file_fd is not None:
                os.close(self._file_fd)
                self._file_fd = None
            os.close(self._parent_fd)
            raise
        finally:
            if self._file_fd is not None:
                fcntl.flock(self._parent_fd, fcntl.LOCK_UN)
        self._lock = threading.RLock()
        self._closed = False
        self._poisoned = False
        self._records = list(self._load())
        self._validate_history(self._records)

    def append(
        self,
        *,
        session_result: SessionResult,
        evaluation_result: EvaluationResultV06,
        evaluation_spec_hash: str,
        retry_index: int,
        recorded_at_ms: int,
    ) -> AttemptLedgerRecord:
        if not isinstance(session_result, SessionResult):
            raise ContractError("session_result: expected SessionResult")
        if not isinstance(evaluation_result, EvaluationResultV06):
            raise ContractError("evaluation_result: expected EvaluationResultV06")
        record = AttemptLedgerRecord(
            attempt_id=session_result.attempt_id,
            session_id=session_result.session_id,
            retry_index=retry_index,
            attempt_validity=evaluation_result.attempt_validity,
            session_result=session_result,
            session_result_hash=session_result.content_hash,
            evaluation_result=evaluation_result,
            evaluation_result_hash=evaluation_result.content_hash,
            evaluation_spec_hash=evaluation_spec_hash,
            recorded_at_ms=recorded_at_ms,
        )
        with self._lock:
            self._ensure_open()
            if self._poisoned:
                raise ContractError("attempt ledger is poisoned")
            fcntl.flock(self._parent_fd, fcntl.LOCK_EX)
            try:
                self._assert_bound_file_locked()
                descriptor = self._file_fd
                if descriptor is None:
                    raise ContractError("attempt ledger is closed")
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                    raw = self._read_all(descriptor)
                    durable = list(self._decode_records(raw))
                    self._assert_durable_continuity(durable)
                    durable = self._preserve_cached_identity(durable)
                    self._validate_history(durable)
                    self._records = durable
                    for existing in durable:
                        if existing.session_id != record.session_id:
                            continue
                        if existing == record:
                            return existing
                        raise ContractError("session_id: conflicting ledger record")

                    history = [
                        item for item in durable if item.attempt_id == record.attempt_id
                    ]
                    if any(item.attempt_validity == "valid" for item in history):
                        raise ContractError("attempt_id: valid result already exists")
                    expected_retry = len(history) + 1
                    if record.retry_index != expected_retry:
                        raise ContractError(
                            f"retry_index: expected {expected_retry}, got {record.retry_index}"
                        )
                    encoded = (canonical_json(record.to_dict()) + "\n").encode("utf-8")
                    original_size = len(raw)
                    try:
                        os.lseek(descriptor, 0, os.SEEK_END)
                        self._write_all(descriptor, encoded)
                        os.fsync(descriptor)
                        self._assert_bound_file_locked()
                        os.fsync(self._parent_fd)
                    except Exception as exc:  # noqa: BLE001 - append rollback boundary.
                        try:
                            os.ftruncate(descriptor, original_size)
                            os.fsync(descriptor)
                        except Exception:  # noqa: BLE001 - preserve primary failure.
                            pass
                        reconciled = self._reconcile_failed_append(
                            descriptor,
                            durable=durable,
                            record=record,
                        )
                        if reconciled == "committed":
                            self._records.append(record)
                            return record
                        if reconciled == "rolled_back":
                            if isinstance(exc, ContractError):
                                raise
                            raise ContractError("attempt ledger append failed") from exc
                        self._poisoned = True
                        raise ContractError(
                            "attempt ledger append has uncertain state and is poisoned"
                        ) from exc
                    self._records.append(record)
                    return record
                finally:
                    try:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                    except OSError:
                        pass
            finally:
                fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    def records(self, attempt_id: str | None = None) -> tuple[AttemptLedgerRecord, ...]:
        with self._lock:
            self._ensure_open()
            self._refresh_locked()
            if attempt_id is None:
                return tuple(self._records)
            require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
            return tuple(item for item in self._records if item.attempt_id == attempt_id)

    def latest_valid(self, attempt_id: str) -> AttemptLedgerRecord | None:
        valid = [
            item
            for item in self.records(attempt_id)
            if item.attempt_validity == "valid"
        ]
        return valid[-1] if valid else None

    def decide(self, attempt_id: str, resume_policy: str) -> ResumeDecision:
        require_str(attempt_id, "attempt_id", pattern=ATTEMPT_PATTERN)
        selected_policy = require_enum(
            resume_policy, "resume_policy", RESUME_POLICIES
        )
        history = self.records(attempt_id)
        if not history:
            return ResumeDecision("run", 1, "no prior attempt")
        retry_index = history[-1].retry_index
        if any(item.attempt_validity == "valid" for item in history):
            if selected_policy == "never":
                return ResumeDecision(
                    "blocked", retry_index, "resume policy forbids existing history"
                )
            return ResumeDecision("skip", retry_index, "valid attempt already complete")
        if selected_policy == "retry_infrastructure":
            return ResumeDecision(
                "run", retry_index + 1, "retrying infrastructure-invalid attempt"
            )
        if selected_policy == "skip_valid":
            return ResumeDecision(
                "blocked", retry_index, "infrastructure retry is not enabled"
            )
        return ResumeDecision(
            "blocked", retry_index, "resume policy forbids existing history"
        )

    def _load(self) -> tuple[AttemptLedgerRecord, ...]:
        if self._file_fd is None:
            raise ContractError("attempt ledger is closed")
        fcntl.flock(self._parent_fd, fcntl.LOCK_SH)
        try:
            self._assert_bound_file_locked()
            descriptor = self._file_fd
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH)
                return self._decode_records(self._read_all(descriptor))
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    @staticmethod
    def _decode_records(raw: bytes) -> tuple[AttemptLedgerRecord, ...]:
        if raw and not raw.endswith(b"\n"):
            raise ContractError("attempt ledger: missing final newline")
        records: list[AttemptLedgerRecord] = []
        for line_number, encoded_line in enumerate(raw.splitlines(), start=1):
            if not encoded_line:
                raise ContractError(f"attempt ledger line {line_number}: empty line")
            try:
                line = encoded_line.decode("utf-8")
                payload = json.loads(line)
                if canonical_json(payload).encode("utf-8") != encoded_line:
                    raise ContractError("expected canonical JSON")
                record = AttemptLedgerRecord.from_dict(payload)
            except (UnicodeDecodeError, json.JSONDecodeError, ContractError) as exc:
                raise ContractError(
                    f"attempt ledger line {line_number}: invalid record"
                ) from exc
            records.append(record)
        return tuple(records)

    @staticmethod
    def _validate_history(records: list[AttemptLedgerRecord]) -> None:
        sessions: set[str] = set()
        by_attempt: dict[str, list[AttemptLedgerRecord]] = {}
        for record in records:
            if record.session_id in sessions:
                raise ContractError("attempt ledger: duplicate session_id")
            sessions.add(record.session_id)
            history = by_attempt.setdefault(record.attempt_id, [])
            expected = len(history) + 1
            if record.retry_index != expected:
                raise ContractError(
                    f"attempt ledger: retry_index expected {expected} for {record.attempt_id}"
                )
            if any(item.attempt_validity == "valid" for item in history):
                raise ContractError("attempt ledger: record follows valid result")
            history.append(record)

    @staticmethod
    def _read_all(descriptor: int) -> bytes:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("ledger path: expected regular file")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractError("attempt ledger append failed")
            view = view[written:]

    def close(self) -> None:
        try:
            lock = self._lock
        except AttributeError:
            return
        with lock:
            if self._closed:
                return
            self._closed = True
            if self._file_fd is not None:
                os.close(self._file_fd)
                self._file_fd = None
            os.close(self._parent_fd)
            self._parent_fd = -1

    def _reconcile_failed_append(
        self,
        descriptor: int,
        *,
        durable: list[AttemptLedgerRecord],
        record: AttemptLedgerRecord,
    ) -> str:
        try:
            observed = list(self._decode_records(self._read_all(descriptor)))
        except Exception:  # noqa: BLE001 - malformed tail is uncertain.
            return "uncertain"
        if observed == durable:
            state = "rolled_back"
        elif observed == [*durable, record]:
            state = "committed"
        else:
            return "uncertain"
        try:
            self._assert_bound_file_locked()
            os.fsync(descriptor)
            os.fsync(self._parent_fd)
        except Exception:  # noqa: BLE001 - durability remains uncertain.
            return "uncertain"
        return state

    def _assert_bound_file_locked(self) -> None:
        if self._file_fd is None:
            raise ContractError("attempt ledger is closed")
        try:
            path_metadata = os.stat(
                self._filename,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ContractError(
                "attempt ledger durable history was lost or replaced"
            ) from exc
        descriptor_metadata = os.fstat(self._file_fd)
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise ContractError(
                "attempt ledger durable history was lost or replaced"
            )

    def _refresh_locked(self) -> None:
        durable = list(self._load())
        self._assert_durable_continuity(durable)
        self._records = self._preserve_cached_identity(durable)

    def _assert_durable_continuity(
        self,
        durable: list[AttemptLedgerRecord],
    ) -> None:
        cached = self._records
        if len(durable) < len(cached) or durable[: len(cached)] != cached:
            raise ContractError("attempt ledger durable history was lost or replaced")

    def _preserve_cached_identity(
        self,
        durable: list[AttemptLedgerRecord],
    ) -> list[AttemptLedgerRecord]:
        cached_by_session = {item.session_id: item for item in self._records}
        return [
            cached_by_session.get(item.session_id, item)
            if cached_by_session.get(item.session_id) == item
            else item
            for item in durable
        ]

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise ContractError("attempt ledger is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort descriptor cleanup.
            pass


def parse_attempt_ledger(raw: bytes) -> tuple[AttemptLedgerRecord, ...]:
    """Parse and validate durable ledger bytes without creating or opening a path."""

    if not isinstance(raw, bytes):
        raise ContractError("attempt ledger: expected bytes")
    records = AttemptLedger._decode_records(raw)
    AttemptLedger._validate_history(list(records))
    return records


__all__ = [
    "AttemptLedger",
    "AttemptLedgerRecord",
    "ResumeDecision",
    "parse_attempt_ledger",
]
