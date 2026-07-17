from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import fcntl
import json
import os
from pathlib import Path
import stat
import threading

from op_bench.runtime.artifacts import PublicArtifactStore
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import (
    ActionObservation,
    ActionRequest,
    EventRecord,
    SCHEMA_VERSION,
)
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_int, require_str


class EventJournal:
    """Server-owned append-only EventRecord sequence and public hash chain."""

    def __init__(
        self,
        session_id: str,
        *,
        clock_ms: Callable[[], int],
        events_path: Path | None = None,
        artifact_store: PublicArtifactStore | None = None,
        max_inline_bytes: int = 4096,
    ) -> None:
        require_str(session_id, "session_id")
        if not callable(clock_ms):
            raise ContractError("clock_ms: expected callable")
        if events_path is not None and not isinstance(events_path, Path):
            raise ContractError("events_path: expected Path")
        if artifact_store is not None and not isinstance(artifact_store, PublicArtifactStore):
            raise ContractError("artifact_store: expected PublicArtifactStore")
        require_int(max_inline_bytes, "max_inline_bytes", minimum=1)
        self.session_id = session_id
        self._clock_ms = clock_ms
        self._events_path = events_path
        self._durable_configured = events_path is not None
        self._artifact_store = artifact_store
        self._max_inline_bytes = max_inline_bytes
        self._lock = threading.RLock()
        self._parent_fd: int | None = None
        self._file_fd: int | None = None
        self._filename: str | None = None
        if events_path is not None:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            if events_path.parent.is_symlink():
                raise ContractError("events_path parent: symlink is denied")
            directory_flags = os.O_RDONLY | os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                directory_flags |= os.O_NOFOLLOW
            try:
                self._parent_fd = os.open(events_path.parent, directory_flags)
            except OSError as exc:
                raise ContractError("events_path parent: expected real directory") from exc
            self._filename = events_path.name
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
                        "events_path: symlink or invalid file is denied"
                    ) from exc
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    os.close(descriptor)
                    raise ContractError("events_path: expected regular file")
                self._file_fd = descriptor
                os.fsync(self._parent_fd)
            except Exception:
                if self._file_fd is not None:
                    os.close(self._file_fd)
                    self._file_fd = None
                os.close(self._parent_fd)
                self._parent_fd = None
                raise
            finally:
                if self._parent_fd is not None:
                    fcntl.flock(self._parent_fd, fcntl.LOCK_UN)
        self._closed = False
        self._poisoned = False
        self._records = list(self._load_existing())
        self._terminal_emitted = any(
            record.event_type == "terminal_emitted" for record in self._records
        )

    @property
    def records(self) -> tuple[EventRecord, ...]:
        with self._lock:
            self._ensure_open()
            if self._durable_configured:
                durable = self._load_existing()
                self._assert_durable_continuity(durable)
                self._records = list(durable)
                self._terminal_emitted = any(
                    record.event_type == "terminal_emitted" for record in durable
                )
            return tuple(self._records)

    @staticmethod
    def event_hash_for(
        *,
        session_id: str,
        sequence: int,
        occurred_at_ms: int,
        event_type: str,
        public_payload: Mapping[str, object],
        previous_event_hash: str | None,
    ) -> str:
        return canonical_sha256(
            {
                "contract_type": "event_record",
                "schema_version": SCHEMA_VERSION,
                "session_id": session_id,
                "sequence": sequence,
                "occurred_at_ms": occurred_at_ms,
                "event_type": event_type,
                "public_payload": dict(public_payload),
                "previous_event_hash": previous_event_hash,
            }
        )

    def append(self, event_type: str, public_payload: Mapping[str, object]) -> EventRecord:
        return self.append_batch(((event_type, public_payload),))[0]

    def append_batch(
        self,
        events: Sequence[tuple[str, Mapping[str, object]]],
    ) -> tuple[EventRecord, ...]:
        if not isinstance(events, Sequence) or not events:
            raise ContractError("events: expected non-empty sequence")
        prepared: list[tuple[str, dict[str, object]]] = []
        for index, item in enumerate(events):
            if not isinstance(item, tuple) or len(item) != 2:
                raise ContractError(f"events[{index}]: expected (event_type, payload)")
            event_type, public_payload = item
            require_str(event_type, f"events[{index}].event_type")
            if not isinstance(public_payload, Mapping):
                raise ContractError(f"events[{index}].public_payload: expected object")
            if any(not isinstance(key, str) for key in public_payload):
                raise ContractError(
                    f"events[{index}].public_payload: object keys must be strings"
                )
            payload = dict(public_payload)
            assert_public_artifact_safe(payload)
            prepared.append((event_type, payload))
        with self._lock:
            self._ensure_open()
            if self._poisoned:
                raise ContractError("event journal is poisoned")
            if not self._durable_configured:
                existing = tuple(self._records)
                appended = self._build_records(existing, prepared)
                self._records.extend(appended)
            else:
                existing, appended = self._append_durable_batch(prepared)
                self._records = [*existing, *appended]
            self._terminal_emitted = any(
                record.event_type == "terminal_emitted" for record in self._records
            )
            return appended

    def _build_records(
        self,
        existing: Sequence[EventRecord],
        events: Sequence[tuple[str, dict[str, object]]],
    ) -> tuple[EventRecord, ...]:
        terminal_seen = any(record.event_type == "terminal_emitted" for record in existing)
        if terminal_seen:
            raise ContractError("terminal event already exists")
        built: list[EventRecord] = []
        previous = existing[-1].event_hash if existing else None
        for event_type, payload in events:
            if terminal_seen:
                raise ContractError("terminal event must be the final event")
            sequence = len(existing) + len(built) + 1
            occurred_at_ms = require_int(self._clock_ms(), "clock_ms", minimum=0)
            event_hash = self.event_hash_for(
                session_id=self.session_id,
                sequence=sequence,
                occurred_at_ms=occurred_at_ms,
                event_type=event_type,
                public_payload=payload,
                previous_event_hash=previous,
            )
            record = EventRecord(
                session_id=self.session_id,
                sequence=sequence,
                occurred_at_ms=occurred_at_ms,
                event_type=event_type,
                public_payload=payload,
                previous_event_hash=previous,
                event_hash=event_hash,
            )
            built.append(record)
            previous = event_hash
            if event_type == "terminal_emitted":
                terminal_seen = True
        return tuple(built)

    def record_action_requested(self, request: ActionRequest) -> None:
        if not isinstance(request, ActionRequest):
            raise ContractError("request: expected ActionRequest")
        if request.session_id != self.session_id:
            raise ContractError("request: session does not match EventJournal")
        common = {
            "action_id": request.action_id,
            "action_name": request.action_name,
            "request_hash": request.content_hash,
            "client_sequence": request.client_sequence,
            "deadline_ms": request.deadline_ms,
        }
        events: list[tuple[str, Mapping[str, object]]] = []
        if request.action_name == "session_finish":
            events.append(("finish_requested", common))
        events.append(("action_requested", common))
        if request.action_name == "test_run":
            events.append(
                ("test_started", {
                    "action_id": request.action_id,
                    "request_hash": request.content_hash,
                }),
            )
        self.append_batch(tuple(events))

    def record_action_observed(
        self,
        request: ActionRequest,
        observation: ActionObservation,
    ) -> None:
        if not isinstance(request, ActionRequest):
            raise ContractError("request: expected ActionRequest")
        if not isinstance(observation, ActionObservation):
            raise ContractError("observation: expected ActionObservation")
        if request.session_id != self.session_id or observation.session_id != self.session_id:
            raise ContractError("action exchange: session does not match EventJournal")
        if request.action_id != observation.action_id:
            raise ContractError("action exchange: action_id mismatch")
        encoded = observation.to_dict()
        data = encoded["data"]
        payload: dict[str, object] = {
            "action_id": request.action_id,
            "action_name": request.action_name,
            "request_hash": request.content_hash,
            "observation_hash": observation.content_hash,
            "ok": observation.ok,
            "error_code": observation.error_code,
            "budget_delta": encoded["budget_delta"],
            "mutation_state": observation.mutation_state,
        }
        if data:
            data_bytes = canonical_json(data).encode("utf-8")
            if len(data_bytes) <= self._max_inline_bytes:
                payload["data"] = data
            else:
                if self._artifact_store is None:
                    raise ContractError("large event data requires an artifact store")
                reference = self._artifact_store.put_json(
                    f"action-observation-{request.action_id}",
                    data,
                )
                payload["data_artifact"] = reference.to_dict()
        events: list[tuple[str, Mapping[str, object]]] = [("action_observed", payload)]
        if request.action_name == "test_run":
            events.append(
                ("test_completed", {
                    "action_id": request.action_id,
                    "observation_hash": observation.content_hash,
                    "ok": observation.ok,
                    "error_code": observation.error_code,
                }),
            )
        events.append(
            ("budget_updated", {
                "action_id": request.action_id,
                "observation_hash": observation.content_hash,
                "budget_delta": encoded["budget_delta"],
            }),
        )
        if observation.error_code == "budget_exhausted":
            events.append(
                ("budget_exhausted", {
                    "action_id": request.action_id,
                    "observation_hash": observation.content_hash,
                }),
            )
        self.append_batch(tuple(events))

    def _load_existing(self) -> tuple[EventRecord, ...]:
        if self._parent_fd is None or self._file_fd is None:
            return ()
        fcntl.flock(self._parent_fd, fcntl.LOCK_SH)
        try:
            self._assert_bound_file_locked()
            descriptor = self._file_fd
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH)
                raw = self._read_all(descriptor)
                return self._decode_records(raw)
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    def _decode_records(self, raw: bytes) -> tuple[EventRecord, ...]:
        if raw and not raw.endswith(b"\n"):
            raise ContractError("events.jsonl: missing final newline")
        records: list[EventRecord] = []
        for line_number, encoded_line in enumerate(raw.splitlines(), start=1):
            if not encoded_line:
                raise ContractError(f"events.jsonl line {line_number}: empty line")
            try:
                line = encoded_line.decode("utf-8")
                payload = json.loads(line)
                if canonical_json(payload).encode("utf-8") != encoded_line:
                    raise ContractError("expected canonical JSON")
                record = EventRecord.from_dict(payload, path=f"events.jsonl line {line_number}")
                if record.session_id != self.session_id:
                    raise ContractError("session_id does not match journal")
                assert_public_artifact_safe(record.to_dict())
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"events.jsonl line {line_number}: invalid event"
                ) from exc
            except ContractError as exc:
                raise ContractError(
                    f"events.jsonl line {line_number}: {exc}"
                ) from exc
            records.append(record)
        issues = verify_event_chain(records)
        if issues:
            raise ContractError(f"events.jsonl: invalid event chain: {issues[0]}")
        terminals = sum(record.event_type == "terminal_emitted" for record in records)
        if terminals > 1:
            raise ContractError("events.jsonl: duplicate terminal event")
        if terminals == 1 and records[-1].event_type != "terminal_emitted":
            raise ContractError("events.jsonl: terminal event must be final")
        return tuple(records)

    def _append_durable_batch(
        self,
        events: Sequence[tuple[str, dict[str, object]]],
    ) -> tuple[tuple[EventRecord, ...], tuple[EventRecord, ...]]:
        if self._parent_fd is None or self._file_fd is None:
            raise ContractError("event journal durable path is not configured")
        fcntl.flock(self._parent_fd, fcntl.LOCK_EX)
        try:
            self._assert_bound_file_locked()
            descriptor = self._file_fd
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise ContractError("events_path: expected regular file")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                raw = self._read_all(descriptor)
                existing = self._decode_records(raw)
                self._assert_durable_continuity(existing)
                appended = self._build_records(existing, events)
                encoded = b"".join(
                    (canonical_json(record.to_dict()) + "\n").encode("utf-8")
                    for record in appended
                )
                original_size = len(raw)
                try:
                    os.lseek(descriptor, 0, os.SEEK_END)
                    self._write_all(descriptor, encoded)
                    os.fsync(descriptor)
                    self._assert_bound_file_locked()
                    os.fsync(self._parent_fd)
                except Exception as exc:  # noqa: BLE001 - rollback is atomic boundary.
                    try:
                        os.ftruncate(descriptor, original_size)
                        os.fsync(descriptor)
                    except Exception:  # noqa: BLE001 - preserve primary failure.
                        pass
                    reconciled = self._reconcile_failed_append(
                        descriptor,
                        existing=existing,
                        appended=appended,
                    )
                    if reconciled == "committed":
                        return existing, appended
                    if reconciled == "rolled_back":
                        if isinstance(exc, ContractError):
                            raise
                        raise ContractError("event journal append failed") from exc
                    self._poisoned = True
                    raise ContractError(
                        "event journal append has uncertain state and is poisoned"
                    ) from exc
                return existing, appended
            finally:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            fcntl.flock(self._parent_fd, fcntl.LOCK_UN)

    def _reconcile_failed_append(
        self,
        descriptor: int,
        *,
        existing: Sequence[EventRecord],
        appended: Sequence[EventRecord],
    ) -> str:
        try:
            raw = self._read_all(descriptor)
            durable = self._decode_records(raw)
        except Exception:  # noqa: BLE001 - malformed tail is an uncertain commit.
            return "uncertain"
        if durable == tuple(existing):
            state = "rolled_back"
        elif durable == (*existing, *appended):
            state = "committed"
        else:
            return "uncertain"
        try:
            self._assert_bound_file_locked()
            os.fsync(descriptor)
            if self._parent_fd is None:
                return "uncertain"
            os.fsync(self._parent_fd)
        except Exception:  # noqa: BLE001 - durability is still uncertain.
            return "uncertain"
        return state

    def _assert_bound_file_locked(self) -> None:
        if self._parent_fd is None or self._file_fd is None or self._filename is None:
            raise ContractError("event journal is closed")
        try:
            path_metadata = os.stat(
                self._filename,
                dir_fd=self._parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ContractError(
                "event journal durable history was lost or replaced"
            ) from exc
        descriptor_metadata = os.fstat(self._file_fd)
        if (
            not stat.S_ISREG(path_metadata.st_mode)
            or path_metadata.st_dev != descriptor_metadata.st_dev
            or path_metadata.st_ino != descriptor_metadata.st_ino
        ):
            raise ContractError(
                "event journal durable history was lost or replaced"
            )

    @staticmethod
    def _read_all(descriptor: int) -> bytes:
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
                raise ContractError("event journal append failed")
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
            file_descriptor = self._file_fd
            parent_descriptor = self._parent_fd
            self._file_fd = None
            self._parent_fd = None
            if file_descriptor is not None:
                os.close(file_descriptor)
            if parent_descriptor is not None:
                os.close(parent_descriptor)

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise ContractError("event journal is closed")

    def _assert_durable_continuity(
        self,
        durable: Sequence[EventRecord],
    ) -> None:
        cached = tuple(getattr(self, "_records", ()))
        if len(durable) < len(cached) or tuple(durable[: len(cached)]) != cached:
            raise ContractError("event journal durable history was lost or replaced")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort descriptor cleanup.
            pass


def verify_event_chain(records: Sequence[EventRecord]) -> tuple[str, ...]:
    issues: list[str] = []
    previous: str | None = None
    session_id: str | None = None
    for index, record in enumerate(records, start=1):
        if not isinstance(record, EventRecord):
            issues.append(f"event {index}: expected EventRecord")
            continue
        if session_id is None:
            session_id = record.session_id
        elif record.session_id != session_id:
            issues.append(f"event {index}: session_id changed")
        if record.sequence != index:
            issues.append(f"event {index}: expected sequence {index}, got {record.sequence}")
        if record.previous_event_hash != previous:
            issues.append(f"event {index}: previous_event_hash mismatch")
        payload = record.to_dict()["public_payload"]
        expected = EventJournal.event_hash_for(
            session_id=record.session_id,
            sequence=record.sequence,
            occurred_at_ms=record.occurred_at_ms,
            event_type=record.event_type,
            public_payload=payload,
            previous_event_hash=record.previous_event_hash,
        )
        if record.event_hash != expected:
            issues.append(f"event {index}: event_hash mismatch")
        previous = record.event_hash
    return tuple(issues)


def verify_action_pairing(records: Sequence[EventRecord]) -> tuple[str, ...]:
    requested: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    request_identity: dict[str, tuple[object, object, int]] = {}
    issues: list[str] = []
    for index, record in enumerate(records, start=1):
        if record.event_type not in {"action_requested", "action_observed"}:
            continue
        action_id = record.public_payload.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            issues.append(f"event {index}: action event missing action_id")
            continue
        if record.event_type == "action_requested":
            requested[action_id] += 1
            if requested[action_id] == 1:
                request_identity[action_id] = (
                    record.public_payload.get("request_hash"),
                    record.public_payload.get("action_name"),
                    index,
                )
        else:
            observed[action_id] += 1
            if requested[action_id] == 0:
                issues.append(f"action {action_id!r}: observation without request")
                continue
            request_hash, action_name, request_index = request_identity[action_id]
            if request_index >= index:
                issues.append(f"action {action_id!r}: observation precedes request")
            if record.public_payload.get("request_hash") != request_hash:
                issues.append(f"action {action_id!r}: request_hash mismatch")
            if record.public_payload.get("action_name") != action_name:
                issues.append(f"action {action_id!r}: action_name mismatch")
    for action_id in sorted(set(requested) | set(observed)):
        if requested[action_id] != 1:
            issues.append(
                f"action {action_id!r}: expected one request, got {requested[action_id]}"
            )
        if observed[action_id] != 1:
            issues.append(
                f"action {action_id!r}: expected one observation, got {observed[action_id]}"
            )
    return tuple(issues)


__all__ = ["EventJournal", "verify_action_pairing", "verify_event_chain"]
