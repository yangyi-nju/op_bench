from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Callable, Sequence

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import (
    AgentTaskView,
    EvaluationSpec,
    EvaluationResultV06,
    EventRecord,
    IntegrityCheck,
    IntegrityReport,
    SCHEMA_VERSION,
    SHA256_PATTERN,
    SessionResult,
)
from op_bench.runtime.evaluation import (
    CompletedEvaluation,
    PrivateEvaluationEvidence,
    validate_evaluation_semantics,
    validate_no_patch_artifact,
    validate_session_evaluation_binding,
)
from op_bench.runtime.events import verify_action_pairing, verify_event_chain
from op_bench.runtime.manifest import RunManifest
from op_bench.runtime.mcp import MCP_PROTOCOL_VERSIONS, McpAdapterTrace
from op_bench.runtime.resume import (
    AttemptLedger,
    AttemptLedgerRecord,
    parse_attempt_ledger,
)
from op_bench.runtime.resources import (
    RuntimeCleanupReport,
    RuntimeResourceHandle,
    RuntimeResourceRecord,
    parse_runtime_lease_store,
    parse_runtime_resource_ledger,
    verify_runtime_cleanup,
    verify_runtime_resource_ownership,
)
from op_bench.runtime.run_artifacts import retry_directory_name
from op_bench.runtime.session import TERMINATION_PRIORITY, termination_attribution
from op_bench.runtime.summary import SelectedAttempt, rebuild_results, rebuild_summary
from op_bench.runtime.task_view import agent_task_view_identity, assert_public_artifact_safe
from op_bench.runtime.validation import ContractError, require_exact_fields, require_str


_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_CHECK_IDS = (
    "manifest_identity",
    "expected_observed_matrix",
    "retry_audit",
    "task_view_identity",
    "event_chain",
    "action_pairing",
    "lifecycle_terminal",
    "runtime_resource_ownership",
    "runtime_cleanup",
    "session_patch_evaluation_identity",
    "public_private_evaluation_identity",
    "evaluation_protocol_scoring_identity",
    "results_rebuild",
    "summary_rebuild",
)
_ATTEMPT_CHECK_IDS = frozenset(
    {
        "retry_audit",
        "task_view_identity",
        "event_chain",
        "action_pairing",
        "lifecycle_terminal",
        "runtime_resource_ownership",
        "runtime_cleanup",
        "session_patch_evaluation_identity",
        "public_private_evaluation_identity",
        "evaluation_protocol_scoring_identity",
    }
)


@dataclass(frozen=True)
class _CheckEvidence:
    message: str
    expected_hash: str | None = None
    actual_hash: str | None = None


class _ReadOnlyRun:
    """Descriptor-bound artifact reader that never creates or rewrites evidence."""

    def __init__(self, run_root: Path) -> None:
        if not isinstance(run_root, Path):
            raise ContractError("run_root: expected Path")
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            self._root_fd = os.open(run_root, flags)
        except OSError as exc:
            raise ContractError("run_root: expected real directory") from exc
        self.run_root = run_root
        self._attempts_fd: int | None = None
        try:
            self._assert_path_binding(run_root, self._root_fd, "run_root")
            self._attempts_fd = os.open("attempts", flags, dir_fd=self._root_fd)
            self._assert_child_binding(
                self._root_fd,
                "attempts",
                self._attempts_fd,
                directory=True,
                label="attempts",
            )
        except Exception:
            if self._attempts_fd is not None:
                os.close(self._attempts_fd)
            os.close(self._root_fd)
            raise

    def close(self) -> None:
        if self._attempts_fd is not None:
            os.close(self._attempts_fd)
            self._attempts_fd = None
        if getattr(self, "_root_fd", -1) >= 0:
            os.close(self._root_fd)
            self._root_fd = -1

    def root_file(self, name: str) -> bytes:
        self._assert_path_binding(self.run_root, self._root_fd, "run_root")
        return self._read_file(self._root_fd, name, name)

    def attempt_ids(self) -> tuple[str, ...]:
        if self._attempts_fd is None:
            raise ContractError("attempts: reader is closed")
        try:
            names = os.listdir(self._attempts_fd)
        except OSError as exc:
            raise ContractError("attempts: cannot list directory") from exc
        return tuple(sorted(names))

    def retry_indices(self, attempt_id: str) -> tuple[int, ...]:
        attempt_fd = self._open_attempt(attempt_id)
        try:
            retries_fd = self._open_child_directory(
                attempt_fd,
                "retries",
                "retries directory",
            )
        finally:
            os.close(attempt_fd)
        try:
            names = os.listdir(retries_fd)
            result: list[int] = []
            for name in names:
                if not name.startswith("retry-"):
                    raise ContractError("retries directory: unexpected entry")
                suffix = name.removeprefix("retry-")
                if not suffix.isdigit():
                    raise ContractError("retries directory: invalid retry entry")
                index = int(suffix)
                if name != retry_directory_name(index):
                    raise ContractError("retries directory: noncanonical retry entry")
                descriptor = self._open_child_directory(
                    retries_fd,
                    name,
                    "retry directory",
                )
                os.close(descriptor)
                result.append(index)
            return tuple(sorted(result))
        finally:
            os.close(retries_fd)

    def retry_file(self, attempt_id: str, retry_index: int, name: str) -> bytes:
        descriptor = self._open_retry(attempt_id, retry_index)
        try:
            value = self._read_file(descriptor, name, name)
            assert value is not None
            return value
        finally:
            os.close(descriptor)

    def retry_file_optional(
        self,
        attempt_id: str,
        retry_index: int,
        name: str,
    ) -> bytes | None:
        descriptor = self._open_retry(attempt_id, retry_index)
        try:
            return self._read_file(descriptor, name, name, optional=True)
        finally:
            os.close(descriptor)

    def _open_attempt(self, attempt_id: str) -> int:
        require_str(attempt_id, "attempt_id")
        if self._attempts_fd is None:
            raise ContractError("attempts: reader is closed")
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(attempt_id, flags, dir_fd=self._attempts_fd)
        except OSError as exc:
            raise ContractError("attempt directory: missing or invalid") from exc
        try:
            self._assert_child_binding(
                self._attempts_fd,
                attempt_id,
                descriptor,
                directory=True,
                label="attempt directory",
            )
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _open_retry(self, attempt_id: str, retry_index: int) -> int:
        attempt_fd = self._open_attempt(attempt_id)
        try:
            retries_fd = self._open_child_directory(
                attempt_fd,
                "retries",
                "retries directory",
            )
        finally:
            os.close(attempt_fd)
        try:
            return self._open_child_directory(
                retries_fd,
                retry_directory_name(retry_index),
                "retry directory",
            )
        finally:
            os.close(retries_fd)

    @staticmethod
    def _open_child_directory(parent_fd: int, name: str, label: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ContractError(f"{label}: missing or invalid") from exc
        try:
            _ReadOnlyRun._assert_child_binding(
                parent_fd,
                name,
                descriptor,
                directory=True,
                label=label,
            )
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    @staticmethod
    def _read_file(
        directory_fd: int,
        name: str,
        label: str,
        *,
        optional: bool = False,
    ) -> bytes | None:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            if optional:
                return None
            raise ContractError(f"{label}: missing artifact") from None
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENXIO, errno.EISDIR}:
                raise ContractError(f"{label}: expected regular file") from exc
            raise ContractError(f"{label}: cannot open artifact") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError(f"{label}: expected regular file")
            if metadata.st_size > _MAX_ARTIFACT_BYTES:
                raise ContractError(f"{label}: artifact exceeds size limit")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > _MAX_ARTIFACT_BYTES:
                    raise ContractError(f"{label}: artifact exceeds size limit")
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_dev != metadata.st_dev
                or current.st_ino != metadata.st_ino
            ):
                raise ContractError(f"{label}: artifact binding changed")
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _assert_path_binding(path: Path, descriptor: int, label: str) -> None:
        try:
            current = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(f"{label}: binding was lost") from exc
        bound = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != bound.st_dev
            or current.st_ino != bound.st_ino
        ):
            raise ContractError(f"{label}: binding changed")

    @staticmethod
    def _assert_child_binding(
        parent_fd: int,
        name: str,
        descriptor: int,
        *,
        directory: bool,
        label: str,
    ) -> None:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(f"{label}: binding was lost") from exc
        bound = os.fstat(descriptor)
        expected_type = stat.S_ISDIR if directory else stat.S_ISREG
        if (
            not expected_type(current.st_mode)
            or current.st_dev != bound.st_dev
            or current.st_ino != bound.st_ino
        ):
            raise ContractError(f"{label}: binding changed")


class _IntegrityState:
    def __init__(self, reader: _ReadOnlyRun, manifest: RunManifest) -> None:
        self.reader = reader
        self.manifest = manifest
        self.expected = {
            item.attempt_id: item for item in manifest.expected_attempts
        }
        self.tasks = {item.task.identifier: item for item in manifest.tasks}

    def ledger_records(self) -> tuple[AttemptLedgerRecord, ...]:
        return parse_attempt_ledger(self.reader.root_file("attempts.jsonl"))

    def selected(self) -> tuple[SelectedAttempt, ...]:
        return _selected_attempts_from_records(self.ledger_records(), self.manifest)

    def task_view(self, attempt_id: str, retry_index: int) -> AgentTaskView:
        payload = _canonical_json_object(
            self.reader.retry_file(attempt_id, retry_index, "agent_task_view.json"),
            "agent_task_view.json",
            public=True,
        )
        return AgentTaskView.from_dict(payload, path="agent_task_view.json")

    def session(self, attempt_id: str, retry_index: int) -> SessionResult:
        payload = _canonical_json_object(
            self.reader.retry_file(attempt_id, retry_index, "session_result.json"),
            "session_result.json",
            public=True,
        )
        return SessionResult.from_dict(payload, path="session_result.json")

    def agent_spec(self, attempt_id: str):
        expected = self.expected[attempt_id]
        return next(
            agent for agent in self.manifest.agents if agent.agent == expected.agent
        )

    def adapter_trace(
        self,
        attempt_id: str,
        retry_index: int,
    ) -> McpAdapterTrace | None:
        raw = self.reader.retry_file_optional(
            attempt_id,
            retry_index,
            "adapter_trace.json",
        )
        if raw is None:
            return None
        payload = _canonical_json_object(
            raw,
            "adapter_trace.json",
            public=True,
        )
        return McpAdapterTrace.from_dict(payload)

    def events(self, attempt_id: str, retry_index: int) -> tuple[EventRecord, ...]:
        raw = self.reader.retry_file(attempt_id, retry_index, "events.jsonl")
        payloads = _canonical_json_lines(raw, "events.jsonl", public=True)
        records = tuple(
            EventRecord.from_dict(payload, path=f"events.jsonl line {index}")
            for index, payload in enumerate(payloads, start=1)
        )
        session_id = self.session(attempt_id, retry_index).session_id
        if any(record.session_id != session_id for record in records):
            raise ContractError("events.jsonl: session_id mismatch")
        return records

    def runtime_profile_hash(self, attempt_id: str) -> str:
        expected = self.expected[attempt_id]
        return self.tasks[expected.task.identifier].runtime.content_hash

    def runtime_resource_evidence(
        self,
        attempt_id: str,
        retry_index: int,
    ) -> tuple[
        tuple[RuntimeResourceRecord, ...],
        tuple[RuntimeResourceHandle, ...],
        RuntimeCleanupReport,
    ]:
        profile_hash = self.runtime_profile_hash(attempt_id)
        records = parse_runtime_resource_ledger(
            self.reader.retry_file(
                attempt_id,
                retry_index,
                "runtime_resources.jsonl",
            ),
            attempt_id=attempt_id,
            retry_index=retry_index,
            runtime_profile_hash=profile_hash,
        )
        handles = parse_runtime_lease_store(
            self.reader.retry_file(
                attempt_id,
                retry_index,
                "private_runtime_resources.json",
            ),
            attempt_id=attempt_id,
            retry_index=retry_index,
            runtime_profile_hash=profile_hash,
        )
        cleanup = RuntimeCleanupReport.from_dict(
            _canonical_json_object(
                self.reader.retry_file(
                    attempt_id,
                    retry_index,
                    "runtime_cleanup.json",
                ),
                "runtime_cleanup.json",
                public=True,
            )
        )
        return records, handles, cleanup

    def evaluations(
        self,
        attempt_id: str,
        retry_index: int,
    ) -> tuple[
        EvaluationResultV06,
        str,
        CompletedEvaluation,
        dict[str, object],
        dict[str, object],
    ]:
        public_payload = _canonical_json_object(
            self.reader.retry_file(
                attempt_id,
                retry_index,
                "public_evaluation.json",
            ),
            "public_evaluation.json",
            public=True,
        )
        private_payload = _canonical_json_object(
            self.reader.retry_file(
                attempt_id,
                retry_index,
                "private_evaluation.json",
            ),
            "private_evaluation.json",
            public=False,
        )
        public_result, public_spec_hash = _parse_public_evaluation(public_payload)
        private_completed = _parse_private_evaluation(private_payload)
        return (
            public_result,
            public_spec_hash,
            private_completed,
            public_payload,
            private_payload,
        )


def selected_attempts_from_ledger(
    ledger: AttemptLedger,
    manifest: RunManifest,
) -> tuple[SelectedAttempt, ...]:
    if not isinstance(ledger, AttemptLedger):
        raise ContractError("ledger: expected AttemptLedger")
    if not isinstance(manifest, RunManifest):
        raise ContractError("manifest: expected RunManifest")
    return _selected_attempts_from_records(ledger.records(), manifest)


def load_run_manifest_artifact(run_root: Path) -> RunManifest:
    """Read one descriptor-bound run Manifest without mutating its run root."""
    reader = _ReadOnlyRun(run_root)
    try:
        payload = _canonical_json_object(
            reader.root_file("run_manifest.json"),
            "run_manifest.json",
            public=False,
        )
        return RunManifest.from_dict(payload, path="run_manifest.json")
    finally:
        reader.close()


def verify_run_artifacts(
    run_root: Path,
    expected_manifest: RunManifest,
) -> IntegrityReport:
    if not isinstance(expected_manifest, RunManifest):
        raise ContractError("expected_manifest: expected RunManifest")
    try:
        reader = _ReadOnlyRun(run_root)
    except Exception:  # noqa: BLE001 - artifact failures become stable checks.
        checks = tuple(_failed_check(check_id) for check_id in _CHECK_IDS)
        return IntegrityReport(
            run_id=expected_manifest.cohort_id,
            status="failed",
            checks=checks,
        )
    try:
        state = _IntegrityState(reader, expected_manifest)
        functions: tuple[tuple[str, Callable[[], _CheckEvidence]], ...] = (
            ("manifest_identity", lambda: _check_manifest(state)),
            ("expected_observed_matrix", lambda: _check_matrix(state)),
            ("retry_audit", lambda: _check_retry_audit(state)),
            ("task_view_identity", lambda: _check_task_views(state)),
            ("event_chain", lambda: _check_event_chains(state)),
            ("action_pairing", lambda: _check_action_pairing(state)),
            ("lifecycle_terminal", lambda: _check_lifecycle(state)),
            (
                "runtime_resource_ownership",
                lambda: _check_runtime_resource_ownership(state),
            ),
            ("runtime_cleanup", lambda: _check_runtime_cleanup(state)),
            (
                "session_patch_evaluation_identity",
                lambda: _check_session_patch_evaluation(state),
            ),
            (
                "public_private_evaluation_identity",
                lambda: _check_public_private_evaluation(state),
            ),
            (
                "evaluation_protocol_scoring_identity",
                lambda: _check_protocol_scoring(state),
            ),
            ("results_rebuild", lambda: _check_results(state)),
            ("summary_rebuild", lambda: _check_summary(state)),
        )
        checks = tuple(_run_check(check_id, function) for check_id, function in functions)
    finally:
        reader.close()
    return IntegrityReport(
        run_id=expected_manifest.cohort_id,
        status="passed" if all(item.status == "passed" for item in checks) else "failed",
        checks=checks,
    )


def persist_integrity_reports(
    run_root: Path,
    manifest: RunManifest,
    report: IntegrityReport,
) -> None:
    from op_bench.runtime.run_artifacts import AttemptArtifactStore

    if not isinstance(manifest, RunManifest):
        raise ContractError("manifest: expected RunManifest")
    if not isinstance(report, IntegrityReport):
        raise ContractError("report: expected IntegrityReport")
    if report.run_id != manifest.cohort_id:
        raise ContractError("report: run_id does not match manifest")
    if report.status != "passed":
        raise ContractError("report: only passed integrity reports may be persisted")
    attempt_checks = tuple(
        check for check in report.checks if check.check_id in _ATTEMPT_CHECK_IDS
    )
    attempt_report = IntegrityReport(
        run_id=manifest.cohort_id,
        status=(
            "passed"
            if all(check.status == "passed" for check in attempt_checks)
            else "failed"
        ),
        checks=attempt_checks,
    )
    store = AttemptArtifactStore(run_root, manifest)
    try:
        for expected in manifest.expected_attempts:
            store.write_integrity(expected.attempt_id, attempt_report)
        store.write_run_integrity(report)
    finally:
        store.close()


def _check_manifest(state: _IntegrityState) -> _CheckEvidence:
    raw = state.reader.root_file("run_manifest.json")
    payload = _canonical_json_object(raw, "run_manifest.json", public=False)
    observed = RunManifest.from_dict(payload, path="run_manifest.json")
    if observed != state.manifest:
        raise ContractError("run_manifest.json: differs from expected manifest")
    return _CheckEvidence(
        "frozen manifest identity matches",
        expected_hash=state.manifest.content_hash,
        actual_hash=observed.content_hash,
    )


def _check_matrix(state: _IntegrityState) -> _CheckEvidence:
    expected = set(state.expected)
    observed = set(state.reader.attempt_ids())
    if observed != expected:
        raise ContractError("attempt directory matrix differs from manifest")
    return _CheckEvidence("expected and observed attempt matrix matches")


def _check_retry_audit(state: _IntegrityState) -> _CheckEvidence:
    records = state.ledger_records()
    expected = set(state.expected)
    observed = {record.attempt_id for record in records}
    if observed != expected:
        raise ContractError("attempt ledger matrix differs from manifest")
    by_attempt: dict[str, list[int]] = {}
    for record in records:
        by_attempt.setdefault(record.attempt_id, []).append(record.retry_index)
    for attempt_id in expected:
        if state.reader.retry_indices(attempt_id) != tuple(by_attempt[attempt_id]):
            raise ContractError("retry artifact matrix differs from ledger")
    _selected_attempts_from_records(records, state.manifest)
    return _CheckEvidence("retry audit is append-only and complete")


def _check_task_views(state: _IntegrityState) -> _CheckEvidence:
    for record in state.ledger_records():
        expected = state.expected[record.attempt_id]
        if (
            agent_task_view_identity(
                state.task_view(record.attempt_id, record.retry_index)
            )
            != expected.task_view
        ):
            raise ContractError("task view identity mismatch")
    return _CheckEvidence("all task view identities match the manifest")


def _check_event_chains(state: _IntegrityState) -> _CheckEvidence:
    for ledger_record in state.ledger_records():
        events = state.events(
            ledger_record.attempt_id,
            ledger_record.retry_index,
        )
        issues = verify_event_chain(events)
        if issues:
            raise ContractError("event chain is invalid")
    return _CheckEvidence("all EventRecord chains are canonical and complete")


def _check_action_pairing(state: _IntegrityState) -> _CheckEvidence:
    for record in state.ledger_records():
        if verify_action_pairing(
            state.events(record.attempt_id, record.retry_index)
        ):
            raise ContractError("action request/observation pairing is invalid")
    return _CheckEvidence("all action requests have one matching observation")


def _check_lifecycle(state: _IntegrityState) -> _CheckEvidence:
    for ledger_record in state.ledger_records():
        attempt_id = ledger_record.attempt_id
        retry_index = ledger_record.retry_index
        records = state.events(attempt_id, retry_index)
        event_types = [event.event_type for event in records]
        required = (
            "session_terminal_emitted",
            "evaluation_started",
            "evaluation_completed",
            "terminal_emitted",
        )
        if any(event_types.count(event_type) != 1 for event_type in required):
            raise ContractError("lifecycle terminal counts are invalid")
        positions = tuple(event_types.index(event_type) for event_type in required)
        expected_positions = tuple(range(len(records) - len(required), len(records)))
        if positions != expected_positions:
            raise ContractError("lifecycle terminal ordering is invalid")
        session = state.session(attempt_id, retry_index)
        _verify_runtime_event_grammar(records, session)
        public_result, spec_hash, _, public_payload, private_payload = state.evaluations(
            attempt_id,
            retry_index,
        )
        _verify_adapter_trace(
            state,
            attempt_id,
            retry_index,
            records,
            session,
            public_result,
        )
        by_type = {record.event_type: record for record in records}
        session_terminal = by_type["session_terminal_emitted"].public_payload
        started = by_type["evaluation_started"].public_payload
        completed = by_type["evaluation_completed"].public_payload
        terminal = by_type["terminal_emitted"].public_payload
        if session_terminal.get("session_result_hash") != session.content_hash:
            raise ContractError("session terminal hash mismatch")
        if (
            started.get("session_result_hash") != session.content_hash
            or started.get("evaluation_spec_hash") != spec_hash
            or started.get("attempt_id") != attempt_id
        ):
            raise ContractError("evaluation start binding mismatch")
        expected_patch = None if session.final_patch is None else session.final_patch.to_dict()
        if started.get("patch") != expected_patch:
            raise ContractError("evaluation start patch mismatch")
        if (
            completed.get("evaluation_result_hash") != public_result.content_hash
            or completed.get("public_evaluation_hash")
            != canonical_sha256(public_payload)
            or completed.get("private_evaluation_hash")
            != canonical_sha256(private_payload)
            or completed.get("attempt_validity") != public_result.attempt_validity
            or completed.get("evaluation_outcome") != public_result.evaluation_outcome
        ):
            raise ContractError("evaluation completion binding mismatch")
        if (
            terminal.get("attempt_id") != attempt_id
            or terminal.get("session_result_hash") != session.content_hash
            or terminal.get("evaluation_result_hash") != public_result.content_hash
            or terminal.get("attempt_validity") != public_result.attempt_validity
            or terminal.get("agent_terminal") != public_result.agent_terminal
            or terminal.get("evaluation_outcome") != public_result.evaluation_outcome
            or ledger_record.evaluation_result != public_result
        ):
            raise ContractError("final terminal binding mismatch")
    return _CheckEvidence("session, evaluation, and final terminals are bound")


def _verify_adapter_trace(
    state: _IntegrityState,
    attempt_id: str,
    retry_index: int,
    records: Sequence[EventRecord],
    session: SessionResult,
    result: EvaluationResultV06,
) -> None:
    agent = state.agent_spec(attempt_id)
    trace = state.adapter_trace(attempt_id, retry_index)
    if agent.adapter.identifier != "codex_mcp_canonical":
        if trace is not None:
            raise ContractError("adapter trace is forbidden for a non-MCP Adapter")
        return
    if trace is None:
        if result.attempt_validity == "valid":
            raise ContractError("MCP adapter trace is missing for a valid Attempt")
        return

    if trace.adapter_id != agent.adapter.identifier:
        raise ContractError("MCP adapter identity mismatch")
    if trace.model_id != agent.model.identifier:
        raise ContractError("MCP model identity mismatch")
    expected_agent_digest = canonical_sha256(
        {
            "adapter_id": trace.adapter_id,
            "model_id": trace.model_id,
            "codex_cli_version": trace.codex_cli_version,
        }
    )
    expected_model_digest = canonical_sha256(
        {
            "adapter_id": trace.adapter_id,
            "model_id": trace.model_id,
        }
    )
    expected_adapter_digest = canonical_sha256(
        {
            "protocol": "action-v1",
            "transport": "mcp-stdio",
            "mcp_protocol_versions": list(MCP_PROTOCOL_VERSIONS),
            "codex_cli_version": trace.codex_cli_version,
        }
    )
    if agent.agent.digest != expected_agent_digest:
        raise ContractError("MCP Agent identity does not bind adapter metadata")
    if agent.model.digest != expected_model_digest:
        raise ContractError("MCP model digest does not bind exact model")
    if agent.adapter.digest != expected_adapter_digest:
        raise ContractError("MCP Adapter digest does not bind exact CLI/protocol")

    action_requests = [
        record for record in records if record.event_type == "action_requested"
    ]
    if trace.tools_call_count != len(action_requests):
        raise ContractError("MCP tool-call count does not match Action trace")
    if result.attempt_validity == "valid" and (
        trace.initialize_count != 1
        or trace.tools_list_count < 1
        or trace.negotiated_protocol_version not in MCP_PROTOCOL_VERSIONS
    ):
        raise ContractError("MCP initialization metadata is incomplete")
    if session.terminal_reason == "agent_finished" and (
        trace.server_terminal_status not in {"completed", "client_closed"}
    ):
        raise ContractError("MCP server terminal does not match Agent finish")
    if session.terminal_reason == "timeout" and (
        trace.server_terminal_status not in {"terminated", "killed"}
    ):
        raise ContractError("MCP server terminal does not match Agent timeout")


def _check_runtime_resource_ownership(
    state: _IntegrityState,
) -> _CheckEvidence:
    for ledger_record in state.ledger_records():
        records, handles, _ = state.runtime_resource_evidence(
            ledger_record.attempt_id,
            ledger_record.retry_index,
        )
        verify_runtime_resource_ownership(records, handles)
    return _CheckEvidence("all runtime resources have exact private ownership evidence")


def _check_runtime_cleanup(state: _IntegrityState) -> _CheckEvidence:
    for ledger_record in state.ledger_records():
        records, _, cleanup = state.runtime_resource_evidence(
            ledger_record.attempt_id,
            ledger_record.retry_index,
        )
        verify_runtime_cleanup(records, cleanup)
    return _CheckEvidence("all runtime resources have terminal cleanup evidence")


def _verify_runtime_event_grammar(
    records: Sequence[EventRecord],
    session: SessionResult,
) -> None:
    event_types = [record.event_type for record in records]
    if not records or event_types[0] != "session_created":
        raise ContractError("lifecycle must begin with session_created")
    if event_types.count("session_created") != 1:
        raise ContractError("lifecycle has duplicate session_created")

    core = ("session_prepared", "session_started", "agent_launched")
    positions: list[int] = []
    prior_present = True
    for event_type in core:
        count = event_types.count(event_type)
        if count > 1:
            raise ContractError("lifecycle core event is duplicated")
        present = count == 1
        if present and not prior_present:
            raise ContractError("lifecycle core event skipped a prior state")
        if present:
            positions.append(event_types.index(event_type))
        prior_present = present
    if positions != sorted(positions):
        raise ContractError("lifecycle core events are out of order")

    requested = {
        record.public_payload.get("action_id"): record
        for record in records
        if record.event_type == "action_requested"
    }
    observed = {
        record.public_payload.get("action_id"): record
        for record in records
        if record.event_type == "action_observed"
    }
    if requested and event_types.count("agent_launched") != 1:
        raise ContractError("action evidence requires agent_launched")
    launch_sequence = (
        None
        if not requested
        else records[event_types.index("agent_launched")].sequence
    )
    budget_events = [
        record for record in records if record.event_type == "budget_updated"
    ]
    for action_id, observation in observed.items():
        matching = [
            record
            for record in budget_events
            if record.public_payload.get("action_id") == action_id
            and record.public_payload.get("observation_hash")
            == observation.public_payload.get("observation_hash")
        ]
        if len(matching) != 1:
            raise ContractError("action observation budget evidence mismatch")
        if matching[0].sequence <= observation.sequence:
            raise ContractError("budget evidence precedes its observation")
        if (
            matching[0].public_payload.get("budget_delta")
            != observation.public_payload.get("budget_delta")
        ):
            raise ContractError("budget evidence payload mismatch")
    if len(budget_events) != len(observed):
        raise ContractError("budget evidence has no matching observation")

    for action_id, request in requested.items():
        observation = observed.get(action_id)
        if observation is None:
            raise ContractError("action request has no observation")
        if launch_sequence is None or request.sequence <= launch_sequence:
            raise ContractError("action request precedes Agent launch")
        budget = next(
            record
            for record in budget_events
            if record.public_payload.get("action_id") == action_id
        )
        if request.public_payload.get("action_name") != "test_run":
            continue
        started = [
            record
            for record in records
            if record.event_type == "test_started"
            and record.public_payload.get("action_id") == action_id
            and record.public_payload.get("request_hash")
            == request.public_payload.get("request_hash")
        ]
        completed = [
            record
            for record in records
            if record.event_type == "test_completed"
            and record.public_payload.get("action_id") == action_id
            and record.public_payload.get("observation_hash")
            == observation.public_payload.get("observation_hash")
        ]
        if len(started) != 1 or len(completed) != 1:
            raise ContractError("test action lifecycle evidence mismatch")
        if (
            completed[0].public_payload.get("ok")
            != observation.public_payload.get("ok")
            or completed[0].public_payload.get("error_code")
            != observation.public_payload.get("error_code")
        ):
            raise ContractError("test completion payload mismatch")
        if not (
            request.sequence
            < started[0].sequence
            < observation.sequence
            < completed[0].sequence
            < budget.sequence
        ):
            raise ContractError("test action lifecycle ordering is invalid")

    _verify_stop_evidence(
        records,
        requested=requested,
        observed=observed,
        terminal_reason=session.terminal_reason,
    )

    if event_types.count("patch_freeze_started") != 1:
        raise ContractError("lifecycle requires one patch_freeze_started")
    freeze_completed = event_types.count("patch_freeze_completed")
    freeze_failed = event_types.count("patch_freeze_failed")
    if freeze_completed + freeze_failed != 1:
        raise ContractError("lifecycle requires one patch freeze outcome")
    freeze_start = event_types.index("patch_freeze_started")
    freeze_outcome_type = (
        "patch_freeze_completed" if freeze_completed else "patch_freeze_failed"
    )
    freeze_outcome = event_types.index(freeze_outcome_type)
    session_terminal = event_types.index("session_terminal_emitted")
    if not (
        freeze_outcome == freeze_start + 1
        and session_terminal == freeze_outcome + 1
    ):
        raise ContractError("patch freeze lifecycle ordering is invalid")
    if session.final_patch is None:
        if not freeze_failed:
            raise ContractError("missing Session patch requires patch_freeze_failed")
    else:
        if not freeze_completed:
            raise ContractError("Session patch requires patch_freeze_completed")
        freeze_payload = records[freeze_outcome].public_payload
        if freeze_payload.get("patch") != session.final_patch.to_dict():
            raise ContractError("patch freeze result identity mismatch")

    terminal = records[session_terminal].public_payload
    attribution = termination_attribution(session.terminal_reason)
    expected_patch = (
        None if session.final_patch is None else session.final_patch.to_dict()
    )
    if (
        terminal.get("attempt_id") != session.attempt_id
        or terminal.get("terminal_reason") != session.terminal_reason
        or terminal.get("session_result_hash") != session.content_hash
        or terminal.get("final_patch") != expected_patch
        or terminal.get("session_validity") != attribution.attempt_validity
    ):
        raise ContractError("Session terminal payload is incomplete or inconsistent")


def _verify_stop_evidence(
    records: Sequence[EventRecord],
    *,
    requested: dict[object, EventRecord],
    observed: dict[object, EventRecord],
    terminal_reason: str,
) -> None:
    candidates: set[str] = set()

    finish_events = [
        record for record in records if record.event_type == "finish_requested"
    ]
    if len(finish_events) > 1:
        raise ContractError("lifecycle has duplicate finish_requested")
    if finish_events:
        finish = finish_events[0]
        if "reason" in finish.public_payload:
            if finish.public_payload.get("reason") != "agent_finished":
                raise ContractError("finish_requested reason is invalid")
            candidates.add("agent_finished")
        else:
            action_id = finish.public_payload.get("action_id")
            request = requested.get(action_id)
            observation = observed.get(action_id)
            if (
                request is None
                or observation is None
                or request.public_payload.get("action_name") != "session_finish"
                or finish.public_payload.get("action_name") != "session_finish"
                or finish.public_payload.get("request_hash")
                != request.public_payload.get("request_hash")
                or not finish.sequence < request.sequence < observation.sequence
            ):
                raise ContractError("finish_requested action binding is invalid")
            if observation.public_payload.get("ok") is True:
                candidates.add("agent_finished")

    exited_events = [
        record for record in records if record.event_type == "agent_exited"
    ]
    if len(exited_events) > 1:
        raise ContractError("lifecycle has duplicate agent_exited")
    if exited_events:
        exit_code = exited_events[0].public_payload.get("exit_code")
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise ContractError("agent_exited exit_code is invalid")
        candidates.add("agent_exited")

    budget_events = [
        record for record in records if record.event_type == "budget_exhausted"
    ]
    if len(budget_events) > 1:
        raise ContractError("lifecycle has duplicate budget_exhausted")
    if budget_events:
        exhausted = budget_events[0]
        if "reason" in exhausted.public_payload:
            if exhausted.public_payload.get("reason") != "budget_exhausted":
                raise ContractError("budget_exhausted reason is invalid")
        else:
            action_id = exhausted.public_payload.get("action_id")
            observation = observed.get(action_id)
            matching_budget_updates = [
                record
                for record in records
                if record.event_type == "budget_updated"
                and record.public_payload.get("action_id") == action_id
                and record.public_payload.get("observation_hash")
                == exhausted.public_payload.get("observation_hash")
            ]
            if (
                observation is None
                or observation.public_payload.get("error_code")
                != "budget_exhausted"
                or exhausted.public_payload.get("observation_hash")
                != observation.public_payload.get("observation_hash")
                or exhausted.sequence <= observation.sequence
                or len(matching_budget_updates) != 1
                or exhausted.sequence <= matching_budget_updates[0].sequence
            ):
                raise ContractError("budget_exhausted action binding is invalid")
        candidates.add("budget_exhausted")

    for event_type, reason in (
        ("timeout_requested", "timeout"),
        ("cancel_requested", "cancelled"),
    ):
        events = [record for record in records if record.event_type == event_type]
        if len(events) > 1:
            raise ContractError(f"lifecycle has duplicate {event_type}")
        if events:
            if events[0].public_payload.get("reason") != reason:
                raise ContractError(f"{event_type} reason is invalid")
            candidates.add(reason)

    observable_infrastructure_errors = {
        "workspace_error",
        "runtime_error",
        "platform_error",
    }
    for observation in observed.values():
        error_code = observation.public_payload.get("error_code")
        if error_code in observable_infrastructure_errors:
            candidates.add(error_code)
    if any(record.event_type == "patch_freeze_failed" for record in records):
        candidates.add("workspace_error")

    evidence_required = {
        "agent_finished",
        "agent_exited",
        "budget_exhausted",
        "timeout",
        "cancelled",
    }
    if terminal_reason in evidence_required and terminal_reason not in candidates:
        raise ContractError("terminal reason has no matching stop evidence")

    priority = {
        reason: index for index, reason in enumerate(TERMINATION_PRIORITY)
    }
    selected_priority = priority[terminal_reason]
    if any(priority[reason] < selected_priority for reason in candidates):
        raise ContractError("stop evidence conflicts with terminal reason priority")


def _check_session_patch_evaluation(state: _IntegrityState) -> _CheckEvidence:
    for record in state.ledger_records():
        attempt_id = record.attempt_id
        retry_index = record.retry_index
        session = state.session(attempt_id, retry_index)
        result, _, _, _, _ = state.evaluations(attempt_id, retry_index)
        if session != record.session_result or result != record.evaluation_result:
            raise ContractError("ledger and attempt artifacts differ")
        validate_session_evaluation_binding(session, result)
        raw_patch = state.reader.retry_file_optional(
            attempt_id,
            retry_index,
            "final.patch",
        )
        if session.final_patch is None:
            if raw_patch is not None:
                raise ContractError("unexpected patch artifact")
        else:
            if raw_patch is None:
                raise ContractError("missing patch artifact")
            actual = "sha256:" + hashlib.sha256(raw_patch).hexdigest()
            if actual != session.final_patch.digest:
                raise ContractError("patch byte identity mismatch")
        validate_no_patch_artifact(result, raw_patch)
    return _CheckEvidence("Session, patch bytes, ledger, and Evaluation match")


def _check_public_private_evaluation(state: _IntegrityState) -> _CheckEvidence:
    tasks = {task.task.identifier: task for task in state.manifest.tasks}
    for record in state.ledger_records():
        attempt_id = record.attempt_id
        retry_index = record.retry_index
        expected = state.expected[attempt_id]
        public, public_spec_hash, private, public_payload, private_payload = (
            state.evaluations(attempt_id, retry_index)
        )
        if public != private.result or public != record.evaluation_result:
            raise ContractError("public/private/ledger Evaluation Result mismatch")
        if (
            public_spec_hash != private.evaluation_spec_hash
            or public_spec_hash != record.evaluation_spec_hash
        ):
            raise ContractError("Evaluation Spec identity mismatch")
        spec = private.evaluation_spec
        task = tasks[expected.task.identifier]
        if (
            spec.session_id != record.session_id
            or spec.attempt_id != attempt_id
            or spec.task != expected.task
            or spec.source != task.source
            or spec.frozen_patch != public.patch
            or spec.hidden_test_asset != task.hidden_test_asset
            or spec.public_tests != task.public_tests
            or spec.fail_to_pass != task.fail_to_pass
            or spec.pass_to_pass != task.pass_to_pass
            or spec.runtime != task.runtime
            or spec.timeout_ms != task.runtime.timeout_ms
            or spec.evaluation != state.manifest.evaluation
            or spec.scoring != state.manifest.scoring
        ):
            raise ContractError("Evaluation Spec graph binding mismatch")
        validate_evaluation_semantics(
            public,
            spec,
            private.private_evidence,
        )
        events = {
            item.event_type: item
            for item in state.events(attempt_id, retry_index)
        }
        completed = events["evaluation_completed"].public_payload
        if (
            completed.get("public_evaluation_hash") != canonical_sha256(public_payload)
            or completed.get("private_evaluation_hash") != canonical_sha256(private_payload)
        ):
            raise ContractError("public/private artifact hash mismatch")
        evidence = private.private_evidence
        if evidence is not None:
            if (
                evidence.source != task.source
                or evidence.patch != public.patch
                or evidence.hidden_test_asset != task.hidden_test_asset
                or not evidence.cleanup_completed
            ):
                raise ContractError("private Evaluation evidence identity mismatch")
    return _CheckEvidence("public and private Evaluation artifacts are bound")


def _check_protocol_scoring(state: _IntegrityState) -> _CheckEvidence:
    for record in state.ledger_records():
        result = record.evaluation_result
        if result.evaluation != state.manifest.evaluation:
            raise ContractError("evaluation protocol identity mismatch")
        if result.scoring != state.manifest.scoring:
            raise ContractError("scoring protocol identity mismatch")
    return _CheckEvidence("evaluation and scoring protocol identities match")


def _check_results(state: _IntegrityState) -> _CheckEvidence:
    expected = rebuild_results(state.manifest, state.selected())
    actual = state.reader.root_file("results.jsonl")
    if actual != expected:
        raise ContractError("results.jsonl differs from deterministic rebuild")
    return _CheckEvidence(
        "results.jsonl exactly matches deterministic rebuild",
        expected_hash=_raw_hash(expected),
        actual_hash=_raw_hash(actual),
    )


def _check_summary(state: _IntegrityState) -> _CheckEvidence:
    expected = (
        canonical_json(rebuild_summary(state.manifest, state.selected())) + "\n"
    ).encode("utf-8")
    actual = state.reader.root_file("summary.json")
    if actual != expected:
        raise ContractError("summary.json differs from deterministic rebuild")
    return _CheckEvidence(
        "summary.json exactly matches deterministic rebuild",
        expected_hash=_raw_hash(expected),
        actual_hash=_raw_hash(actual),
    )


def _selected_attempts_from_records(
    records: Sequence[AttemptLedgerRecord],
    manifest: RunManifest,
) -> tuple[SelectedAttempt, ...]:
    selected_records = _selected_records(records, manifest)
    return tuple(
        SelectedAttempt(
            attempt_id=expected.attempt_id,
            retry_index=selected_records[expected.attempt_id].retry_index,
            evaluation_spec_hash=(
                selected_records[expected.attempt_id].evaluation_spec_hash
            ),
            evaluation_result=(
                selected_records[expected.attempt_id].evaluation_result
            ),
        )
        for expected in manifest.expected_attempts
        if expected.attempt_id in selected_records
    )


def _selected_records(
    records: Sequence[AttemptLedgerRecord],
    manifest: RunManifest,
) -> dict[str, AttemptLedgerRecord]:
    expected = {item.attempt_id for item in manifest.expected_attempts}
    histories: dict[str, list[AttemptLedgerRecord]] = {}
    for record in records:
        if not isinstance(record, AttemptLedgerRecord):
            raise ContractError("ledger records: expected AttemptLedgerRecord")
        if record.attempt_id not in expected:
            raise ContractError("ledger records: unexpected attempt_id")
        histories.setdefault(record.attempt_id, []).append(record)
    selected: dict[str, AttemptLedgerRecord] = {}
    for attempt_id, history in histories.items():
        valid = [item for item in history if item.attempt_validity == "valid"]
        selected[attempt_id] = valid[-1] if valid else history[-1]
    return selected


def _parse_public_evaluation(
    payload: object,
) -> tuple[EvaluationResultV06, str]:
    data = require_exact_fields(
        payload,
        "public_evaluation",
        (
            "record_type",
            "schema_version",
            "evaluation_spec_hash",
            "evaluation_result",
            "evaluation_result_hash",
        ),
    )
    if data["record_type"] != "public_evaluation":
        raise ContractError("record_type: expected 'public_evaluation'")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
    spec_hash = require_str(
        data["evaluation_spec_hash"],
        "evaluation_spec_hash",
        pattern=SHA256_PATTERN,
    )
    result = EvaluationResultV06.from_dict(
        data["evaluation_result"], path="public_evaluation.evaluation_result"
    )
    result_hash = require_str(
        data["evaluation_result_hash"],
        "evaluation_result_hash",
        pattern=SHA256_PATTERN,
    )
    if result_hash != result.content_hash:
        raise ContractError("evaluation_result_hash: does not match result")
    return result, spec_hash


def _parse_private_evaluation(payload: object) -> CompletedEvaluation:
    data = require_exact_fields(
        payload,
        "private_evaluation",
        (
            "record_type",
            "schema_version",
            "evaluation_spec_hash",
            "evaluation_spec",
            "evaluation_result",
            "evaluation_result_hash",
            "private_evidence",
        ),
    )
    if data["record_type"] != "private_evaluation":
        raise ContractError("record_type: expected 'private_evaluation'")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
    result = EvaluationResultV06.from_dict(
        data["evaluation_result"], path="private_evaluation.evaluation_result"
    )
    result_hash = require_str(
        data["evaluation_result_hash"],
        "evaluation_result_hash",
        pattern=SHA256_PATTERN,
    )
    if result_hash != result.content_hash:
        raise ContractError("evaluation_result_hash: does not match result")
    private_value = data["private_evidence"]
    evidence = (
        None
        if private_value is None
        else PrivateEvaluationEvidence.from_dict(private_value)
    )
    spec = EvaluationSpec.from_dict(
        data["evaluation_spec"],
        path="private_evaluation.evaluation_spec",
    )
    spec_hash = require_str(
        data["evaluation_spec_hash"],
        "evaluation_spec_hash",
        pattern=SHA256_PATTERN,
    )
    if spec_hash != spec.content_hash:
        raise ContractError("evaluation_spec_hash: does not match EvaluationSpec")
    return CompletedEvaluation(
        result=result,
        private_evidence=evidence,
        evaluation_spec=spec,
    )


def _canonical_json_object(
    raw: bytes,
    label: str,
    *,
    public: bool,
) -> dict[str, object]:
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ContractError(f"{label}: expected one canonical JSON line")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{label}: expected object")
    if (canonical_json(payload) + "\n").encode("utf-8") != raw:
        raise ContractError(f"{label}: expected canonical JSON")
    if public:
        assert_public_artifact_safe(payload)
    return payload


def _canonical_json_lines(
    raw: bytes,
    label: str,
    *,
    public: bool,
) -> tuple[dict[str, object], ...]:
    if not raw or not raw.endswith(b"\n"):
        raise ContractError(f"{label}: missing final newline")
    payloads: list[dict[str, object]] = []
    for index, encoded in enumerate(raw.splitlines(), start=1):
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{label} line {index}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"{label} line {index}: expected object")
        if canonical_json(payload).encode("utf-8") != encoded:
            raise ContractError(f"{label} line {index}: expected canonical JSON")
        if public:
            assert_public_artifact_safe(payload)
        payloads.append(payload)
    return tuple(payloads)


def _run_check(
    check_id: str,
    function: Callable[[], _CheckEvidence],
) -> IntegrityCheck:
    try:
        evidence = function()
    except Exception:  # noqa: BLE001 - corrupt local evidence is a failed check.
        return _failed_check(check_id)
    return IntegrityCheck(
        check_id=check_id,
        status="passed",
        message=evidence.message,
        expected_hash=evidence.expected_hash,
        actual_hash=evidence.actual_hash,
    )


def _failed_check(check_id: str) -> IntegrityCheck:
    return IntegrityCheck(
        check_id=check_id,
        status="failed",
        message=f"{check_id} verification failed",
        expected_hash=None,
        actual_hash=None,
    )


def _raw_hash(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


__all__ = [
    "load_run_manifest_artifact",
    "persist_integrity_reports",
    "selected_attempts_from_ledger",
    "verify_run_artifacts",
]
