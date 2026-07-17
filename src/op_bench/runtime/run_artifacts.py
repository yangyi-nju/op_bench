from __future__ import annotations

from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import threading

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import (
    AgentTaskView,
    EvaluationSpec,
    EvaluationResultV06,
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
from op_bench.runtime.manifest import ExpectedAttempt, RunManifest
from op_bench.runtime.resources import RuntimeCleanupReport
from op_bench.runtime.task_view import agent_task_view_identity, assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_int,
    require_list,
    require_str,
)
from op_bench.runtime.workspace import FrozenPatch, PatchArtifact


_MAX_ARTIFACT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class EvaluationArtifactHashes:
    public_evaluation_hash: str
    private_evaluation_hash: str

    def __post_init__(self) -> None:
        require_str(
            self.public_evaluation_hash,
            "public_evaluation_hash",
            pattern=SHA256_PATTERN,
        )
        require_str(
            self.private_evaluation_hash,
            "private_evaluation_hash",
            pattern=SHA256_PATTERN,
        )


@dataclass(frozen=True)
class AttemptArtifactIndex:
    attempt_id: str
    session_result_hash: str
    patch_hash: str | None
    evaluation_result_hash: str
    private_evidence_hash: str
    terminal_event_hash: str

    def __post_init__(self) -> None:
        require_str(self.attempt_id, "attempt_id")
        for value, path in (
            (self.session_result_hash, "session_result_hash"),
            (self.evaluation_result_hash, "evaluation_result_hash"),
            (self.private_evidence_hash, "private_evidence_hash"),
            (self.terminal_event_hash, "terminal_event_hash"),
        ):
            require_str(value, path, pattern=SHA256_PATTERN)
        if self.patch_hash is not None:
            require_str(self.patch_hash, "patch_hash", pattern=SHA256_PATTERN)


class AttemptArtifactStore:
    """Descriptor-bound canonical artifact layout for one v0.6 cohort."""

    def __init__(self, run_root: Path, manifest: RunManifest) -> None:
        if not isinstance(run_root, Path):
            raise ContractError("run_root: expected Path")
        if not isinstance(manifest, RunManifest):
            raise ContractError("manifest: expected RunManifest")
        if run_root.is_symlink():
            raise ContractError("run_root: symlink is denied")
        run_root.mkdir(parents=True, exist_ok=True)
        if run_root.is_symlink() or not run_root.is_dir():
            raise ContractError("run_root: expected real directory")
        self.run_root = run_root
        self.manifest = manifest
        self._expected = {
            item.attempt_id: item for item in manifest.expected_attempts
        }
        self._lock = threading.RLock()
        self._closed = False
        self._attempt_fds: dict[str, int] = {}
        self._retry_fds: dict[tuple[str, int], int] = {}
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            self._root_fd = os.open(run_root, directory_flags)
        except OSError as exc:
            raise ContractError("run_root: expected real directory") from exc
        self._attempts_fd: int | None = None
        try:
            try:
                os.mkdir("attempts", 0o700, dir_fd=self._root_fd)
                os.fsync(self._root_fd)
            except FileExistsError:
                pass
            self._attempts_fd = os.open(
                "attempts",
                directory_flags,
                dir_fd=self._root_fd,
            )
            self._assert_root_binding()
            self._assert_child_directory_binding(
                self._root_fd,
                "attempts",
                self._attempts_fd,
                "attempts",
            )
        except Exception:
            if self._attempts_fd is not None:
                os.close(self._attempts_fd)
            os.close(self._root_fd)
            raise

    def events_path(self, attempt_id: str, *, retry_index: int = 1) -> Path:
        with self._lock:
            self._ensure_open()
            self._retry_fd(attempt_id, retry_index)
            return (
                self.run_root
                / "attempts"
                / attempt_id
                / "retries"
                / retry_directory_name(retry_index)
                / "events.jsonl"
            )

    def runtime_resources_path(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> Path:
        return self._retry_path(attempt_id, retry_index, "runtime_resources.jsonl")

    def private_runtime_resources_path(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> Path:
        return self._retry_path(
            attempt_id,
            retry_index,
            "private_runtime_resources.json",
        )

    def write_runtime_cleanup(
        self,
        attempt_id: str,
        report: RuntimeCleanupReport,
        *,
        retry_index: int = 1,
    ) -> None:
        if not isinstance(report, RuntimeCleanupReport):
            raise ContractError("report: expected RuntimeCleanupReport")
        if report.attempt_id != attempt_id or report.retry_index != retry_index:
            raise ContractError("runtime cleanup: attempt/retry identity mismatch")
        if report.runtime_profile_hash != self._runtime_profile_hash(attempt_id):
            raise ContractError("runtime cleanup: Runtime Profile identity mismatch")
        payload = report.to_dict()
        assert_public_artifact_safe(payload)
        self._atomic_write(
            self._retry_fd(attempt_id, retry_index),
            "runtime_cleanup.json",
            _json_bytes(payload),
            label="runtime_cleanup.json",
        )

    def read_runtime_cleanup(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> RuntimeCleanupReport:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "runtime_cleanup.json",
            public=True,
        )
        report = RuntimeCleanupReport.from_dict(payload)
        if report.attempt_id != attempt_id or report.retry_index != retry_index:
            raise ContractError("runtime cleanup: attempt/retry identity mismatch")
        if report.runtime_profile_hash != self._runtime_profile_hash(attempt_id):
            raise ContractError("runtime cleanup: Runtime Profile identity mismatch")
        return report

    def write_runtime_conformance(
        self,
        attempt_id: str,
        payload: object,
        *,
        retry_index: int = 1,
    ) -> None:
        encoded = _runtime_conformance_payload(payload)
        assert_public_artifact_safe(encoded)
        self._atomic_write(
            self._retry_fd(attempt_id, retry_index),
            "runtime_conformance.json",
            _json_bytes(encoded),
            label="runtime_conformance.json",
        )

    def read_runtime_conformance(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> dict[str, object]:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "runtime_conformance.json",
            public=True,
        )
        return _runtime_conformance_payload(payload)

    def write_run_manifest(self) -> None:
        payload = self.manifest.to_dict()
        self._write_root("run_manifest.json", _json_bytes(payload))

    def write_session_inputs(
        self,
        attempt_id: str,
        task_view: AgentTaskView,
        session_result: SessionResult,
        frozen_patch: FrozenPatch | None,
        patch_artifact: PatchArtifact | None,
        *,
        retry_index: int = 1,
    ) -> None:
        expected = self._expected_attempt(attempt_id)
        if not isinstance(task_view, AgentTaskView):
            raise ContractError("task_view: expected AgentTaskView")
        if agent_task_view_identity(task_view) != expected.task_view:
            raise ContractError("task_view: does not match expected attempt")
        if not isinstance(session_result, SessionResult):
            raise ContractError("session_result: expected SessionResult")
        if session_result.attempt_id != attempt_id:
            raise ContractError("session_result: attempt_id mismatch")

        patch_bytes: bytes | None
        if frozen_patch is None and patch_artifact is None:
            if session_result.final_patch is not None:
                raise ContractError("patch handoff: missing frozen patch")
            patch_bytes = None
        elif isinstance(frozen_patch, FrozenPatch) and isinstance(
            patch_artifact, PatchArtifact
        ):
            if session_result.final_patch != frozen_patch.patch:
                raise ContractError("patch identity mismatch across session/frozen")
            if patch_artifact.patch != frozen_patch.patch:
                raise ContractError("patch identity mismatch across frozen/artifact")
            if patch_artifact.workspace != frozen_patch.workspace:
                raise ContractError("workspace identity mismatch for patch artifact")
            if patch_artifact.size_bytes != len(frozen_patch.patch_bytes):
                raise ContractError("patch artifact size mismatch")
            if patch_artifact.changed_paths != frozen_patch.changed_paths:
                raise ContractError("patch artifact changed_paths mismatch")
            if patch_artifact.empty != frozen_patch.empty:
                raise ContractError("patch artifact empty state mismatch")
            patch_bytes = frozen_patch.patch_bytes
        else:
            raise ContractError("patch handoff: partial patch bundle")

        task_view_payload = task_view.to_dict()
        result_payload = session_result.to_dict()
        assert_public_artifact_safe(task_view_payload)
        assert_public_artifact_safe(result_payload)
        if patch_bytes is not None:
            try:
                patch_text = patch_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ContractError("public patch: expected UTF-8") from exc
            assert_public_artifact_safe(patch_text)

        attempt_fd = self._retry_fd(attempt_id, retry_index)
        self._atomic_write(
            attempt_fd,
            "agent_task_view.json",
            _json_bytes(task_view_payload),
            label="agent_task_view.json",
        )
        self._atomic_write(
            attempt_fd,
            "session_result.json",
            _json_bytes(result_payload),
            label="session_result.json",
        )
        if patch_bytes is not None:
            self._atomic_write(
                attempt_fd,
                "final.patch",
                patch_bytes,
                label="final.patch",
            )

    def write_evaluation(
        self,
        attempt_id: str,
        completed: CompletedEvaluation,
        *,
        retry_index: int = 1,
    ) -> EvaluationArtifactHashes:
        expected = self._expected_attempt(attempt_id)
        if not isinstance(completed, CompletedEvaluation):
            raise ContractError("completed: expected CompletedEvaluation")
        result = completed.result
        if result.attempt_id != attempt_id:
            raise ContractError("evaluation_result: attempt_id mismatch")
        spec = completed.evaluation_spec
        if spec.attempt_id != attempt_id or spec.session_id != result.session_id:
            raise ContractError("evaluation_spec: session/attempt mismatch")
        if spec.task != expected.task:
            raise ContractError("evaluation_spec: task identity mismatch")
        task = next(
            item for item in self.manifest.tasks if item.task == expected.task
        )
        if (
            spec.source != task.source
            or spec.hidden_test_asset != task.hidden_test_asset
            or spec.public_tests != task.public_tests
            or spec.fail_to_pass != task.fail_to_pass
            or spec.pass_to_pass != task.pass_to_pass
            or spec.runtime != task.runtime
            or spec.timeout_ms != task.runtime.timeout_ms
        ):
            raise ContractError("evaluation_spec: task evidence mismatch")
        session = self.read_session_result(
            attempt_id,
            retry_index=retry_index,
        )
        validate_session_evaluation_binding(session, result)
        validate_evaluation_semantics(
            result,
            spec,
            completed.private_evidence,
        )
        patch_bytes = (
            None
            if session.final_patch is None
            else self.read_patch(attempt_id, retry_index=retry_index)
        )
        validate_no_patch_artifact(result, patch_bytes)
        if result.scoring != self.manifest.scoring:
            raise ContractError("evaluation_result: scoring identity mismatch")
        if result.evaluation != self.manifest.evaluation:
            raise ContractError("evaluation_result: evaluation identity mismatch")
        if completed.private_evidence is not None:
            evidence = completed.private_evidence
            if evidence.patch != result.patch:
                raise ContractError("private evaluation: patch identity mismatch")

        public_payload = {
            "record_type": "public_evaluation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_spec_hash": completed.evaluation_spec_hash,
            "evaluation_result": result.to_dict(),
            "evaluation_result_hash": result.content_hash,
        }
        private_payload = {
            "record_type": "private_evaluation",
            "schema_version": SCHEMA_VERSION,
            "evaluation_spec_hash": completed.evaluation_spec_hash,
            "evaluation_spec": spec.to_dict(),
            "evaluation_result": result.to_dict(),
            "evaluation_result_hash": result.content_hash,
            "private_evidence": (
                None
                if completed.private_evidence is None
                else completed.private_evidence.to_dict()
            ),
        }
        assert_public_artifact_safe(public_payload)
        public_bytes = _json_bytes(public_payload)
        private_bytes = _json_bytes(private_payload)
        attempt_fd = self._retry_fd(attempt_id, retry_index)
        self._atomic_write(
            attempt_fd,
            "public_evaluation.json",
            public_bytes,
            label="public_evaluation.json",
        )
        self._atomic_write(
            attempt_fd,
            "private_evaluation.json",
            private_bytes,
            label="private_evaluation.json",
        )
        return EvaluationArtifactHashes(
            public_evaluation_hash=canonical_sha256(public_payload),
            private_evaluation_hash=canonical_sha256(private_payload),
        )

    def read_public_evaluation(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> EvaluationResultV06:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "public_evaluation.json",
            public=True,
        )
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
        require_str(
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
        return result

    def read_private_evaluation(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> CompletedEvaluation:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "private_evaluation.json",
            public=False,
        )
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
        private = (
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
            private_evidence=private,
            evaluation_spec=spec,
        )

    def read_session_result(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> SessionResult:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "session_result.json",
            public=True,
        )
        return SessionResult.from_dict(payload, path="session_result.json")

    def read_agent_task_view(
        self,
        attempt_id: str,
        *,
        retry_index: int = 1,
    ) -> AgentTaskView:
        payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "agent_task_view.json",
            public=True,
        )
        return AgentTaskView.from_dict(payload, path="agent_task_view.json")

    def read_patch(self, attempt_id: str, *, retry_index: int = 1) -> bytes:
        attempt_fd = self._retry_fd(attempt_id, retry_index)
        raw = self._read_file(attempt_fd, "final.patch", "final.patch")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError("public patch: expected UTF-8") from exc
        assert_public_artifact_safe(text)
        return raw

    def build_index(
        self,
        attempt_id: str,
        terminal_event_hash: str,
        *,
        retry_index: int = 1,
    ) -> AttemptArtifactIndex:
        require_str(
            terminal_event_hash,
            "terminal_event_hash",
            pattern=SHA256_PATTERN,
        )
        session = self.read_session_result(attempt_id, retry_index=retry_index)
        private_payload = self._read_attempt_json(
            attempt_id,
            retry_index,
            "private_evaluation.json",
            public=False,
        )
        result = self.read_public_evaluation(attempt_id, retry_index=retry_index)
        return AttemptArtifactIndex(
            attempt_id=attempt_id,
            session_result_hash=session.content_hash,
            patch_hash=None if session.final_patch is None else session.final_patch.digest,
            evaluation_result_hash=result.content_hash,
            private_evidence_hash=canonical_sha256(private_payload),
            terminal_event_hash=terminal_event_hash,
        )

    def write_integrity(
        self,
        attempt_id: str,
        report: IntegrityReport,
    ) -> None:
        self._expected_attempt(attempt_id)
        if not isinstance(report, IntegrityReport):
            raise ContractError("report: expected IntegrityReport")
        payload = report.to_dict()
        assert_public_artifact_safe(payload)
        self._atomic_write(
            self._attempt_fd(attempt_id),
            "integrity.json",
            _json_bytes(payload),
            label="integrity.json",
            replace_existing=True,
        )

    def write_run_integrity(self, report: IntegrityReport) -> None:
        if not isinstance(report, IntegrityReport):
            raise ContractError("report: expected IntegrityReport")
        payload = report.to_dict()
        assert_public_artifact_safe(payload)
        self._write_root(
            "integrity.json",
            _json_bytes(payload),
            replace_existing=True,
        )

    def write_results_bytes(self, raw: bytes) -> None:
        if not isinstance(raw, bytes):
            raise ContractError("results: expected bytes")
        if raw and not raw.endswith(b"\n"):
            raise ContractError("results.jsonl: missing final newline")
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"results.jsonl line {line_number}: invalid JSON"
                ) from exc
            if canonical_json(payload).encode("utf-8") != line:
                raise ContractError(
                    f"results.jsonl line {line_number}: expected canonical JSON"
                )
            assert_public_artifact_safe(payload)
        self._write_root("results.jsonl", raw, replace_existing=True)

    def write_summary_bytes(self, raw: bytes) -> None:
        if not isinstance(raw, bytes):
            raise ContractError("summary: expected bytes")
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise ContractError("summary.json: expected one canonical JSON line")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("summary.json: invalid JSON") from exc
        if _json_bytes(payload) != raw:
            raise ContractError("summary.json: expected canonical JSON")
        assert_public_artifact_safe(payload)
        self._write_root("summary.json", raw, replace_existing=True)

    def read_results_bytes(self) -> bytes:
        self._ensure_open()
        self._assert_root_binding()
        raw = self._read_file(self._root_fd, "results.jsonl", "results.jsonl")
        if raw and not raw.endswith(b"\n"):
            raise ContractError("results.jsonl: missing final newline")
        for line_number, line in enumerate(raw.splitlines(), start=1):
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError(
                    f"results.jsonl line {line_number}: invalid JSON"
                ) from exc
            if canonical_json(payload).encode("utf-8") != line:
                raise ContractError(
                    f"results.jsonl line {line_number}: expected canonical JSON"
                )
            assert_public_artifact_safe(payload)
        return raw

    def read_summary_bytes(self) -> bytes:
        self._ensure_open()
        self._assert_root_binding()
        raw = self._read_file(self._root_fd, "summary.json", "summary.json")
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise ContractError("summary.json: expected one canonical JSON line")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("summary.json: invalid JSON") from exc
        if _json_bytes(payload) != raw:
            raise ContractError("summary.json: expected canonical JSON")
        assert_public_artifact_safe(payload)
        return raw

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for descriptor in self._retry_fds.values():
                os.close(descriptor)
            self._retry_fds.clear()
            for descriptor in self._attempt_fds.values():
                os.close(descriptor)
            self._attempt_fds.clear()
            if self._attempts_fd is not None:
                os.close(self._attempts_fd)
                self._attempts_fd = None
            os.close(self._root_fd)
            self._closed = True

    def _retry_path(
        self,
        attempt_id: str,
        retry_index: int,
        name: str,
    ) -> Path:
        with self._lock:
            self._ensure_open()
            self._retry_fd(attempt_id, retry_index)
            return (
                self.run_root
                / "attempts"
                / attempt_id
                / "retries"
                / retry_directory_name(retry_index)
                / name
            )

    def _runtime_profile_hash(self, attempt_id: str) -> str:
        expected = self._expected_attempt(attempt_id)
        for task in self.manifest.tasks:
            if task.task == expected.task:
                return task.runtime.content_hash
        raise ContractError("attempt_id: Task Runtime Profile is missing")

    def _expected_attempt(self, attempt_id: str) -> ExpectedAttempt:
        require_str(attempt_id, "attempt_id")
        try:
            return self._expected[attempt_id]
        except KeyError as exc:
            raise ContractError("attempt_id: not present in frozen manifest") from exc

    def _attempt_fd(self, attempt_id: str) -> int:
        self._expected_attempt(attempt_id)
        self._ensure_open()
        if self._attempts_fd is None:
            raise ContractError("artifact store is closed")
        cached = self._attempt_fds.get(attempt_id)
        if cached is not None:
            self._assert_child_directory_binding(
                self._attempts_fd,
                attempt_id,
                cached,
                "attempt directory",
            )
            return cached
        try:
            os.mkdir(attempt_id, 0o700, dir_fd=self._attempts_fd)
            os.fsync(self._attempts_fd)
        except FileExistsError:
            pass
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(
                attempt_id,
                flags,
                dir_fd=self._attempts_fd,
            )
        except OSError as exc:
            raise ContractError("attempt directory: symlink or invalid directory") from exc
        self._assert_child_directory_binding(
            self._attempts_fd,
            attempt_id,
            descriptor,
            "attempt directory",
        )
        self._attempt_fds[attempt_id] = descriptor
        return descriptor

    def _retry_fd(self, attempt_id: str, retry_index: int) -> int:
        require_int(retry_index, "retry_index", minimum=1)
        key = (attempt_id, retry_index)
        cached = self._retry_fds.get(key)
        if cached is not None:
            attempt_fd = self._attempt_fd(attempt_id)
            retries_fd = self._open_child_directory(
                attempt_fd,
                "retries",
                "retries directory",
            )
            try:
                self._assert_child_directory_binding(
                    retries_fd,
                    retry_directory_name(retry_index),
                    cached,
                    "retry directory",
                )
            finally:
                os.close(retries_fd)
            return cached

        attempt_fd = self._attempt_fd(attempt_id)
        try:
            os.mkdir("retries", 0o700, dir_fd=attempt_fd)
            os.fsync(attempt_fd)
        except FileExistsError:
            pass
        retries_fd = self._open_child_directory(
            attempt_fd,
            "retries",
            "retries directory",
        )
        name = retry_directory_name(retry_index)
        try:
            try:
                os.mkdir(name, 0o700, dir_fd=retries_fd)
                os.fsync(retries_fd)
            except FileExistsError:
                pass
            descriptor = self._open_child_directory(
                retries_fd,
                name,
                "retry directory",
            )
        finally:
            os.close(retries_fd)
        self._retry_fds[key] = descriptor
        return descriptor

    @staticmethod
    def _open_child_directory(parent_fd: int, name: str, label: str) -> int:
        flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ContractError(f"{label}: symlink or invalid directory") from exc
        try:
            AttemptArtifactStore._assert_child_directory_binding(
                parent_fd,
                name,
                descriptor,
                label,
            )
        except Exception:
            os.close(descriptor)
            raise
        return descriptor

    def _write_root(
        self,
        name: str,
        raw: bytes,
        *,
        replace_existing: bool = False,
    ) -> None:
        self._ensure_open()
        self._assert_root_binding()
        self._atomic_write(
            self._root_fd,
            name,
            raw,
            label=name,
            replace_existing=replace_existing,
        )

    def _read_attempt_json(
        self,
        attempt_id: str,
        retry_index: int,
        name: str,
        *,
        public: bool,
    ) -> dict[str, object]:
        raw = self._read_file(self._retry_fd(attempt_id, retry_index), name, name)
        if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
            raise ContractError(f"{name}: expected one canonical JSON line")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{name}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ContractError(f"{name}: expected object")
        if _json_bytes(payload) != raw:
            raise ContractError(f"{name}: expected canonical JSON")
        if public:
            assert_public_artifact_safe(payload)
        return payload

    def _atomic_write(
        self,
        directory_fd: int,
        name: str,
        raw: bytes,
        *,
        label: str,
        replace_existing: bool = False,
    ) -> None:
        with self._lock:
            self._ensure_open()
            fcntl.flock(directory_fd, fcntl.LOCK_EX)
            temporary: str | None = None
            try:
                existing = self._read_optional_file(directory_fd, name, label)
                if existing is not None:
                    if existing == raw:
                        return
                    if not replace_existing:
                        raise ContractError(
                            f"{label}: conflicting artifact already exists"
                        )
                temporary = f".{name}.{secrets.token_hex(12)}.tmp"
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                if hasattr(os, "O_NOFOLLOW"):
                    flags |= os.O_NOFOLLOW
                descriptor = os.open(
                    temporary,
                    flags,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    _write_all(descriptor, raw)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                os.replace(
                    temporary,
                    name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                )
                temporary = None
                os.fsync(directory_fd)
            finally:
                if temporary is not None:
                    try:
                        os.unlink(temporary, dir_fd=directory_fd)
                    except FileNotFoundError:
                        pass
                fcntl.flock(directory_fd, fcntl.LOCK_UN)

    def _read_file(self, directory_fd: int, name: str, label: str) -> bytes:
        with self._lock:
            self._ensure_open()
            fcntl.flock(directory_fd, fcntl.LOCK_SH)
            try:
                value = self._read_optional_file(directory_fd, name, label)
                if value is None:
                    raise ContractError(f"{label}: missing artifact")
                return value
            finally:
                fcntl.flock(directory_fd, fcntl.LOCK_UN)

    @staticmethod
    def _read_optional_file(
        directory_fd: int,
        name: str,
        label: str,
    ) -> bytes | None:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        try:
            descriptor = os.open(name, flags, dir_fd=directory_fd)
        except FileNotFoundError:
            return None
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
            raw = b""
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                raw += chunk
                if len(raw) > _MAX_ARTIFACT_BYTES:
                    raise ContractError(f"{label}: artifact exceeds size limit")
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(current.st_mode)
                or current.st_dev != metadata.st_dev
                or current.st_ino != metadata.st_ino
            ):
                raise ContractError(f"{label}: artifact binding changed")
            return raw
        finally:
            os.close(descriptor)

    def _assert_root_binding(self) -> None:
        try:
            current = os.stat(self.run_root, follow_symlinks=False)
        except OSError as exc:
            raise ContractError("run_root: binding was lost or replaced") from exc
        bound = os.fstat(self._root_fd)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != bound.st_dev
            or current.st_ino != bound.st_ino
        ):
            raise ContractError("run_root: binding was lost or replaced")

    @staticmethod
    def _assert_child_directory_binding(
        parent_fd: int,
        name: str,
        descriptor: int,
        label: str,
    ) -> None:
        try:
            current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise ContractError(f"{label}: binding was lost or replaced") from exc
        bound = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(current.st_mode)
            or current.st_dev != bound.st_dev
            or current.st_ino != bound.st_ino
        ):
            raise ContractError(f"{label}: binding was lost or replaced")

    def _ensure_open(self) -> None:
        if self._closed:
            raise ContractError("artifact store is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _runtime_conformance_payload(value: object) -> dict[str, object]:
    data = require_exact_fields(
        value,
        "runtime_conformance",
        ("report_type", "schema_version", "status", "entries"),
    )
    if require_str(data["report_type"], "report_type") != "runtime_conformance":
        raise ContractError("report_type: expected 'runtime_conformance'")
    if require_str(data["schema_version"], "schema_version") != SCHEMA_VERSION:
        raise ContractError(f"schema_version: expected {SCHEMA_VERSION!r}")
    status = require_enum(
        data["status"],
        "status",
        ("not_applicable", "passed", "failed", "blocked"),
    )
    entries = require_list(data["entries"], "entries")
    canonical_json(entries)
    return {
        "report_type": "runtime_conformance",
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "entries": entries,
    }


def retry_directory_name(retry_index: int) -> str:
    require_int(retry_index, "retry_index", minimum=1)
    return f"retry-{retry_index:04d}"


def _write_all(descriptor: int, raw: bytes) -> None:
    offset = 0
    while offset < len(raw):
        written = os.write(descriptor, raw[offset:])
        if written <= 0:
            raise OSError("artifact write made no progress")
        offset += written


__all__ = [
    "AttemptArtifactIndex",
    "AttemptArtifactStore",
    "EvaluationArtifactHashes",
    "retry_directory_name",
]
