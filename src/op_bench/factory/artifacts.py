from __future__ import annotations

from collections.abc import Mapping
import errno
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
from typing import Union
import uuid

from op_bench.factory.contracts import (
    CandidateRecord,
    DatasetFreezeManifest,
    DecisionRecord,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
)
from op_bench.factory.prompt_quality import PromptQualityEvidence
from op_bench.factory.complexity import ComplexityEvidence
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError


FactoryContract = Union[
    CandidateRecord,
    DecisionRecord,
    FactoryAdmissionRecord,
    DatasetFreezeManifest,
    PromptQualityEvidence,
    ComplexityEvidence,
]

_CONTRACT_TYPES = {
    CandidateRecord.contract_type: CandidateRecord,
    DecisionRecord.contract_type: DecisionRecord,
    FactoryAdmissionRecord.contract_type: FactoryAdmissionRecord,
    DatasetFreezeManifest.contract_type: DatasetFreezeManifest,
    PromptQualityEvidence.contract_type: PromptQualityEvidence,
    ComplexityEvidence.contract_type: ComplexityEvidence,
}
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_SENSITIVE_FACTORY_KEYS = {
    "api_key",
    "authorization",
    "credential",
    "credentials",
    "gold_patch_contents",
    "hidden_test_contents",
    "password",
    "secret",
    "target_handle",
    "token",
}
_SENSITIVE_FACTORY_TEXT = (
    re.compile(r"(?:^|[\s\"'])/(?:Users|home|private|tmp)/"),
    re.compile(r"(?:^|[\s\"'])[A-Za-z]:\\"),
    re.compile(r"\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
)


def _validate_relative_json_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ContractError("relative_path: expected non-empty string")
    if "\\" in value:
        raise ContractError("relative_path: expected normalized POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or path.as_posix() != value
        or path.suffix != ".json"
    ):
        raise ContractError(
            "relative_path: expected normalized relative .json path"
        )
    return path


def _contract_identity(contract: FactoryContract) -> str:
    if isinstance(contract, CandidateRecord):
        return contract.candidate_id
    if isinstance(contract, DecisionRecord):
        return contract.decision_id
    if isinstance(contract, FactoryAdmissionRecord):
        return contract.admission_id
    if isinstance(contract, DatasetFreezeManifest):
        return contract.freeze_id
    if isinstance(contract, PromptQualityEvidence):
        return contract.task_id
    if isinstance(contract, ComplexityEvidence):
        return contract.task_id
    from op_bench.factory.quality_release import (
        QualityCandidateDecision,
        QualityCandidateRecord,
    )

    if isinstance(contract, QualityCandidateRecord):
        return contract.candidate_id
    if isinstance(contract, QualityCandidateDecision):
        return contract.decision_id
    raise ContractError("contract: unsupported Factory contract")


def _assert_factory_public_safe(
    value: object,
    *,
    path: str = "$",
    allow_private_provenance_text: bool = False,
) -> None:
    if isinstance(value, str):
        patterns = (
            _SENSITIVE_FACTORY_TEXT[2:]
            if allow_private_provenance_text
            else _SENSITIVE_FACTORY_TEXT
        )
        for pattern in patterns:
            if pattern.search(value):
                raise ContractError(
                    f"factory artifact {path}: sensitive text is denied"
                )
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _SENSITIVE_FACTORY_KEYS:
                raise ContractError(
                    f"factory artifact {path}.{key}: sensitive field is denied"
                )
            _assert_factory_public_safe(
                item,
                path=f"{path}.{key}",
                allow_private_provenance_text=allow_private_provenance_text,
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_factory_public_safe(
                item,
                path=f"{path}[{index}]",
                allow_private_provenance_text=allow_private_provenance_text,
            )
        return
    raise ContractError(
        f"factory artifact {path}: unsupported value type {type(value).__name__}"
    )


def _parse_contract_bytes(content: bytes) -> FactoryContract:
    try:
        decoded = content.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("artifact: invalid JSON") from exc
    if canonical_json(value).encode("utf-8") != content:
        raise ContractError("artifact: expected canonical JSON")
    if not isinstance(value, Mapping):
        raise ContractError("artifact: expected JSON object")
    contract_type = value.get("contract_type")
    contract_class = _CONTRACT_TYPES.get(contract_type)
    if contract_class is None and contract_type in (
        "quality_candidate",
        "quality_candidate_decision",
    ):
        from op_bench.factory.quality_release import (
            QualityCandidateDecision,
            QualityCandidateRecord,
        )

        contract_class = {
            QualityCandidateRecord.contract_type: QualityCandidateRecord,
            QualityCandidateDecision.contract_type: QualityCandidateDecision,
        }[contract_type]
    if contract_class is None:
        raise ContractError(
            f"artifact: unsupported contract_type {contract_type!r}"
        )
    contract = contract_class.from_dict(value)
    _assert_factory_public_safe(
        contract.to_dict(),
        allow_private_provenance_text=contract_type == "quality_candidate",
    )
    return contract


def load_factory_contract(path: Path) -> FactoryContract:
    return _parse_contract_bytes(load_regular_file_bytes(path))


def load_regular_file_bytes(path: Path) -> bytes:
    """Read a real regular file without following a final symlink."""

    if not isinstance(path, Path):
        raise ContractError("artifact path: expected Path")
    if path.is_symlink():
        raise ContractError("artifact path: symlink is denied")
    flags = os.O_RDONLY | _NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ContractError("artifact path: symlink is denied") from exc
        raise ContractError("artifact: cannot be opened") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("artifact: expected regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def load_canonical_json_artifact(path: Path) -> Mapping[str, object]:
    """Load an exact canonical JSON object from a no-symlink regular file."""

    content = load_regular_file_bytes(path)
    try:
        decoded = content.decode("utf-8")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError("artifact: invalid JSON") from exc
    if canonical_json(value).encode("utf-8") != content:
        raise ContractError("artifact: expected canonical JSON")
    if not isinstance(value, Mapping):
        raise ContractError("artifact: expected JSON object")
    return value


class FactoryArtifactStore:
    """Descriptor-relative immutable storage for canonical Factory contracts."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ContractError("artifact root: expected Path")
        if root.is_symlink():
            raise ContractError("artifact root: symlink is denied")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            descriptor = os.open(root, _DIRECTORY_FLAGS)
        except OSError as exc:
            raise ContractError("artifact root: expected real directory") from exc
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            os.close(descriptor)
            raise ContractError("artifact root: expected directory")
        self.root = root
        self._root_fd = descriptor
        self._lock = threading.RLock()
        self._closed = False

    def write_contract(
        self,
        relative_path: str,
        contract: FactoryContract,
    ) -> FactoryArtifactReference:
        self._ensure_open()
        path = _validate_relative_json_path(relative_path)
        from op_bench.factory.quality_release import (
            QualityCandidateDecision,
            QualityCandidateRecord,
        )

        if not isinstance(
            contract,
            (
                CandidateRecord,
                DecisionRecord,
                FactoryAdmissionRecord,
                DatasetFreezeManifest,
                PromptQualityEvidence,
                ComplexityEvidence,
                QualityCandidateRecord,
                QualityCandidateDecision,
            ),
        ):
            raise ContractError("contract: unsupported Factory contract")
        payload = contract.to_dict()
        _assert_factory_public_safe(
            payload,
            allow_private_provenance_text=isinstance(
                contract, QualityCandidateRecord
            ),
        )
        encoded = canonical_json(payload).encode("utf-8")
        reference = FactoryArtifactReference(
            artifact_type=contract.contract_type,
            artifact_id=_contract_identity(contract),
            content_hash=contract.content_hash,
            relative_path=relative_path,
        )
        installed = self._write_encoded(path, encoded)
        self._verify_content(reference, installed)
        return reference

    def write_json(
        self,
        relative_path: str,
        value: object,
        *,
        artifact_type: str,
        artifact_id: str,
    ) -> FactoryArtifactReference:
        self._ensure_open()
        path = _validate_relative_json_path(relative_path)
        _assert_factory_public_safe(value)
        encoded = canonical_json(value).encode("utf-8")
        reference = FactoryArtifactReference(
            artifact_type=artifact_type,
            artifact_id=artifact_id,
            content_hash=canonical_sha256(value),
            relative_path=relative_path,
        )
        installed = self._write_encoded(path, encoded)
        try:
            decoded = json.loads(installed.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("artifact: invalid JSON") from exc
        if canonical_json(decoded).encode("utf-8") != installed:
            raise ContractError("artifact: expected canonical JSON")
        if canonical_sha256(decoded) != reference.content_hash:
            raise ContractError("artifact content hash mismatch")
        return reference

    def _write_encoded(
        self,
        path: PurePosixPath,
        encoded: bytes,
    ) -> bytes:
        with self._lock:
            parent_fd, filename = self._open_parent(path, create=True)
            try:
                existing = self._read_file(parent_fd, filename, missing_ok=True)
                if existing is not None:
                    if existing != encoded:
                        raise ContractError(
                            "artifact destination is immutable and contains different bytes"
                        )
                    return existing

                temporary = f".factory-{uuid.uuid4().hex}.tmp"
                descriptor: int | None = None
                try:
                    descriptor = os.open(
                        temporary,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW,
                        0o600,
                        dir_fd=parent_fd,
                    )
                    self._write_all(descriptor, encoded)
                    os.fsync(descriptor)
                    os.close(descriptor)
                    descriptor = None
                    try:
                        os.link(
                            temporary,
                            filename,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        raced = self._read_file(
                            parent_fd,
                            filename,
                            missing_ok=False,
                        )
                        if raced != encoded:
                            raise ContractError(
                                "artifact destination is immutable and contains different bytes"
                            )
                    except OSError as exc:
                        raise ContractError(
                            "artifact atomic install failed"
                        ) from exc
                    os.fsync(parent_fd)
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
                    try:
                        os.unlink(temporary, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                    except FileNotFoundError:
                        pass
                installed = self._read_file(
                    parent_fd,
                    filename,
                    missing_ok=False,
                )
                return installed
            finally:
                os.close(parent_fd)

    def read_contract(
        self,
        reference: FactoryArtifactReference,
    ) -> FactoryContract:
        self._ensure_open()
        if not isinstance(reference, FactoryArtifactReference):
            raise ContractError(
                "reference: expected FactoryArtifactReference"
            )
        path = _validate_relative_json_path(reference.relative_path)
        with self._lock:
            parent_fd, filename = self._open_parent(path, create=False)
            try:
                content = self._read_file(
                    parent_fd,
                    filename,
                    missing_ok=False,
                )
            finally:
                os.close(parent_fd)
        return self._verify_content(reference, content)

    def verify_reference(
        self,
        reference: FactoryArtifactReference,
    ) -> None:
        self.read_contract(reference)

    def close(self) -> None:
        with getattr(self, "_lock", threading.RLock()):
            if getattr(self, "_closed", True):
                return
            self._closed = True
            os.close(self._root_fd)

    def __enter__(self) -> "FactoryArtifactStore":
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise ContractError("artifact store is closed")

    def _open_parent(
        self,
        path: PurePosixPath,
        *,
        create: bool,
    ) -> tuple[int, str]:
        current = os.dup(self._root_fd)
        try:
            for component in path.parts[:-1]:
                try:
                    selected = os.open(
                        component,
                        _DIRECTORY_FLAGS,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    if not create:
                        raise ContractError("artifact is missing")
                    os.mkdir(component, 0o700, dir_fd=current)
                    os.fsync(current)
                    selected = os.open(
                        component,
                        _DIRECTORY_FLAGS,
                        dir_fd=current,
                    )
                except OSError as exc:
                    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
                        raise ContractError(
                            "artifact ancestor symlink is denied"
                        ) from exc
                    raise ContractError(
                        "artifact parent directory cannot be opened"
                    ) from exc
                metadata = os.fstat(selected)
                if not stat.S_ISDIR(metadata.st_mode):
                    os.close(selected)
                    raise ContractError(
                        "artifact ancestor: expected directory"
                    )
                if stat.S_IMODE(metadata.st_mode) != 0o700:
                    os.close(selected)
                    raise ContractError("artifact directory mode drift")
                os.close(current)
                current = selected
            return current, path.name
        except Exception:
            os.close(current)
            raise

    @staticmethod
    def _read_file(
        parent_fd: int,
        filename: str,
        *,
        missing_ok: bool,
    ) -> bytes | None:
        try:
            descriptor = os.open(
                filename,
                os.O_RDONLY | _NOFOLLOW,
                dir_fd=parent_fd,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise ContractError("artifact is missing")
        except OSError as exc:
            if exc.errno in (errno.ELOOP, errno.EMLINK):
                raise ContractError("artifact target symlink is denied") from exc
            raise ContractError("artifact cannot be opened") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ContractError("artifact: expected regular file")
            if stat.S_IMODE(metadata.st_mode) != 0o600:
                raise ContractError("artifact file mode drift")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 64 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        remaining = memoryview(content)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise ContractError("artifact write failed")
            remaining = remaining[written:]

    @staticmethod
    def _verify_content(
        reference: FactoryArtifactReference,
        content: bytes,
    ) -> FactoryContract:
        contract = _parse_contract_bytes(content)
        if contract.contract_type != reference.artifact_type:
            raise ContractError("artifact type mismatch")
        if _contract_identity(contract) != reference.artifact_id:
            raise ContractError("artifact identity mismatch")
        if contract.content_hash != reference.content_hash:
            raise ContractError("artifact content hash mismatch")
        return contract
