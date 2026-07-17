from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import uuid

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_int,
    require_str,
)


_ARTIFACT_ID_PATTERN = r"public/sha256-[0-9a-f]{64}\.json"
_DIGEST_PATTERN = r"sha256:[0-9a-f]{64}"


@dataclass(frozen=True)
class ArtifactReference:
    artifact_id: str
    digest: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        require_str(self.artifact_id, "artifact_id", pattern=_ARTIFACT_ID_PATTERN)
        require_str(self.digest, "digest", pattern=_DIGEST_PATTERN)
        expected = self.artifact_id.removeprefix("public/sha256-").removesuffix(".json")
        if self.digest != f"sha256:{expected}":
            raise ContractError("digest: does not match artifact_id")
        require_int(self.size_bytes, "size_bytes", minimum=0)
        if self.media_type != "application/json":
            raise ContractError("media_type: expected 'application/json'")

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ArtifactReference":
        data = require_exact_fields(
            value,
            "artifact_reference",
            ("artifact_id", "digest", "size_bytes", "media_type"),
        )
        return cls(
            artifact_id=require_str(data["artifact_id"], "artifact_id"),
            digest=require_str(data["digest"], "digest"),
            size_bytes=require_int(data["size_bytes"], "size_bytes", minimum=0),
            media_type=require_str(data["media_type"], "media_type"),
        )


class PublicArtifactStore:
    """Content-addressed storage for JSON values safe to expose publicly."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise ContractError("artifact root: expected Path")
        if root.is_symlink():
            raise ContractError("artifact root: symlink is denied")
        root.mkdir(parents=True, exist_ok=True)
        metadata = root.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ContractError("artifact root: symlink is denied")
        if not stat.S_ISDIR(metadata.st_mode):
            raise ContractError("artifact root: expected directory")
        self.root = root
        self._public = root / "public"
        directory_flags = os.O_RDONLY | os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            directory_flags |= os.O_NOFOLLOW
        try:
            root_fd = os.open(root, directory_flags)
        except OSError as exc:
            raise ContractError("artifact root: expected real directory") from exc
        public_fd: int | None = None
        try:
            try:
                os.mkdir("public", mode=0o700, dir_fd=root_fd)
            except FileExistsError:
                pass
            try:
                public_fd = os.open(
                    "public",
                    directory_flags,
                    dir_fd=root_fd,
                )
            except OSError as exc:
                raise ContractError(
                    "artifact public directory: expected real directory"
                ) from exc
            public_metadata = os.fstat(public_fd)
            if not stat.S_ISDIR(public_metadata.st_mode):
                raise ContractError("artifact public directory: expected real directory")
            os.fsync(root_fd)
        except Exception:
            if public_fd is not None:
                os.close(public_fd)
            os.close(root_fd)
            raise
        self._root_fd = root_fd
        self._public_fd = public_fd
        self._lock = threading.RLock()
        self._closed = False

    def put_json(self, logical_name: str, value: object) -> ArtifactReference:
        self._ensure_open()
        require_str(logical_name, "logical_name")
        assert_public_artifact_safe(value)
        encoded = canonical_json(value).encode("utf-8")
        digest_hex = hashlib.sha256(encoded).hexdigest()
        filename = f"sha256-{digest_hex}.json"
        reference = ArtifactReference(
            artifact_id=f"public/{filename}",
            digest=f"sha256:{digest_hex}",
            size_bytes=len(encoded),
            media_type="application/json",
        )
        with self._lock:
            self._ensure_open()
            try:
                existing = self._read_filename(filename)
            except ContractError as exc:
                if "missing" not in str(exc):
                    raise
            else:
                self._assert_public_json(reference, existing)
                return reference

            temporary = f".artifact-{uuid.uuid4().hex}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                try:
                    descriptor = os.open(
                        temporary,
                        flags,
                        0o600,
                        dir_fd=self._public_fd,
                    )
                except OSError as exc:
                    raise ContractError(
                        "artifact temporary file cannot be created"
                    ) from exc
                try:
                    self._write_all(descriptor, encoded)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
                try:
                    os.link(
                        temporary,
                        filename,
                        src_dir_fd=self._public_fd,
                        dst_dir_fd=self._public_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
                os.fsync(self._public_fd)
            finally:
                try:
                    os.unlink(temporary, dir_fd=self._public_fd)
                    os.fsync(self._public_fd)
                except FileNotFoundError:
                    pass
            self._assert_public_json(reference, self._read_filename(filename))
            return reference

    def read_bytes(self, reference: ArtifactReference | Mapping[str, object]) -> bytes:
        self._ensure_open()
        if isinstance(reference, Mapping):
            reference = ArtifactReference.from_dict(reference)
        if not isinstance(reference, ArtifactReference):
            raise ContractError("reference: expected ArtifactReference")
        filename = reference.artifact_id.removeprefix("public/")
        with self._lock:
            self._ensure_open()
            content = self._read_filename(filename)
            self._assert_public_json(reference, content)
            return content

    def _read_filename(self, filename: str) -> bytes:
        if re.fullmatch(r"sha256-[0-9a-f]{64}\.json", filename) is None:
            raise ContractError("artifact filename: invalid")
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(filename, flags, dir_fd=self._public_fd)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOENT}:
                raise ContractError("artifact is missing or is a symlink") from exc
            raise ContractError("artifact cannot be opened") from exc
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
            return b"".join(chunks)
        finally:
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, content: bytes) -> None:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise ContractError("artifact write failed")
            view = view[written:]

    @classmethod
    def _assert_public_json(
        cls,
        reference: ArtifactReference,
        content: bytes,
    ) -> None:
        cls._assert_bytes(reference, content)
        try:
            decoded = content.decode("utf-8")
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("artifact: invalid JSON") from exc
        if canonical_json(value).encode("utf-8") != content:
            raise ContractError("artifact: expected canonical JSON")
        assert_public_artifact_safe(value)

    def close(self) -> None:
        try:
            lock = self._lock
        except AttributeError:
            return
        with lock:
            if self._closed:
                return
            self._closed = True
            os.close(self._public_fd)
            os.close(self._root_fd)

    def _ensure_open(self) -> None:
        if getattr(self, "_closed", True):
            raise ContractError("artifact store is closed")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - best-effort descriptor cleanup.
            pass

    @staticmethod
    def _assert_bytes(reference: ArtifactReference, content: bytes) -> None:
        if len(content) != reference.size_bytes:
            raise ContractError("artifact size mismatch")
        actual = f"sha256:{hashlib.sha256(content).hexdigest()}"
        if actual != reference.digest:
            raise ContractError("artifact digest mismatch")


__all__ = ["ArtifactReference", "PublicArtifactStore"]
