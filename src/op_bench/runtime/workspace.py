from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import subprocess
import tempfile
import threading
from typing import Iterator, Mapping

from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import ContentIdentity, EvaluationSpec, SessionResult
from op_bench.runtime.validation import ContractError, require_bool, require_int, require_str


class WorkspaceError(ContractError):
    """Base error for an authoritative workspace boundary."""


class WorkspacePolicyError(WorkspaceError):
    """Raised when a filesystem operation violates the frozen workspace policy."""


class WorkspaceStateError(WorkspaceError):
    """Raised when an operation is incompatible with the workspace lifecycle."""


@dataclass(frozen=True)
class WorkspacePolicy:
    policy_id: str
    writable_paths: tuple[str, ...]
    patch_paths: tuple[str, ...]
    allowed_modes: tuple[int, ...]
    max_read_bytes: int
    max_write_bytes: int
    max_file_bytes: int
    max_patch_bytes: int
    allow_binary: bool

    def __post_init__(self) -> None:
        require_str(self.policy_id, "policy_id")
        _validate_scopes(self.writable_paths, "writable_paths")
        _validate_scopes(self.patch_paths, "patch_paths")
        if not isinstance(self.allowed_modes, tuple) or not self.allowed_modes:
            raise WorkspacePolicyError("allowed_modes: expected non-empty tuple")
        seen_modes: set[int] = set()
        for index, mode in enumerate(self.allowed_modes):
            if isinstance(mode, bool) or not isinstance(mode, int):
                raise WorkspacePolicyError(f"allowed_modes[{index}]: expected integer")
            if mode < 0 or mode > 0o777:
                raise WorkspacePolicyError(f"allowed_modes[{index}]: invalid regular-file mode")
            if mode in seen_modes:
                raise WorkspacePolicyError(f"allowed_modes: duplicate value {oct(mode)}")
            seen_modes.add(mode)
        for name, value in (
            ("max_read_bytes", self.max_read_bytes),
            ("max_write_bytes", self.max_write_bytes),
            ("max_file_bytes", self.max_file_bytes),
            ("max_patch_bytes", self.max_patch_bytes),
        ):
            try:
                require_int(value, name, minimum=1)
            except ContractError as exc:
                raise WorkspacePolicyError(str(exc)) from exc
        try:
            require_bool(self.allow_binary, "allow_binary")
        except ContractError as exc:
            raise WorkspacePolicyError(str(exc)) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_version": "workspace-policy-v1",
            "policy_id": self.policy_id,
            "writable_paths": list(self.writable_paths),
            "patch_paths": list(self.patch_paths),
            "allowed_modes": list(self.allowed_modes),
            "max_read_bytes": self.max_read_bytes,
            "max_write_bytes": self.max_write_bytes,
            "max_file_bytes": self.max_file_bytes,
            "max_patch_bytes": self.max_patch_bytes,
            "allow_binary": self.allow_binary,
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class WorkspaceRead:
    workspace: ContentIdentity
    path: str
    content: bytes
    mode: int


@dataclass(frozen=True)
class WorkspaceEntry:
    workspace: ContentIdentity
    path: str
    entry_type: str
    mode: int
    size_bytes: int

    def __post_init__(self) -> None:
        _require_identity(self.workspace, "workspace", "workspace")
        _normalize_relative_path(self.path)
        if self.entry_type not in {"file", "directory"}:
            raise ContractError("entry_type: expected file or directory")
        require_int(self.mode, "mode", minimum=0)
        require_int(self.size_bytes, "size_bytes", minimum=0)


@dataclass(frozen=True)
class WorkspaceMutation:
    workspace: ContentIdentity
    path: str
    changed: bool


@dataclass(frozen=True)
class WorkspacePatchMutation:
    workspace: ContentIdentity
    paths: tuple[str, ...]
    changed: bool


@dataclass(frozen=True)
class _FileSnapshot:
    content: bytes
    mode: int


@dataclass(frozen=True)
class WorkspaceBinding:
    workspace: ContentIdentity
    operation: str
    identifier: str


@dataclass(frozen=True)
class WorkspaceDiff:
    workspace: ContentIdentity
    patch_bytes: bytes


@dataclass(frozen=True)
class FrozenPatch:
    workspace: ContentIdentity
    source: ContentIdentity
    base_commit: str
    patch: ContentIdentity
    patch_bytes: bytes
    changed_paths: tuple[str, ...]
    empty: bool

    def __post_init__(self) -> None:
        _require_identity(self.workspace, "workspace", "workspace")
        _require_identity(self.source, "source", "source")
        _require_identity(self.patch, "patch", "patch")
        require_str(self.base_commit, "base_commit")
        if not isinstance(self.patch_bytes, bytes):
            raise ContractError("patch_bytes: expected bytes")
        if not isinstance(self.changed_paths, tuple):
            raise ContractError("changed_paths: expected tuple")
        normalized_paths = tuple(_normalize_relative_path(path) for path in self.changed_paths)
        if normalized_paths != tuple(sorted(set(normalized_paths))):
            raise ContractError("changed_paths: expected sorted unique canonical paths")
        try:
            require_bool(self.empty, "empty")
        except ContractError as exc:
            raise ContractError(str(exc)) from exc
        if self.empty != (len(self.patch_bytes) == 0 and not self.changed_paths):
            raise ContractError("empty: does not match patch bytes and changed paths")
        expected = raw_patch_identity(self.patch_bytes, identifier=self.patch.identifier)
        if self.patch != expected:
            raise ContractError("patch bytes do not match patch identity")


@dataclass(frozen=True)
class PatchArtifact:
    artifact_id: str
    workspace: ContentIdentity
    patch: ContentIdentity
    size_bytes: int
    changed_paths: tuple[str, ...]
    empty: bool

    def __post_init__(self) -> None:
        require_str(self.artifact_id, "artifact_id")
        _require_identity(self.workspace, "workspace", "workspace")
        _require_identity(self.patch, "patch", "patch")
        require_int(self.size_bytes, "size_bytes", minimum=0)
        if not isinstance(self.changed_paths, tuple):
            raise ContractError("changed_paths: expected tuple")
        normalized_paths = tuple(_normalize_relative_path(path) for path in self.changed_paths)
        if normalized_paths != tuple(sorted(set(normalized_paths))):
            raise ContractError("changed_paths: expected sorted unique canonical paths")
        require_bool(self.empty, "empty")
        if self.empty != (self.size_bytes == 0 and not self.changed_paths):
            raise ContractError("empty: does not match artifact metadata")


class AuthoritativeWorkspace:
    """The single mutable filesystem authority for one evaluation attempt."""

    def __init__(
        self,
        *,
        root: Path,
        source: ContentIdentity,
        policy: WorkspacePolicy,
        base_commit: str,
        materialization_mode: str,
        identity: ContentIdentity,
        root_fd: int,
    ) -> None:
        self._root = root
        self._root_fd = root_fd
        self.source = source
        self.policy = policy
        self.base_commit = base_commit
        self.materialization_mode = materialization_mode
        self.identity = identity
        self._condition = threading.Condition(threading.RLock())
        self._mutation_lock = threading.Lock()
        self._state = "open"
        self._active_mutations = 0
        self._base_snapshots: dict[str, _FileSnapshot] = {}
        self._frozen_patch: FrozenPatch | None = None
        self._freeze_error: BaseException | None = None

    @classmethod
    def open(
        cls,
        root: Path | str,
        *,
        source: ContentIdentity,
        policy: WorkspacePolicy,
        materialization_mode: str = "local_git_copy",
    ) -> "AuthoritativeWorkspace":
        if not isinstance(source, ContentIdentity) or source.identity_type != "source":
            raise WorkspacePolicyError("source: expected source identity")
        if not isinstance(policy, WorkspacePolicy):
            raise WorkspacePolicyError("policy: expected WorkspacePolicy")
        try:
            require_str(materialization_mode, "materialization_mode")
        except ContractError as exc:
            raise WorkspacePolicyError(str(exc)) from exc

        requested_root = Path(root)
        if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
            raise WorkspacePolicyError(
                "workspace requires descriptor-relative O_NOFOLLOW/O_DIRECTORY support"
            )
        try:
            root_lstat = requested_root.lstat()
        except OSError as exc:
            raise WorkspacePolicyError(f"workspace root is unavailable: {exc}") from exc
        if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
            raise WorkspacePolicyError("workspace root: expected non-symlink directory")
        resolved_root = requested_root.resolve(strict=True)

        try:
            root_fd = os.open(
                resolved_root,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            )
        except OSError as exc:
            raise WorkspacePolicyError(
                f"workspace root cannot be opened safely: {exc.strerror or 'denied'}"
            ) from exc
        try:
            descriptor_root = os.fstat(root_fd)
            current_root = resolved_root.lstat()
            if not _same_inode(descriptor_root, current_root) or not stat.S_ISDIR(
                descriptor_root.st_mode
            ):
                raise WorkspacePolicyError("workspace root changed while authority was created")

            _reject_hidden_index_flags(resolved_root)
            status = _git(
                resolved_root,
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            )
            if status.stdout:
                raise WorkspacePolicyError("workspace must be clean before authority is established")
            base_commit = _git(resolved_root, "rev-parse", "--verify", "HEAD").stdout.decode(
                "ascii"
            ).strip()
            if not base_commit or any(
                character not in "0123456789abcdef" for character in base_commit
            ):
                raise WorkspacePolicyError("workspace base commit is not a canonical Git object id")
            base_snapshots = _commit_scope_snapshots(
                resolved_root,
                base_commit,
                policy,
            )
            rebound_root = resolved_root.lstat()
            if not _same_inode(descriptor_root, rebound_root):
                raise WorkspacePolicyError(
                    "workspace root changed while base commit was inspected"
                )

            identity_digest = canonical_sha256(
                {
                    "identity_version": "authoritative-workspace-v1",
                    "source": source.to_dict(),
                    "base_commit": base_commit,
                    "materialization_mode": materialization_mode,
                    "policy": policy.to_dict(),
                }
            )
            workspace_identity = ContentIdentity(
                identity_type="workspace",
                identifier=f"{source.identifier}:authoritative-workspace-v1",
                digest=identity_digest,
                digest_kind="canonical_config",
            )
            workspace = cls(
                root=resolved_root,
                source=source,
                policy=policy,
                base_commit=base_commit,
                materialization_mode=materialization_mode,
                identity=workspace_identity,
                root_fd=root_fd,
            )
            try:
                current_snapshots = workspace._snapshot_scope_files()
                if current_snapshots != base_snapshots:
                    raise WorkspacePolicyError(
                        "workspace patch scope does not match recorded base commit"
                    )
                workspace._base_snapshots = base_snapshots
            except BaseException:
                workspace.close()
                root_fd = -1
                raise
            return workspace
        except BaseException:
            if root_fd >= 0:
                os.close(root_fd)
            raise

    def close(self) -> None:
        descriptor = getattr(self, "_root_fd", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._root_fd = -1

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    def read(self, path: str, *, max_bytes: int | None = None) -> WorkspaceRead:
        requested_limit = self.policy.max_read_bytes if max_bytes is None else max_bytes
        try:
            require_int(requested_limit, "max_bytes", minimum=1)
        except ContractError as exc:
            raise WorkspacePolicyError(str(exc)) from exc
        limit = min(requested_limit, self.policy.max_read_bytes)
        with self._parent_directory(path) as (normalized, parent_fd, name):
            descriptor = _open_readonly_at(parent_fd, name, normalized)
            try:
                metadata = os.fstat(descriptor)
                mode = _validate_regular_metadata(metadata, self.policy, normalized)
                if metadata.st_size > limit:
                    raise WorkspacePolicyError(
                        f"path {normalized!r}: size {metadata.st_size} exceeds read limit {limit}"
                    )
                content = os.read(descriptor, limit + 1)
                self._assert_entry_binding(parent_fd, name, metadata, normalized)
            finally:
                os.close(descriptor)
            self._assert_parent_binding(normalized, parent_fd)
        if len(content) > limit:
            raise WorkspacePolicyError(f"path {normalized!r}: content exceeds read limit {limit}")
        return WorkspaceRead(self.identity, normalized, content, mode)

    def list_entries(
        self,
        path: str,
        *,
        recursive: bool,
        max_entries: int,
        max_depth: int,
    ) -> tuple[WorkspaceEntry, ...]:
        try:
            require_bool(recursive, "recursive")
            require_int(max_entries, "max_entries", minimum=1)
            require_int(max_depth, "max_depth", minimum=1)
        except ContractError as exc:
            raise WorkspacePolicyError(str(exc)) from exc
        normalized = "." if path == "." else _normalize_relative_path(path)
        entries: list[WorkspaceEntry] = []
        with self._directory(normalized) as directory_fd:
            self._collect_entries(
                directory_fd,
                prefix=normalized,
                recursive=recursive,
                max_entries=max_entries,
                max_depth=max_depth,
                depth=1,
                entries=entries,
            )
        self._assert_root_binding()
        return tuple(entries)

    def write(
        self,
        path: str,
        content: bytes | str,
        *,
        mode: int | None = None,
    ) -> WorkspaceMutation:
        with self._mutation_admission():
            if isinstance(content, str):
                encoded = content.encode("utf-8")
            elif isinstance(content, bytes):
                encoded = content
            else:
                raise WorkspacePolicyError("content: expected bytes or string")
            if len(encoded) > self.policy.max_write_bytes:
                raise WorkspacePolicyError(
                    f"content: exceeds max_write_bytes {self.policy.max_write_bytes}"
                )
            if len(encoded) > self.policy.max_file_bytes:
                raise WorkspacePolicyError(
                    f"content: exceeds max_file_bytes {self.policy.max_file_bytes}"
                )
            if not self.policy.allow_binary and _looks_binary(encoded):
                raise WorkspacePolicyError("content: binary writes are denied")

            with self._parent_directory(path) as (normalized, parent_fd, name):
                self._require_writable(normalized)
                existing: bytes | None = None
                existing_mode: int | None = None
                entry = _lstat_at(parent_fd, name, normalized, missing_ok=True)
                if entry is not None:
                    descriptor = _open_readonly_at(parent_fd, name, normalized)
                    try:
                        metadata = os.fstat(descriptor)
                        existing_mode = _validate_regular_metadata(
                            metadata, self.policy, normalized
                        )
                        existing = _read_all(descriptor, self.policy.max_file_bytes)
                        self._assert_entry_binding(parent_fd, name, metadata, normalized)
                    finally:
                        os.close(descriptor)

                selected_mode = (
                    existing_mode if mode is None and existing_mode is not None else mode
                )
                if selected_mode is None:
                    selected_mode = 0o644
                self._validate_requested_mode(selected_mode)
                if existing == encoded and existing_mode == selected_mode:
                    return WorkspaceMutation(self.identity, normalized, False)

                self._atomic_replace(parent_fd, name, encoded, selected_mode)
                self._assert_parent_binding(normalized, parent_fd)
                return WorkspaceMutation(self.identity, normalized, True)

    def delete(self, path: str) -> WorkspaceMutation:
        with self._mutation_admission():
            with self._parent_directory(path) as (normalized, parent_fd, name):
                self._require_writable(normalized)
                entry = _lstat_at(parent_fd, name, normalized, missing_ok=True)
                if entry is None:
                    return WorkspaceMutation(self.identity, normalized, False)
                descriptor = _open_readonly_at(parent_fd, name, normalized)
                try:
                    metadata = os.fstat(descriptor)
                    _validate_regular_metadata(metadata, self.policy, normalized)
                    self._assert_entry_binding(parent_fd, name, metadata, normalized)
                finally:
                    os.close(descriptor)
                try:
                    os.unlink(name, dir_fd=parent_fd)
                except OSError as exc:
                    raise WorkspacePolicyError(
                        f"path {normalized!r}: cannot delete safely: {exc.strerror or 'denied'}"
                    ) from exc
                self._assert_parent_binding(normalized, parent_fd)
                return WorkspaceMutation(self.identity, normalized, True)

    def apply_patch(self, patch: bytes | str) -> WorkspacePatchMutation:
        with self._mutation_admission():
            if isinstance(patch, str):
                patch_bytes = patch.encode("utf-8")
            elif isinstance(patch, bytes):
                patch_bytes = patch
            else:
                raise WorkspacePolicyError("patch: expected bytes or string")
            if len(patch_bytes) > self.policy.max_write_bytes:
                raise WorkspacePolicyError(
                    f"patch exceeds max_write_bytes {self.policy.max_write_bytes}"
                )
            if len(patch_bytes) > self.policy.max_patch_bytes:
                raise WorkspacePolicyError(
                    f"patch exceeds max_patch_bytes {self.policy.max_patch_bytes}"
                )
            if not patch_bytes:
                return WorkspacePatchMutation(self.identity, (), False)
            if not self.policy.allow_binary and b"GIT binary patch" in patch_bytes:
                raise WorkspacePolicyError("binary patch is denied")
            _validate_patch_modes(patch_bytes, self.policy)
            paths = _patch_paths_from_bytes(patch_bytes)
            if not paths:
                raise WorkspacePolicyError("patch: expected at least one changed path")
            snapshots: dict[str, _FileSnapshot | None] = {}
            for path in paths:
                self._require_writable(path)
                snapshots[path] = self._snapshot(path)

            staged = self._stage_patch(patch_bytes, paths, snapshots)
            with ExitStack() as stack:
                targets: list[tuple[str, int, str]] = []
                for path in paths:
                    normalized, parent_fd, name = stack.enter_context(
                        self._parent_directory(path)
                    )
                    if self._snapshot_at(parent_fd, name, normalized) != snapshots[path]:
                        raise WorkspacePolicyError(
                            f"path {path!r}: changed while patch was staged"
                        )
                    targets.append((normalized, parent_fd, name))

                applied_paths: list[tuple[str, int, str]] = []
                try:
                    for path, parent_fd, name in targets:
                        result = staged[path]
                        if result is None:
                            current = _lstat_at(parent_fd, name, path, missing_ok=True)
                            if current is not None:
                                if stat.S_ISLNK(current.st_mode):
                                    raise WorkspacePolicyError(
                                        f"path {path!r}: symlink is denied"
                                    )
                                os.unlink(name, dir_fd=parent_fd)
                        else:
                            self._atomic_replace(
                                parent_fd,
                                name,
                                result.content,
                                result.mode,
                            )
                        applied_paths.append((path, parent_fd, name))
                    for path, parent_fd, _ in targets:
                        self._assert_parent_binding(path, parent_fd)
                except BaseException:
                    for path, parent_fd, name in reversed(applied_paths):
                        original = snapshots[path]
                        if original is None:
                            try:
                                os.unlink(name, dir_fd=parent_fd)
                            except FileNotFoundError:
                                pass
                        else:
                            self._atomic_replace(
                                parent_fd,
                                name,
                                original.content,
                                original.mode,
                            )
                    raise
            return WorkspacePatchMutation(self.identity, paths, True)

    def _snapshot(self, path: str) -> _FileSnapshot | None:
        with self._parent_directory(path) as (normalized, parent_fd, name):
            result = self._snapshot_at(parent_fd, name, normalized)
            self._assert_parent_binding(normalized, parent_fd)
            return result

    def _snapshot_at(
        self,
        parent_fd: int,
        name: str,
        path: str,
    ) -> _FileSnapshot | None:
        entry = _lstat_at(parent_fd, name, path, missing_ok=True)
        if entry is None:
            return None
        descriptor = _open_readonly_at(parent_fd, name, path)
        try:
            metadata = os.fstat(descriptor)
            mode = _validate_regular_metadata(metadata, self.policy, path)
            content = _read_all(descriptor, self.policy.max_file_bytes)
            self._assert_entry_binding(parent_fd, name, metadata, path)
        finally:
            os.close(descriptor)
        if not self.policy.allow_binary and _looks_binary(content):
            raise WorkspacePolicyError(f"path {path!r}: binary file is denied")
        return _FileSnapshot(content=content, mode=mode)

    def _stage_patch(
        self,
        patch_bytes: bytes,
        paths: tuple[str, ...],
        snapshots: dict[str, _FileSnapshot | None],
    ) -> dict[str, _FileSnapshot | None]:
        with tempfile.TemporaryDirectory(prefix="opbench-workspace-apply-") as temporary:
            stage_root = Path(temporary) / "stage"
            stage_root.mkdir()
            for path, snapshot in snapshots.items():
                if snapshot is None:
                    continue
                candidate = stage_root.joinpath(*PurePosixPath(path).parts)
                candidate.parent.mkdir(parents=True, exist_ok=True)
                candidate.write_bytes(snapshot.content)
                candidate.chmod(snapshot.mode)
            patch_path = Path(temporary) / "candidate.patch"
            patch_path.write_bytes(patch_bytes)
            check = _git(
                stage_root,
                "apply",
                "--check",
                "--whitespace=nowarn",
                str(patch_path),
                check=False,
            )
            if check.returncode != 0:
                detail = check.stderr.decode("utf-8", errors="replace").strip()
                raise WorkspacePolicyError(f"patch does not apply: {detail}")
            applied = _git(
                stage_root,
                "apply",
                "--whitespace=nowarn",
                str(patch_path),
                check=False,
            )
            if applied.returncode != 0:
                detail = applied.stderr.decode("utf-8", errors="replace").strip()
                raise WorkspacePolicyError(f"patch application failed: {detail}")

            results: dict[str, _FileSnapshot | None] = {}
            for path in paths:
                candidate = stage_root.joinpath(*PurePosixPath(path).parts)
                try:
                    metadata = candidate.lstat()
                except FileNotFoundError:
                    results[path] = None
                    continue
                mode = _validate_regular_metadata(metadata, self.policy, path)
                content = candidate.read_bytes()
                if len(content) > self.policy.max_file_bytes:
                    raise WorkspacePolicyError(f"path {path!r}: exceeds max_file_bytes")
                if not self.policy.allow_binary and _looks_binary(content):
                    raise WorkspacePolicyError(f"path {path!r}: binary file is denied")
                results[path] = _FileSnapshot(content=content, mode=mode)
            return results

    def _snapshot_scope_files(self) -> dict[str, _FileSnapshot]:
        self._assert_root_binding()
        snapshots: dict[str, _FileSnapshot] = {}
        for scope in self.policy.patch_paths:
            if scope.endswith("/"):
                directory = scope[:-1]
                probe = f"{directory}/.opbench-scope-probe"
                with self._parent_directory(probe) as (normalized, directory_fd, _):
                    self._walk_scope_directory(directory_fd, directory, snapshots)
                    self._assert_parent_binding(normalized, directory_fd)
            else:
                snapshot = self._snapshot(scope)
                if snapshot is not None:
                    prior = snapshots.get(scope)
                    if prior is not None and prior != snapshot:
                        raise WorkspacePolicyError(
                            f"patch scope {scope!r}: overlapping snapshots disagree"
                        )
                    snapshots[scope] = snapshot
        self._assert_root_binding()
        return snapshots

    def _walk_scope_directory(
        self,
        directory_fd: int,
        prefix: str,
        snapshots: dict[str, _FileSnapshot],
    ) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise WorkspacePolicyError(
                f"patch scope {prefix!r}: cannot enumerate safely"
            ) from exc
        for name in names:
            path = f"{prefix}/{name}"
            _normalize_relative_path(path)
            metadata = _lstat_at(directory_fd, name, path, missing_ok=False)
            assert metadata is not None
            if stat.S_ISLNK(metadata.st_mode):
                raise WorkspacePolicyError(f"path {path!r}: symlink is denied")
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_fd = os.open(
                        name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise WorkspacePolicyError(
                        f"path {path!r}: directory binding changed"
                    ) from exc
                try:
                    opened_directory = os.fstat(child_fd)
                    if not _same_inode(metadata, opened_directory):
                        raise WorkspacePolicyError(
                            f"path {path!r}: directory binding changed"
                        )
                    self._walk_scope_directory(child_fd, path, snapshots)
                    self._assert_entry_binding(
                        directory_fd,
                        name,
                        opened_directory,
                        path,
                    )
                finally:
                    os.close(child_fd)
                continue
            descriptor = _open_readonly_at(directory_fd, name, path)
            try:
                opened = os.fstat(descriptor)
                mode = _validate_regular_metadata(opened, self.policy, path)
                content = _read_all(descriptor, self.policy.max_file_bytes)
                self._assert_entry_binding(directory_fd, name, opened, path)
            finally:
                os.close(descriptor)
            if not self.policy.allow_binary and _looks_binary(content):
                raise WorkspacePolicyError(f"path {path!r}: binary file is denied")
            snapshots[path] = _FileSnapshot(content=content, mode=mode)

    def _capture_authoritative_patch(self) -> tuple[tuple[str, ...], bytes]:
        current = self._snapshot_scope_files()
        changed_paths = tuple(
            sorted(
                path
                for path in set(self._base_snapshots) | set(current)
                if self._base_snapshots.get(path) != current.get(path)
            )
        )
        if not changed_paths:
            return (), b""

        with tempfile.TemporaryDirectory(prefix="opbench-canonical-diff-") as temporary:
            stage_root = Path(temporary) / "stage"
            stage_root.mkdir()
            _materialize_snapshots(stage_root, self._base_snapshots)
            _git(stage_root, "init", "--quiet")
            _git(stage_root, "add", "--all")
            for path in sorted(set(self._base_snapshots) - set(current)):
                stage_root.joinpath(*PurePosixPath(path).parts).unlink()
            _materialize_snapshots(stage_root, current)

            arguments = (
                "diff",
                "--no-ext-diff",
                "--no-color",
                "--no-textconv",
                "--binary",
                "--full-index",
                "--no-renames",
                "--diff-algorithm=myers",
                "--no-indent-heuristic",
                "--unified=3",
                "--src-prefix=a/",
                "--dst-prefix=b/",
                "--",
                ".",
            )
            tracked = _git(stage_root, *arguments).stdout
            additions: list[bytes] = []
            for path in sorted(set(current) - set(self._base_snapshots)):
                result = _git(
                    stage_root,
                    "diff",
                    "--no-ext-diff",
                    "--no-color",
                    "--no-textconv",
                    "--binary",
                    "--full-index",
                    "--no-index",
                    "--diff-algorithm=myers",
                    "--no-indent-heuristic",
                    "--unified=3",
                    "--src-prefix=a/",
                    "--dst-prefix=b/",
                    "--",
                    "/dev/null",
                    path,
                    check=False,
                )
                if result.returncode not in (0, 1):
                    detail = result.stderr.decode("utf-8", errors="replace").strip()
                    raise WorkspaceError(f"cannot diff new path {path!r}: {detail}")
                additions.append(result.stdout)
            patch_bytes = tracked + b"".join(additions)

        parsed_paths = _patch_paths_from_bytes(patch_bytes)
        if parsed_paths != changed_paths:
            raise WorkspacePolicyError(
                "canonical descriptor snapshot diff does not match changed paths"
            )
        return changed_paths, patch_bytes

    def bind_test(self, selector_id: str) -> WorkspaceBinding:
        try:
            require_str(selector_id, "selector_id")
        except ContractError as exc:
            raise WorkspacePolicyError(str(exc)) from exc
        return WorkspaceBinding(self.identity, "test", selector_id)

    def diff(self) -> WorkspaceDiff:
        with self._condition:
            while self._state == "freezing":
                self._condition.wait()
            if self._state == "frozen":
                assert self._frozen_patch is not None
                return WorkspaceDiff(self.identity, self._frozen_patch.patch_bytes)
            if self._state == "freeze_failed":
                self._raise_freeze_error()
        _, patch_bytes = self._capture_authoritative_patch()
        return WorkspaceDiff(self.identity, patch_bytes)

    def freeze(self) -> FrozenPatch:
        with self._condition:
            while self._state == "freezing":
                self._condition.wait()
            if self._state == "frozen":
                assert self._frozen_patch is not None
                return self._frozen_patch
            if self._state == "freeze_failed":
                self._raise_freeze_error()
            if self._state != "open":
                raise WorkspaceStateError(f"cannot freeze workspace in state {self._state!r}")
            self._state = "freezing"
            while self._active_mutations:
                self._condition.wait()

        try:
            frozen = self._build_frozen_patch()
        except BaseException as exc:
            with self._condition:
                self._freeze_error = exc
                self._state = "freeze_failed"
                self._condition.notify_all()
            raise

        with self._condition:
            self._frozen_patch = frozen
            self._state = "frozen"
            self._condition.notify_all()
            return frozen

    def _build_frozen_patch(self) -> FrozenPatch:
        changed_paths, patch_bytes = self._capture_authoritative_patch()
        if len(patch_bytes) > self.policy.max_patch_bytes:
            raise WorkspacePolicyError(
                f"patch exceeds max_patch_bytes {self.policy.max_patch_bytes}"
            )
        if bool(changed_paths) != bool(patch_bytes):
            raise WorkspacePolicyError("canonical patch does not match changed path set")
        if not self.policy.allow_binary and b"GIT binary patch" in patch_bytes:
            raise WorkspacePolicyError("binary patch is denied")

        for path in changed_paths:
            if not _path_in_scopes(path, self.policy.patch_paths):
                raise WorkspacePolicyError(f"path {path!r}: outside patch scope")
            self._validate_frozen_path(path)

        parsed_paths = _patch_paths_from_bytes(patch_bytes)
        if parsed_paths != changed_paths:
            raise WorkspacePolicyError(
                "canonical patch paths do not match authoritative Git status"
            )
        self._verify_strict_clean_apply(patch_bytes)
        patch_identity = raw_patch_identity(
            patch_bytes,
            identifier=f"{self.identity.identifier}:final.patch",
        )
        return FrozenPatch(
            workspace=self.identity,
            source=self.source,
            base_commit=self.base_commit,
            patch=patch_identity,
            patch_bytes=patch_bytes,
            changed_paths=changed_paths,
            empty=not patch_bytes,
        )

    def _validate_frozen_path(self, path: str) -> None:
        current = self._snapshot(path)
        if current is None and path not in self._base_snapshots:
            raise WorkspacePolicyError(f"deleted path {path!r}: missing from base")

    def _verify_strict_clean_apply(self, patch_bytes: bytes) -> None:
        if not patch_bytes:
            return
        with tempfile.TemporaryDirectory(prefix="opbench-clean-apply-") as temporary:
            temporary_root = Path(temporary)
            clean_root = temporary_root / "source"
            clean_root.mkdir()
            _materialize_snapshots(clean_root, self._base_snapshots)
            _git(clean_root, "init", "--quiet")
            _git(clean_root, "add", "--all")
            patch_path = temporary_root / "candidate.patch"
            patch_path.write_bytes(patch_bytes)
            check = _git(
                clean_root,
                "apply",
                "--check",
                "--index",
                "--whitespace=nowarn",
                str(patch_path),
                check=False,
            )
            if check.returncode != 0:
                detail = check.stderr.decode("utf-8", errors="replace").strip()
                raise WorkspacePolicyError(f"patch failed strict clean-base apply: {detail}")

    def _raise_freeze_error(self) -> None:
        assert self._freeze_error is not None
        error = self._freeze_error
        if isinstance(error, WorkspaceError):
            raise type(error)(str(error)) from error
        raise WorkspaceError(f"prior freeze failed: {error}") from error

    @contextmanager
    def _mutation_admission(self) -> Iterator[None]:
        with self._condition:
            if self._state != "open":
                raise WorkspaceStateError(
                    f"workspace does not accept mutations in state {self._state!r}"
                )
        self._mutation_lock.acquire()
        try:
            with self._condition:
                if self._state != "open":
                    raise WorkspaceStateError(
                        f"workspace does not accept mutations in state {self._state!r}"
                    )
                self._active_mutations += 1
            try:
                yield
            finally:
                with self._condition:
                    self._active_mutations -= 1
                    self._condition.notify_all()
        finally:
            self._mutation_lock.release()

    @contextmanager
    def _parent_directory(self, path: str) -> Iterator[tuple[str, int, str]]:
        normalized = _normalize_relative_path(path)
        parts = PurePosixPath(normalized).parts
        self._assert_root_binding()
        descriptor = os.dup(self._root_fd)
        try:
            for component in parts[:-1]:
                try:
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        reason = "symlink or non-directory parent is denied"
                    elif exc.errno == errno.ENOENT:
                        reason = "parent does not exist"
                    else:
                        reason = exc.strerror or "cannot open parent"
                    raise WorkspacePolicyError(
                        f"path {normalized!r}: {reason}"
                    ) from exc
                os.close(descriptor)
                descriptor = next_descriptor
            yield normalized, descriptor, parts[-1]
        finally:
            os.close(descriptor)

    @contextmanager
    def _directory(self, path: str) -> Iterator[int]:
        self._assert_root_binding()
        if path == ".":
            descriptor = os.dup(self._root_fd)
            try:
                yield descriptor
            finally:
                os.close(descriptor)
            return
        with self._parent_directory(path) as (normalized, parent_fd, name):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                    reason = "symlink, special file, or non-directory is denied"
                else:
                    reason = exc.strerror or "cannot open directory"
                raise WorkspacePolicyError(f"path {normalized!r}: {reason}") from exc
            try:
                opened = os.fstat(descriptor)
                self._assert_entry_binding(parent_fd, name, opened, normalized)
                yield descriptor
                self._assert_entry_binding(parent_fd, name, opened, normalized)
                self._assert_parent_binding(normalized, parent_fd)
            finally:
                os.close(descriptor)

    def _collect_entries(
        self,
        directory_fd: int,
        *,
        prefix: str,
        recursive: bool,
        max_entries: int,
        max_depth: int,
        depth: int,
        entries: list[WorkspaceEntry],
    ) -> None:
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise WorkspacePolicyError("workspace directory cannot be enumerated") from exc
        for name in names:
            if prefix == "." and name == ".git":
                continue
            candidate = name if prefix == "." else f"{prefix}/{name}"
            normalized = _normalize_relative_path(candidate)
            metadata = _lstat_at(directory_fd, name, normalized, missing_ok=False)
            assert metadata is not None
            if stat.S_ISLNK(metadata.st_mode):
                raise WorkspacePolicyError(f"path {normalized!r}: symlink is denied")
            if stat.S_ISDIR(metadata.st_mode):
                entry = WorkspaceEntry(
                    workspace=self.identity,
                    path=normalized,
                    entry_type="directory",
                    mode=stat.S_IMODE(metadata.st_mode),
                    size_bytes=0,
                )
            elif stat.S_ISREG(metadata.st_mode):
                mode = _validate_regular_metadata(metadata, self.policy, normalized)
                entry = WorkspaceEntry(
                    workspace=self.identity,
                    path=normalized,
                    entry_type="file",
                    mode=mode,
                    size_bytes=metadata.st_size,
                )
            else:
                raise WorkspacePolicyError(
                    f"path {normalized!r}: special file is denied"
                )
            entries.append(entry)
            if len(entries) > max_entries:
                raise WorkspacePolicyError(
                    f"workspace listing exceeds max_entries {max_entries}"
                )
            if not recursive or not stat.S_ISDIR(metadata.st_mode) or depth >= max_depth:
                continue
            try:
                child_fd = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                raise WorkspacePolicyError(
                    f"path {normalized!r}: directory binding changed"
                ) from exc
            try:
                opened = os.fstat(child_fd)
                if not _same_inode(metadata, opened):
                    raise WorkspacePolicyError(
                        f"path {normalized!r}: directory binding changed"
                    )
                self._collect_entries(
                    child_fd,
                    prefix=normalized,
                    recursive=True,
                    max_entries=max_entries,
                    max_depth=max_depth,
                    depth=depth + 1,
                    entries=entries,
                )
                self._assert_entry_binding(directory_fd, name, opened, normalized)
            finally:
                os.close(child_fd)

    def _assert_root_binding(self) -> None:
        if self._root_fd < 0:
            raise WorkspaceStateError("workspace authority is closed")
        try:
            descriptor_root = os.fstat(self._root_fd)
            current_root = self._root.lstat()
        except OSError as exc:
            raise WorkspacePolicyError("workspace root binding is unavailable") from exc
        if (
            not _same_inode(descriptor_root, current_root)
            or not stat.S_ISDIR(current_root.st_mode)
            or stat.S_ISLNK(current_root.st_mode)
        ):
            raise WorkspacePolicyError("workspace root binding changed")

    def _assert_parent_binding(self, path: str, parent_fd: int) -> None:
        parts = PurePosixPath(path).parts[:-1]
        descriptor = os.dup(self._root_fd)
        try:
            for component in parts:
                try:
                    next_descriptor = os.open(
                        component,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                except OSError as exc:
                    raise WorkspacePolicyError(
                        f"path {path!r}: parent binding changed or became a symlink"
                    ) from exc
                os.close(descriptor)
                descriptor = next_descriptor
            if not _same_inode(os.fstat(descriptor), os.fstat(parent_fd)):
                raise WorkspacePolicyError(f"path {path!r}: parent binding changed")
        finally:
            os.close(descriptor)

    def _assert_entry_binding(
        self,
        parent_fd: int,
        name: str,
        opened_metadata: os.stat_result,
        path: str,
    ) -> None:
        current = _lstat_at(parent_fd, name, path, missing_ok=False)
        assert current is not None
        if stat.S_ISLNK(current.st_mode) or not _same_inode(current, opened_metadata):
            raise WorkspacePolicyError(f"path {path!r}: file binding changed")

    def _require_writable(self, path: str) -> None:
        if not _path_in_scopes(path, self.policy.writable_paths):
            raise WorkspacePolicyError(f"path {path!r}: outside writable scope")

    def _validate_requested_mode(self, mode: int) -> None:
        if isinstance(mode, bool) or not isinstance(mode, int) or mode not in self.policy.allowed_modes:
            raise WorkspacePolicyError(f"mode {mode!r}: not allowed")

    def _atomic_replace(self, parent_fd: int, name: str, content: bytes, mode: int) -> None:
        temporary_name = f".opbench-write-{secrets.token_hex(12)}"
        descriptor = -1
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                mode,
                dir_fd=parent_fd,
            )
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
            raise

def _validate_scopes(scopes: tuple[str, ...], path: str) -> None:
    if not isinstance(scopes, tuple) or not scopes:
        raise WorkspacePolicyError(f"{path}: expected non-empty tuple")
    seen: set[str] = set()
    for index, scope in enumerate(scopes):
        if not isinstance(scope, str) or not scope:
            raise WorkspacePolicyError(f"{path}[{index}]: expected non-empty string")
        directory = scope.endswith("/")
        candidate = scope[:-1] if directory else scope
        try:
            normalized = _normalize_relative_path(candidate)
        except WorkspacePolicyError as exc:
            raise WorkspacePolicyError(f"{path}[{index}]: {exc}") from exc
        canonical = normalized + "/" if directory else normalized
        if canonical != scope:
            raise WorkspacePolicyError(f"{path}[{index}]: path is not canonical")
        if canonical in seen:
            raise WorkspacePolicyError(f"{path}: duplicate scope {canonical!r}")
        seen.add(canonical)


def _normalize_relative_path(path: str) -> str:
    if not isinstance(path, str) or not path:
        raise WorkspacePolicyError("path: expected non-empty relative POSIX path")
    if "\x00" in path or "\\" in path or any(ord(character) < 32 for character in path):
        raise WorkspacePolicyError("path: control characters and backslash are denied")
    if path.startswith(":") or any(character in path for character in "*?[]"):
        raise WorkspacePolicyError("path: Git pathspec magic and glob characters are denied")
    pure = PurePosixPath(path)
    if pure.is_absolute():
        raise WorkspacePolicyError("path: absolute paths are denied")
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise WorkspacePolicyError("path: traversal and non-canonical components are denied")
    if parts[0] == ".git" or ".git" in parts:
        raise WorkspacePolicyError("path: Git metadata is denied")
    normalized = pure.as_posix()
    if normalized != path:
        raise WorkspacePolicyError("path: expected canonical POSIX path")
    return normalized


def _path_in_scopes(path: str, scopes: tuple[str, ...]) -> bool:
    for scope in scopes:
        if scope.endswith("/"):
            if path.startswith(scope) and len(path) > len(scope):
                return True
        elif path == scope:
            return True
    return False


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _lstat_at(
    parent_fd: int,
    name: str,
    path: str,
    *,
    missing_ok: bool,
) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise WorkspacePolicyError(f"path {path!r}: does not exist")
    except OSError as exc:
        raise WorkspacePolicyError(
            f"path {path!r}: cannot inspect safely: {exc.strerror or 'denied'}"
        ) from exc


def _open_readonly_at(parent_fd: int, name: str, path: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    flags |= os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            reason = "symlink is denied"
        else:
            reason = exc.strerror or "denied"
        raise WorkspacePolicyError(
            f"path {path!r}: cannot open non-symlink regular file: {reason}"
        ) from exc


def _validate_regular_metadata(
    metadata: os.stat_result,
    policy: WorkspacePolicy,
    path: str,
) -> int:
    if not stat.S_ISREG(metadata.st_mode):
        raise WorkspacePolicyError(f"path {path!r}: expected regular file")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode not in policy.allowed_modes:
        raise WorkspacePolicyError(f"path {path!r}: mode {oct(mode)} is not allowed")
    if metadata.st_size > policy.max_file_bytes:
        raise WorkspacePolicyError(f"path {path!r}: exceeds max_file_bytes")
    return mode


def _read_all(descriptor: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    remaining = limit + 1
    while remaining > 0:
        chunk = os.read(descriptor, min(remaining, 64 * 1024))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    content = b"".join(chunks)
    if len(content) > limit:
        raise WorkspacePolicyError(f"file exceeds byte limit {limit}")
    return content


def _looks_binary(content: bytes) -> bool:
    return b"\x00" in content


def _reject_hidden_index_flags(root: Path) -> None:
    records = _git(root, "ls-files", "-v", "-z").stdout.split(b"\0")
    for record in records:
        if not record:
            continue
        if len(record) < 3 or record[1:2] != b" ":
            raise WorkspacePolicyError("Git index flag output is malformed")
        tag = record[0]
        if tag == ord("S") or ord("a") <= tag <= ord("z"):
            raise WorkspacePolicyError(
                "workspace Git index flags may not hide tracked content"
            )


def _commit_scope_snapshots(
    root: Path,
    base_commit: str,
    policy: WorkspacePolicy,
) -> dict[str, _FileSnapshot]:
    result = _git(
        root,
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        base_commit,
        "--",
        *policy.patch_paths,
    )
    snapshots: dict[str, _FileSnapshot] = {}
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise WorkspacePolicyError("base commit tree output is malformed")
        raw_mode, object_type, raw_object_id = fields
        try:
            path = raw_path.decode("utf-8", errors="strict")
            object_id = raw_object_id.decode("ascii", errors="strict")
            git_mode = int(raw_mode, 8)
        except (UnicodeDecodeError, ValueError) as exc:
            raise WorkspacePolicyError("base commit tree contains non-canonical data") from exc
        normalized = _normalize_relative_path(path)
        if normalized != path or not _path_in_scopes(normalized, policy.patch_paths):
            raise WorkspacePolicyError(
                f"base commit returned path outside patch scope: {normalized!r}"
            )
        if object_type != b"blob" or git_mode & 0o170000 != 0o100000:
            if git_mode == 0o120000:
                reason = "symlink mode is denied"
            else:
                reason = "non-regular entries are denied"
            raise WorkspacePolicyError(f"path {normalized!r}: {reason}")
        mode = git_mode & 0o777
        if mode not in policy.allowed_modes:
            raise WorkspacePolicyError(
                f"path {normalized!r}: mode {oct(mode)} is not allowed"
            )
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", object_id):
            raise WorkspacePolicyError(
                f"path {normalized!r}: invalid base blob object id"
            )
        size_output = _git(root, "cat-file", "-s", object_id).stdout.strip()
        try:
            size = int(size_output.decode("ascii", errors="strict"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise WorkspacePolicyError(
                f"path {normalized!r}: invalid base blob size"
            ) from exc
        if size < 0 or size > policy.max_file_bytes:
            raise WorkspacePolicyError(f"path {normalized!r}: exceeds max_file_bytes")
        content = _git(root, "cat-file", "blob", object_id).stdout
        if len(content) != size or _git_blob_object_id(content, len(object_id)) != object_id:
            raise WorkspacePolicyError(
                f"path {normalized!r}: base blob content identity mismatch"
            )
        if not policy.allow_binary and _looks_binary(content):
            raise WorkspacePolicyError(f"path {normalized!r}: binary file is denied")
        if normalized in snapshots:
            raise WorkspacePolicyError(f"base commit contains duplicate path {normalized!r}")
        snapshots[normalized] = _FileSnapshot(content=content, mode=mode)
    return snapshots


def _git_blob_object_id(content: bytes, object_id_length: int) -> str:
    if object_id_length == 40:
        digest = hashlib.sha1()
    elif object_id_length == 64:
        digest = hashlib.sha256()
    else:
        raise WorkspacePolicyError("unsupported Git object id length")
    digest.update(f"blob {len(content)}\0".encode("ascii"))
    digest.update(content)
    return digest.hexdigest()


def _git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return _run_git(
        (
            "git",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.filemode=true",
            "-c",
            "core.quotePath=true",
            "-C",
            str(root),
            *arguments,
        ),
        check=check,
    )


def _run_git(
    command: tuple[str, ...],
    *,
    check: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("GIT_CONFIG_KEY_") or name.startswith("GIT_CONFIG_VALUE_"):
            environment.pop(name)
    for name in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG_COUNT",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_WORK_TREE",
    ):
        environment.pop(name, None)
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_ATTR_NOSYSTEM"] = "1"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
    except OSError as exc:
        raise WorkspaceError(f"cannot execute local Git: {exc}") from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkspaceError(f"local Git failed ({' '.join(command[1:])}): {detail}")
    return result


def _patch_paths_from_bytes(patch_bytes: bytes) -> tuple[str, ...]:
    if not patch_bytes:
        return ()
    with tempfile.TemporaryDirectory(prefix="opbench-patch-parse-") as temporary:
        parse_root = Path(temporary) / "parse"
        parse_root.mkdir()
        patch_path = Path(temporary) / "candidate.patch"
        patch_path.write_bytes(patch_bytes)
        result = _git(
            parse_root,
            "apply",
            "--numstat",
            "-z",
            str(patch_path),
            check=False,
        )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise WorkspacePolicyError(f"canonical patch is not parseable: {detail}")
    paths: set[str] = set()
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        fields = record.split(b"\t", 2)
        if len(fields) != 3:
            raise WorkspacePolicyError("canonical patch numstat is malformed")
        try:
            decoded = fields[2].decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise WorkspacePolicyError("canonical patch path is not UTF-8") from exc
        paths.add(_normalize_relative_path(decoded))
    return tuple(sorted(paths))


def _materialize_snapshots(
    root: Path,
    snapshots: Mapping[str, _FileSnapshot],
) -> None:
    for path in sorted(snapshots):
        snapshot = snapshots[path]
        candidate = root.joinpath(*PurePosixPath(path).parts)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(snapshot.content)
        candidate.chmod(snapshot.mode)


def _validate_patch_modes(patch_bytes: bytes, policy: WorkspacePolicy) -> None:
    mode_lines = re.findall(
        rb"^(?:old mode|new mode|new file mode|deleted file mode) ([0-7]{6})$",
        patch_bytes,
        flags=re.MULTILINE,
    )
    for raw_mode in mode_lines:
        git_mode = int(raw_mode, 8)
        if git_mode == 0o120000:
            raise WorkspacePolicyError("patch: symlink mode is denied")
        if git_mode & 0o170000 != 0o100000:
            raise WorkspacePolicyError(f"patch: non-regular mode {raw_mode.decode()} is denied")
        file_mode = git_mode & 0o777
        if file_mode not in policy.allowed_modes:
            raise WorkspacePolicyError(
                f"patch: mode {oct(file_mode)} is not allowed"
            )


def _require_identity(value: object, identity_type: str, path: str) -> ContentIdentity:
    if not isinstance(value, ContentIdentity) or value.identity_type != identity_type:
        raise ContractError(f"{path}: expected {identity_type} identity")
    return value


def raw_patch_identity(patch_bytes: bytes, *, identifier: str) -> ContentIdentity:
    if not isinstance(patch_bytes, bytes):
        raise WorkspacePolicyError("patch_bytes: expected bytes")
    try:
        require_str(identifier, "identifier")
    except ContractError as exc:
        raise WorkspacePolicyError(str(exc)) from exc
    return ContentIdentity(
        identity_type="patch",
        identifier=identifier,
        digest="sha256:" + hashlib.sha256(patch_bytes).hexdigest(),
        digest_kind="content_sha256",
    )


def build_patch_artifact(frozen: FrozenPatch, *, artifact_id: str) -> PatchArtifact:
    if not isinstance(frozen, FrozenPatch):
        raise ContractError("frozen: expected FrozenPatch")
    return PatchArtifact(
        artifact_id=artifact_id,
        workspace=frozen.workspace,
        patch=frozen.patch,
        size_bytes=len(frozen.patch_bytes),
        changed_paths=frozen.changed_paths,
        empty=frozen.empty,
    )


def assert_patch_identity_handoff(
    *,
    frozen: FrozenPatch,
    session_result: SessionResult,
    patch_artifact: PatchArtifact,
    evaluation_spec: EvaluationSpec,
) -> None:
    if not isinstance(frozen, FrozenPatch):
        raise ContractError("frozen: expected FrozenPatch")
    if not isinstance(session_result, SessionResult):
        raise ContractError("session_result: expected SessionResult")
    if not isinstance(patch_artifact, PatchArtifact):
        raise ContractError("patch_artifact: expected PatchArtifact")
    if not isinstance(evaluation_spec, EvaluationSpec):
        raise ContractError("evaluation_spec: expected EvaluationSpec")
    identities = (
        frozen.patch,
        session_result.final_patch,
        patch_artifact.patch,
        evaluation_spec.frozen_patch,
    )
    if any(identity != frozen.patch for identity in identities):
        raise ContractError("patch identity mismatch across frozen/session/artifact/evaluation")
    if patch_artifact.workspace != frozen.workspace:
        raise ContractError("workspace identity mismatch for patch artifact")
    if patch_artifact.size_bytes != len(frozen.patch_bytes):
        raise ContractError("patch artifact size mismatch")
    if patch_artifact.changed_paths != frozen.changed_paths:
        raise ContractError("patch artifact changed_paths mismatch")
    if patch_artifact.empty != frozen.empty:
        raise ContractError("patch artifact empty state mismatch")


__all__ = [
    "AuthoritativeWorkspace",
    "FrozenPatch",
    "PatchArtifact",
    "WorkspaceBinding",
    "WorkspaceDiff",
    "WorkspaceEntry",
    "WorkspaceError",
    "WorkspaceMutation",
    "WorkspacePatchMutation",
    "WorkspacePolicy",
    "WorkspacePolicyError",
    "WorkspaceRead",
    "WorkspaceStateError",
    "assert_patch_identity_handoff",
    "build_patch_artifact",
    "raw_patch_identity",
]
