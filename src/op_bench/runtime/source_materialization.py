from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


@dataclass(frozen=True)
class FrozenSourceSnapshot:
    workspace: Path
    revision: str
    source_tree: str
    snapshot_commit: str


class SourceMaterializationError(Exception):
    """An exact local Git revision could not become an isolated workspace."""


def _validate_archive_entry(name: str, link_target: str | None = None) -> None:
    """Reject archive names or symlink targets that can escape the workspace."""
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise SourceMaterializationError("archive entry has an unsafe name")
    normalized_name = name[:-1] if name.endswith("/") else name
    if not normalized_name:
        raise SourceMaterializationError("archive entry has an empty name")
    entry = PurePosixPath(normalized_name)
    if entry.is_absolute() or any(
        part in ("", ".", "..") for part in normalized_name.split("/")
    ):
        raise SourceMaterializationError("archive entry escapes the workspace")
    if entry.as_posix() != normalized_name:
        raise SourceMaterializationError("archive entry is not normalized")
    if link_target is None:
        return
    if (
        not isinstance(link_target, str)
        or not link_target
        or "\x00" in link_target
        or "\\" in link_target
        or PurePosixPath(link_target).is_absolute()
    ):
        raise SourceMaterializationError("archive symlink has an unsafe target")
    contained_parts = list(entry.parent.parts)
    for part in link_target.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not contained_parts:
                raise SourceMaterializationError("archive symlink escapes the workspace")
            contained_parts.pop()
            continue
        contained_parts.append(part)


def materialize_frozen_git_revision(
    source_directory: Path,
    revision: str,
    workspace: Path,
    *,
    include_submodules: bool = False,
) -> FrozenSourceSnapshot:
    """Create one clean standalone Git workspace from an exact local revision."""
    if not isinstance(source_directory, Path):
        raise SourceMaterializationError("source directory must be a pathlib.Path")
    if source_directory.is_symlink() or not source_directory.is_dir():
        raise SourceMaterializationError("source directory must be a real directory")
    if not isinstance(revision, str) or not revision or "\x00" in revision:
        raise SourceMaterializationError("source revision must be a non-empty string")
    if not isinstance(workspace, Path):
        raise SourceMaterializationError("workspace must be a pathlib.Path")
    if workspace.exists() or workspace.is_symlink():
        raise SourceMaterializationError("workspace already exists")
    if not isinstance(include_submodules, bool):
        raise SourceMaterializationError("include_submodules must be a bool")

    try:
        resolved_commit = _git_text(
            source_directory,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        )
        source_tree = _git_text(
            source_directory,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{resolved_commit}^{{tree}}",
        )
        workspace.mkdir(parents=True, exist_ok=False)
        _stream_safe_git_archive(source_directory, resolved_commit, workspace)
        _git(workspace, "init", "--quiet", "--initial-branch=main")
        _git(workspace, "add", "--force", "--all")
        _restore_gitlink_index_entries(
            source_directory,
            resolved_commit,
            workspace,
        )
        snapshot_tree = _git_text(workspace, "write-tree")
        if snapshot_tree != source_tree:
            raise SourceMaterializationError(
                "materialized Git tree does not match frozen revision"
            )
        snapshot_commit = _install_frozen_commit(
            source_directory,
            workspace,
            resolved_commit,
        )
        _git(workspace, "update-ref", "HEAD", snapshot_commit)
        if include_submodules:
            _materialize_frozen_submodules(
                source_directory,
                resolved_commit,
                workspace,
            )
        if _git_bytes(
            workspace,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--ignored=matching",
            "--ignore-submodules=all",
        ):
            raise SourceMaterializationError("materialized workspace is not clean")
        return FrozenSourceSnapshot(
            workspace=workspace,
            revision=resolved_commit,
            source_tree=source_tree,
            snapshot_commit=snapshot_commit,
        )
    except Exception as exc:  # noqa: BLE001 - normalize the private failure boundary.
        _remove_exact_workspace(workspace)
        if isinstance(exc, SourceMaterializationError):
            raise
        detail = str(exc).strip()
        message = "frozen source materialization failed"
        if detail:
            message = f"{message}: {detail}"
        raise SourceMaterializationError(message) from exc


def _stream_safe_git_archive(
    source_directory: Path,
    revision: str,
    workspace: Path,
) -> None:
    process = subprocess.Popen(
        (
            "git",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(source_directory),
            "archive",
            "--format=tar",
            revision,
        ),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                _extract_archive_member(archive, member, workspace)
        stderr = process.stderr.read()
        return_code = process.wait()
    except Exception:
        process.stdout.close()
        process.terminate()
        process.wait()
        raise
    finally:
        process.stdout.close()
        process.stderr.close()
    if return_code != 0:
        raise SourceMaterializationError(
            "git archive failed: " + stderr.decode("utf-8", errors="replace").strip()
        )


def _extract_archive_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    workspace: Path,
) -> None:
    link_target = member.linkname if member.issym() else None
    _validate_archive_entry(member.name, link_target)
    relative = PurePosixPath(member.name.rstrip("/"))
    destination = workspace.joinpath(*relative.parts)
    _ensure_real_parent_directories(workspace, destination.parent)

    if member.isdir():
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise SourceMaterializationError("archive directory conflicts with an entry")
        destination.mkdir(exist_ok=True)
        destination.chmod(0o755)
        return
    if destination.exists() or destination.is_symlink():
        raise SourceMaterializationError("archive contains a duplicate entry")
    if member.isreg():
        source = archive.extractfile(member)
        if source is None:
            raise SourceMaterializationError("archive regular file has no content")
        with source, destination.open("xb") as target:
            shutil.copyfileobj(source, target)
        destination.chmod(0o755 if member.mode & 0o111 else 0o644)
        return
    if member.issym():
        destination.symlink_to(member.linkname)
        return
    raise SourceMaterializationError("archive contains an unsupported entry type")


def _ensure_real_parent_directories(workspace: Path, parent: Path) -> None:
    relative = parent.relative_to(workspace)
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SourceMaterializationError("archive entry has a symlink parent")
        if current.exists():
            if not current.is_dir():
                raise SourceMaterializationError("archive entry parent is not a directory")
        else:
            current.mkdir()


def _install_frozen_commit(
    source_directory: Path,
    workspace: Path,
    revision: str,
) -> str:
    commit_content = _git(
        source_directory,
        "cat-file",
        "commit",
        revision,
    ).stdout
    installed = _git(
        workspace,
        "hash-object",
        "-t",
        "commit",
        "-w",
        "--stdin",
        input_bytes=commit_content,
    ).stdout.decode("ascii").strip()
    if installed != revision:
        raise SourceMaterializationError("installed commit does not match frozen revision")
    commit_headers = commit_content.split(b"\n\n", 1)[0].splitlines()
    if any(line.startswith(b"parent ") for line in commit_headers):
        (workspace / ".git" / "shallow").write_text(
            revision + "\n",
            encoding="ascii",
        )
    return installed


def _restore_gitlink_index_entries(
    source_directory: Path,
    revision: str,
    workspace: Path,
) -> None:
    for object_id, path in _gitlink_entries(source_directory, revision):
        _git(
            workspace,
            "update-index",
            "--add",
            "--cacheinfo",
            "160000",
            object_id,
            path,
        )


def _materialize_frozen_submodules(
    source_directory: Path,
    revision: str,
    workspace: Path,
) -> None:
    for object_id, path in _gitlink_entries(source_directory, revision):
        source_submodule = _real_submodule_directory(source_directory, path)
        try:
            resolved_commit = _git_text(
                source_submodule,
                "rev-parse",
                "--verify",
                "--end-of-options",
                f"{object_id}^{{commit}}",
            )
        except subprocess.CalledProcessError as exc:
            raise SourceMaterializationError(
                f"frozen submodule commit is unavailable: {path}"
            ) from exc
        if resolved_commit != object_id:
            raise SourceMaterializationError(
                f"frozen submodule commit does not match Gitlink: {path}"
            )
        relative = PurePosixPath(path)
        destination = workspace.joinpath(*relative.parts)
        _ensure_real_parent_directories(workspace, destination.parent)
        if destination.is_symlink() or (
            destination.exists() and not destination.is_dir()
        ):
            raise SourceMaterializationError(
                f"submodule destination is not a real directory: {path}"
            )
        destination.mkdir(exist_ok=True)
        if any(destination.iterdir()):
            raise SourceMaterializationError(
                f"submodule destination is not empty: {path}"
            )
        _stream_safe_git_archive(source_submodule, object_id, destination)
        _materialize_frozen_submodules(
            source_submodule,
            object_id,
            destination,
        )


def _real_submodule_directory(source_directory: Path, path: str) -> Path:
    current = source_directory
    for part in PurePosixPath(path).parts:
        current = current / part
        if current.is_symlink():
            raise SourceMaterializationError(
                f"source submodule path contains a symlink: {path}"
            )
    if not current.is_dir():
        raise SourceMaterializationError(
            f"frozen source submodule is not populated: {path}"
        )
    return current


def _gitlink_entries(
    source_directory: Path,
    revision: str,
) -> tuple[tuple[str, str], ...]:
    gitlinks: list[tuple[str, str]] = []
    entries = _git_bytes(
        source_directory,
        "ls-tree",
        "-r",
        "-z",
        revision,
    )
    for encoded_entry in entries.split(b"\x00"):
        if not encoded_entry:
            continue
        try:
            header, encoded_path = encoded_entry.split(b"\t", 1)
            mode, object_type, encoded_object_id = header.split(b" ", 2)
        except ValueError as exc:
            raise SourceMaterializationError("Git tree entry is malformed") from exc
        if mode != b"160000":
            continue
        if object_type != b"commit":
            raise SourceMaterializationError("Gitlink entry has an invalid object type")
        try:
            path = encoded_path.decode("utf-8")
            object_id = encoded_object_id.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SourceMaterializationError("Gitlink entry is not decodable") from exc
        _validate_archive_entry(path)
        if not object_id or any(
            character not in "0123456789abcdef" for character in object_id
        ):
            raise SourceMaterializationError("Gitlink object id is not canonical")
        gitlinks.append((object_id, path))
    return tuple(gitlinks)


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("ascii").strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    return _git(repository, *arguments).stdout


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = _git_environment()
    if extra_environment is not None:
        environment.update(extra_environment)
    return subprocess.run(
        (
            "git",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(repository),
            *arguments,
        ),
        check=True,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def _git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "LC_ALL": "C",
        "LANG": "C",
    }


def _remove_exact_workspace(workspace: Path) -> None:
    if workspace.is_symlink() or (workspace.exists() and not workspace.is_dir()):
        workspace.unlink()
    elif workspace.is_dir():
        shutil.rmtree(workspace)
