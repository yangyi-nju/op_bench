#!/usr/bin/env python

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.task import TaskManifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable source snapshot for one or more op_bench tasks."
    )
    parser.add_argument(
        "--task",
        action="append",
        required=True,
        help="Task directory containing task.json. May be provided multiple times.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".op_bench_cache/sources",
        help="Root directory for source snapshots.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate an existing snapshot.",
    )
    parser.add_argument(
        "--write-task",
        action="store_true",
        help="Write source.snapshot_path back into each task manifest.",
    )
    parser.add_argument(
        "--from-local-repo",
        help="Use an existing local repository checkout as the source instead of downloading an archive.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file for snapshot preparation evidence.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cache_dir = Path(args.cache_dir).resolve()
    records: list[dict[str, object]] = []

    for task_dir in args.task:
        task_path = Path(task_dir).resolve() / "task.json"
        task = TaskManifest.load(task_path)
        destination = snapshot_destination(cache_dir, task)
        record = prepare_snapshot(
            task,
            task_path,
            destination,
            force=args.force,
            write_task=args.write_task,
            local_repo=Path(args.from_local_repo).resolve() if args.from_local_repo else None,
        )
        records.append(record)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(records, sort_keys=True))
    return 0 if all(record["status"] == "ready" for record in records) else 1


def snapshot_destination(cache_dir: Path, task: TaskManifest) -> Path:
    repo = str(task.data["source"]["repo"]).replace(":", "/").strip("/")
    safe_repo = Path(*[part for part in repo.split("/") if part])
    return cache_dir / safe_repo / task.base_commit / "source"


def prepare_snapshot(
    task: TaskManifest,
    task_path: Path,
    destination: Path,
    *,
    force: bool,
    write_task: bool,
    local_repo: Path | None,
) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    if destination.exists() and force:
        shutil.rmtree(destination)

    if destination.exists():
        status = "ready"
    elif local_repo is not None:
        local_record = prepare_snapshot_from_local_repo(task, local_repo, destination)
        commands.extend(local_record["commands"])
        if local_record["status"] != "ready":
            return record(task, destination, str(local_record["status"]), commands)
        status = "ready"
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        archive_path = destination.parent / "source.tar.gz"
        with tempfile.TemporaryDirectory(prefix=f"op-bench-source-{task.task_id}-") as tmp:
            tmp_path = Path(tmp)
            download = download_archive(task, archive_path)
            commands.append(download)
            if download["exit_code"] != 0:
                return record(task, destination, "download_failed", commands)

            extract_dir = tmp_path / "extract"
            extract_dir.mkdir()
            try:
                extracted_root = extract_archive(archive_path, extract_dir)
                shutil.move(str(extracted_root), str(destination))
            except (OSError, tarfile.TarError) as exc:
                commands.append(
                    {
                        "command": ["op_bench", "extract-source-archive", str(archive_path), str(destination)],
                        "cwd": str(tmp_path),
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": str(exc),
                    }
                )
                return record(task, destination, "extract_failed", commands)

        for command in init_snapshot_git(destination):
            commands.append(command)
            if command["exit_code"] != 0:
                return record(task, destination, "git_init_failed", commands)
        status = "ready"

    if write_task:
        write_snapshot_path(task_path, destination)

    return record(task, destination, status, commands)


def prepare_snapshot_from_local_repo(
    task: TaskManifest,
    local_repo: Path,
    destination: Path,
) -> dict[str, object]:
    commands: list[dict[str, object]] = []
    if not local_repo.exists():
        return record(task, destination, "local_repo_missing", commands)
    if not (local_repo / ".git").exists():
        return record(task, destination, "local_repo_not_git", commands)

    if not commit_exists(local_repo, task.base_commit):
        result = run_command(
            ["git", "fetch", "--depth=1", "origin", task.base_commit],
            local_repo,
            timeout_sec=task.timeout_sec,
        )
        commands.append(result)
        if result["exit_code"] != 0:
            return record(task, destination, "local_repo_checkout_failed", commands)

    for command in [
        ["git", "-c", "advice.detachedHead=false", "checkout", "--detach", task.base_commit],
    ]:
        result = run_command(command, local_repo, timeout_sec=task.timeout_sec)
        commands.append(result)
        if result["exit_code"] != 0:
            return record(task, destination, "local_repo_checkout_failed", commands)

    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_command = [
        "rsync",
        "-a",
        "--delete",
        "--exclude",
        ".git",
        f"{local_repo}/",
        f"{destination}/",
    ]
    result = run_command(copy_command, ROOT, timeout_sec=task.timeout_sec)
    commands.append(result)
    if result["exit_code"] != 0:
        return record(task, destination, "local_repo_copy_failed", commands)

    for command in init_snapshot_git(destination):
        commands.append(command)
        if command["exit_code"] != 0:
            return record(task, destination, "git_init_failed", commands)
    return record(task, destination, "ready", commands)


def commit_exists(local_repo: Path, commit: str) -> bool:
    result = run_command(["git", "cat-file", "-e", f"{commit}^{{commit}}"], local_repo)
    return result["exit_code"] == 0


def download_archive(task: TaskManifest, archive_path: Path) -> dict[str, object]:
    try:
        url = github_archive_url(task)
        if shutil.which("curl"):
            command = [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                "--continue-at",
                "-",
                "--output",
                str(archive_path),
                url,
            ]
            completed = subprocess.run(
                command,
                cwd=archive_path.parent,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=task.timeout_sec,
                check=False,
            )
            return {
                "command": command,
                "cwd": str(archive_path.parent),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        command = ["op_bench", "download-source-archive", url, str(archive_path)]
        request = urllib.request.Request(url, headers={"User-Agent": "op_bench-source-snapshot/0.1"})
        with urllib.request.urlopen(request, timeout=task.timeout_sec) as response:
            with archive_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        return {
            "command": command,
            "cwd": str(archive_path.parent),
            "exit_code": 0,
            "stdout": f"downloaded {archive_path.stat().st_size} bytes\n",
            "stderr": "",
        }
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, http.client.IncompleteRead, subprocess.TimeoutExpired) as exc:
        return {
            "command": ["op_bench", "download-source-archive", str(task.repo_url), str(archive_path)],
            "cwd": str(archive_path.parent),
            "exit_code": 1,
            "stdout": "",
            "stderr": str(exc),
        }


def github_archive_url(task: TaskManifest) -> str:
    repo_url = task.repo_url
    parsed = urlparse(repo_url)
    if parsed.netloc != "github.com":
        raise ValueError(f"source snapshot archive is only implemented for GitHub HTTPS repos: {repo_url}")
    repo = parsed.path.strip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return f"https://github.com/{repo}/archive/{task.base_commit}.tar.gz"


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(extract_dir)
    children = [child for child in extract_dir.iterdir() if child.is_dir()]
    if len(children) != 1:
        raise OSError(f"expected one extracted directory in {extract_dir}, found {len(children)}")
    return children[0]


def init_snapshot_git(destination: Path) -> list[dict[str, object]]:
    commands = [
        ["git", "init", "--quiet"],
        ["git", "config", "user.name", "op_bench"],
        ["git", "config", "user.email", "op_bench@example.invalid"],
        ["git", "add", "-A"],
        ["git", "commit", "--quiet", "-m", "source snapshot"],
    ]
    results: list[dict[str, object]] = []
    for command in commands:
        results.append(run_command(command, destination))
    return results


def run_command(command: list[str], cwd: Path, timeout_sec: int | None = None) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "exit_code": 124,
            "stdout": exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }


def write_snapshot_path(task_path: Path, destination: Path) -> None:
    data = json.loads(task_path.read_text(encoding="utf-8"))
    relative = os.path.relpath(destination, start=task_path.parent)
    data["source"]["snapshot_path"] = relative
    task_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record(
    task: TaskManifest,
    destination: Path,
    status: str,
    commands: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "status": status,
        "snapshot_path": str(destination),
        "commands": commands,
    }


if __name__ == "__main__":
    raise SystemExit(main())
