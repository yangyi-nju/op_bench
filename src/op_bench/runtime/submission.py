"""Create a clean source baseline and freeze final workspace changes.

The capture database stays on the controller, outside the solver workspace.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


def _git(workspace: Path, *args: str, database: Path | None = None, index: Path | None = None,
         input_text: str | None = None, accepted_exit_codes: tuple[int, ...] = (0,)) -> str:
    command = ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
               "-c", "user.name=OpBench", "-c", "user.email=opbench@example.invalid"]
    if database is not None:
        command += ["--git-dir", str(database), "--work-tree", str(workspace)]
    command.extend(args)
    variables = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    variables.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    if index is not None:
        variables["GIT_INDEX_FILE"] = str(index)
    result = subprocess.run(command, cwd=workspace, env=variables, capture_output=True,
                            text=True, errors="replace", timeout=120, input=input_text)
    if result.returncode not in accepted_exit_codes:
        raise RuntimeError(f"Git workspace operation failed: {result.stderr.strip()}")
    return result.stdout


def _force_add(workspace: Path, paths: list[str], database: Path | None = None) -> None:
    if not paths:
        return
    # update-index consumes literal filenames, not pathspecs matched against
    # every tree entry. This remains linear for complete framework sources and
    # deliberately includes selected source even when an ignore rule matches.
    _git(workspace, "update-index", "--add", "-z", "--stdin", database=database,
         input_text="\0".join(paths) + "\0")


def _write_workspace_excludes(database: Path, names: tuple[str, ...]) -> None:
    # Keep only the few reserved subtrees here. Tens of thousands of generated
    # filenames turn ordinary Git status/ls-files into repeated pattern scans;
    # the controller filters those exact paths with a set during final capture.
    lines = []
    for name in names:
        if "\n" in name or "\r" in name:
            raise ValueError("Baseline artifact filenames cannot contain newlines")
        escaped = "".join("\\" + char if char in "\\*?[]#! " else char for char in name)
        lines.append("/" + escaped)
    directory = database / "info"
    directory.mkdir(exist_ok=True)
    (directory / "exclude").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _snapshot_ignore_rules(workspace: Path, database: Path, reserved: tuple[str, ...]) -> None:
    """Preserve Git's own hierarchical ignore semantics before the Agent runs."""
    destination = database / "submission-ignore"
    destination.mkdir()
    for directory, children, files in os.walk(workspace, followlinks=False):
        relative = Path(directory).relative_to(workspace)
        children[:] = [name for name in children if name != ".git" and not any(
            (relative / name).as_posix() == prefix or (relative / name).as_posix().startswith(prefix + "/")
            for prefix in reserved)]
        if ".gitignore" not in files:
            continue
        source = Path(directory) / ".gitignore"
        if source.is_symlink():
            continue  # Git does not follow .gitignore symlinks.
        target = destination / relative / ".gitignore"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def initialize_workspace(workspace: Path, database: Path, original_paths: list[str], *,
                reserved: tuple[str, ...] = (".opbench",)) -> None:
    # The Agent gets a normal one-commit repository. A second Git database,
    # outside the mounted workspace, controls final patch capture.
    present = [name for name in original_paths if (workspace / name).exists() or (workspace / name).is_symlink()]
    _git(workspace, "init", "--quiet")
    _write_workspace_excludes(workspace / ".git", reserved)
    _force_add(workspace, present)
    _git(workspace, "commit", "--quiet", "--allow-empty", "-m", "Task baseline")
    _git(workspace, "init", "--bare", "--quiet", str(database))
    _write_workspace_excludes(database, reserved)
    _force_add(workspace, present, database)
    _git(workspace, "commit", "--quiet", "--allow-empty", "-m", "Evaluation baseline", database=database)
    _snapshot_ignore_rules(workspace, database, reserved)


def freeze_patch(workspace: Path, database: Path, *, reserved: tuple[str, ...] = (".opbench",),
            generated: tuple[str, ...] = ()) -> str:
    generated_paths = set(generated)
    def excluded(name: str) -> bool:
        return (name == ".git" or name.startswith(".git/") or name in generated_paths
                or any(name == prefix or name.startswith(prefix + "/") for prefix in reserved))
    # The controller's original index always defines existing source. New
    # filenames are checked using the initial ignore files in a separate, tiny
    # worktree: editing/deleting/adding .gitignore cannot change capture policy.
    # One whole-tree update captures tracked edits, deletions and mode changes
    # regardless of the Agent's ignore/index changes, without a per-file
    # pathspec list. The controller index contains only original source here.
    _git(workspace, "add", "--update", "--", ".", database=database)
    paths = _git(workspace, "ls-files", "--others", "-z",
                 "--exclude-from=" + str(database / "info/exclude"), database=database).split("\0")
    new = sorted({name for name in paths if name and not excluded(name)})
    ignored = set()
    if new:
        ignored = set(_git(database / "submission-ignore", "check-ignore", "--no-index", "--stdin", "-z",
                           database=database, input_text="\0".join(new) + "\0",
                           accepted_exit_codes=(0, 1)).split("\0"))
    _force_add(workspace, sorted(set(new) - ignored), database)
    # Honor normal `git add -f` for intentionally submitted ignored files.
    # Read only the Agent's index using the controller's own Git configuration;
    # never trust the Agent's HEAD, hooks, repository config or commit history.
    agent_git = workspace / ".git"
    agent_index = agent_git / "index"
    if agent_git.is_dir() and not agent_git.is_symlink() and agent_index.is_file() and not agent_index.is_symlink():
        private_index = database / "agent-index.snapshot"
        shutil.copyfile(agent_index, private_index)
        agent_paths = _git(workspace, "ls-files", "--cached", "-z", database=database, index=private_index).split("\0")
        collected = set(_git(workspace, "ls-files", "--cached", "-z", database=database).split("\0"))
        generated_paths = set(generated)
        additional = []
        for name in agent_paths:
            path = Path(name)
            if (not name or name in collected or name in generated_paths or path.parts[0] == ".git"
                    or any(name == prefix or name.startswith(prefix + "/") for prefix in reserved)):
                continue
            if path.is_absolute() or ".." in path.parts or not (workspace / path).parent.resolve().is_relative_to(workspace):
                raise ValueError("Agent index contains a path outside the task workspace")
            if (workspace / path).exists() or (workspace / path).is_symlink():
                additional.append(name)
        _force_add(workspace, additional, database)
    return _git(workspace, "diff", "--cached", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", database=database)
