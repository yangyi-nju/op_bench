"""Software provenance is recorded separately from task and scoring revisions."""
from pathlib import Path
import subprocess
import tomllib


SOFTWARE_VERSION = "0.8.0.dev1"


def _checkout_root() -> Path | None:
    module = Path(__file__).resolve()
    # An installed package can live in .venv/site-packages underneath an
    # unrelated Git project (or underneath an OpBench checkout itself). Only
    # OpBench's declared source layout is evidence of a source installation.
    if tuple(module.parts[-3:]) != ("src", "op_bench", "provenance.py"):
        return None
    root = module.parents[2]
    try:
        with (root / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
    except (OSError, ValueError):
        return None
    name = project.get("name") if isinstance(project, dict) else None
    if not isinstance(name, str) or name.casefold() != "opbench":
        return None
    # A worktree's .git is a file. Do not walk upward if this root has neither
    # its own repository nor a worktree declaration.
    return root if (root / ".git").exists() else None


def software_identity() -> dict:
    result = {"version": SOFTWARE_VERSION, "git_revision": None,
              "working_tree_modified": None, "installation": "package"}
    directory = _checkout_root()
    if directory is not None:
        try:
            top = subprocess.run(["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                                 capture_output=True, text=True, check=True, timeout=10)
            if Path(top.stdout.strip()).resolve() != directory:
                return result
            revision = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True, timeout=10)
            changed = subprocess.run(["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=normal"],
                                     capture_output=True, text=True, check=True, timeout=10)
            result.update(git_revision=revision.stdout.strip(), working_tree_modified=bool(changed.stdout),
                          installation="checkout")
        except (OSError, subprocess.SubprocessError):
            result["installation"] = "checkout_unavailable"
    return result
