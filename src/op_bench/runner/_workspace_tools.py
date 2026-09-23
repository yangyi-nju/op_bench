"""Trusted stdlib helper sent into the existing task session, never a grader."""
import json
import os
from pathlib import Path
import subprocess
import sys


def execute(root, name, args, reserved):
    root = Path(root).resolve()

    def path(value, *, write=False):
        if not isinstance(value, str) or not value or Path(value).is_absolute():
            raise ValueError("path must be relative to the workspace")
        raw = Path(value)
        if ".." in raw.parts or ".git" in raw.parts:
            raise ValueError("path leaves the public workspace")
        selected = root / raw
        resolved = selected.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("symlink leaves the workspace")
        if ".git" in resolved.relative_to(root).parts:
            raise ValueError("path resolves to Git internals")
        for candidate in (raw.as_posix(), resolved.relative_to(root).as_posix()):
            if any(candidate == item or candidate.startswith(item + "/")
                   for item in reserved if item != ".opbench" or write):
                raise ValueError("reserved runtime path")
        return selected

    def files(base):
        for directory, dirs, names in os.walk(base, followlinks=False):
            allowed = []
            for child in dirs:
                item = Path(directory) / child
                try:
                    checked = path(item.relative_to(root).as_posix())
                except ValueError:
                    continue
                if not checked.is_symlink():
                    allowed.append(child)
            dirs[:] = sorted(allowed)
            for name in sorted(names):
                item = Path(directory) / name
                try:
                    checked = path(item.relative_to(root).as_posix())
                except ValueError:
                    continue
                if checked.is_file() and not checked.is_symlink():
                    yield checked

    if name == "list_files":
        base = path(args.get("path", "."))
        offset, limit = args.get("offset", 0), args.get("limit", 100)
        selected = []
        for index, item in enumerate(files(base)):
            if index < offset:
                continue
            if len(selected) == limit:
                return {"files": selected, "next_offset": offset + limit}
            selected.append(item.relative_to(root).as_posix())
        return {"files": selected, "next_offset": None}
    if name == "read_file":
        selected = path(args["path"])
        start, count = args.get("start_line", 1), args.get("max_lines", 200)
        lines, size, more = [], 0, False
        with selected.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if number < start:
                    continue
                if len(lines) >= count or size + len(line) > 24000:
                    more = True
                    break
                lines.append(f"{number}: {line.rstrip()}")
                size += len(line)
        return {"text": "\n".join(lines), "truncated": more}
    if name == "search":
        matches = []
        for selected in files(path(args.get("path", "."))):
            if selected.stat().st_size > 2 * 1024 * 1024:
                continue
            try:
                with selected.open(encoding="utf-8") as stream:
                    for number, line in enumerate(stream, 1):
                        if args["query"] in line:
                            matches.append({"path": selected.relative_to(root).as_posix(),
                                            "line": number, "text": line.rstrip()[:500]})
                            if len(matches) >= args.get("limit", 40):
                                return {"matches": matches, "truncated": True}
            except UnicodeError:
                continue
        return {"matches": matches, "truncated": False}
    if name in {"write_file", "replace_text", "delete_file"}:
        selected = path(args["path"], write=True)
        if selected.is_symlink():
            raise ValueError("editing symlinks is unsupported")
        if name == "delete_file":
            selected.unlink()
        else:
            if name == "replace_text":
                text = selected.read_text(encoding="utf-8")
                if text.count(args["old_text"]) != 1:
                    raise ValueError("old_text must match exactly once; read the file first")
                text = text.replace(args["old_text"], args["new_text"], 1)
            else:
                text = args["content"]
            selected.parent.mkdir(parents=True, exist_ok=True)
            selected.write_text(text, encoding="utf-8")
            if (root / ".git").is_dir():
                # Explicit tool edits are submissions, including new source
                # matched by an upstream ignore rule. Final capture still
                # excludes reserved paths and baseline-generated artifacts.
                subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "add", "-f", "--",
                                selected.relative_to(root).as_posix()], cwd=root,
                               check=True, capture_output=True, timeout=10)
        return {"updated": args["path"]}
    if name == "diff":
        result = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "diff",
                                 "--no-ext-diff", "--no-textconv", "HEAD"], cwd=root,
                                text=True, capture_output=True, timeout=10)
        untracked = subprocess.run(["git", "ls-files", "--others", "--exclude-standard"],
                                   cwd=root, text=True, capture_output=True, timeout=10)
        return {"diff": result.stdout[:24000], "untracked": untracked.stdout[:4000],
                "truncated": len(result.stdout) > 24000, "exit_code": result.returncode}
    raise ValueError("unknown workspace operation")


if __name__ == "__main__":
    try:
        request = json.load(sys.stdin)
        print(json.dumps(execute(request["workspace"], request["name"],
                                 request["arguments"], request["reserved"]), ensure_ascii=False))
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc)}))
