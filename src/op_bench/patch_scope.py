from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PatchScopeResult:
    status: str
    filtered_patch: str
    out_of_scope_paths: list[str]


_DIFF_PATH_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def extract_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for match in _DIFF_PATH_RE.finditer(patch_text):
        b_path = match.group(2)
        if b_path not in paths:
            paths.append(b_path)
    return paths


def validate_patch_scope(
    patch_text: str,
    allowed_paths: list[str],
    mode: str = "enforced",
) -> PatchScopeResult:
    if not allowed_paths:
        return PatchScopeResult(status="no_scope", filtered_patch=patch_text, out_of_scope_paths=[])

    if not patch_text.strip():
        return PatchScopeResult(status="empty_patch", filtered_patch="", out_of_scope_paths=[])

    modified_paths = extract_patch_paths(patch_text)
    allowed_set = set(allowed_paths)
    out_of_scope = [p for p in modified_paths if p not in allowed_set]

    if not out_of_scope:
        return PatchScopeResult(status="in_scope", filtered_patch=patch_text, out_of_scope_paths=[])

    if mode == "enforced":
        return PatchScopeResult(status="out_of_scope", filtered_patch="", out_of_scope_paths=out_of_scope)

    filtered = _filter_patch(patch_text, allowed_set)
    return PatchScopeResult(status="filtered", filtered_patch=filtered, out_of_scope_paths=out_of_scope)


def _filter_patch(patch_text: str, allowed_set: set[str]) -> str:
    hunks: list[str] = []
    current_hunk_lines: list[str] = []
    current_path: str | None = None

    for line in patch_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current_hunk_lines and current_path in allowed_set:
                hunks.append("".join(current_hunk_lines))
            current_hunk_lines = [line]
            match = _DIFF_PATH_RE.match(line.rstrip("\n"))
            current_path = match.group(2) if match else None
        else:
            current_hunk_lines.append(line)

    if current_hunk_lines and current_path in allowed_set:
        hunks.append("".join(current_hunk_lines))

    return "".join(hunks)
