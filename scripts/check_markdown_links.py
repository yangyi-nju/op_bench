#!/usr/bin/env python3
"""Check current documentation; archived records retain their original links."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
ENTRYPOINTS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "datasets/README.md",
    ROOT / "tasks/README.md",
    ROOT / "runs/README.md",
    ROOT / "runs/archive/README.md",
)


def _historical(document: Path) -> bool:
    relative = document.relative_to(ROOT).as_posix()
    return bool(re.match(r"docs/v0\.[1-7]/", relative)) or relative.startswith("docs/superpowers/") or relative in {
        "docs/project_plan.md", "docs/project_state.md", "docs/history/CHANGELOG-through-v0.7.md",
    }


def _documents(*, include_history: bool = False) -> tuple[Path, ...]:
    return tuple(
        sorted(
            (document for document in {*ENTRYPOINTS, *(ROOT / "docs").rglob("*.md"),
             *(ROOT / "tests/fixtures").rglob("*.md"),
             *(ROOT / "datasets/v0.8_candidates").rglob("*.md")}
             if include_history or not _historical(document)),
            key=lambda path: path.as_posix(),
        )
    )


def _targets(document: Path) -> tuple[str, ...]:
    in_fence = False
    targets: list[str] = []
    for line in document.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            targets.extend(LINK_PATTERN.findall(line))
    return tuple(targets)


def _local_target(raw: str) -> str | None:
    value = raw.strip()
    if value.startswith("<") and ">" in value:
        value = value[1 : value.index(">")]
    else:
        # Markdown permits an optional quoted title after the destination.
        value = value.split(maxsplit=1)[0]
    split = urlsplit(value)
    if split.scheme or split.netloc or value.startswith(("mailto:", "#")):
        return None
    return unquote(split.path) or None


def find_missing_links(*, include_history: bool = False) -> tuple[str, ...]:
    findings: list[str] = []
    for document in _documents(include_history=include_history):
        if not document.is_file():
            findings.append(f"{document.relative_to(ROOT)}: document is missing")
            continue
        for raw in _targets(document):
            relative = _local_target(raw)
            if relative is None:
                continue
            resolved = (document.parent / relative).resolve(strict=False)
            try:
                resolved.relative_to(ROOT)
            except ValueError:
                findings.append(
                    f"{document.relative_to(ROOT)}: link escapes repository: {raw}"
                )
                continue
            if not resolved.exists():
                findings.append(
                    f"{document.relative_to(ROOT)}: missing target: {raw}"
                )
    return tuple(sorted(set(findings)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-history", action="store_true",
                        help="Also inspect original version records against today's tree; retired assets may be missing")
    args = parser.parse_args(argv)
    try:
        findings = find_missing_links(include_history=args.include_history)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"cannot inspect Markdown links: {exc}", file=sys.stderr)
        return 2
    for finding in findings:
        print(finding)
    print(f"Markdown link findings: {len(findings)}")
    if not args.include_history:
        print("Scope: current docs and history navigation; original version records keep historical asset links.")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
