from __future__ import annotations

from dataclasses import dataclass
import ipaddress
from pathlib import Path
import re
from typing import Iterable


PUBLIC_ARTIFACT_DIRECTORIES = frozenset(
    {
        "archives",
        "configs",
        "datasets",
        "docs",
        "environments",
        "factory",
        "runs",
        "sources",
        "tasks",
    }
)
PUBLIC_ROOT_FILES = frozenset(
    {
        "CHANGELOG.md",
        "README.md",
        "README.zh-CN.md",
    }
)


@dataclass(frozen=True, order=True)
class PublicTreePrivacyFinding:
    path: str
    line: int
    code: str


_ABSOLUTE_USER_PATH = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|home)/[^\s\"'`<>]+"
)
_SSH_IDENTITY = re.compile(
    r"(?:~|/(?:Users|home)/[^/\s\"'`<>]+)?/\.ssh(?:/[^\s\"'`<>]+)?"
)
_IPV4 = re.compile(
    r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
    r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?![\d.])"
)
_REMOTE_IDENTITY = re.compile(
    r"\broot@"
    r"(?!localhost\b)(?![A-Za-z0-9.-]*\.example\.invalid\b)"
    r"[A-Za-z0-9][A-Za-z0-9.-]*\b"
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(?:authorization|proxy-authorization)\s*:\s*bearer\s+\S+"),
    re.compile(
        r"(?i)[\"']?(?:api[_-]?key|access[_-]?token|secret[_-]?key)"
        r"[\"']?\s*[:=]\s*[\"'](?P<value>[^\"']{12,})[\"']"
    ),
)
_CONNECTION_FIELD = re.compile(
    r"[\"'](?P<field>hostname|remote_user|identity_file)[\"']"
    r"\s*:\s*[\"'](?P<value>[^\"']*)[\"']"
)


def scan_public_tree(
    root: Path,
    *,
    tracked_paths: Iterable[Path],
) -> tuple[PublicTreePrivacyFinding, ...]:
    """Scan tracked public artifacts without retaining or returning matched text."""

    root = root.resolve()
    findings: set[PublicTreePrivacyFinding] = set()
    for supplied_path in tracked_paths:
        relative = _normalized_relative_path(supplied_path)
        if relative is None:
            findings.add(
                PublicTreePrivacyFinding(
                    path=_safe_path_label(supplied_path),
                    line=0,
                    code="tree.invalid_path",
                )
            )
            continue
        if not _is_public_artifact(relative):
            continue
        display_path = relative.as_posix()
        path = root / relative
        if path.is_symlink():
            findings.add(
                PublicTreePrivacyFinding(
                    path=display_path,
                    line=0,
                    code="tree.symlink",
                )
            )
            continue
        if not path.is_file():
            findings.add(
                PublicTreePrivacyFinding(
                    path=display_path,
                    line=0,
                    code="tree.unreadable",
                )
            )
            continue
        try:
            payload = path.read_bytes()
        except OSError:
            findings.add(
                PublicTreePrivacyFinding(
                    path=display_path,
                    line=0,
                    code="tree.unreadable",
                )
            )
            continue
        if b"\0" in payload:
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for code in _privacy_codes(line):
                findings.add(
                    PublicTreePrivacyFinding(
                        path=display_path,
                        line=line_number,
                        code=code,
                    )
                )
    return tuple(sorted(findings))


def _normalized_relative_path(path: Path) -> Path | None:
    if path.is_absolute() or not path.parts:
        return None
    normalized = Path(*path.parts)
    if any(part in {"", ".", ".."} for part in normalized.parts):
        return None
    return normalized


def _safe_path_label(path: Path) -> str:
    if path.is_absolute():
        return "<absolute-path>"
    safe_parts = tuple(part for part in path.parts if part not in {"", ".", ".."})
    return Path(*safe_parts).as_posix() if safe_parts else "<invalid-path>"


def _is_public_artifact(path: Path) -> bool:
    return (
        len(path.parts) == 1 and path.name in PUBLIC_ROOT_FILES
    ) or path.parts[0] in PUBLIC_ARTIFACT_DIRECTORIES


def _privacy_codes(line: str) -> set[str]:
    codes: set[str] = set()
    if any(
        not _is_safe_absolute_user_path(match.group(0))
        for match in _ABSOLUTE_USER_PATH.finditer(line)
    ):
        codes.add("private.absolute_user_path")
    if any(
        not _is_safe_ssh_identity(match.group(0))
        for match in _SSH_IDENTITY.finditer(line)
    ):
        codes.add("private.ssh_identity")
    if any(
        not _is_safe_ipv4(match.group(0))
        for match in _IPV4.finditer(line)
    ):
        codes.add("private.ipv4")
    if _REMOTE_IDENTITY.search(line):
        codes.add("private.remote_identity")
    if any(pattern.search(line) for pattern in _CREDENTIAL_PATTERNS):
        codes.add("private.credential")
    for match in _CONNECTION_FIELD.finditer(line):
        field = match.group("field")
        value = match.group("value")
        if not _is_safe_connection_placeholder(field, value):
            codes.add(f"private.{field}_field")
    return codes


def _is_safe_absolute_user_path(value: str) -> bool:
    parts = Path(value).parts
    if len(parts) < 3:
        return False
    owner = parts[2].lower()
    return owner in {
        "example",
        "example-user",
        "example_user",
        "opbench",
        "user",
        "username",
        "your-user",
        "your_user",
    }


def _is_safe_ssh_identity(value: str) -> bool:
    lowered = value.lower()
    if "example" in lowered or "<" in lowered or "your" in lowered:
        return True
    if lowered.startswith("/users/") or lowered.startswith("/home/"):
        return _is_safe_absolute_user_path(value)
    return lowered in {
        "~/.ssh/authorized_keys",
        "~/.ssh/config",
        "~/.ssh/id_ecdsa",
        "~/.ssh/id_ed25519",
        "~/.ssh/id_rsa",
        "~/.ssh/known_hosts",
    }


def _is_safe_ipv4(value: str) -> bool:
    address = ipaddress.IPv4Address(value)
    documentation_networks = (
        ipaddress.IPv4Network("192.0.2.0/24"),
        ipaddress.IPv4Network("198.51.100.0/24"),
        ipaddress.IPv4Network("203.0.113.0/24"),
    )
    return address.is_loopback or any(
        address in network for network in documentation_networks
    )


def _is_safe_connection_placeholder(field: str, value: str) -> bool:
    if field == "hostname":
        return value == "localhost" or value.endswith(".example.invalid")
    if field == "remote_user":
        return value in {"example-user", "example_user", "<user>"}
    if field == "identity_file":
        return value in {
            "/tmp/example-key",
            "/path/to/example-key",
            "<identity-file>",
        }
    return False
