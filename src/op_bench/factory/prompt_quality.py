"""Deterministic Prompt non-leakage evidence for Factory admission."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
import re
import shlex
from types import MappingProxyType
from typing import ClassVar

from op_bench.runtime.canonical import JsonValue, canonical_sha256
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt
from op_bench.runtime.contracts import AgentTaskView
from op_bench.runtime.task_view import assert_public_artifact_safe
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_list,
    require_mapping,
    require_str,
)


_SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
_UTC_SECONDS_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z"
)
_IDENTIFIER = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_PYTHON_SYMBOL = re.compile(
    r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_CPP_CLASS_SYMBOL = re.compile(r"^\s*(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_QUOTED_LITERAL = re.compile(r"(?P<quote>['\"])(?P<value>(?:\\.|(?!\1).)*)\1")
_COMPARISON_LITERAL = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=|<=|>=|<|>)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')"
)
_COMPARISON_FULL = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*\s*(?:==|!=|<=|>=|<|>)\s*"
    r"(?:[A-Za-z_][A-Za-z0-9_]*|-?[0-9]+|\"(?:\\.|[^\"])*\"|'(?:\\.|[^'])*')"
)
_HUNK_HEADER = re.compile(
    r"@@ -[0-9]+(?:,[0-9]+)? \+[0-9]+(?:,[0-9]+)? @@(?: .*)?"
)
_HUNK_COUNTS = re.compile(
    r"@@ -[0-9]+(?:,([0-9]+))? \+[0-9]+(?:,([0-9]+))? @@"
)
_CPP_SIGNATURE_NAME = re.compile(r"([~A-Za-z_][A-Za-z0-9_:~]*)\s*\(")
_CPP_CONTROL = re.compile(r"^\s*(?:if|for|while|switch|catch)\b")
_PULL_REQUEST = re.compile(r"\bpr\s*#\s*[0-9]+\b", re.IGNORECASE)
_PULL_URL = re.compile(
    r"https?://(?:www\.)?github\.com/[^\s/]+/[^\s/]+/pulls?/[0-9]+(?:[^\s]*)?",
    re.IGNORECASE,
)
_COMMIT_URL = re.compile(
    r"https?://(?:www\.)?github\.com/[^\s/]+/[^\s/]+/commit/[0-9a-f]+(?:[^\s]*)?",
    re.IGNORECASE,
)
_PRIVATE_PROVENANCE = re.compile(
    r"\b(?:"
    r"gold(?:[ _.-]+patch)?"
    r"|hidden(?:[ _.-]+tests?)?"
    r"|admission(?:[ _.-]+evidence)?"
    r")\b",
    re.IGNORECASE,
)
_SOLUTION_INSTRUCTION = re.compile(
    r"\b(?:"
    r"(?:modify|change|edit|update)\s+(?:the\s+)?(?:file\s+)?[A-Za-z0-9_.\-/]+"
    r"|replace\s+[A-Za-z_][A-Za-z0-9_]*(?:\s*[+*/-]\s*[A-Za-z_0-9]+)+"
    r"|reuse\s+[A-Za-z_][A-Za-z0-9_]*"
    r"|register\s+[A-Za-z_][A-Za-z0-9_]*"
    r"|add\s+(?:an?\s+)?(?:condition|check|branch|guard|case)\b"
    r")",
    re.IGNORECASE,
)
_SOURCE_KEYWORDS = frozenset(
    {
        "alignas",
        "alignof",
        "and",
        "asm",
        "auto",
        "await",
        "bool",
        "break",
        "case",
        "catch",
        "char",
        "class",
        "const",
        "continue",
        "def",
        "default",
        "delete",
        "do",
        "double",
        "elif",
        "else",
        "enum",
        "except",
        "explicit",
        "export",
        "false",
        "finally",
        "float",
        "for",
        "friend",
        "if",
        "import",
        "inline",
        "int",
        "lambda",
        "long",
        "namespace",
        "new",
        "noexcept",
        "none",
        "not",
        "nullptr",
        "operator",
        "or",
        "pass",
        "private",
        "protected",
        "public",
        "raise",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "template",
        "this",
        "throw",
        "true",
        "try",
        "typedef",
        "typename",
        "union",
        "unsigned",
        "virtual",
        "void",
        "while",
        "with",
        "yield",
    }
)


def _sorted_unique(values: tuple[str, ...], *, path: str) -> tuple[str, ...]:
    normalized = tuple(require_str(value, f"{path}[{index}]") for index, value in enumerate(values))
    return tuple(sorted(set(normalized)))


def _normalize_path(value: str) -> str | None:
    text = value.strip().split("\t", 1)[0].strip()
    if text in {"", "/dev/null"}:
        return None
    if text.startswith(("a/", "b/")):
        text = text[2:]
    path = PurePosixPath(text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


@dataclass(frozen=True)
class PrivateAnswerIndex:
    """Private, deterministic answer-side facts used only for Prompt scanning."""

    changed_paths: tuple[str, ...]
    added_symbols: tuple[str, ...]
    distinctive_literals: tuple[str, ...]
    hidden_selectors: tuple[str, ...]
    internal_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in (
            "changed_paths",
            "added_symbols",
            "distinctive_literals",
            "hidden_selectors",
            "internal_names",
        ):
            value = getattr(self, field)
            if not isinstance(value, tuple):
                raise ContractError(f"private_answer_index.{field}: expected tuple")
            object.__setattr__(
                self,
                field,
                _sorted_unique(value, path=f"private_answer_index.{field}"),
            )


@dataclass(frozen=True)
class PromptFinding:
    code: str
    severity: str
    public_field: str
    matched_value_hash: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("code", "severity", "public_field", "matched_value_hash")

    def __post_init__(self) -> None:
        require_str(
            self.code,
            "prompt_finding.code",
            pattern=r"(?:answer|provenance|solution)\.[a-z0-9][a-z0-9._-]*",
        )
        require_enum(self.severity, "prompt_finding.severity", ("defer", "reject"))
        require_str(
            self.public_field,
            "prompt_finding.public_field",
            pattern=r"[a-z][a-z0-9_]*",
        )
        require_str(
            self.matched_value_hash,
            "prompt_finding.matched_value_hash",
            pattern=_SHA256_PATTERN,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "code": self.code,
            "severity": self.severity,
            "public_field": self.public_field,
            "matched_value_hash": self.matched_value_hash,
        }

    @classmethod
    def from_dict(cls, value: object, *, path: str = "prompt_finding") -> "PromptFinding":
        data = require_exact_fields(value, path, cls.wire_fields())
        return cls(
            code=require_str(data["code"], f"{path}.code"),
            severity=require_str(data["severity"], f"{path}.severity"),
            public_field=require_str(data["public_field"], f"{path}.public_field"),
            matched_value_hash=require_str(
                data["matched_value_hash"], f"{path}.matched_value_hash"
            ),
        )


def _finding(code: str, field: str, matched: str) -> PromptFinding:
    return PromptFinding(
        code=code,
        severity="reject",
        public_field=field,
        matched_value_hash=canonical_sha256({"normalized_match": matched.casefold()}),
    )


def empty_private_index() -> PrivateAnswerIndex:
    return PrivateAnswerIndex((), (), (), ())


def _header_path(value: str, *, prefix: str, path: str, allow_null: bool) -> str | None:
    text = value.strip().split("\t", 1)[0].strip()
    if text == "/dev/null":
        if allow_null:
            return None
        raise ContractError(f"{path}: invalid diff header path")
    if not text.startswith(prefix):
        raise ContractError(f"{path}: invalid diff header path")
    normalized = _normalize_path(text)
    if normalized is None:
        raise ContractError(f"{path}: invalid diff header path")
    return normalized


def _parse_hunk_header(line: str, *, path: str) -> tuple[int, int]:
    if _HUNK_HEADER.fullmatch(line) is None:
        raise ContractError(f"{path}: malformed hunk header")
    counts = _HUNK_COUNTS.match(line)
    assert counts is not None
    return (
        1 if counts.group(1) is None else int(counts.group(1)),
        1 if counts.group(2) is None else int(counts.group(2)),
    )


def _parse_unified_diff(patch: str, *, path: str) -> tuple[set[str], tuple[str, ...]]:
    """Parse paths and added source lines from a complete unified diff."""

    paths: set[str] = set()
    added: list[str] = []
    saw_diff = False
    header_paths: tuple[str | None, str | None] | None = None
    old_remaining = 0
    new_remaining = 0
    in_hunk = False

    def finish_hunk() -> None:
        if in_hunk and (old_remaining != 0 or new_remaining != 0):
            raise ContractError(f"{path}: incomplete hunk")

    for line_number, line in enumerate(patch.splitlines(), start=1):
        line_path = f"{path}:{line_number}"
        if in_hunk and old_remaining == 0 and new_remaining == 0:
            in_hunk = False
        if in_hunk:
            if line.startswith("\\ No newline at end of file"):
                continue
            if line.startswith("+"):
                new_remaining -= 1
                added.append(line[1:])
            elif line.startswith("-"):
                old_remaining -= 1
            elif line.startswith(" "):
                old_remaining -= 1
                new_remaining -= 1
            else:
                raise ContractError(f"{line_path}: invalid hunk line")
            if old_remaining < 0 or new_remaining < 0:
                raise ContractError(f"{line_path}: hunk line count overflow")
            continue

        if line.startswith("diff --git "):
            finish_hunk()
            try:
                parts = shlex.split(line)
            except ValueError as exc:
                raise ContractError(f"{line_path}: malformed diff header") from exc
            if len(parts) != 4 or parts[:2] != ["diff", "--git"]:
                raise ContractError(f"{line_path}: malformed diff header")
            left = _header_path(
                parts[2], prefix="a/", path=line_path, allow_null=False
            )
            right = _header_path(
                parts[3], prefix="b/", path=line_path, allow_null=False
            )
            assert left is not None and right is not None
            paths.update((left, right))
            header_paths = (left, right)
            saw_diff = True
            continue
        if line.startswith(("--- ", "+++ ")):
            if header_paths is None:
                raise ContractError(f"{line_path}: diff header is required")
            prefix = "a/" if line.startswith("--- ") else "b/"
            normalized = _header_path(
                line[4:],
                prefix=prefix,
                path=line_path,
                allow_null=True,
            )
            if normalized is not None:
                paths.add(normalized)
            continue
        if line.startswith("@@"):
            if header_paths is None:
                raise ContractError(f"{line_path}: diff header is required")
            finish_hunk()
            old_remaining, new_remaining = _parse_hunk_header(line, path=line_path)
            in_hunk = True
            continue
        if line.startswith(("GIT binary patch", "Binary files ")):
            raise ContractError(f"{line_path}: unsupported diff payload")
        if line.startswith(("+", "-", " ")):
            raise ContractError(f"{line_path}: source line outside hunk")

    finish_hunk()
    if patch and not saw_diff:
        raise ContractError(f"{path}: diff header is required")
    return paths, tuple(added)


def _source_symbols(lines: tuple[str, ...]) -> set[str]:
    symbols: set[str] = set()
    for index, line in enumerate(lines):
        for pattern in (_PYTHON_SYMBOL, _CPP_CLASS_SYMBOL):
            match = pattern.match(line)
            if match is not None:
                symbols.add(match.group(1))
        if "(" not in line or _CPP_CONTROL.match(line) is not None:
            continue
        signature = line.strip()
        for next_line in lines[index + 1 : index + 9]:
            if "{" in signature or ";" in signature:
                break
            signature = f"{signature} {next_line.strip()}"
        if "{" not in signature or ";" in signature:
            continue
        declaration = re.split(
            r"(?<!:):(?!:)",
            signature.split("{", 1)[0],
            maxsplit=1,
        )[0]
        names = _CPP_SIGNATURE_NAME.findall(declaration)
        if names:
            symbols.add(names[-1].split("::")[-1].removeprefix("~"))
    return symbols


def _distinctive_literals(lines: tuple[str, ...]) -> set[str]:
    literals: set[str] = set()
    for line in lines:
        for match in _QUOTED_LITERAL.finditer(line):
            value = match.group("value")
            if len(value) >= 6:
                literals.add(value)
        for match in _COMPARISON_LITERAL.finditer(line):
            value = _normalize_comparison(match.group(0))
            if len(value) >= 6:
                literals.add(value)
    return literals


def build_private_answer_index(
    *,
    gold_patch: str,
    hidden_test_patch: str,
    patch_scope: tuple[str, ...],
    hidden_selectors: tuple[str, ...],
) -> PrivateAnswerIndex:
    """Extract answer-side facts from diffs without loading framework code."""

    for name, value in (("gold_patch", gold_patch), ("hidden_test_patch", hidden_test_patch)):
        require_str(value, name, min_length=0)
    if not isinstance(patch_scope, tuple) or not isinstance(hidden_selectors, tuple):
        raise ContractError("private_answer_index: patch_scope and hidden_selectors must be tuples")

    gold_paths, gold_lines = _parse_unified_diff(gold_patch, path="gold_patch")
    hidden_paths, hidden_lines = _parse_unified_diff(
        hidden_test_patch,
        path="hidden_test_patch",
    )
    paths = gold_paths | hidden_paths
    for index, path in enumerate(patch_scope):
        normalized = _normalize_path(require_str(path, f"patch_scope[{index}]"))
        if normalized is None:
            raise ContractError(f"patch_scope[{index}]: expected normalized relative path")
        paths.add(normalized)

    lines = gold_lines + hidden_lines
    symbols = _source_symbols(lines)
    identifier_counts = Counter(
        identifier
        for line in lines
        for identifier in _IDENTIFIER.findall(line)
        if len(identifier) >= 8 and identifier.casefold() not in _SOURCE_KEYWORDS
    )
    internal_names = {
        identifier
        for identifier, count in identifier_counts.items()
        if count <= 2 and identifier not in symbols
    }
    return PrivateAnswerIndex(
        changed_paths=tuple(paths),
        added_symbols=tuple(symbols),
        distinctive_literals=tuple(_distinctive_literals(lines)),
        hidden_selectors=hidden_selectors,
        internal_names=tuple(internal_names),
    )


def _text_contains(text: str, matched: str) -> bool:
    return matched.casefold() in text.casefold()


def _normalize_comparison(value: str) -> str:
    return re.sub(r"\s*(==|!=|<=|>=|<|>)\s*", r" \1 ", value).strip()


def _literal_overlaps(text: str, literal: str) -> bool:
    if _COMPARISON_FULL.fullmatch(literal) is None:
        return _text_contains(text, literal)
    normalized = _normalize_comparison(literal).casefold()
    return any(
        _normalize_comparison(match.group(0)).casefold() == normalized
        for match in _COMPARISON_LITERAL.finditer(text)
    )


def _text_contains_symbol(text: str, matched: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9_]){re.escape(matched.casefold())}(?![a-z0-9_])",
        text.casefold(),
    ) is not None


def scan_rendered_prompt(
    rendered_prompt: str,
    private_index: PrivateAnswerIndex,
) -> tuple[PromptFinding, ...]:
    """Return public, hashed findings for leakage in an exact rendered Prompt."""

    require_str(rendered_prompt, "rendered_prompt", min_length=0)
    if not isinstance(private_index, PrivateAnswerIndex):
        raise ContractError("private_index: expected PrivateAnswerIndex")

    findings: set[PromptFinding] = set()
    public_field = "rendered_prompt"
    for pattern, code in (
        (_PULL_REQUEST, "provenance.pull_request"),
        (_PULL_URL, "provenance.pull_request"),
        (_COMMIT_URL, "provenance.commit"),
        (_PRIVATE_PROVENANCE, "provenance.private_term"),
        (_SOLUTION_INSTRUCTION, "solution.instruction"),
    ):
        for match in pattern.finditer(rendered_prompt):
            findings.add(_finding(code, public_field, match.group(0)))

    for path in private_index.changed_paths:
        if _text_contains(rendered_prompt, path):
            findings.add(_finding("answer.changed_path", public_field, path))
    for symbol in private_index.added_symbols:
        if _text_contains_symbol(rendered_prompt, symbol):
            findings.add(_finding("answer.symbol", public_field, symbol))
    for literal in private_index.distinctive_literals:
        if _literal_overlaps(rendered_prompt, literal):
            findings.add(_finding("answer.distinctive_literal", public_field, literal))
    for selector in private_index.hidden_selectors:
        if _text_contains(rendered_prompt, selector):
            findings.add(_finding("answer.hidden_selector", public_field, selector))
    for name in private_index.internal_names:
        if _text_contains_symbol(rendered_prompt, name):
            findings.add(_finding("answer.internal_name", public_field, name))

    return tuple(
        sorted(
            findings,
            key=lambda finding: (
                finding.code,
                finding.public_field,
                finding.matched_value_hash,
            ),
        )
    )


def _validate_utc_seconds(value: object, path: str) -> str:
    text = require_str(value, path)
    if _UTC_SECONDS_PATTERN.fullmatch(text) is None:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: invalid UTC timestamp") from exc
    return text


def _review(value: object, *, path: str, decision: str) -> Mapping[str, str]:
    data = require_exact_fields(value, path, ("decision", "reviewer", "reviewed_at"))
    if require_str(data["decision"], f"{path}.decision") != decision:
        raise ContractError(f"{path}.decision: expected {decision!r}")
    return MappingProxyType(
        {
            "decision": decision,
            "reviewer": require_str(data["reviewer"], f"{path}.reviewer"),
            "reviewed_at": _validate_utc_seconds(data["reviewed_at"], f"{path}.reviewed_at"),
        }
    )


@dataclass(frozen=True, init=False)
class PromptQualityEvidence:
    contract_type: ClassVar[str] = "prompt_quality"
    schema_version: ClassVar[str] = "v1"

    task_id: str
    public_task_id: str
    prompt_hash: str
    agent_task_view_hash: str
    scanner_version: str
    findings: tuple[PromptFinding, ...]
    blind_review: Mapping[str, object]
    semantic_review: Mapping[str, object]
    decision: str
    created_at: str
    content_hash: str = ""

    def __init__(
        self,
        *,
        task_id: str,
        public_task_id: str,
        rendered_prompt: str,
        agent_task_view: AgentTaskView | Mapping[str, object],
        private_index: PrivateAnswerIndex,
        scanner_version: str,
        blind_review: Mapping[str, object],
        semantic_review: Mapping[str, object],
        decision: str,
        created_at: str,
    ) -> None:
        """Construct source-bound evidence; hashes and findings are never inputs."""

        require_str(rendered_prompt, "rendered_prompt", min_length=0)
        payload = _agent_task_view_payload(agent_task_view)
        canonical_prompt = render_mcp_prompt(payload)
        if rendered_prompt != canonical_prompt:
            raise ContractError(
                "rendered_prompt: does not match canonical render of agent_task_view"
            )
        task_identity = payload["task"]
        assert isinstance(task_identity, dict)
        if public_task_id != task_identity["identifier"]:
            raise ContractError(
                "prompt_quality.public_task_id: does not match agent_task_view"
            )
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "public_task_id", public_task_id)
        object.__setattr__(self, "prompt_hash", canonical_sha256(canonical_prompt))
        object.__setattr__(self, "agent_task_view_hash", canonical_sha256(payload))
        object.__setattr__(self, "scanner_version", scanner_version)
        object.__setattr__(
            self,
            "findings",
            scan_rendered_prompt(canonical_prompt, private_index),
        )
        object.__setattr__(self, "blind_review", blind_review)
        object.__setattr__(self, "semantic_review", semantic_review)
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "content_hash", "")
        self._validate_stored()

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "task_id",
            "public_task_id",
            "prompt_hash",
            "agent_task_view_hash",
            "scanner_version",
            "findings",
            "blind_review",
            "semantic_review",
            "decision",
            "created_at",
            "content_hash",
        )

    def _validate_stored(self) -> None:
        require_str(self.task_id, "prompt_quality.task_id")
        require_str(self.public_task_id, "prompt_quality.public_task_id")
        require_str(self.prompt_hash, "prompt_quality.prompt_hash", pattern=_SHA256_PATTERN)
        require_str(
            self.agent_task_view_hash,
            "prompt_quality.agent_task_view_hash",
            pattern=_SHA256_PATTERN,
        )
        require_str(self.scanner_version, "prompt_quality.scanner_version", pattern=r"[a-z0-9][a-z0-9._-]*")
        if not isinstance(self.findings, tuple) or not all(
            isinstance(finding, PromptFinding) for finding in self.findings
        ):
            raise ContractError("prompt_quality.findings: expected PromptFinding tuple")
        finding_key = lambda finding: (
            finding.code,
            finding.public_field,
            finding.matched_value_hash,
        )
        if self.findings != tuple(sorted(self.findings, key=finding_key)) or len(set(self.findings)) != len(self.findings):
            raise ContractError("prompt_quality.findings: expected sorted, unique findings")
        blind_review = _review(self.blind_review, path="prompt_quality.blind_review", decision="accepted")
        semantic_review = _review(
            self.semantic_review,
            path="prompt_quality.semantic_review",
            decision="equivalent",
        )
        if blind_review["reviewer"] == semantic_review["reviewer"]:
            raise ContractError("prompt_quality: reviews require different reviewers")
        object.__setattr__(self, "blind_review", blind_review)
        object.__setattr__(self, "semantic_review", semantic_review)
        require_enum(self.decision, "prompt_quality.decision", ("accepted", "deferred", "rejected"))
        if self.decision == "accepted" and any(
            finding.severity == "reject" for finding in self.findings
        ):
            raise ContractError("prompt_quality.decision: acceptance requires no high-confidence finding")
        _validate_utc_seconds(self.created_at, "prompt_quality.created_at")

        expected_hash = canonical_sha256(self._payload_without_hash())
        if self.content_hash == "":
            object.__setattr__(self, "content_hash", expected_hash)
        else:
            stored_hash = require_str(
                self.content_hash,
                "prompt_quality.content_hash",
                pattern=_SHA256_PATTERN,
            )
            if stored_hash != expected_hash:
                raise ContractError("prompt_quality.content_hash: payload hash mismatch")

    def _payload_without_hash(self) -> dict[str, JsonValue]:
        return {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "public_task_id": self.public_task_id,
            "prompt_hash": self.prompt_hash,
            "agent_task_view_hash": self.agent_task_view_hash,
            "scanner_version": self.scanner_version,
            "findings": [finding.to_dict() for finding in self.findings],
            "blind_review": dict(self.blind_review),
            "semantic_review": dict(self.semantic_review),
            "decision": self.decision,
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, JsonValue]:
        payload = self._payload_without_hash()
        payload["content_hash"] = self.content_hash
        return payload

    @classmethod
    def from_dict(cls, value: object, *, path: str = "prompt_quality") -> "PromptQualityEvidence":
        data = require_exact_fields(value, path, cls.wire_fields())
        if require_str(data["contract_type"], f"{path}.contract_type") != cls.contract_type:
            raise ContractError(f"{path}.contract_type: expected {cls.contract_type!r}")
        if require_str(data["schema_version"], f"{path}.schema_version") != cls.schema_version:
            raise ContractError(f"{path}.schema_version: expected {cls.schema_version!r}")
        findings = tuple(
            PromptFinding.from_dict(item, path=f"{path}.findings[{index}]")
            for index, item in enumerate(require_list(data["findings"], f"{path}.findings"))
        )
        evidence = object.__new__(cls)
        for name, item in (
            ("task_id", require_str(data["task_id"], f"{path}.task_id")),
            ("public_task_id", require_str(data["public_task_id"], f"{path}.public_task_id")),
            ("prompt_hash", require_str(data["prompt_hash"], f"{path}.prompt_hash")),
            (
                "agent_task_view_hash",
                require_str(data["agent_task_view_hash"], f"{path}.agent_task_view_hash"),
            ),
            ("scanner_version", require_str(data["scanner_version"], f"{path}.scanner_version")),
            ("findings", findings),
            ("blind_review", require_mapping(data["blind_review"], f"{path}.blind_review")),
            ("semantic_review", require_mapping(data["semantic_review"], f"{path}.semantic_review")),
            ("decision", require_str(data["decision"], f"{path}.decision")),
            ("created_at", _validate_utc_seconds(data["created_at"], f"{path}.created_at")),
            ("content_hash", require_str(data["content_hash"], f"{path}.content_hash")),
        ):
            object.__setattr__(evidence, name, item)
        evidence._validate_stored()
        return evidence


def _agent_task_view_payload(value: object) -> dict[str, JsonValue]:
    if isinstance(value, AgentTaskView):
        selected = value
    elif isinstance(value, Mapping):
        selected = AgentTaskView.from_dict(value, path="agent_task_view")
    else:
        raise ContractError("agent_task_view: expected AgentTaskView")
    payload = selected.to_dict()
    assert_public_artifact_safe(payload)
    canonical_sha256(payload)
    return payload


def build_prompt_quality_evidence(
    *,
    task_id: str,
    public_task_id: str,
    rendered_prompt: str,
    agent_task_view: AgentTaskView | Mapping[str, object],
    private_index: PrivateAnswerIndex,
    scanner_version: str,
    blind_review: Mapping[str, object],
    semantic_review: Mapping[str, object],
    decision: str,
    created_at: str,
) -> PromptQualityEvidence:
    """Build evidence by recomputing every Prompt-facing acceptance claim."""

    return PromptQualityEvidence(
        task_id=task_id,
        public_task_id=public_task_id,
        rendered_prompt=rendered_prompt,
        agent_task_view=agent_task_view,
        private_index=private_index,
        scanner_version=scanner_version,
        blind_review=blind_review,
        semantic_review=semantic_review,
        decision=decision,
        created_at=created_at,
    )


def validate_prompt_quality_evidence(
    evidence: PromptQualityEvidence,
    *,
    rendered_prompt: str,
    agent_task_view: AgentTaskView | Mapping[str, object],
    private_index: PrivateAnswerIndex,
) -> None:
    """Fail closed unless stored evidence exactly matches the source inputs."""

    if not isinstance(evidence, PromptQualityEvidence):
        raise ContractError("prompt_quality: expected PromptQualityEvidence")
    require_str(rendered_prompt, "rendered_prompt", min_length=0)
    payload = _agent_task_view_payload(agent_task_view)
    canonical_prompt = render_mcp_prompt(payload)
    if rendered_prompt != canonical_prompt:
        raise ContractError(
            "rendered_prompt: does not match canonical render of agent_task_view"
        )
    expected_prompt_hash = canonical_sha256(canonical_prompt)
    if evidence.prompt_hash != expected_prompt_hash:
        raise ContractError("prompt_quality.prompt_hash: does not match rendered_prompt")
    expected_view_hash = canonical_sha256(payload)
    if evidence.agent_task_view_hash != expected_view_hash:
        raise ContractError(
            "prompt_quality.agent_task_view_hash: does not match agent_task_view"
        )
    expected_findings = scan_rendered_prompt(canonical_prompt, private_index)
    if evidence.findings != expected_findings:
        raise ContractError("prompt_quality.findings: do not match rendered_prompt scan")
