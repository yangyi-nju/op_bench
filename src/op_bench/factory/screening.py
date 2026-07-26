from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    ScreeningFinding,
    derived_disposition,
    finding_sort_key,
)
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError, require_int, require_str


RULE_IDS = (
    "v07-revert-title",
    "v07-required-dates",
    "v07-stable-author-window",
    "v07-change-kind",
    "v07-normal-diff-size",
    "v07-minimum-behavioral-change",
    "v07-executable-test-delta",
    "v07-commit-identities",
    "v07-source-availability",
    "v07-runtime-support",
    "v07-boundary-taxonomy",
    "v07-root-cause-review",
)


def _parse_timestamp(value: str, path: str) -> datetime:
    require_str(value, path)
    try:
        return datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{path}: expected UTC RFC3339 seconds") from exc


@dataclass(frozen=True)
class ScreeningRuleSet:
    version: str
    author_date_start: str
    author_date_end: str
    max_changed_files: int
    max_changed_lines: int
    min_behavioral_changed_lines: int
    rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_str(self.version, "version", pattern=r"v[0-9]+")
        _parse_timestamp(self.author_date_start, "author_date_start")
        _parse_timestamp(self.author_date_end, "author_date_end")
        if self.author_date_start > self.author_date_end:
            raise ContractError("author date window: start must not exceed end")
        require_int(self.max_changed_files, "max_changed_files", minimum=1)
        require_int(self.max_changed_lines, "max_changed_lines", minimum=1)
        require_int(
            self.min_behavioral_changed_lines,
            "min_behavioral_changed_lines",
            minimum=1,
        )
        if self.min_behavioral_changed_lines > self.max_changed_lines:
            raise ContractError(
                "min_behavioral_changed_lines: must not exceed max_changed_lines"
            )
        if not isinstance(self.rule_ids, tuple) or not self.rule_ids:
            raise ContractError("rule_ids: expected non-empty tuple")
        if len(set(self.rule_ids)) != len(self.rule_ids):
            raise ContractError("rule_ids: duplicate value")
        if tuple(sorted(self.rule_ids)) != self.rule_ids:
            raise ContractError("rule_ids: expected sorted values")

    def identity_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "author_date_start": self.author_date_start,
            "author_date_end": self.author_date_end,
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
            "min_behavioral_changed_lines": self.min_behavioral_changed_lines,
            "rule_ids": list(self.rule_ids),
        }

    @property
    def rule_set_hash(self) -> str:
        return canonical_sha256(self.identity_payload())

    @property
    def rule_set_id(self) -> str:
        return "screening:v1:" + self.rule_set_hash.removeprefix("sha256:")


V07_BOUNDARY_SCREENING_V1 = ScreeningRuleSet(
    version="v1",
    author_date_start="2024-01-01T00:00:00Z",
    author_date_end="2025-04-30T23:59:59Z",
    max_changed_files=3,
    max_changed_lines=200,
    min_behavioral_changed_lines=20,
    rule_ids=tuple(sorted(RULE_IDS)),
)


def derive_disposition(findings: Sequence[ScreeningFinding]) -> str:
    return derived_disposition(tuple(findings))


def _finding(
    *,
    code: str,
    severity: str,
    rule_id: str,
    message: str,
    observed: object,
    expected: dict[str, object],
) -> ScreeningFinding:
    return ScreeningFinding(
        code=code,
        severity=severity,
        rule_id=rule_id,
        message=message,
        observed=observed,
        expected=expected,
    )


def screen_candidate(
    candidate: CandidateRecord,
    rules: ScreeningRuleSet = V07_BOUNDARY_SCREENING_V1,
) -> DecisionRecord:
    if not isinstance(candidate, CandidateRecord):
        raise ContractError("candidate: expected CandidateRecord")
    if not isinstance(rules, ScreeningRuleSet):
        raise ContractError("rules: expected ScreeningRuleSet")

    findings: list[ScreeningFinding] = []
    title = candidate.title.casefold()
    if title.startswith("revert") or title.startswith("reland"):
        findings.append(
            _finding(
                code="metadata.revert_or_reland",
                severity="reject",
                rule_id="v07-revert-title",
                message="Revert and reland changes are not standalone bug candidates.",
                observed=candidate.title,
                expected={"title_prefix_excluded": ["revert", "reland"]},
            )
        )

    if candidate.author_date is None or candidate.merge_date is None:
        findings.append(
            _finding(
                code="metadata.missing_date",
                severity="defer",
                rule_id="v07-required-dates",
                message="Author and merge dates are required for stable-window screening.",
                observed={
                    "author_date": candidate.author_date,
                    "merge_date": candidate.merge_date,
                },
                expected={"author_date": "RFC3339", "merge_date": "RFC3339"},
            )
        )
    elif not (
        rules.author_date_start
        <= candidate.author_date
        <= rules.author_date_end
    ):
        if candidate.environment_freeze is None:
            findings.append(
                _finding(
                    code="window.outside_stable",
                    severity="reject",
                    rule_id="v07-stable-author-window",
                    message="Author date is outside the stable candidate window.",
                    observed=candidate.author_date,
                    expected={
                        "minimum": rules.author_date_start,
                        "maximum": rules.author_date_end,
                    },
                )
            )
        else:
            findings.append(
                _finding(
                    code="window.environment_freeze_exception",
                    severity="warning",
                    rule_id="v07-stable-author-window",
                    message="An explicit environment freeze permits window review.",
                    observed=candidate.author_date,
                    expected={
                        "environment_freeze": candidate.environment_freeze.content_hash
                    },
                )
            )

    if candidate.change_kind != "bugfix":
        findings.append(
            _finding(
                code="change.non_bug_change",
                severity="reject",
                rule_id="v07-change-kind",
                message="Pure refactor, cleanup, and feature changes are excluded.",
                observed=candidate.change_kind,
                expected={"allowed": ["bugfix"]},
            )
        )

    if (
        candidate.total_files > rules.max_changed_files
        or candidate.total_changed_lines > rules.max_changed_lines
    ):
        findings.append(
            _finding(
                code="change.large_diff_requires_review",
                severity="defer",
                rule_id="v07-normal-diff-size",
                message="The change exceeds normal automatic screening thresholds.",
                observed={
                    "files": candidate.total_files,
                    "changed_lines": candidate.total_changed_lines,
                },
                expected={
                    "maximum_files": rules.max_changed_files,
                    "maximum_changed_lines": rules.max_changed_lines,
                },
            )
        )

    has_changed_test = any(item.is_test for item in candidate.changed_files)
    has_executable_test = has_changed_test or candidate.external_test is not None
    if (
        candidate.total_changed_lines < rules.min_behavioral_changed_lines
        and not has_executable_test
    ):
        findings.append(
            _finding(
                code="change.below_behavioral_threshold",
                severity="reject",
                rule_id="v07-minimum-behavioral-change",
                message="A tiny change without an executable test delta is excluded.",
                observed=candidate.total_changed_lines,
                expected={
                    "minimum_changed_lines": rules.min_behavioral_changed_lines,
                    "or_executable_test": True,
                },
            )
        )
    if not has_executable_test:
        findings.append(
            _finding(
                code="test.missing_executable_delta",
                severity="reject",
                rule_id="v07-executable-test-delta",
                message="A changed test or explicit external test is required.",
                observed=False,
                expected={"executable_test_delta": True},
            )
        )

    if candidate.base_commit is None or candidate.merge_commit is None:
        findings.append(
            _finding(
                code="source.missing_commit_identity",
                severity="defer",
                rule_id="v07-commit-identities",
                message="Base and merge commit identities are required.",
                observed={
                    "base_commit": candidate.base_commit,
                    "merge_commit": candidate.merge_commit,
                },
                expected={"base_commit": "git_sha1", "merge_commit": "git_sha1"},
            )
        )
    if not candidate.source_available:
        findings.append(
            _finding(
                code="source.unavailable",
                severity="defer",
                rule_id="v07-source-availability",
                message="The required source snapshot is not available.",
                observed=False,
                expected={"source_available": True},
            )
        )
    if not candidate.runtime_supported:
        findings.append(
            _finding(
                code="runtime.unsupported",
                severity="defer",
                rule_id="v07-runtime-support",
                message="The candidate runtime requirements are not currently supported.",
                observed=False,
                expected={"runtime_supported": True},
            )
        )
    if candidate.proposed_dimension != "boundary":
        findings.append(
            _finding(
                code="taxonomy.not_boundary",
                severity="reject",
                rule_id="v07-boundary-taxonomy",
                message="The v0.7 Boundary screener only accepts Boundary candidates.",
                observed=candidate.proposed_dimension,
                expected={"dimension": "boundary"},
            )
        )

    findings.append(
        _finding(
            code="review.root_cause_required",
            severity="warning",
            rule_id="v07-root-cause-review",
            message="Root-cause classification requires human review.",
            observed=False,
            expected={"reviewed": True},
        )
    )
    ordered = tuple(sorted(findings, key=finding_sort_key))
    disposition = derive_disposition(ordered)
    decision_id = DecisionRecord.decision_id_for(
        candidate_id=candidate.candidate_id,
        candidate_content_hash=candidate.content_hash,
        rule_set_hash=rules.rule_set_hash,
        decision_source="automation",
        prior_decision_hash=None,
    )
    return DecisionRecord(
        decision_id=decision_id,
        candidate_id=candidate.candidate_id,
        candidate_content_hash=candidate.content_hash,
        rule_set_id=rules.rule_set_id,
        rule_set_hash=rules.rule_set_hash,
        target_dimension=candidate.proposed_dimension,
        target_subclass=candidate.proposed_subclass,
        findings=ordered,
        disposition=disposition,
        decision_source="automation",
        prior_decision=None,
        created_at=candidate.created_at,
    )
