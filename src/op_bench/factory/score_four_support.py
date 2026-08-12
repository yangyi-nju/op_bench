"""Typed, immutable support evidence for retained score-four historical Tasks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType

from op_bench.factory.artifacts import load_regular_file_bytes
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError


_HASH = re.compile(r"sha256:[0-9a-f]{64}")
_ATTEMPT = re.compile(r"attempt:v1:[0-9a-f]{64}")
_PUBLIC_TASK_IDS = (
    "opbench-v07-t0003",
    "opbench-v07-t0022",
    "opbench-v07-t0024",
)
_PILOT_ARTIFACT_HASH = (
    "sha256:c11e707818156ed319f6ac9c31ba475a2cbf0d36df86d417f8e93f2bfc93048f"
)
_SECOND_REVIEW_ARTIFACT_HASH = (
    "sha256:9a1b74984480ffed6982152a7a4647ab17aab40475e2a2b2b63980942d316add"
)
_SECOND_REVIEW_CONTENT_HASH = (
    "sha256:70f6e32eff7f92c1458e3642c4c88d193bfba53cbf2aa7469cb41a0b69b43438"
)
_SECOND_REVIEW_SOURCE_HASH = (
    "sha256:daa89c88fb5e3af71fa0cd0048bb1864b73ad21cbbeec7efebc639548b108401"
)
_SECOND_REVIEWER = "complexity-second-reviewer-v07-independent-01"
_EXPECTED_PILOT_FACTS = {
    "opbench-v07-t0003": {
        "duration_ms": 153723,
        "evaluation_outcome": "resolved",
        "expected_attempt_id": (
            "attempt:v1:"
            "b07d1351633af5af631177d756697034194742126c4b7d99270ccea19c03c123"
        ),
        "task_view_sha256": (
            "sha256:"
            "c1641016e38611cbc08adb3b25dbabf5c6978d82cdcb480a0b963dba723893ae"
        ),
        "terminal_reason": "agent_finished",
        "validity": "valid",
    },
    "opbench-v07-t0022": {
        "duration_ms": 119072,
        "evaluation_outcome": "f2p_failed",
        "expected_attempt_id": (
            "attempt:v1:"
            "a2dfc5603607621b6be06ab89d33fed78c879b231b6858c7e64d99cd6e9c5529"
        ),
        "task_view_sha256": (
            "sha256:"
            "c758facff675e2c93ec1709099cb8e86df439288e7e51d942f49ba4b69587ceb"
        ),
        "terminal_reason": "agent_finished",
        "validity": "valid",
    },
    "opbench-v07-t0024": {
        "duration_ms": 126484,
        "evaluation_outcome": "f2p_failed",
        "expected_attempt_id": (
            "attempt:v1:"
            "c7c9fc5b8ae74e2b50f7c26078133a2d8f1abbfe454650a26e1899b8580e9773"
        ),
        "task_view_sha256": (
            "sha256:"
            "8f7ca083b730aec84db0ccb4d1a4e1c585925facc36073a45b5a568065c9adc6"
        ),
        "terminal_reason": "agent_finished",
        "validity": "valid",
    },
}


@dataclass(frozen=True)
class ScoreFourSupport:
    """One exact pilot fact and its independent second-review decision."""

    public_task_id: str
    pilot_fact: Mapping[str, object]
    pilot_record_hash: str
    second_review_record_hash: str
    counts_toward_final: bool
    pilot_decision: str
    second_review: bool
    reviewer: str

    def blind_pilot_binding(self) -> dict[str, object]:
        return {
            **self.pilot_fact,
            "public_task_id": self.public_task_id,
            "counts_toward_final": self.counts_toward_final,
            "decision": self.pilot_decision,
            "complexity_evidence_decision": "accepted",
            "complexity_evidence_severity": "none",
            "factual_evidence_hash": _PILOT_ARTIFACT_HASH,
            "factual_record_hash": self.pilot_record_hash,
            "second_review_artifact_hash": _SECOND_REVIEW_ARTIFACT_HASH,
            "second_review_content_hash": _SECOND_REVIEW_CONTENT_HASH,
            "second_review_record_hash": self.second_review_record_hash,
            "second_review_source_hash": _SECOND_REVIEW_SOURCE_HASH,
            "reviewer": self.reviewer,
        }

    def second_review_binding(self) -> dict[str, object]:
        return {
            "public_task_id": self.public_task_id,
            "artifact_hash": _SECOND_REVIEW_ARTIFACT_HASH,
            "artifact_content_hash": _SECOND_REVIEW_CONTENT_HASH,
            "record_hash": self.second_review_record_hash,
            "source_sha256": _SECOND_REVIEW_SOURCE_HASH,
            "reviewer": self.reviewer,
            "pilot_decision": self.pilot_decision,
            "complexity_evidence_decision": "accepted",
            "complexity_evidence_severity": "none",
            "second_review": self.second_review,
        }


def load_score_four_support(
    pilot_path: Path,
    second_review_path: Path,
) -> Mapping[str, ScoreFourSupport]:
    """Load the two tracked score-four support artifacts without raw-run inputs."""

    pilot_bytes, pilot = _load_json_object(pilot_path, "score-four pilot")
    second_bytes, second = _load_json_object(
        second_review_path,
        "score-four second review",
    )
    pilot_records = _validate_pilot(pilot)
    second_records = _validate_second_review(second)
    _expect(
        _bytes_hash(pilot_bytes) == _PILOT_ARTIFACT_HASH,
        "score-four pilot: artifact bytes hash mismatch",
    )
    _expect(
        _bytes_hash(second_bytes) == _SECOND_REVIEW_ARTIFACT_HASH,
        "score-four second review: artifact bytes hash mismatch",
    )

    support: dict[str, ScoreFourSupport] = {}
    for public_task_id in _PUBLIC_TASK_IDS:
        pilot_record = pilot_records[public_task_id]
        second_record = second_records[public_task_id]
        support[public_task_id] = ScoreFourSupport(
            public_task_id=public_task_id,
            pilot_fact=MappingProxyType(dict(pilot_record)),
            pilot_record_hash=canonical_sha256(pilot_record),
            second_review_record_hash=canonical_sha256(second_record),
            counts_toward_final=False,
            pilot_decision="accepted",
            second_review=True,
            reviewer=_SECOND_REVIEWER,
        )
    return MappingProxyType(support)


def validate_score_four_review_binding(
    support: Mapping[str, ScoreFourSupport],
    *,
    public_task_id: str,
    prompt_review: object,
    complexity_review: object,
    source_evidence: object,
) -> None:
    """Bind a combined retained review to the two independently loaded artifacts."""

    item = support.get(public_task_id)
    if item is None:
        raise ContractError("score-four review: public Task ID lacks support")
    prompt = _mapping(prompt_review, "score-four review.prompt")
    complexity = _mapping(complexity_review, "score-four review.complexity")
    sources = _mapping(source_evidence, "score-four review.source_evidence")
    if set(sources) != {
        "blind_pilot",
        "blind_review",
        "second_complexity_review",
        "semantic_review",
    }:
        raise ContractError(
            "score-four review.source_evidence: unexpected contract fields"
        )
    _expect(
        complexity.get("second_review") is True,
        "score-four review.complexity.second_review: true required",
    )
    _expect_exact(
        complexity.get("blind_pilot"),
        item.blind_pilot_binding(),
        "score-four review.complexity.blind_pilot",
    )
    _expect_exact(
        sources.get("blind_pilot"),
        item.blind_pilot_binding(),
        "score-four review.source_evidence.blind_pilot",
    )
    _expect_exact(
        sources.get("second_complexity_review"),
        item.second_review_binding(),
        "score-four review.source_evidence.second_complexity_review",
    )

    prompt_blind = _mapping(
        prompt.get("blind_review"),
        "score-four review.prompt.blind_review",
    )
    prompt_semantic = _mapping(
        prompt.get("semantic_review"),
        "score-four review.prompt.semantic_review",
    )
    source_blind = _mapping(
        sources.get("blind_review"),
        "score-four review.source_evidence.blind_review",
    )
    source_semantic = _mapping(
        sources.get("semantic_review"),
        "score-four review.source_evidence.semantic_review",
    )
    blind_reviewer = _nonempty(prompt_blind.get("reviewer"), "blind reviewer")
    semantic_reviewer = _nonempty(
        complexity.get("reviewer"),
        "complexity reviewer",
    )
    _expect(
        source_blind.get("reviewer") == blind_reviewer,
        "score-four review: blind source reviewer mismatch",
    )
    _expect(
        prompt_semantic.get("reviewer") == semantic_reviewer
        and source_semantic.get("reviewer") == semantic_reviewer,
        "score-four review: semantic reviewer mismatch",
    )
    _expect(
        item.reviewer not in {blind_reviewer, semantic_reviewer},
        "score-four review: second reviewer must be independent",
    )


def _validate_pilot(
    value: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    required = {
        "artifact_hashes",
        "cohort_id",
        "comparability_key",
        "contract_type",
        "counts_toward_final",
        "execution_status",
        "expected_attempt_count",
        "manifest_hash",
        "missing_attempt_count",
        "observed_attempt_count",
        "outcomes_are_factual_only",
        "results_hash",
        "schema_version",
        "tasks",
    }
    _exact_fields(value, required, "score-four pilot")
    _expect(
        value["contract_type"] == "historical_blind_pilot_factual_evidence"
        and value["schema_version"] == "v1",
        "score-four pilot: unsupported contract",
    )
    _expect(
        value["counts_toward_final"] is False
        and value["outcomes_are_factual_only"] is True,
        "score-four pilot: factual non-counting evidence required",
    )
    _expect(
        value["execution_status"] == "completed"
        and value["expected_attempt_count"] == 3
        and value["observed_attempt_count"] == 3
        and value["missing_attempt_count"] == 0,
        "score-four pilot: exact completed attempt counts required",
    )
    artifact_hashes = _mapping(
        value["artifact_hashes"],
        "score-four pilot.artifact_hashes",
    )
    _exact_fields(
        artifact_hashes,
        {
            "attempts_jsonl",
            "integrity_json",
            "preparation_contract_json",
            "results_jsonl",
            "run_manifest_json",
            "summary_json",
        },
        "score-four pilot.artifact_hashes",
    )
    for field, digest in artifact_hashes.items():
        _hash(digest, f"score-four pilot.artifact_hashes.{field}")
    comparability_key = _hash(
        value["comparability_key"],
        "score-four pilot.comparability_key",
    )
    _expect(
        value["cohort_id"] == f"cohort:v1:{comparability_key.removeprefix('sha256:')}",
        "score-four pilot.cohort_id: comparability mismatch",
    )
    _hash(value["manifest_hash"], "score-four pilot.manifest_hash")
    results_hash = _hash(value["results_hash"], "score-four pilot.results_hash")
    _expect(
        results_hash == artifact_hashes["results_jsonl"],
        "score-four pilot.results_hash: artifact mismatch",
    )
    entries = value["tasks"]
    _expect(
        isinstance(entries, list) and len(entries) == 3,
        "score-four pilot.tasks: expected exactly 3 records",
    )
    records: dict[str, Mapping[str, object]] = {}
    for index, entry in enumerate(entries):
        record = _mapping(entry, f"score-four pilot.tasks[{index}]")
        _exact_fields(
            record,
            {
                "duration_ms",
                "evaluation_outcome",
                "expected_attempt_id",
                "public_task_id",
                "task_view_sha256",
                "terminal_reason",
                "validity",
            },
            f"score-four pilot.tasks[{index}]",
        )
        public_task_id = _nonempty(
            record["public_task_id"],
            f"score-four pilot.tasks[{index}].public_task_id",
        )
        _expect(
            public_task_id in _PUBLIC_TASK_IDS and public_task_id not in records,
            "score-four pilot.tasks: duplicate or unexpected public Task ID",
        )
        _expect(
            isinstance(record["duration_ms"], int)
            and not isinstance(record["duration_ms"], bool)
            and record["duration_ms"] > 0,
            f"score-four pilot.tasks[{index}].duration_ms: positive integer required",
        )
        _expect(
            isinstance(record["expected_attempt_id"], str)
            and _ATTEMPT.fullmatch(record["expected_attempt_id"]) is not None,
            f"score-four pilot.tasks[{index}].expected_attempt_id: invalid",
        )
        _hash(
            record["task_view_sha256"],
            f"score-four pilot.tasks[{index}].task_view_sha256",
        )
        expected = {
            **_EXPECTED_PILOT_FACTS[public_task_id],
            "public_task_id": public_task_id,
        }
        _expect_exact(
            record,
            expected,
            f"score-four pilot.tasks[{index}]",
        )
        records[public_task_id] = record
    _expect(
        tuple(records) == _PUBLIC_TASK_IDS,
        "score-four pilot.tasks: canonical Task order required",
    )
    return records


def _validate_second_review(
    value: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    _exact_fields(
        value,
        {
            "content_hash",
            "contract_type",
            "created_at",
            "records",
            "reviewer",
            "schema_version",
            "source_sha256",
        },
        "score-four second review",
    )
    _expect(
        value["contract_type"] == "historical_second_complexity_review"
        and value["schema_version"] == "v1",
        "score-four second review: unsupported contract",
    )
    _expect(
        value["content_hash"]
        == canonical_sha256(
            {
                key: item
                for key, item in value.items()
                if key != "content_hash"
            }
        )
        == _SECOND_REVIEW_CONTENT_HASH,
        "score-four second review.content_hash: mismatch",
    )
    _expect(
        value["reviewer"] == _SECOND_REVIEWER
        and value["source_sha256"] == _SECOND_REVIEW_SOURCE_HASH,
        "score-four second review: reviewer or source mismatch",
    )
    entries = value["records"]
    _expect(
        isinstance(entries, list) and len(entries) == 3,
        "score-four second review.records: expected exactly 3 records",
    )
    records: dict[str, Mapping[str, object]] = {}
    for index, entry in enumerate(entries):
        record = _mapping(entry, f"score-four second review.records[{index}]")
        _exact_fields(
            record,
            {
                "complexity_evidence_decision",
                "complexity_evidence_severity",
                "pilot_decision",
                "public_task_id",
                "reviewer",
                "second_review",
                "source_sha256",
            },
            f"score-four second review.records[{index}]",
        )
        public_task_id = _nonempty(
            record["public_task_id"],
            f"score-four second review.records[{index}].public_task_id",
        )
        _expect(
            public_task_id in _PUBLIC_TASK_IDS and public_task_id not in records,
            "score-four second review.records: duplicate or unexpected public Task ID",
        )
        _expect(
            record
            == {
                "complexity_evidence_decision": "accepted",
                "complexity_evidence_severity": "none",
                "pilot_decision": "accepted",
                "public_task_id": public_task_id,
                "reviewer": _SECOND_REVIEWER,
                "second_review": True,
                "source_sha256": _SECOND_REVIEW_SOURCE_HASH,
            },
            f"score-four second review.records[{index}]: decision mismatch",
        )
        records[public_task_id] = record
    _expect(
        tuple(records) == _PUBLIC_TASK_IDS,
        "score-four second review.records: canonical Task order required",
    )
    return records


def _load_json_object(
    path: Path,
    label: str,
) -> tuple[bytes, Mapping[str, object]]:
    encoded = load_regular_file_bytes(path)
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{label}: invalid JSON") from exc
    _expect(isinstance(value, Mapping), f"{label}: expected object")
    _expect(
        encoded == (canonical_json(value) + "\n").encode("utf-8"),
        f"{label}: expected canonical JSON bytes with terminal newline",
    )
    return encoded, value


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    return value


def _exact_fields(
    value: Mapping[str, object],
    required: set[str],
    path: str,
) -> None:
    if set(value) != required:
        raise ContractError(f"{path}: unexpected contract fields")


def _expect_exact(value: object, expected: object, path: str) -> None:
    if value != expected:
        raise ContractError(f"{path}: support binding mismatch")


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _hash(value: object, path: str) -> str:
    text = _nonempty(value, path)
    if _HASH.fullmatch(text) is None:
        raise ContractError(f"{path}: expected sha256 digest")
    return text


def _nonempty(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{path}: expected non-empty string")
    return value


def _bytes_hash(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"
