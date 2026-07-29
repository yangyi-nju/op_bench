from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from op_bench.factory.artifacts import load_factory_contract
from op_bench.factory.quality_release import (
    HARD_CANDIDATE_REJECTION_REASONS,
    QualityCandidateDecision,
    QualityCandidateRecord,
    QualityMainHistoryReversalFinding,
    derive_quality_main_history_findings,
    validate_candidate_index,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/screen_v07_quality_candidates.py"
CAPTURES = ROOT / "factory/v0.7/p8/captures.json"
RECEIPTS = ROOT / "factory/v0.7/p8/acquisition_receipts.json"
HISTORICAL = ROOT / "factory/v0.7/p7/historical_readmission.json"
INDEX = ROOT / "factory/v0.7/p8/screening/screening_index.json"
CREATED_AT = "2026-07-29T00:00:00Z"
HISTORICAL_PRS = {
    int(record["task_id"].split("__", 2)[1])
    for record in json.loads(HISTORICAL.read_text(encoding="utf-8"))["records"]
}
FORBIDDEN_FINAL_OR_PUBLIC_FIELDS = {
    "admission",
    "agent_prompt",
    "agent_task_view",
    "complexity",
    "difficulty",
    "final_taxonomy",
    "gold_patch",
    "hidden_test",
    "prompt",
    "public_task_id",
    "root_cause_tags",
    "task_view",
    "taxonomy",
}


def _capture(
    *,
    number: int = 190001,
    merge: str = "b" * 40,
    base: str = "a" * 40,
    change_kind: str = "bugfix",
    source_available: bool = True,
    runtime_supported: bool = True,
    required_hardware: list[str] | None = None,
    behavioral_test_evidence: bool = True,
    changed_file_count: int = 2,
    base_ref_name: str = "main",
    head_ref_name: str = "fix-example",
    title: str = "Fix compiled backward metadata mismatch",
    description: str = "A compiled backward path returned the wrong metadata.",
) -> dict[str, object]:
    changed_files = [
        {
            "path": "test/test_example.py",
            "additions": 18,
            "deletions": 1,
            "change_type": "MODIFIED",
            "is_test": True,
        },
        {
            "path": "torch/example.py",
            "additions": 12,
            "deletions": 5,
            "change_type": "MODIFIED",
            "is_test": False,
        },
    ]
    return {
        "repository": "pytorch/pytorch",
        "pr_number": number,
        "pr_url": f"https://github.com/pytorch/pytorch/pull/{number}",
        "base_commit": base,
        "merge_commit": merge,
        "base_ref_name": base_ref_name,
        "head_ref_name": head_ref_name,
        "merged_at": "2026-07-20T01:02:03Z",
        "title": title,
        "description": description,
        "linked_issues": [
            {
                "number": number - 1,
                "url": f"https://github.com/pytorch/pytorch/issues/{number - 1}",
            }
        ],
        "changed_files": changed_files,
        "changed_file_count": changed_file_count,
        "behavioral_test_evidence": behavioral_test_evidence,
        "change_kind": change_kind,
        "source_available": source_available,
        "runtime_supported": runtime_supported,
        "required_hardware": (
            ["cpu", "cuda"] if required_hardware is None else required_hardware
        ),
        "execution_hints": {
            "devices": ["cpu", "cuda"],
            "modes": ["eager", "compile"],
            "phases": ["forward", "backward"],
            "distributed": False,
        },
        "proposed_contract_families": ["tensor_metadata", "gradient"],
        "proposed_trigger_tags": [
            "mixed_dtype_or_precision_mode",
            "dynamic_shape",
        ],
        "preliminary_review_reasons": [],
    }


def _changed_files_hash(candidate: dict[str, object]) -> str:
    return canonical_sha256(
        [
            {
                "path": item["path"],
                "additions": item["additions"],
                "deletions": item["deletions"],
                "change_type": item["change_type"],
            }
            for item in candidate["changed_files"]
        ]
    )


def _receipt(candidate: dict[str, object]) -> dict[str, object]:
    resolved_marker = (
        "Pull Request resolved: "
        f"https://github.com/pytorch/pytorch/pull/{candidate['pr_number']}"
    )
    selected_commit_message = (
        f"{candidate['title']}\n\n{resolved_marker}"
    )
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_acquisition_receipt",
        "schema_version": "v1",
        "repository": candidate["repository"],
        "pr_number": candidate["pr_number"],
        "pr_url": candidate["pr_url"],
        "merged_at": candidate["merged_at"],
        "merge_commit": candidate["merge_commit"],
        "base_commit": candidate["base_commit"],
        "base_ref_name": candidate["base_ref_name"],
        "head_ref_name": candidate["head_ref_name"],
        "resolver_mode": "main_history_exact_resolved_marker",
        "main_history_head_oid": "f" * 40,
        "main_history_commit": candidate["merge_commit"],
        "main_history_first_parent": candidate["base_commit"],
        "main_history_committed_at": candidate["merged_at"],
        "resolved_marker": resolved_marker,
        "selected_commit_message": selected_commit_message,
        "selected_commit_message_hash": canonical_sha256(
            selected_commit_message
        ),
        "main_history_exact_marker_hits": [
            {
                "oid": candidate["merge_commit"],
                "first_parent_oid": candidate["base_commit"],
                "committed_at": candidate["merged_at"],
                "message_hash": canonical_sha256(
                    selected_commit_message
                ),
            }
        ],
        "main_history_reversal_findings": [],
        "files_total_count": candidate["changed_file_count"],
        "files_captured_node_count": candidate["changed_file_count"],
        "files_has_next_page": False,
        "files_pagination_complete": True,
        "changed_files_hash": _changed_files_hash(candidate),
        "capture_method": "authenticated_read_only_gh_api_graphql",
        "captured_at": CREATED_AT,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _receipt_set(candidates: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_acquisition_receipt_set",
        "schema_version": "v1",
        "repository": "pytorch/pytorch",
        "captured_at": CREATED_AT,
        "capture_method": "authenticated_read_only_gh_api_graphql",
        "main_history_scan": {
            "ref_name": "main",
            "head_oid": "f" * 40,
            "since": "2026-03-01T00:00:00Z",
            "page_count": 1,
            "commit_count": 1,
            "final_has_next_page": False,
            "scanned_commit_facts_hash": canonical_sha256(
                [{"oid": "f" * 40}]
            ),
            "finding_rule_version": "v1",
        },
        "receipts": [_receipt(candidate) for candidate in candidates],
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _capture_set(
    candidates: list[dict[str, object]],
    receipt_set_hash: str,
) -> dict[str, object]:
    receipts = {
        receipt["pr_number"]: receipt
        for receipt in _receipt_set(candidates)["receipts"]
    }
    bound_candidates = [
        {
            **candidate,
            "acquisition_receipt_hash": receipts[
                candidate["pr_number"]
            ]["content_hash"],
        }
        for candidate in candidates
    ]
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_capture_set",
        "schema_version": "v1",
        "repository": "pytorch/pytorch",
        "captured_at": "2026-07-29T00:00:00Z",
        "acquisition_receipt_set_hash": receipt_set_hash,
        "acquisition": {
            "connector_first": True,
            "connector_queries": [
                "CPU eager bug fix test",
                "CUDA backward bug fix test",
            ],
            "bulk_method": "authenticated_read_only_gh_api_graphql",
            "merge_commit_rule": "pull_request.mergeCommit.oid",
            "base_commit_rule": "landed_commit.parents(first:1).nodes[0].oid",
            "changed_files_rule": "pull_request.files complete connection",
        },
        "candidates": bound_candidates,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _write_acquisition_inputs(
    capture_path: Path,
    receipt_path: Path,
    candidates: list[dict[str, object]],
) -> None:
    receipt_set = _receipt_set(candidates)
    receipt_path.write_bytes(canonical_json(receipt_set).encode("utf-8"))
    capture_path.write_bytes(
        canonical_json(
            _capture_set(candidates, receipt_set["content_hash"])
        ).encode("utf-8")
    )


def _run_screen(
    capture_path: Path,
    output_dir: Path,
    *,
    receipt_path: Path | None = None,
    historical_path: Path = HISTORICAL,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        (
            str(ROOT / ".venv/bin/python"),
            str(SCRIPT),
            "--input",
            str(capture_path.resolve()),
            "--receipts",
            str(
                (
                    receipt_path
                    if receipt_path is not None
                    else capture_path.with_name("receipts.json")
                ).resolve()
            ),
            "--historical-index",
            str(historical_path.resolve()),
            "--output-dir",
            str(output_dir.resolve()),
            "--created-at",
            CREATED_AT,
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _assert_no_forbidden_keys(
    testcase: unittest.TestCase,
    value: object,
) -> None:
    if isinstance(value, dict):
        testcase.assertFalse(
            set(value) & FORBIDDEN_FINAL_OR_PUBLIC_FIELDS,
            set(value) & FORBIDDEN_FINAL_OR_PUBLIC_FIELDS,
        )
        for item in value.values():
            _assert_no_forbidden_keys(testcase, item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_forbidden_keys(testcase, item)


class QualityCandidateContractTests(unittest.TestCase):
    def test_reversal_finding_targets_are_kind_exclusive(self) -> None:
        finding = {
            "event_oid": "d" * 40,
            "committed_at": "2026-07-20T01:02:04Z",
            "message_hash": "sha256:" + "d" * 64,
            "kind": "revert_commit",
            "target_pr_number": 190777,
            "target_commit_oid": "c" * 40,
        }

        with self.assertRaisesRegex(ContractError, "target_pr_number"):
            QualityMainHistoryReversalFinding.from_dict(
                finding,
                path="finding",
            )

    def test_schema_tracks_candidate_decision_and_index_wire_contracts(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v07_quality_release.schema.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = schema["$defs"]["quality_candidate"]
        decision = schema["$defs"]["quality_candidate_decision"]
        receipt = schema["$defs"]["quality_candidate_acquisition_receipt"]
        receipt_set = schema[
            "$defs"
        ]["quality_candidate_acquisition_receipt_set"]
        index = schema["$defs"]["quality_candidate_screening_index"]
        self.assertEqual(
            set(candidate["required"]),
            set(QualityCandidateRecord.wire_fields()),
        )
        self.assertEqual(
            set(candidate["properties"]),
            set(QualityCandidateRecord.wire_fields()),
        )
        self.assertEqual(
            set(decision["required"]),
            set(QualityCandidateDecision.wire_fields()),
        )
        self.assertEqual(
            set(decision["properties"]),
            set(QualityCandidateDecision.wire_fields()),
        )
        self.assertEqual(
            set(receipt["required"]),
            {
                "contract_type",
                "schema_version",
                "repository",
                "pr_number",
                "pr_url",
                "merged_at",
                "merge_commit",
                "base_commit",
                "base_ref_name",
                "head_ref_name",
                "resolver_mode",
                "main_history_head_oid",
                "main_history_commit",
                "main_history_first_parent",
                "main_history_committed_at",
                "resolved_marker",
                "selected_commit_message",
                "selected_commit_message_hash",
                "main_history_exact_marker_hits",
                "main_history_reversal_findings",
                "files_total_count",
                "files_captured_node_count",
                "files_has_next_page",
                "files_pagination_complete",
                "changed_files_hash",
                "capture_method",
                "captured_at",
                "content_hash",
            },
        )
        self.assertEqual(
            set(receipt_set["required"]),
            set(receipt_set["properties"]),
        )
        self.assertEqual(set(index["required"]), set(index["properties"]))

    def test_cli_writes_typed_canonical_candidate_decision_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            first = temporary / "first"
            second = temporary / "second"
            _write_acquisition_inputs(captures, receipts, [_capture()])

            accepted = _run_screen(
                captures, first, receipt_path=receipts
            )
            repeated = _run_screen(
                captures, second, receipt_path=receipts
            )

            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertEqual(_tree_hash(first), _tree_hash(second))
            index = json.loads(
                (first / "screening_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(index["historical_k"], 14)
            self.assertEqual(index["required_candidate_count"], 108)
            self.assertEqual(index["candidate_count"], 1)
            self.assertEqual(
                index["disposition_counts"],
                {
                    "accepted_for_build": 1,
                    "deferred_for_review": 0,
                    "hard_rejected": 0,
                },
            )
            entry = index["records"][0]
            candidate = load_factory_contract(
                first / entry["candidate"]["relative_path"]
            )
            decision = load_factory_contract(
                first / entry["decision"]["relative_path"]
            )
            self.assertIsInstance(candidate, QualityCandidateRecord)
            self.assertIsInstance(decision, QualityCandidateDecision)
            self.assertEqual(candidate.candidate_status, "accepted_for_build")
            self.assertEqual(decision.disposition, "accepted_for_build")
            self.assertEqual(
                index["content_hash"],
                canonical_sha256(
                    {
                        key: value
                        for key, value in index.items()
                        if key != "content_hash"
                    }
                ),
            )

    def test_hard_rejection_reasons_are_exact_and_preliminary(self) -> None:
        mutations = {
            "source unavailable": {
                "source_available": False,
                "reason": "source.unavailable",
            },
            "unsupported runtime": {
                "runtime_supported": False,
                "reason": "runtime.unsupported_cpu_cuda",
            },
            "cleanup": {
                "change_kind": "cleanup",
                "reason": "change.documentation_cleanup_refactor_only",
            },
            "no test": {
                "behavioral_test_evidence": False,
                "reason": "test.no_behavioral_evidence",
            },
            "outside hardware": {
                "required_hardware": ["xpu"],
                "reason": "runtime.hardware_outside_v07_scope",
            },
        }
        self.assertEqual(
            HARD_CANDIDATE_REJECTION_REASONS,
            (
                "change.documentation_cleanup_refactor_only",
                "duplicate.exact_provenance",
                "runtime.hardware_outside_v07_scope",
                "runtime.unsupported_cpu_cuda",
                "source.missing_immutable_commits",
                "source.unavailable",
                "test.no_behavioral_evidence",
            ),
        )
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures = temporary / "captures.json"
                receipts = temporary / "receipts.json"
                candidate = _capture()
                mutated_field = next(
                    key for key in mutation if key != "reason"
                )
                candidate[mutated_field] = next(
                    value
                    for key, value in mutation.items()
                    if key != "reason"
                )
                if name == "no test":
                    candidate["changed_files"] = [
                        {
                            **item,
                            "is_test": False,
                        }
                        for item in candidate["changed_files"]
                    ]
                _write_acquisition_inputs(
                    captures, receipts, [candidate]
                )
                output = temporary / "screening"
                completed = _run_screen(
                    captures, output, receipt_path=receipts
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                index = json.loads(
                    (output / "screening_index.json").read_text(encoding="utf-8")
                )
                self.assertEqual(index["disposition_counts"]["hard_rejected"], 1)
                decision = load_factory_contract(
                    output / index["records"][0]["decision"]["relative_path"]
                )
                self.assertIn(mutation["reason"], decision.hard_rejection_reasons)
                _assert_no_forbidden_keys(self, decision.to_dict())

    def test_malformed_sha_duplicate_pr_and_incomplete_files_fail_closed(self) -> None:
        cases = {
            "missing immutable main provenance": (
                [_capture(base=None)],
                "main_history_first_parent",
            ),
            "malformed sha": (
                [_capture(base="abc")],
                "base_commit",
            ),
            "duplicate PR": (
                [_capture(), _capture(merge="c" * 40)],
                "duplicate",
            ),
            "incomplete files": (
                [_capture(changed_file_count=3)],
                "changed_file_count",
            ),
        }
        for name, (candidates, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures = temporary / "captures.json"
                receipts = temporary / "receipts.json"
                _write_acquisition_inputs(captures, receipts, candidates)
                completed = _run_screen(
                    captures,
                    temporary / "screening",
                    receipt_path=receipts,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)

    def test_main_ref_and_backport_markers_fail_closed(self) -> None:
        cases = {
            "release base": (
                _capture(base_ref_name="release/2.7"),
                "base_ref_name",
            ),
            "cherry-pick head": (
                _capture(head_ref_name="cherry-pick-189999"),
                "head_ref_name",
            ),
            "backport body": (
                _capture(
                    description=(
                        "Backport of https://github.com/pytorch/pytorch/"
                        "pull/189999"
                    )
                ),
                "backport",
            ),
        }
        for name, (candidate, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures = temporary / "captures.json"
                receipts = temporary / "receipts.json"
                _write_acquisition_inputs(
                    captures, receipts, [candidate]
                )
                completed = _run_screen(
                    captures,
                    temporary / "screening",
                    receipt_path=receipts,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr.casefold())

    def test_receipts_bind_commits_refs_and_complete_file_metadata(self) -> None:
        cases = {
            "wrong base": lambda candidate: candidate.__setitem__(
                "base_commit", "c" * 40
            ),
            "wrong merge": lambda candidate: candidate.__setitem__(
                "merge_commit", "d" * 40
            ),
            "truncated files": lambda candidate: (
                candidate["changed_files"].pop(),
                candidate.__setitem__("changed_file_count", 1),
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures = temporary / "captures.json"
                receipts = temporary / "receipts.json"
                _write_acquisition_inputs(
                    captures, receipts, [_capture()]
                )
                payload = json.loads(
                    captures.read_text(encoding="utf-8")
                )
                mutate(payload["candidates"][0])
                payload["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "content_hash"
                    }
                )
                captures.write_bytes(
                    canonical_json(payload).encode("utf-8")
                )

                completed = _run_screen(
                    captures,
                    temporary / "screening",
                    receipt_path=receipts,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("receipt", completed.stderr.casefold())

    def test_receipts_reject_mutated_main_history_resolver_evidence(self) -> None:
        cases = {
            "resolver": lambda receipt: receipt.__setitem__(
                "resolver_mode", "pull_request_commits"
            ),
            "marker identity": lambda receipt: receipt.__setitem__(
                "resolved_marker",
                "Pull Request resolved: "
                "https://github.com/pytorch/pytorch/pull/190002",
            ),
            "message digest": lambda receipt: receipt.__setitem__(
                "selected_commit_message_hash",
                "sha256:" + "c" * 64,
            ),
            "message marker removed": lambda receipt: (
                receipt.__setitem__(
                    "selected_commit_message",
                    "Fix without resolved marker",
                ),
                receipt.__setitem__(
                    "selected_commit_message_hash",
                    canonical_sha256("Fix without resolved marker"),
                ),
            ),
            "marker uniqueness": lambda receipt: (
                receipt.__setitem__(
                    "main_history_exact_marker_hits",
                    [
                        *receipt["main_history_exact_marker_hits"],
                        {
                            "oid": "f" * 40,
                            "first_parent_oid": "a" * 40,
                            "committed_at": (
                                "2026-07-20T01:02:02Z"
                            ),
                            "message_hash": "sha256:" + "f" * 64,
                        },
                    ],
                ),
            ),
            "selected marker member": lambda receipt: (
                receipt["main_history_exact_marker_hits"][0].__setitem__(
                    "oid", "f" * 40
                )
            ),
            "reversal findings": lambda receipt: (
                receipt.__setitem__(
                    "main_history_reversal_findings",
                    [
                        {
                            "event_oid": "d" * 40,
                            "committed_at": (
                                "2026-07-20T01:02:04Z"
                            ),
                            "message_hash": "sha256:" + "d" * 64,
                            "kind": "revert_commit",
                            "target_pr_number": None,
                            "target_commit_oid": (
                                receipt["main_history_commit"]
                            ),
                        }
                    ],
                )
            ),
            "history commit": lambda receipt: receipt.__setitem__(
                "main_history_commit", "d" * 40
            ),
            "history parent": lambda receipt: receipt.__setitem__(
                "main_history_first_parent", "e" * 40
            ),
            "history timestamp": lambda receipt: receipt.__setitem__(
                "main_history_committed_at", "2026-07-20T01:02:04Z"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures_path = temporary / "captures.json"
                receipts_path = temporary / "receipts.json"
                _write_acquisition_inputs(
                    captures_path, receipts_path, [_capture()]
                )
                receipts = json.loads(
                    receipts_path.read_text(encoding="utf-8")
                )
                mutate(receipts["receipts"][0])
                receipts["receipts"][0]["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in receipts["receipts"][0].items()
                        if key != "content_hash"
                    }
                )
                receipts["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in receipts.items()
                        if key != "content_hash"
                    }
                )
                receipts_path.write_bytes(
                    canonical_json(receipts).encode("utf-8")
                )

                completed = _run_screen(
                    captures_path,
                    temporary / "screening",
                    receipt_path=receipts_path,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertRegex(
                    completed.stderr.casefold(),
                    r"main_history|resolver|marker|message|receipt",
                )

    def test_receipt_set_requires_complete_fixed_main_history_scan(
        self,
    ) -> None:
        cases = {
            "incomplete pagination": lambda scan: scan.__setitem__(
                "final_has_next_page", True
            ),
            "late epoch": lambda scan: scan.__setitem__(
                "since", "2026-04-01T00:00:00Z"
            ),
            "head mismatch": lambda scan: scan.__setitem__(
                "head_oid", "c" * 40
            ),
            "page count mismatch": lambda scan: scan.__setitem__(
                "commit_count", 101
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                captures_path = temporary / "captures.json"
                receipts_path = temporary / "receipts.json"
                _write_acquisition_inputs(
                    captures_path, receipts_path, [_capture()]
                )
                receipts = json.loads(
                    receipts_path.read_text(encoding="utf-8")
                )
                mutate(receipts["main_history_scan"])
                receipts["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in receipts.items()
                        if key != "content_hash"
                    }
                )
                receipts_path.write_bytes(
                    canonical_json(receipts).encode("utf-8")
                )
                captures = json.loads(
                    captures_path.read_text(encoding="utf-8")
                )
                captures[
                    "acquisition_receipt_set_hash"
                ] = receipts["content_hash"]
                captures["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in captures.items()
                        if key != "content_hash"
                    }
                )
                captures_path.write_bytes(
                    canonical_json(captures).encode("utf-8")
                )

                completed = _run_screen(
                    captures_path,
                    temporary / "screening",
                    receipt_path=receipts_path,
                )

                self.assertNotEqual(completed.returncode, 0)
                self.assertRegex(
                    completed.stderr.casefold(),
                    r"main[_ ]history",
                )

    def test_historical_k_must_equal_retained_dispositions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            historical = temporary / "historical.json"
            _write_acquisition_inputs(captures, receipts, [_capture()])
            historical_payload = json.loads(
                HISTORICAL.read_text(encoding="utf-8")
            )
            historical_payload["k"] = 15
            historical_payload["required_candidate_count"] = 105
            historical_payload["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in historical_payload.items()
                    if key != "content_hash"
                }
            )
            historical.write_bytes(
                canonical_json(historical_payload).encode("utf-8")
            )

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
                historical_path=historical,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("historical", completed.stderr.casefold())

    def test_different_pr_with_same_landed_provenance_is_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            first = _capture(number=190001)
            duplicate = _capture(number=190002)
            _write_acquisition_inputs(
                captures, receipts, [first, duplicate]
            )

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            index = json.loads(
                (
                    temporary / "screening/screening_index.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(index["capture_count"], 2)
            self.assertEqual(index["candidate_count"], 1)
            self.assertEqual(index["exclusion_count"], 1)
            self.assertEqual(
                index["exclusions"],
                [
                    {
                        "repository": "pytorch/pytorch",
                        "pr_number": 190002,
                        "kept_pr_number": 190001,
                        "base_commit": "a" * 40,
                        "merge_commit": "b" * 40,
                        "reason": "duplicate.exact_provenance",
                    }
                ],
            )

    def test_rocm_only_evidence_is_hard_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            candidate = _capture(
                title="Fix HIP kernel launch configuration",
                description=(
                    "ROCm raised a HIP error in hipLaunchKernel. "
                    "The regression is specific to the HIP runtime."
                ),
                required_hardware=["cuda"],
            )
            candidate["execution_hints"]["phases"] = ["forward"]
            candidate["proposed_contract_families"] = ["api_behavior"]
            _write_acquisition_inputs(
                captures, receipts, [candidate]
            )

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            index = json.loads(
                (
                    temporary / "screening/screening_index.json"
                ).read_text(encoding="utf-8")
            )
            decision = load_factory_contract(
                temporary
                / "screening"
                / index["records"][0]["decision"]["relative_path"]
            )
            self.assertEqual(decision.disposition, "hard_rejected")
            self.assertIn(
                "runtime.hardware_outside_v07_scope",
                decision.hard_rejection_reasons,
            )

    def test_scope_reversals_and_low_signal_changes_are_classified(self) -> None:
        hard_cases = {
            "nested DDP": (
                "Fix nested DDP module context",
                "Nested DistributedDataParallel state was cleared.",
                "runtime.hardware_outside_v07_scope",
            ),
            "TorchElastic": (
                "Add TorchElastic signal failure hook",
                "Enrich torch.distributed.elastic worker failures.",
                "runtime.hardware_outside_v07_scope",
            ),
            "explicit AMD": (
                "[AOTI] Fix constant copy ordering (AMD)",
                "NVIDIA uses legacy stream ordering, while ROCm differs.",
                "runtime.hardware_outside_v07_scope",
            ),
            "grammar only": (
                "Fix grammar in pooling error message",
                "Correct a missing word in the exception text.",
                "change.documentation_cleanup_refactor_only",
            ),
        }
        review_cases = {
            "warning": "Fix NumPy DeprecationWarning",
            "unit test": "Fix unit test failure in sparse to()",
            "repro": "Fix runnable repro script variable",
            "benchmark infra": (
                "Fix IndentationError in imports_for_benchmark_kernel"
            ),
        }
        candidates: list[dict[str, object]] = []
        expected: dict[int, tuple[str, str | None]] = {}
        for offset, (_, (title, description, reason)) in enumerate(
            hard_cases.items()
        ):
            number = 191000 + offset
            candidate = _capture(
                number=number,
                merge=f"{offset + 1:x}" * 40,
                title=title,
                description=description,
            )
            candidate["execution_hints"]["phases"] = ["forward"]
            candidate["proposed_contract_families"] = ["api_behavior"]
            candidates.append(candidate)
            expected[number] = ("hard_rejected", reason)
        for offset, (_, title) in enumerate(review_cases.items()):
            number = 191100 + offset
            candidate = _capture(
                number=number,
                merge=f"{offset + 12:x}" * 40,
                title=title,
                description="The affected helper now runs correctly.",
            )
            candidate["execution_hints"]["phases"] = ["forward"]
            candidate["proposed_contract_families"] = ["api_behavior"]
            candidates.append(candidate)
            expected[number] = ("deferred_for_review", None)
        distributed_collective = _capture(
            number=191200,
            merge="6" * 40,
            title="Fix BucketMode scheduling",
            description="Repair reduce_scatter and all_gather bucket ordering.",
        )
        distributed_collective["execution_hints"]["phases"] = ["forward"]
        distributed_collective["proposed_contract_families"] = [
            "api_behavior"
        ]
        distributed_collective["changed_files"] = [
            {
                "path": "test/distributed/test_inductor_collectives.py",
                "additions": 18,
                "deletions": 1,
                "change_type": "MODIFIED",
                "is_test": True,
            },
            {
                "path": "torch/_inductor/collectives.py",
                "additions": 12,
                "deletions": 5,
                "change_type": "MODIFIED",
                "is_test": False,
            },
        ]
        candidates.append(distributed_collective)
        expected[191200] = (
            "hard_rejected",
            "runtime.hardware_outside_v07_scope",
        )
        repro_harness = _capture(
            number=191201,
            merge="7" * 40,
            title="Fix FX graph runnable output",
            description="Correct generated runnable script state.",
        )
        repro_harness["execution_hints"]["phases"] = ["forward"]
        repro_harness["proposed_contract_families"] = ["api_behavior"]
        repro_harness["changed_files"][1]["path"] = (
            "torch/_dynamo/repro/after_aot.py"
        )
        candidates.append(repro_harness)
        expected[191201] = ("deferred_for_review", None)
        fbcode = _capture(
            number=191202,
            merge="8" * 40,
            title="Fix missing libtorch symbols in FBCODE",
            description="Repair test_gpu_cpp_wrapper linkage.",
        )
        fbcode["execution_hints"]["phases"] = ["forward"]
        fbcode["proposed_contract_families"] = ["api_behavior"]
        candidates.append(fbcode)
        expected[191202] = ("deferred_for_review", None)
        distributed_production = _capture(
            number=191203,
            merge="9" * 40,
            title="Fix flight recorder builder ordering",
            description="Repair the builder state transition.",
        )
        distributed_production["execution_hints"]["phases"] = ["forward"]
        distributed_production["proposed_contract_families"] = [
            "api_behavior"
        ]
        distributed_production["changed_files"][1]["path"] = (
            "torch/distributed/flight_recorder/components/builder.py"
        )
        candidates.append(distributed_production)
        expected[191203] = (
            "hard_rejected",
            "runtime.hardware_outside_v07_scope",
        )

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            _write_acquisition_inputs(captures, receipts, candidates)
            output = temporary / "screening"
            completed = _run_screen(
                captures, output, receipt_path=receipts
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            index = json.loads(
                (output / "screening_index.json").read_text(encoding="utf-8")
            )
            for item in index["records"]:
                decision = load_factory_contract(
                    output / item["decision"]["relative_path"]
                )
                candidate_record = load_factory_contract(
                    output / item["candidate"]["relative_path"]
                )
                disposition, reason = expected[item["pr_number"]]
                self.assertEqual(decision.disposition, disposition)
                if reason is not None:
                    self.assertIn(reason, decision.hard_rejection_reasons)
                else:
                    self.assertTrue(decision.preliminary_review_reasons)
                if item["pr_number"] in {
                    191000,
                    191001,
                    191200,
                    191203,
                }:
                    self.assertTrue(
                        candidate_record.execution_hints.distributed
                    )

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            _write_acquisition_inputs(
                captures,
                receipts,
                [
                    _capture(
                        title="Back out a previous compiler fix",
                        description="Rollback after a regression.",
                    )
                ],
            )
            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertRegex(
                completed.stderr.casefold(),
                r"primary|reversal",
            )

    def test_reversal_history_finding_cannot_be_screened(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            _write_acquisition_inputs(
                captures,
                receipts,
                [_capture(number=190778)],
            )
            receipt_set = json.loads(
                receipts.read_text(encoding="utf-8")
            )
            receipt = receipt_set["receipts"][0]
            receipt["main_history_reversal_findings"] = [
                {
                    "event_oid": "d" * 40,
                    "committed_at": "2026-07-20T01:02:04Z",
                    "message_hash": "sha256:" + "d" * 64,
                    "kind": "revert_commit",
                    "target_pr_number": None,
                    "target_commit_oid": receipt[
                        "main_history_commit"
                    ],
                }
            ]
            receipt["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "content_hash"
                }
            )
            receipt_set["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt_set.items()
                    if key != "content_hash"
                }
            )
            receipts.write_bytes(
                canonical_json(receipt_set).encode("utf-8")
            )
            capture_set = json.loads(
                captures.read_text(encoding="utf-8")
            )
            capture_set["candidates"][0][
                "acquisition_receipt_hash"
            ] = receipt["content_hash"]
            capture_set[
                "acquisition_receipt_set_hash"
            ] = receipt_set["content_hash"]
            capture_set["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in capture_set.items()
                    if key != "content_hash"
                }
            )
            captures.write_bytes(
                canonical_json(capture_set).encode("utf-8")
            )

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertRegex(
                completed.stderr.casefold(),
                r"revers|primary",
            )

    def test_second_exact_marker_after_revert_is_rejected_generically(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            candidate = _capture(
                number=190777,
                merge="e" * 40,
                title="Preserve CUDA errors across the ABI boundary",
            )
            _write_acquisition_inputs(captures, receipts, [candidate])
            receipt_set = json.loads(
                receipts.read_text(encoding="utf-8")
            )
            receipt = receipt_set["receipts"][0]
            exact_marker = (
                "Pull Request resolved: "
                "https://github.com/pytorch/pytorch/pull/190777"
            )
            marker_hits, reversal_findings = (
                derive_quality_main_history_findings(
                    190777,
                    (
                        {
                            "oid": "c" * 40,
                            "first_parent_oid": "a" * 40,
                            "committed_at": "2026-07-19T01:02:03Z",
                            "message": (
                                "Preserve CUDA errors across the ABI "
                                f"boundary\n\n{exact_marker}"
                            ),
                        },
                        {
                            "oid": "d" * 40,
                            "first_parent_oid": "c" * 40,
                            "committed_at": "2026-07-20T00:02:03Z",
                            "message": (
                                'Revert "Preserve CUDA errors across the '
                                'ABI boundary (#190777)"'
                            ),
                        },
                        {
                            "oid": "e" * 40,
                            "first_parent_oid": receipt[
                                "main_history_first_parent"
                            ],
                            "committed_at": receipt[
                                "main_history_committed_at"
                            ],
                            "message": receipt[
                                "selected_commit_message"
                            ],
                        },
                    ),
                )
            )
            self.assertEqual(len(marker_hits), 2)
            self.assertEqual(len(reversal_findings), 1)
            self.assertEqual(reversal_findings[0].kind, "revert_pr")
            self.assertEqual(
                reversal_findings[0].target_pr_number,
                190777,
            )
            receipt["main_history_exact_marker_hits"] = [
                item.to_dict() for item in marker_hits
            ]
            receipt["main_history_reversal_findings"] = [
                item.to_dict() for item in reversal_findings
            ]
            receipt["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "content_hash"
                }
            )
            receipt_set["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt_set.items()
                    if key != "content_hash"
                }
            )
            receipts.write_bytes(
                canonical_json(receipt_set).encode("utf-8")
            )
            capture_set = json.loads(
                captures.read_text(encoding="utf-8")
            )
            capture_set["candidates"][0][
                "acquisition_receipt_hash"
            ] = receipt["content_hash"]
            capture_set[
                "acquisition_receipt_set_hash"
            ] = receipt_set["content_hash"]
            capture_set["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in capture_set.items()
                    if key != "content_hash"
                }
            )
            captures.write_bytes(
                canonical_json(capture_set).encode("utf-8")
            )

            # The selected second marker's own headline is deliberately not
            # a reversal; only the complete history finding exposes it.
            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertRegex(
                completed.stderr.casefold(),
                r"marker|revers|history",
            )

    def test_backward_compatibility_is_not_gradient_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            candidate = _capture(
                title="Fix NumPy protocol backward compatibility",
                description="Preserve backward compatibility for array wrapping.",
            )
            _write_acquisition_inputs(captures, receipts, [candidate])

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("behavioral evidence", completed.stderr.casefold())

    def test_source_path_only_autograd_cannot_claim_gradient_or_backward(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            candidate = _capture(
                title="Fix return metadata mismatch",
                description="The public API returned stale metadata.",
            )
            candidate["changed_files"][1][
                "path"
            ] = "torch/autograd/example.py"
            candidate["execution_hints"]["phases"] = ["backward"]
            candidate["proposed_contract_families"] = [
                "tensor_metadata",
                "gradient",
            ]
            _write_acquisition_inputs(
                captures, receipts, [candidate]
            )

            completed = _run_screen(
                captures,
                temporary / "screening",
                receipt_path=receipts,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("behavioral evidence", completed.stderr.casefold())

    def test_factory_loader_rejects_candidate_self_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            receipts = temporary / "receipts.json"
            output = temporary / "screening"
            _write_acquisition_inputs(captures, receipts, [_capture()])
            completed = _run_screen(
                captures, output, receipt_path=receipts
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            path = next((output / "candidates").glob("*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["title"] = "tampered"
            path.write_bytes(canonical_json(payload).encode("utf-8"))
            with self.assertRaisesRegex(ContractError, "content_hash"):
                load_factory_contract(path)

    def test_relocated_index_still_binds_official_acquisition_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory) / "screening"
            shutil.copytree(INDEX.parent, relocated)
            index_path = relocated / "screening_index.json"
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            payload["capture_set_hash"] = "sha256:" + "c" * 64
            payload["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in payload.items()
                    if key != "content_hash"
                }
            )
            index_path.write_bytes(
                canonical_json(payload).encode("utf-8")
            )

            errors = validate_candidate_index(ROOT, index_path)

            self.assertTrue(errors)
            self.assertTrue(
                any(
                    "official" in error or "capture" in error
                    for error in errors
                ),
                errors,
            )

    def test_pinned_roots_reject_coherent_official_capture_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            p8 = temporary_root / "factory/v0.7/p8"
            p8.mkdir(parents=True)
            p7 = temporary_root / "factory/v0.7/p7"
            p7.mkdir(parents=True)
            shutil.copy2(CAPTURES, p8 / "captures.json")
            shutil.copy2(RECEIPTS, p8 / "acquisition_receipts.json")
            shutil.copytree(INDEX.parent, p8 / "screening")
            shutil.copy2(HISTORICAL, p7 / "historical_readmission.json")
            captures_path = p8 / "captures.json"
            receipts_path = p8 / "acquisition_receipts.json"
            captures = json.loads(
                captures_path.read_text(encoding="utf-8")
            )
            receipts = json.loads(
                receipts_path.read_text(encoding="utf-8")
            )
            pr_number = captures["candidates"][0]["pr_number"]
            capture = captures["candidates"][0]
            receipt = next(
                item
                for item in receipts["receipts"]
                if item["pr_number"] == pr_number
            )
            capture["base_commit"] = "c" * 40
            receipt["base_commit"] = "c" * 40
            receipt["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "content_hash"
                }
            )
            receipts["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipts.items()
                    if key != "content_hash"
                }
            )
            capture["acquisition_receipt_hash"] = receipt["content_hash"]
            captures[
                "acquisition_receipt_set_hash"
            ] = receipts["content_hash"]
            captures["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in captures.items()
                    if key != "content_hash"
                }
            )
            receipts_path.write_bytes(
                canonical_json(receipts).encode("utf-8")
            )
            captures_path.write_bytes(
                canonical_json(captures).encode("utf-8")
            )

            errors = validate_candidate_index(
                temporary_root,
                p8 / "screening/screening_index.json",
            )

            self.assertTrue(
                any("pinned" in error for error in errors),
                errors,
            )

    def test_pin_rejects_coherent_main_history_scan_fact_omission(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            p8 = temporary_root / "factory/v0.7/p8"
            p8.mkdir(parents=True)
            p7 = temporary_root / "factory/v0.7/p7"
            p7.mkdir(parents=True)
            captures_path = p8 / "captures.json"
            receipts_path = p8 / "acquisition_receipts.json"
            historical_path = p7 / "historical_readmission.json"
            shutil.copy2(CAPTURES, captures_path)
            shutil.copy2(RECEIPTS, receipts_path)
            shutil.copy2(HISTORICAL, historical_path)
            captures = json.loads(
                captures_path.read_text(encoding="utf-8")
            )
            receipts = json.loads(
                receipts_path.read_text(encoding="utf-8")
            )

            # Simulate omitting a commit fact from the complete main scan,
            # then coherently recompute every ordinary acquisition hash.
            receipts["main_history_scan"][
                "scanned_commit_facts_hash"
            ] = "sha256:" + "c" * 64
            receipts["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipts.items()
                    if key != "content_hash"
                }
            )
            captures[
                "acquisition_receipt_set_hash"
            ] = receipts["content_hash"]
            captures["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in captures.items()
                    if key != "content_hash"
                }
            )
            receipts_path.write_bytes(
                canonical_json(receipts).encode("utf-8")
            )
            captures_path.write_bytes(
                canonical_json(captures).encode("utf-8")
            )
            screening = p8 / "screening"
            rebuilt = _run_screen(
                captures_path,
                screening,
                receipt_path=receipts_path,
                historical_path=historical_path,
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            relocated = temporary_root / "relocated"
            shutil.copytree(screening, relocated)

            official_errors = validate_candidate_index(
                temporary_root,
                screening / "screening_index.json",
            )
            relocated_errors = validate_candidate_index(
                temporary_root,
                relocated / "screening_index.json",
            )

            self.assertTrue(
                any("pinned" in error for error in official_errors),
                official_errors,
            )
            self.assertTrue(
                any("pinned" in error for error in relocated_errors),
                relocated_errors,
            )

    def test_pin_rejects_coherent_pr_branch_substitution_everywhere(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            p8 = temporary_root / "factory/v0.7/p8"
            p8.mkdir(parents=True)
            p7 = temporary_root / "factory/v0.7/p7"
            p7.mkdir(parents=True)
            captures_path = p8 / "captures.json"
            receipts_path = p8 / "acquisition_receipts.json"
            historical_path = p7 / "historical_readmission.json"
            shutil.copy2(CAPTURES, captures_path)
            shutil.copy2(RECEIPTS, receipts_path)
            shutil.copy2(HISTORICAL, historical_path)
            captures = json.loads(
                captures_path.read_text(encoding="utf-8")
            )
            receipts = json.loads(
                receipts_path.read_text(encoding="utf-8")
            )
            capture = next(
                item
                for item in captures["candidates"]
                if item["pr_number"] == 177673
            )
            receipt = next(
                item
                for item in receipts["receipts"]
                if item["pr_number"] == 177673
            )
            branch_commit = "028b47f6900c5cc332134112d5ea83a9b2c22f1b"
            branch_parent = "b459a7c37fa76721b8d302dafd41fefb1842140f"
            branch_time = "2026-03-25T08:08:27Z"
            marker = (
                "Pull Request resolved: "
                "https://github.com/pytorch/pytorch/pull/177673"
            )
            message = f"{capture['title']}\n\n{marker}"
            for payload in (capture, receipt):
                payload["merge_commit"] = branch_commit
                payload["base_commit"] = branch_parent
                payload["merged_at"] = branch_time
            receipt["main_history_commit"] = branch_commit
            receipt["main_history_first_parent"] = branch_parent
            receipt["main_history_committed_at"] = branch_time
            receipt["main_history_exact_marker_hits"] = [
                {
                    "oid": branch_commit,
                    "first_parent_oid": branch_parent,
                    "committed_at": branch_time,
                    "message_hash": canonical_sha256(message),
                }
            ]
            receipt["resolved_marker"] = marker
            receipt["selected_commit_message"] = message
            receipt["selected_commit_message_hash"] = canonical_sha256(
                message
            )
            receipt["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "content_hash"
                }
            )
            receipts["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in receipts.items()
                    if key != "content_hash"
                }
            )
            capture["acquisition_receipt_hash"] = receipt["content_hash"]
            captures[
                "acquisition_receipt_set_hash"
            ] = receipts["content_hash"]
            captures["content_hash"] = canonical_sha256(
                {
                    key: value
                    for key, value in captures.items()
                    if key != "content_hash"
                }
            )
            receipts_path.write_bytes(
                canonical_json(receipts).encode("utf-8")
            )
            captures_path.write_bytes(
                canonical_json(captures).encode("utf-8")
            )
            screening = p8 / "screening"
            rebuilt = _run_screen(
                captures_path,
                screening,
                receipt_path=receipts_path,
                historical_path=historical_path,
            )
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stderr)
            relocated = temporary_root / "relocated"
            shutil.copytree(screening, relocated)

            official_errors = validate_candidate_index(
                temporary_root,
                screening / "screening_index.json",
            )
            relocated_errors = validate_candidate_index(
                temporary_root,
                relocated / "screening_index.json",
            )

            self.assertTrue(
                any("pinned" in error for error in official_errors),
                official_errors,
            )
            self.assertTrue(
                any("pinned" in error for error in relocated_errors),
                relocated_errors,
            )


class FrozenQualityCandidateFunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capture_set = json.loads(CAPTURES.read_text(encoding="utf-8"))
        cls.receipt_set = json.loads(RECEIPTS.read_text(encoding="utf-8"))
        cls.index = json.loads(INDEX.read_text(encoding="utf-8"))

    def test_candidate_pool_is_three_times_the_actual_deficit(self) -> None:
        historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
        self.assertGreaterEqual(
            self.index["candidate_count"],
            3 * (50 - historical["k"]),
        )
        self.assertEqual(
            self.index["candidate_count"],
            sum(self.index["disposition_counts"].values()),
        )
        self.assertEqual(
            self.index["eligible_candidate_count"],
            self.index["disposition_counts"]["accepted_for_build"]
            + self.index["disposition_counts"]["deferred_for_review"],
        )
        self.assertEqual(
            self.index["capture_count"],
            len(self.capture_set["candidates"])
            + self.index["exclusion_count"],
        )

    def test_real_provenance_is_unique_nonhistorical_and_complete(self) -> None:
        candidates = self.capture_set["candidates"]
        self.assertFalse(
            {182045, 186379}
            & {item["pr_number"] for item in candidates},
        )
        dispositions = {
            item["pr_number"]: item["disposition"]
            for item in self.index["records"]
        }
        self.assertEqual(
            len({item["pr_number"] for item in candidates}),
            len(candidates),
        )
        self.assertEqual(
            len(
                {
                    (
                        item["repository"],
                        item["base_commit"],
                        item["merge_commit"],
                    )
                    for item in candidates
                }
            ),
            len(candidates),
        )
        self.assertFalse(
            {item["pr_number"] for item in candidates} & HISTORICAL_PRS
        )
        receipts = {
            item["pr_number"]: item
            for item in self.receipt_set["receipts"]
        }
        scan = self.receipt_set["main_history_scan"]
        self.assertEqual(scan["ref_name"], "main")
        self.assertEqual(scan["since"], "2026-03-01T00:00:00Z")
        self.assertFalse(scan["final_has_next_page"])
        self.assertGreater(scan["page_count"], 0)
        self.assertGreater(scan["commit_count"], 0)
        self.assertRegex(
            scan["scanned_commit_facts_hash"],
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertEqual(
            set(receipts),
            {item["pr_number"] for item in candidates},
        )
        self.assertEqual(
            self.capture_set["acquisition_receipt_set_hash"],
            self.receipt_set["content_hash"],
        )
        for candidate in candidates:
            self.assertEqual(candidate["repository"], "pytorch/pytorch")
            self.assertEqual(candidate["base_ref_name"], "main")
            self.assertNotRegex(
                candidate["head_ref_name"],
                r"(?i)(?:^|[-_/])(?:cherry(?:-pick)?|backport|cp)"
                r"(?:[-_/]|$)|release",
            )
            self.assertNotRegex(
                candidate["title"],
                r"(?i)\b(?:revert(?:ed|ing)?|back\s+out|rollback)\b",
            )
            self.assertRegex(candidate["base_commit"], r"^[0-9a-f]{40}$")
            self.assertRegex(candidate["merge_commit"], r"^[0-9a-f]{40}$")
            self.assertNotEqual(
                candidate["base_commit"],
                candidate["merge_commit"],
            )
            self.assertEqual(
                candidate["changed_file_count"],
                len(candidate["changed_files"]),
            )
            self.assertTrue(candidate["changed_files"])
            receipt = receipts[candidate["pr_number"]]
            self.assertEqual(
                receipt["main_history_head_oid"],
                scan["head_oid"],
            )
            self.assertGreaterEqual(
                receipt["main_history_committed_at"],
                scan["since"],
            )
            self.assertEqual(
                candidate["acquisition_receipt_hash"],
                receipt["content_hash"],
            )
            for field in (
                "repository",
                "pr_number",
                "pr_url",
                "merged_at",
                "merge_commit",
                "base_commit",
                "base_ref_name",
                "head_ref_name",
            ):
                self.assertEqual(candidate[field], receipt[field])
            self.assertTrue(receipt["files_pagination_complete"])
            self.assertFalse(receipt["files_has_next_page"])
            self.assertEqual(
                receipt["files_total_count"],
                len(candidate["changed_files"]),
            )
            self.assertEqual(
                receipt["files_captured_node_count"],
                len(candidate["changed_files"]),
            )
            self.assertEqual(
                receipt["changed_files_hash"],
                _changed_files_hash(candidate),
            )
            marker = (
                "Pull Request resolved: "
                f"https://github.com/pytorch/pytorch/pull/"
                f"{candidate['pr_number']}"
            )
            self.assertEqual(
                receipt["resolver_mode"],
                "main_history_exact_resolved_marker",
            )
            self.assertEqual(receipt["resolved_marker"], marker)
            self.assertIn(
                marker,
                receipt["selected_commit_message"].splitlines(),
            )
            self.assertEqual(
                receipt["selected_commit_message_hash"],
                canonical_sha256(receipt["selected_commit_message"]),
            )
            self.assertEqual(
                receipt["main_history_commit"],
                candidate["merge_commit"],
            )
            self.assertEqual(
                receipt["main_history_first_parent"],
                candidate["base_commit"],
            )
            self.assertEqual(
                receipt["main_history_committed_at"],
                candidate["merged_at"],
            )
            self.assertEqual(
                receipt["main_history_exact_marker_hits"],
                [
                    {
                        "oid": candidate["merge_commit"],
                        "first_parent_oid": candidate["base_commit"],
                        "committed_at": candidate["merged_at"],
                        "message_hash": receipt[
                            "selected_commit_message_hash"
                        ],
                    }
                ],
            )
            self.assertEqual(
                receipt["main_history_reversal_findings"],
                [],
            )
            if dispositions[candidate["pr_number"]] != "hard_rejected":
                self.assertTrue(candidate["behavioral_test_evidence"])
                self.assertTrue(
                    any(
                        item["is_test"]
                        for item in candidate["changed_files"]
                    )
                )

    def test_accepted_pool_has_no_known_scope_or_low_signal_leaks(self) -> None:
        candidates = {
            item["pr_number"]: item
            for item in self.capture_set["candidates"]
        }
        forbidden_title = re.compile(
            r"\b(?:revert(?:ed|ing)?|back\s+out|rollback|"
            r"ddp|DistributedDataParallel|TorchElastic|"
            r"rocm|hip|amd|grammar|spelling|typo|"
            r"DeprecationWarning|unit test failure|runnable repro|"
            r"repro scripts?|imports_for_benchmark_kernel|"
            r"IndentationError)\b",
            re.IGNORECASE,
        )
        for item in self.index["records"]:
            if item["disposition"] != "accepted_for_build":
                continue
            candidate = candidates[item["pr_number"]]
            self.assertIsNone(
                forbidden_title.search(candidate["title"]),
                (candidate["pr_number"], candidate["title"]),
            )
            scope_evidence = "\n".join(
                (
                    candidate["title"],
                    candidate["description"],
                    *(
                        changed["path"]
                        for changed in candidate["changed_files"]
                    ),
                )
            )
            self.assertNotRegex(
                scope_evidence,
                r"(?i)\b(?:ddp|DistributedDataParallel|TorchElastic)\b|"
                r"torch(?:/|\.)distributed(?:/|\.)elastic",
            )

    def test_production_distributed_path_is_uniformly_scope_rejected(self) -> None:
        candidate = next(
            item
            for item in self.capture_set["candidates"]
            if item["pr_number"] == 177076
        )
        record = next(
            item
            for item in self.index["records"]
            if item["pr_number"] == 177076
        )
        candidate_record = load_factory_contract(
            INDEX.parent / record["candidate"]["relative_path"]
        )
        decision = load_factory_contract(
            INDEX.parent / record["decision"]["relative_path"]
        )
        self.assertTrue(
            any(
                item["path"].startswith("torch/distributed/")
                for item in candidate["changed_files"]
            )
        )
        self.assertTrue(candidate_record.execution_hints.distributed)
        self.assertEqual(decision.disposition, "hard_rejected")
        self.assertIn(
            "runtime.hardware_outside_v07_scope",
            decision.hard_rejection_reasons,
        )

    def test_candidate_capture_covers_contracts_contexts_and_triggers(self) -> None:
        families = set(self.index["proposed_contract_families"])
        self.assertEqual(
            families,
            {
                "api_behavior",
                "efficiency_safety",
                "gradient",
                "mutation_state",
                "result",
                "tensor_metadata",
            },
        )
        context = self.index["execution_context_summary"]
        self.assertEqual(set(context["devices"]), {"cpu", "cuda"})
        self.assertEqual(set(context["modes"]), {"eager", "compile"})
        self.assertEqual(set(context["phases"]), {"forward", "backward"})
        self.assertGreaterEqual(context["candidate_counts"]["cuda"], 18)
        self.assertGreaterEqual(context["candidate_counts"]["compile"], 18)
        self.assertGreaterEqual(context["candidate_counts"]["backward"], 12)
        self.assertGreaterEqual(len(self.index["proposed_trigger_tags"]), 7)

    def test_artifact_tree_is_canonical_private_and_revalidates(self) -> None:
        self.assertEqual(validate_candidate_index(ROOT, INDEX), ())
        for path in [
            CAPTURES,
            RECEIPTS,
            INDEX,
            *(INDEX.parent / "candidates").glob("*.json"),
            *(INDEX.parent / "decisions").glob("*.json"),
        ]:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path.read_bytes(), canonical_json(value).encode("utf-8"))
            _assert_no_forbidden_keys(self, value)


if __name__ == "__main__":
    unittest.main()
