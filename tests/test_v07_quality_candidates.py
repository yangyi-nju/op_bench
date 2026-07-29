from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from op_bench.factory.artifacts import load_factory_contract
from op_bench.factory.quality_release import (
    HARD_CANDIDATE_REJECTION_REASONS,
    QualityCandidateDecision,
    QualityCandidateRecord,
    validate_candidate_index,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/screen_v07_quality_candidates.py"
CAPTURES = ROOT / "factory/v0.7/p8/captures.json"
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
        "merged_at": "2026-07-20T01:02:03Z",
        "title": "Fix compiled backward metadata mismatch",
        "description": "A compiled backward path returned the wrong metadata.",
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


def _capture_set(candidates: list[dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_capture_set",
        "schema_version": "v1",
        "repository": "pytorch/pytorch",
        "captured_at": "2026-07-29T00:00:00Z",
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
        "candidates": candidates,
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


def _write_capture_set(path: Path, candidates: list[dict[str, object]]) -> None:
    path.write_bytes(canonical_json(_capture_set(candidates)).encode("utf-8"))


def _run_screen(
    capture_path: Path,
    output_dir: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        (
            str(ROOT / ".venv/bin/python"),
            str(SCRIPT),
            "--input",
            str(capture_path.resolve()),
            "--historical-index",
            str(HISTORICAL),
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
    def test_schema_tracks_candidate_decision_and_index_wire_contracts(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v07_quality_release.schema.json").read_text(
                encoding="utf-8"
            )
        )
        candidate = schema["$defs"]["quality_candidate"]
        decision = schema["$defs"]["quality_candidate_decision"]
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
        self.assertEqual(set(index["required"]), set(index["properties"]))

    def test_cli_writes_typed_canonical_candidate_decision_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            first = temporary / "first"
            second = temporary / "second"
            _write_capture_set(captures, [_capture()])

            accepted = _run_screen(captures, first)
            repeated = _run_screen(captures, second)

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
            self.assertEqual(
                validate_candidate_index(
                    ROOT,
                    first / "screening_index.json",
                    require_minimum=False,
                ),
                (),
            )

    def test_hard_rejection_reasons_are_exact_and_preliminary(self) -> None:
        mutations = {
            "missing immutable commits": {
                "base_commit": None,
                "reason": "source.missing_immutable_commits",
            },
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
                _write_capture_set(captures, [candidate])
                output = temporary / "screening"
                completed = _run_screen(captures, output)
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
                _write_capture_set(captures, candidates)
                completed = _run_screen(captures, temporary / "screening")
                self.assertNotEqual(completed.returncode, 0)
                self.assertIn(message, completed.stderr)

    def test_factory_loader_rejects_candidate_self_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            captures = temporary / "captures.json"
            output = temporary / "screening"
            _write_capture_set(captures, [_capture()])
            completed = _run_screen(captures, output)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            path = next((output / "candidates").glob("*.json"))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["title"] = "tampered"
            path.write_bytes(canonical_json(payload).encode("utf-8"))
            with self.assertRaisesRegex(ContractError, "content_hash"):
                load_factory_contract(path)


class FrozenQualityCandidateFunnelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.capture_set = json.loads(CAPTURES.read_text(encoding="utf-8"))
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
            len(self.capture_set["candidates"]),
        )

    def test_real_provenance_is_unique_nonhistorical_and_complete(self) -> None:
        candidates = self.capture_set["candidates"]
        dispositions = {
            item["pr_number"]: item["disposition"]
            for item in self.index["records"]
        }
        self.assertEqual(
            len({item["pr_number"] for item in candidates}),
            len(candidates),
        )
        self.assertFalse(
            {item["pr_number"] for item in candidates} & HISTORICAL_PRS
        )
        for candidate in candidates:
            self.assertEqual(candidate["repository"], "pytorch/pytorch")
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
            if dispositions[candidate["pr_number"]] != "hard_rejected":
                self.assertTrue(candidate["behavioral_test_evidence"])
                self.assertTrue(
                    any(
                        item["is_test"]
                        for item in candidate["changed_files"]
                    )
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
            INDEX,
            *(INDEX.parent / "candidates").glob("*.json"),
            *(INDEX.parent / "decisions").glob("*.json"),
        ]:
            value = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(path.read_bytes(), canonical_json(value).encode("utf-8"))
            _assert_no_forbidden_keys(self, value)


if __name__ == "__main__":
    unittest.main()
