from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from op_bench.factory.prompt_quality import (
    PrivateAnswerIndex,
    PromptQualityEvidence,
    build_private_answer_index,
    empty_private_index,
    scan_rendered_prompt,
)
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64


class PromptQualityScannerTests(unittest.TestCase):
    def test_scanner_rejects_provenance_paths_symbols_and_hidden_selectors(self) -> None:
        index = PrivateAnswerIndex(
            changed_paths=("torch/_decomp/decompositions.py",),
            added_symbols=("computeStorageNbytes",),
            distinctive_literals=("numel == 0",),
            hidden_selectors=("HiddenTests.test_empty",),
        )
        prompt = """
        PR #143792 says to modify torch/_decomp/decompositions.py.
        Reuse computeStorageNbytes when numel == 0.
        Run HiddenTests.test_empty.
        """

        codes = {item.code for item in scan_rendered_prompt(prompt, index)}

        self.assertEqual(
            codes,
            {
                "answer.changed_path",
                "answer.distinctive_literal",
                "answer.hidden_selector",
                "answer.symbol",
                "provenance.pull_request",
                "solution.instruction",
            },
        )

    def test_scanner_allows_public_api_and_natural_stack_trace(self) -> None:
        prompt = (
            "torch.addmv returns the wrong shape for an empty matrix.\n"
            'Traceback: File "torch/runtime.py", line 8, in addmv'
        )

        self.assertEqual(scan_rendered_prompt(prompt, empty_private_index()), ())

    def test_private_index_extracts_sorted_diff_facts_without_source_imports(self) -> None:
        gold_patch = """diff --git a/torch/foo.py b/torch/foo.py
--- a/torch/foo.py
+++ b/torch/foo.py
@@ -1 +1,4 @@
+def compute_storage_size(value):
+    if value == \"empty matrix\":
+        return unique_helper(value)
"""
        hidden_patch = """diff --git a/test/test_foo.py b/test/test_foo.py
--- a/test/test_foo.py
+++ b/test/test_foo.py
@@ -1 +1,2 @@
+class HiddenProbe:
+    pass
"""

        index = build_private_answer_index(
            gold_patch=gold_patch,
            hidden_test_patch=hidden_patch,
            patch_scope=("torch/foo.py",),
            hidden_selectors=("HiddenProbe.test_empty",),
        )

        self.assertEqual(
            index.changed_paths,
            ("test/test_foo.py", "torch/foo.py"),
        )
        self.assertIn("compute_storage_size", index.added_symbols)
        self.assertIn("HiddenProbe", index.added_symbols)
        self.assertIn("unique_helper", index.internal_names)
        self.assertIn("empty matrix", index.distinctive_literals)
        self.assertEqual(index.hidden_selectors, ("HiddenProbe.test_empty",))


def evidence_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": "prompt_quality",
        "schema_version": "v1",
        "task_id": "pytorch__empty_addmv",
        "public_task_id": "task-v07-empty-addmv",
        "prompt_hash": SHA_A,
        "agent_task_view_hash": SHA_B,
        "scanner_version": "prompt-overlap-v1",
        "findings": [],
        "blind_review": {
            "decision": "accepted",
            "reviewer": "reviewer-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        "semantic_review": {
            "decision": "equivalent",
            "reviewer": "curator-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        "decision": "accepted",
        "created_at": "2026-07-29T00:00:00Z",
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


class PromptQualityEvidenceTests(unittest.TestCase):
    def test_evidence_round_trips_and_schema_matches_wire_contract(self) -> None:
        selected = PromptQualityEvidence.from_dict(evidence_payload())

        self.assertEqual(PromptQualityEvidence.from_dict(selected.to_dict()), selected)
        schema = json.loads(
            (ROOT / "schemas" / "prompt_quality.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(PromptQualityEvidence.wire_fields()))
        self.assertEqual(set(schema["properties"]), set(PromptQualityEvidence.wire_fields()))

    def test_evidence_mutations_fail_closed(self) -> None:
        selected = evidence_payload()
        mutations: tuple[tuple[str, object], ...] = (
            (
                "findings",
                [
                    {
                        "code": "answer.symbol",
                        "severity": "reject",
                        "public_field": "rendered_prompt",
                        "matched_value_hash": SHA_A,
                    }
                ],
            ),
            ("blind_review.decision", "rejected"),
            ("prompt_hash", SHA_B),
            ("content_hash", SHA_A),
        )
        for path, value in mutations:
            with self.subTest(path=path):
                payload = copy.deepcopy(selected)
                target: object = payload
                parts = path.split(".")
                for part in parts[:-1]:
                    target = target[part]  # type: ignore[index]
                target[parts[-1]] = value  # type: ignore[index]

                with self.assertRaises(ContractError):
                    PromptQualityEvidence.from_dict(payload)

    def test_accepted_evidence_requires_clear_scan_and_distinct_reviewers(self) -> None:
        payload = evidence_payload()
        payload["semantic_review"] = {
            "decision": "equivalent",
            "reviewer": "reviewer-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        }
        payload["content_hash"] = canonical_sha256(
            {key: value for key, value in payload.items() if key != "content_hash"}
        )

        with self.assertRaisesRegex(ContractError, "different reviewers"):
            PromptQualityEvidence.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
