from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

from op_bench.factory.prompt_quality import (
    PrivateAnswerIndex,
    PromptQualityEvidence,
    build_private_answer_index,
    build_prompt_quality_evidence,
    empty_private_index,
    scan_rendered_prompt,
    validate_prompt_quality_evidence,
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
@@ -0,0 +1,3 @@
+def compute_storage_size(value):
+    if value == \"empty matrix\":
+        return unique_helper(value)
"""
        hidden_patch = """diff --git a/test/test_foo.py b/test/test_foo.py
--- a/test/test_foo.py
+++ b/test/test_foo.py
@@ -0,0 +1,2 @@
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

    def test_private_index_rejects_malformed_or_unsafe_diff_headers(self) -> None:
        for patch in (
            'diff --git "a/torch/foo.py b/torch/foo.py',
            "diff --git a/../private.py b/torch/foo.py",
        ):
            with self.subTest(patch=patch):
                with self.assertRaisesRegex(ContractError, "diff header"):
                    build_private_answer_index(
                        gold_patch=patch,
                        hidden_test_patch="",
                        patch_scope=(),
                        hidden_selectors=(),
                    )

    def test_private_index_rejects_unsupported_binary_diff_payloads(self) -> None:
        patch = """diff --git a/torch/foo.py b/torch/foo.py
GIT binary patch
literal 12
abcdef
"""

        with self.assertRaisesRegex(ContractError, "unsupported diff payload"):
            build_private_answer_index(
                gold_patch=patch,
                hidden_test_patch="",
                patch_scope=(),
                hidden_selectors=(),
            )

    def test_private_index_uses_hunks_and_keeps_added_lines_beginning_with_plus_plus(self) -> None:
        patch = """diff --git a/torch/foo.py b/torch/foo.py
--- a/torch/foo.py
+++ b/torch/foo.py
@@ -0,0 +1 @@
+++counterValue = 1
"""

        index = build_private_answer_index(
            gold_patch=patch,
            hidden_test_patch="",
            patch_scope=(),
            hidden_selectors=(),
        )

        self.assertIn("counterValue", index.internal_names)

    def test_private_index_extracts_brace_next_line_cpp_functions_and_constructors(self) -> None:
        patch = """diff --git a/aten/src/foo.cpp b/aten/src/foo.cpp
--- a/aten/src/foo.cpp
+++ b/aten/src/foo.cpp
@@ -0,0 +1,9 @@
+Tensor Widget::computeStorageNbytes(
+    int64_t count)
+{
+    return Tensor();
+}
+Widget::Widget(
+    int64_t count)
+{
+}
"""

        index = build_private_answer_index(
            gold_patch=patch,
            hidden_test_patch="",
            patch_scope=(),
            hidden_selectors=(),
        )

        self.assertIn("computeStorageNbytes", index.added_symbols)
        self.assertIn("Widget", index.added_symbols)

    def test_private_index_uses_constructor_name_before_initializer_list(self) -> None:
        patch = """diff --git a/aten/src/foo.cpp b/aten/src/foo.cpp
--- a/aten/src/foo.cpp
+++ b/aten/src/foo.cpp
@@ -0,0 +1,4 @@
+Widget::Widget(int value)
+    : value_(value)
+{
+}
"""

        index = build_private_answer_index(
            gold_patch=patch,
            hidden_test_patch="",
            patch_scope=(),
            hidden_selectors=(),
        )

        self.assertIn("Widget", index.added_symbols)
        self.assertNotIn("value_", index.added_symbols)

    def test_scanner_normalizes_comparison_literal_spacing(self) -> None:
        index = PrivateAnswerIndex(
            changed_paths=(),
            added_symbols=(),
            distinctive_literals=("numel == 0",),
            hidden_selectors=(),
        )

        findings = scan_rendered_prompt("The result differs when numel==0.", index)

        self.assertEqual([item.code for item in findings], ["answer.distinctive_literal"])


def evidence_payload() -> dict[str, object]:
    return build_prompt_quality_evidence(
        task_id="pytorch__empty_addmv",
        public_task_id="task-v07-empty-addmv",
        rendered_prompt="The public behavior differs for an empty matrix.",
        agent_task_view={"statement_body": "The public behavior differs for an empty matrix."},
        private_index=empty_private_index(),
        scanner_version="prompt-overlap-v1",
        blind_review={
            "decision": "accepted",
            "reviewer": "reviewer-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        semantic_review={
            "decision": "equivalent",
            "reviewer": "curator-id",
            "reviewed_at": "2026-07-29T00:00:00Z",
        },
        decision="accepted",
        created_at="2026-07-29T00:00:00Z",
    ).to_dict()


class PromptQualityEvidenceTests(unittest.TestCase):
    def test_accepted_evidence_cannot_be_directly_asserted_without_source_inputs(self) -> None:
        with self.assertRaises(TypeError):
            PromptQualityEvidence(
                task_id="pytorch__empty_addmv",
                public_task_id="task-v07-empty-addmv",
                prompt_hash=SHA_A,
                agent_task_view_hash=SHA_B,
                scanner_version="prompt-overlap-v1",
                findings=(),
                blind_review={
                    "decision": "accepted",
                    "reviewer": "reviewer-id",
                    "reviewed_at": "2026-07-29T00:00:00Z",
                },
                semantic_review={
                    "decision": "equivalent",
                    "reviewer": "curator-id",
                    "reviewed_at": "2026-07-29T00:00:00Z",
                },
                decision="accepted",
                created_at="2026-07-29T00:00:00Z",
            )

    def test_evidence_recomputes_exact_prompt_view_and_scan_claims(self) -> None:
        prompt = "The public behavior differs for an empty matrix."
        view = {"statement_body": prompt, "runtime_hint": "cpu"}
        index = empty_private_index()
        evidence = build_prompt_quality_evidence(
            task_id="pytorch__empty_addmv",
            public_task_id="task-v07-empty-addmv",
            rendered_prompt=prompt,
            agent_task_view=view,
            private_index=index,
            scanner_version="prompt-overlap-v1",
            blind_review={
                "decision": "accepted",
                "reviewer": "reviewer-id",
                "reviewed_at": "2026-07-29T00:00:00Z",
            },
            semantic_review={
                "decision": "equivalent",
                "reviewer": "curator-id",
                "reviewed_at": "2026-07-29T00:00:00Z",
            },
            decision="accepted",
            created_at="2026-07-29T00:00:00Z",
        )

        self.assertEqual(evidence.prompt_hash, canonical_sha256(prompt))
        self.assertEqual(evidence.agent_task_view_hash, canonical_sha256(view))
        self.assertEqual(evidence.findings, ())
        validate_prompt_quality_evidence(
            evidence,
            rendered_prompt=prompt,
            agent_task_view=view,
            private_index=index,
        )
        with self.assertRaisesRegex(ContractError, "findings|prompt_hash"):
            validate_prompt_quality_evidence(
                evidence,
                rendered_prompt="Reuse computeStorageNbytes for the empty matrix.",
                agent_task_view=view,
                private_index=PrivateAnswerIndex(
                    changed_paths=(),
                    added_symbols=("computeStorageNbytes",),
                    distinctive_literals=(),
                    hidden_selectors=(),
                ),
            )

    def test_evidence_builder_rejects_accepted_private_answer_overlap(self) -> None:
        with self.assertRaisesRegex(ContractError, "acceptance"):
            build_prompt_quality_evidence(
                task_id="pytorch__empty_addmv",
                public_task_id="task-v07-empty-addmv",
                rendered_prompt="Reuse computeStorageNbytes for the empty matrix.",
                agent_task_view={"statement_body": "public"},
                private_index=PrivateAnswerIndex(
                    changed_paths=(),
                    added_symbols=("computeStorageNbytes",),
                    distinctive_literals=(),
                    hidden_selectors=(),
                ),
                scanner_version="prompt-overlap-v1",
                blind_review={
                    "decision": "accepted",
                    "reviewer": "reviewer-id",
                    "reviewed_at": "2026-07-29T00:00:00Z",
                },
                semantic_review={
                    "decision": "equivalent",
                    "reviewer": "curator-id",
                    "reviewed_at": "2026-07-29T00:00:00Z",
                },
                decision="accepted",
                created_at="2026-07-29T00:00:00Z",
            )

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
