from __future__ import annotations

import unittest

from op_bench.factory.taxonomy import (
    BOUNDARY_KEYWORD_PACKS,
    keyword_pack,
    match_keyword_packs,
    validate_problem_taxonomy,
)


class BoundaryTaxonomyTests(unittest.TestCase):
    def test_registry_contains_exact_b1_through_b5(self) -> None:
        self.assertEqual(
            tuple(pack.subclass for pack in BOUNDARY_KEYWORD_PACKS),
            ("B1", "B2", "B3", "B4", "B5"),
        )
        self.assertEqual(
            tuple(pack.pack_id for pack in BOUNDARY_KEYWORD_PACKS),
            (
                "boundary-b1-v1",
                "boundary-b2-v1",
                "boundary-b3-v1",
                "boundary-b4-v1",
                "boundary-b5-v1",
            ),
        )

    def test_matching_is_case_insensitive_and_exclusions_win(self) -> None:
        self.assertIn(
            "boundary-b1-v1",
            match_keyword_packs(
                "Fix ZERO SIZE tensor reduction",
                ("aten/reduce.py",),
            ),
        )
        self.assertNotIn(
            "boundary-b5-v1",
            match_keyword_packs(
                "Performance-only CUDA grid limit launch tuning",
                ("benchmarks/cuda_launch.py",),
            ),
        )

    def test_each_boundary_subclass_has_a_literal_positive_match(self) -> None:
        cases = (
            ("boundary-b1-v1", "empty tensor reduction"),
            ("boundary-b2-v1", "rank 0 scalar tensor"),
            ("boundary-b3-v1", "numel overflow in index arithmetic"),
            ("boundary-b4-v1", "invalid dim is out of range"),
            ("boundary-b5-v1", "CUDA grid limit causes launch failure"),
        )

        for expected, text in cases:
            with self.subTest(pack_id=expected):
                self.assertIn(expected, match_keyword_packs(text, ()))

    def test_generic_words_and_non_bug_work_do_not_match(self) -> None:
        cases = (
            ("boundary-b1-v1", "Return an empty list from a documentation helper"),
            ("boundary-b2-v1", "Rename a scalar helper during cleanup-only work"),
            ("boundary-b3-v1", "Add a large tensor performance benchmark"),
            ("boundary-b4-v1", "Document the axis argument"),
            ("boundary-b5-v1", "Tune CUDA launch performance-only behavior"),
        )

        for unexpected, text in cases:
            with self.subTest(pack_id=unexpected):
                self.assertNotIn(unexpected, match_keyword_packs(text, ()))

    def test_ambiguous_text_returns_each_matching_pack_in_registry_order(self) -> None:
        self.assertEqual(
            match_keyword_packs(
                "zero size scalar rank 0 input has an invalid dim out of range; "
                "numel overflow reaches the CUDA grid limit",
                ("aten/src/ATen/native/cuda/Reduce.cu",),
            ),
            (
                "boundary-b1-v1",
                "boundary-b2-v1",
                "boundary-b3-v1",
                "boundary-b4-v1",
                "boundary-b5-v1",
            ),
        )

    def test_keyword_pack_lookup_rejects_unknown_identity(self) -> None:
        self.assertEqual(keyword_pack("boundary-b3-v1").subclass, "B3")
        with self.assertRaisesRegex(KeyError, "boundary-b9-v1"):
            keyword_pack("boundary-b9-v1")


class ProblemTaxonomyTests(unittest.TestCase):
    def test_historical_operator_may_omit_taxonomy(self) -> None:
        self.assertEqual(
            validate_problem_taxonomy(
                {
                    "framework": "pytorch",
                    "operator_name": "aten.sum.default",
                }
            ),
            (),
        )

    def test_legacy_precision_pair_may_omit_failure_contract(self) -> None:
        self.assertEqual(
            validate_problem_taxonomy(
                {
                    "problem_dimension": "precision",
                    "problem_subclass": "P3",
                }
            ),
            (),
        )

    def test_taxonomy_fields_must_be_provided_together(self) -> None:
        self.assertEqual(
            validate_problem_taxonomy({"problem_dimension": "boundary"}),
            ("operator taxonomy fields must be provided together",),
        )

    def test_dimension_requires_matching_subclass_family(self) -> None:
        boundary = {
            "problem_dimension": "boundary",
            "problem_subclass": "P3",
            "failure_contract": "wrong-result",
        }
        precision = {
            "problem_dimension": "precision",
            "problem_subclass": "B2",
            "failure_contract": "wrong-result",
        }

        self.assertIn(
            "operator.problem_subclass: boundary requires B1..B5",
            validate_problem_taxonomy(boundary),
        )
        self.assertIn(
            "operator.problem_subclass: precision requires P1..P5",
            validate_problem_taxonomy(precision),
        )

    def test_all_supported_subclasses_and_failure_contracts_validate(self) -> None:
        cases = (
            ("boundary", "B1", "wrong-result"),
            ("boundary", "B2", "exception"),
            ("boundary", "B3", "crash-oob"),
            ("boundary", "B4", "silent-acceptance"),
            ("boundary", "B5", "wrong-result"),
            ("precision", "P1", "wrong-result"),
            ("precision", "P2", "exception"),
            ("precision", "P3", "crash-oob"),
            ("precision", "P4", "silent-acceptance"),
            ("precision", "P5", "wrong-result"),
        )

        for dimension, subclass, failure_contract in cases:
            with self.subTest(
                dimension=dimension,
                subclass=subclass,
                failure_contract=failure_contract,
            ):
                self.assertEqual(
                    validate_problem_taxonomy(
                        {
                            "problem_dimension": dimension,
                            "problem_subclass": subclass,
                            "failure_contract": failure_contract,
                        }
                    ),
                    (),
                )

    def test_unsupported_dimension_and_failure_contract_are_rejected(self) -> None:
        errors = validate_problem_taxonomy(
            {
                "problem_dimension": "performance",
                "problem_subclass": "B1",
                "failure_contract": "slow",
            }
        )

        self.assertIn(
            "operator.problem_dimension: expected 'boundary' or 'precision'",
            errors,
        )
        self.assertIn(
            "operator.failure_contract: unsupported value 'slow'",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
