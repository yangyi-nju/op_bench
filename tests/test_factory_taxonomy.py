from __future__ import annotations

import unittest

from op_bench.factory.taxonomy import (
    BOUNDARY_KEYWORD_PACKS,
    derived_slices,
    keyword_pack,
    match_keyword_packs,
    parse_taxonomy_v2,
    validate_problem_taxonomy,
)
from op_bench.runtime.validation import ContractError


def valid_taxonomy_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "taxonomy_version": "v2",
        "contract_family": "result",
        "contract_detail_tags": ["numerical"],
        "trigger_tags": [],
        "execution_context": {
            "devices": ["cpu"],
            "modes": ["eager"],
            "phases": ["forward"],
            "distributed": False,
        },
        "failure_type": "wrong_result",
        "root_cause_tags": [],
        "component_tags": [],
    }
    payload.update(overrides)
    return payload


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


class TaxonomyV2Tests(unittest.TestCase):
    def test_taxonomy_v2_requires_controlled_primary_axes(self) -> None:
        taxonomy = parse_taxonomy_v2(
            {
                "taxonomy_version": "v2",
                "contract_family": "result",
                "contract_detail_tags": ["numerical"],
                "trigger_tags": ["extreme_value_or_size", "device_specific"],
                "execution_context": {
                    "devices": ["cuda"],
                    "modes": ["eager"],
                    "phases": ["forward"],
                    "distributed": False,
                },
                "failure_type": "wrong_result",
                "root_cause_tags": ["overflow"],
                "component_tags": ["aten", "cuda_kernel"],
            }
        )

        self.assertEqual(derived_slices(taxonomy), ("boundary", "device", "precision"))

    def test_required_unknown_and_other_are_rejected(self) -> None:
        for field, value in (
            ("contract_family", "other"),
            ("failure_type", "unknown"),
        ):
            payload = valid_taxonomy_payload()
            payload[field] = value
            with self.subTest(field=field):
                with self.assertRaises(ContractError):
                    parse_taxonomy_v2(payload)

    def test_boundary_trigger_tags_map_to_boundary_slice(self) -> None:
        for trigger in (
            "empty_or_zero",
            "scalar_or_low_rank",
            "extreme_value_or_size",
            "invalid_or_endpoint_parameter",
            "noncontiguous_or_special_layout",
            "dynamic_shape",
        ):
            with self.subTest(trigger=trigger):
                payload = valid_taxonomy_payload(trigger_tags=[trigger])
                self.assertIn(
                    "boundary", derived_slices(parse_taxonomy_v2(payload))
                )

    def test_tuple_fields_require_registry_order_without_duplicates(self) -> None:
        contexts = (
            {
                "devices": ["cuda", "cpu"],
                "modes": ["compile", "eager"],
                "phases": ["backward", "forward"],
                "distributed": False,
            },
            {
                "devices": ["cpu", "cpu"],
                "modes": ["eager"],
                "phases": ["forward"],
                "distributed": False,
            },
        )

        for context in contexts:
            with self.subTest(context=context):
                payload = valid_taxonomy_payload(execution_context=context)
                with self.assertRaises(ContractError):
                    parse_taxonomy_v2(payload)


if __name__ == "__main__":
    unittest.main()
