from __future__ import annotations

import math
import subprocess
import sys
import unittest

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError, require_int


class CanonicalJsonTests(unittest.TestCase):
    def test_key_order_does_not_change_bytes_or_hash(self) -> None:
        left = {"beta": [2, {"value": "算子"}], "alpha": 1}
        right = {"alpha": 1, "beta": [2, {"value": "算子"}]}

        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(canonical_sha256(left), canonical_sha256(right))
        self.assertEqual(
            canonical_json(left),
            '{"alpha":1,"beta":[2,{"value":"算子"}]}',
        )

    def test_hash_is_reconstructed_in_a_fresh_process(self) -> None:
        payload = {"schema_version": "v1", "values": [1, "two", False, None]}
        expected = canonical_sha256(payload)
        script = (
            "from op_bench.runtime.canonical import canonical_sha256; "
            f"print(canonical_sha256({payload!r}))"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), expected)
        self.assertRegex(expected, r"^sha256:[0-9a-f]{64}$")

    def test_rejects_non_string_object_keys(self) -> None:
        with self.assertRaisesRegex(ContractError, r"\$: object keys must be strings"):
            canonical_json({1: "value"})

    def test_rejects_non_finite_numbers(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContractError, r"\$\.value: floats are not canonical"):
                    canonical_json({"value": value})

    def test_rejects_finite_floats_to_keep_wire_numbers_integral(self) -> None:
        with self.assertRaisesRegex(ContractError, r"\$\.value: floats are not canonical"):
            canonical_json({"value": 1.5})


class StrictValidationTests(unittest.TestCase):
    def test_bool_is_not_accepted_as_an_integer(self) -> None:
        with self.assertRaisesRegex(ContractError, "budget.max_actions: expected integer"):
            require_int(True, "budget.max_actions", minimum=0)

    def test_integer_minimum_is_enforced(self) -> None:
        with self.assertRaisesRegex(ContractError, "budget.max_actions: must be >= 1"):
            require_int(0, "budget.max_actions", minimum=1)


if __name__ == "__main__":
    unittest.main()
