"""Trusted comparison of finite numeric observations, without candidate imports.

The wire format is one shape and a flat array of binary64-representable real
numbers. Empty tensors are allowed when a shape dimension is zero; a scalar has
shape [] and exactly one value. This checks values and shape, not a program's
internal dtype, device, algorithm, or general resistance to malicious code.
"""
from __future__ import annotations

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path


MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_ELEMENTS = 1_000_000


class NumericFormatError(ValueError):
    pass


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise NumericFormatError("Duplicate JSON object key")
        result[key] = value
    return result


def parse_document(data: bytes):
    if len(data) > MAX_DOCUMENT_BYTES:
        raise NumericFormatError("Numeric document exceeds its byte limit")
    try:
        return json.loads(data, object_pairs_hook=_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(NumericFormatError("Nonfinite JSON number")))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise NumericFormatError(f"Invalid numeric JSON: {exc}") from exc


def read_document(path: Path):
    with path.open("rb") as stream:
        return parse_document(stream.read(MAX_DOCUMENT_BYTES + 1))


def finite_number(value, name):
    if type(value) not in {int, float}:
        raise NumericFormatError(f"{name} must be a real number, not a boolean or string")
    try:
        converted = float(value)
    except (ValueError, OverflowError) as exc:
        raise NumericFormatError(f"{name} is outside the supported finite numeric range") from exc
    if not math.isfinite(converted):
        raise NumericFormatError(f"{name} must be finite")
    return converted


def tensor(value):
    if not isinstance(value, dict) or set(value) != {"shape", "values"}:
        raise NumericFormatError("Numeric observation must contain exactly shape and values")
    shape, values = value["shape"], value["values"]
    if not isinstance(shape, list) or len(shape) > 64 or any(
            type(dimension) is not int or dimension < 0 or dimension > MAX_ELEMENTS for dimension in shape):
        raise NumericFormatError("shape requires at most 64 nonnegative integer dimensions within the element limit")
    count = math.prod(shape)
    if count > MAX_ELEMENTS or not isinstance(values, list) or len(values) != count:
        raise NumericFormatError("values length must equal the shape product within the element limit")
    return {"shape": shape, "values": [finite_number(item, "value") for item in values]}


def validate_oracle(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "input", "expected", "atol", "rtol"}:
        raise NumericFormatError("Oracle requires schema_version, input, expected, atol and rtol")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise NumericFormatError("Unsupported numeric oracle schema_version")
    # Re-encoding validates finite input JSON, including otherwise overflowing
    # exponents decoded as float infinity. Inputs may include booleans and text;
    # output values always use the stricter finite-real contract above.
    try:
        encoded = json.dumps(value["input"], allow_nan=False).encode() + b"\n"
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise NumericFormatError(f"Invalid numeric case input: {exc}") from exc
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise NumericFormatError("Numeric input exceeds its byte limit")
    expected = tensor(value["expected"])
    atol, rtol = finite_number(value["atol"], "atol"), finite_number(value["rtol"], "rtol")
    if atol < 0 or rtol < 0:
        raise NumericFormatError("Numeric tolerances must be nonnegative")
    normalized = {"schema_version": 1, "input": value["input"], "expected": expected, "atol": atol, "rtol": rtol}
    if len(json.dumps(normalized, allow_nan=False, separators=(",", ":")).encode()) + 1 > MAX_DOCUMENT_BYTES:
        raise NumericFormatError("Normalized numeric oracle exceeds its byte limit")
    return normalized


def load_oracle(grader_dir: Path, relative: str):
    name = Path(relative)
    path = (grader_dir / name).resolve()
    if name.is_absolute() or ".." in name.parts or not path.is_relative_to(grader_dir.resolve()):
        raise NumericFormatError("Numeric oracle must stay inside the private grader directory")
    return validate_oracle(read_document(path))


def judge(oracle: dict, output: bytes) -> dict:
    """Compare an untrusted byte stream with an already validated private oracle."""
    if not output.strip():
        return {"passed": False, "reason": "missing_numeric_observation", "valid_observation": False}
    try:
        actual = tensor(parse_document(output))
    except NumericFormatError as exc:
        return {"passed": False, "reason": "invalid_numeric_observation", "valid_observation": False,
                "detail": str(exc)}
    expected = oracle["expected"]
    if actual["shape"] != expected["shape"]:
        return {"passed": False, "reason": "numeric_shape_mismatch", "valid_observation": True,
                "expected_shape": expected["shape"], "observed_shape": actual["shape"]}
    failures, examples, maximum_error = 0, [], Decimal(0)
    # Exact binary64 inputs need up to 1074 fractional binary places. This
    # decimal precision avoids overflow and tolerance-boundary rounding in the
    # comparison; it is not a numerical reference implementation of an operator.
    with localcontext() as context:
        context.prec = 2200
        absolute, relative = Decimal.from_float(oracle["atol"]), Decimal.from_float(oracle["rtol"])
        for index, (observed, reference) in enumerate(zip(actual["values"], expected["values"])):
            target = Decimal.from_float(reference)
            difference = abs(Decimal.from_float(observed) - target)
            maximum_error = max(maximum_error, difference)
            if difference > absolute + relative * abs(target):
                failures += 1
                if len(examples) < 8:
                    examples.append({"index": index, "observed": observed, "expected": reference})
    return {"passed": failures == 0, "reason": None if failures == 0 else "numeric_value_mismatch",
            "valid_observation": True, "compared_values": len(actual["values"]),
            "mismatched_values": failures, "max_absolute_error_decimal": str(maximum_error),
            "examples": examples}


def process_judgment(oracle: dict, output: bytes, process: dict) -> dict:
    if process.get("exit_code") != 0 or process.get("timed_out") or process.get("output_limited"):
        return {"passed": False, "reason": "numeric_process_failed", "valid_observation": None}
    return judge(oracle, output)
