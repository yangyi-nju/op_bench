from __future__ import annotations

import hashlib
import json
from typing import Any

from op_bench.runtime.validation import ContractError


JsonValue = Any


def canonical_json(value: JsonValue) -> str:
    """Return the single canonical JSON representation used for v0.6 identity.

    Wire contracts intentionally exclude floating-point numbers. Runtime and
    score measurements that need fractional values are encoded as integral
    units (for example milliseconds) so hashes are stable across JSON stacks.
    """

    _validate_json_value(value, path="$")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: JsonValue) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _validate_json_value(value: JsonValue, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        raise ContractError(f"{path}: floats are not canonical")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ContractError(f"{path}: object keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ContractError(f"{path}: unsupported JSON value type {type(value).__name__}")
