from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


class ContractError(ValueError):
    """Raised when a runtime contract is not canonical or fails validation."""


def require_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{path}: expected integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{path}: must be >= {minimum}")
    return value


def require_optional_int(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
) -> int | None:
    if value is None:
        return None
    return require_int(value, path, minimum=minimum)


def require_bool(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ContractError(f"{path}: expected boolean")
    return value


def require_str(
    value: object,
    path: str,
    *,
    min_length: int = 1,
    pattern: str | None = None,
) -> str:
    if not isinstance(value, str):
        raise ContractError(f"{path}: expected string")
    if len(value) < min_length:
        raise ContractError(f"{path}: must contain at least {min_length} character(s)")
    if pattern is not None and re.fullmatch(pattern, value) is None:
        raise ContractError(f"{path}: does not match required pattern")
    return value


def require_enum(value: object, path: str, allowed: Iterable[str]) -> str:
    text = require_str(value, path)
    allowed_values = tuple(allowed)
    if text not in allowed_values:
        raise ContractError(f"{path}: unsupported value {text!r}")
    return text


def require_mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path}: expected object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ContractError(f"{path}: object keys must be strings")
        result[key] = item
    return result


def require_exact_fields(
    value: object,
    path: str,
    fields: Iterable[str],
) -> dict[str, Any]:
    data = require_mapping(value, path)
    expected = set(fields)
    actual = set(data)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ContractError(f"{path}: missing fields {missing}")
    if unknown:
        raise ContractError(f"{path}: unknown fields {unknown}")
    return data


def require_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{path}: expected array")
    return value


def require_str_tuple(
    value: object,
    path: str,
    *,
    allowed: Iterable[str] | None = None,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    items = require_list(value, path)
    if not allow_empty and not items:
        raise ContractError(f"{path}: must contain at least one value")
    result: list[str] = []
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        text = require_enum(item, item_path, allowed) if allowed is not None else require_str(item, item_path)
        if text in result:
            raise ContractError(f"{path}: duplicate value {text!r}")
        result.append(text)
    return tuple(result)
