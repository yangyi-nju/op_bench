from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.validation import ContractError


class SchemaValidationError(ContractError):
    """Raised when a schema artifact or instance fails closed validation."""


_SUPPORTED_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "oneOf",
    "const",
    "enum",
    "type",
    "required",
    "properties",
    "additionalProperties",
    "items",
    "minItems",
    "uniqueItems",
    "minimum",
    "minLength",
    "pattern",
}


def load_runtime_schema(path: Path | str | None = None) -> dict[str, Any]:
    schema_path = (
        Path(path)
        if path is not None
        else Path(__file__).resolve().parents[3] / "schemas" / "runtime_contracts.schema.json"
    )
    try:
        with schema_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaValidationError(f"cannot load runtime schema {schema_path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SchemaValidationError("runtime schema: expected object")
    _validate_schema_shape(value, "$")
    return value


def validate_schema_instance(
    instance: object,
    schema: dict[str, Any],
    definition: str | None = None,
) -> None:
    if not isinstance(schema, dict):
        raise SchemaValidationError("schema: expected object")
    _validate_schema_shape(schema, "$")
    selected = schema
    if definition is not None:
        definitions = schema.get("$defs")
        if not isinstance(definitions, dict) or definition not in definitions:
            raise SchemaValidationError(f"schema: unknown definition {definition!r}")
        selected = definitions[definition]
        if not isinstance(selected, dict):
            raise SchemaValidationError(f"schema definition {definition!r}: expected object")
    _validate(instance, selected, schema, "$")


def parse_runtime_contract(instance: object, schema: dict[str, Any] | None = None) -> object:
    selected_schema = schema or load_runtime_schema()
    validate_schema_instance(instance, selected_schema)
    if not isinstance(instance, dict):
        raise SchemaValidationError("$: expected object")
    contract_type = instance.get("contract_type")
    if not isinstance(contract_type, str):
        raise SchemaValidationError("$.contract_type: expected string")

    from op_bench.runtime.contracts import (
        ActionObservation,
        ActionRequest,
        AgentSpec,
        AgentTaskView,
        BudgetDelta,
        BudgetPolicy,
        CapabilityPolicy,
        ContentIdentity,
        EvaluationResultV06,
        EvaluationSpec,
        EventRecord,
        FullTaskSpec,
        IntegrityCheck,
        IntegrityReport,
        RuntimeProfile,
        SessionResult,
        SessionSpec,
        TestExecutionSummary,
        TestSelector,
    )
    from op_bench.runtime.manifest import ExpectedAttempt, RunManifest

    contract_classes = (
        ContentIdentity,
        CapabilityPolicy,
        BudgetPolicy,
        RuntimeProfile,
        TestSelector,
        FullTaskSpec,
        AgentTaskView,
        AgentSpec,
        ActionRequest,
        BudgetDelta,
        ActionObservation,
        EventRecord,
        SessionSpec,
        EvaluationSpec,
        SessionResult,
        TestExecutionSummary,
        EvaluationResultV06,
        IntegrityCheck,
        IntegrityReport,
        ExpectedAttempt,
        RunManifest,
    )
    by_type = {item.contract_type: item for item in contract_classes}
    try:
        contract_class = by_type[contract_type]
    except KeyError as exc:
        raise SchemaValidationError(f"$.contract_type: unsupported value {contract_type!r}") from exc
    return contract_class.from_dict(instance)


def _validate(instance: object, rule: dict[str, Any], root: dict[str, Any], path: str) -> None:
    if "$ref" in rule:
        target = _resolve_ref(rule["$ref"], root)
        _validate(instance, target, root, path)

    if "oneOf" in rule:
        choices = rule["oneOf"]
        if not isinstance(choices, list) or not choices:
            raise SchemaValidationError(f"{path}: oneOf must be a non-empty array")
        matches = 0
        errors: list[str] = []
        for choice in choices:
            if not isinstance(choice, dict):
                raise SchemaValidationError(f"{path}: oneOf entry must be an object")
            try:
                _validate(instance, choice, root, path)
            except SchemaValidationError as exc:
                errors.append(str(exc))
            else:
                matches += 1
        if matches != 1:
            detail = f"; first mismatch: {errors[0]}" if matches == 0 and errors else ""
            raise SchemaValidationError(
                f"{path}: expected exactly one oneOf match, got {matches}{detail}"
            )

    if "const" in rule and not _json_equal(instance, rule["const"]):
        raise SchemaValidationError(f"{path}: expected constant {rule['const']!r}")

    if "enum" in rule:
        values = rule["enum"]
        if not isinstance(values, list) or not any(_json_equal(instance, item) for item in values):
            raise SchemaValidationError(f"{path}: value {instance!r} is not in enum")

    expected_type = rule.get("type")
    if expected_type is not None:
        if not isinstance(expected_type, str):
            raise SchemaValidationError(f"{path}: schema type must be a string")
        if not _matches_type(instance, expected_type):
            raise SchemaValidationError(f"{path}: expected {expected_type}")

    if expected_type == "object" or (
        expected_type is None
        and isinstance(instance, dict)
        and any(key in rule for key in ("required", "properties", "additionalProperties"))
    ):
        _validate_object(instance, rule, root, path)
    elif expected_type == "array":
        _validate_array(instance, rule, root, path)
    elif expected_type == "string":
        _validate_string(instance, rule, path)
    elif expected_type == "integer":
        _validate_integer(instance, rule, path)


def _validate_object(
    instance: object,
    rule: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    assert isinstance(instance, dict)
    required = rule.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise SchemaValidationError(f"{path}: required must be an array of strings")
    for name in required:
        if name not in instance:
            raise SchemaValidationError(f"{path}: missing required property {name!r}")

    properties = rule.get("properties", {})
    if not isinstance(properties, dict):
        raise SchemaValidationError(f"{path}: properties must be an object")
    for name, child_rule in properties.items():
        if not isinstance(child_rule, dict):
            raise SchemaValidationError(f"{path}: property schema {name!r} must be an object")
        if name in instance:
            _validate(instance[name], child_rule, root, _child_path(path, name))

    extras = [name for name in instance if name not in properties]
    additional = rule.get("additionalProperties", True)
    if additional is False and extras:
        raise SchemaValidationError(f"{path}: unknown properties {sorted(extras)!r}")
    if isinstance(additional, dict):
        for name in extras:
            _validate(instance[name], additional, root, _child_path(path, name))
    elif additional not in (True, False):
        raise SchemaValidationError(f"{path}: additionalProperties must be boolean or object")


def _validate_array(
    instance: object,
    rule: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    assert isinstance(instance, list)
    minimum = rule.get("minItems")
    if minimum is not None:
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise SchemaValidationError(f"{path}: minItems must be a non-negative integer")
        if len(instance) < minimum:
            raise SchemaValidationError(f"{path}: expected at least {minimum} items")
    if rule.get("uniqueItems", False):
        encoded = [canonical_json(item) for item in instance]
        if len(encoded) != len(set(encoded)):
            raise SchemaValidationError(f"{path}: expected unique items")
    items = rule.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise SchemaValidationError(f"{path}: items must be an object")
        for index, item in enumerate(instance):
            _validate(item, items, root, f"{path}[{index}]")


def _validate_string(instance: object, rule: dict[str, Any], path: str) -> None:
    assert isinstance(instance, str)
    minimum = rule.get("minLength")
    if minimum is not None:
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise SchemaValidationError(f"{path}: minLength must be a non-negative integer")
        if len(instance) < minimum:
            raise SchemaValidationError(f"{path}: expected length >= {minimum}")
    pattern = rule.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise SchemaValidationError(f"{path}: pattern must be a string")
        try:
            matched = re.search(pattern, instance) is not None
        except re.error as exc:
            raise SchemaValidationError(f"{path}: invalid schema pattern: {exc}") from exc
        if not matched:
            raise SchemaValidationError(f"{path}: value does not match required pattern")


def _validate_integer(instance: object, rule: dict[str, Any], path: str) -> None:
    assert isinstance(instance, int) and not isinstance(instance, bool)
    minimum = rule.get("minimum")
    if minimum is not None:
        if not isinstance(minimum, int) or isinstance(minimum, bool):
            raise SchemaValidationError(f"{path}: minimum must be an integer")
        if instance < minimum:
            raise SchemaValidationError(f"{path}: expected value >= {minimum}")


def _resolve_ref(reference: object, root: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        raise SchemaValidationError(f"schema: unsupported reference {reference!r}")
    name = reference.removeprefix("#/$defs/")
    if not name or "/" in name:
        raise SchemaValidationError(f"schema: unsupported reference {reference!r}")
    definitions = root.get("$defs")
    if not isinstance(definitions, dict) or name not in definitions:
        raise SchemaValidationError(f"schema: unresolved reference {reference!r}")
    target = definitions[name]
    if not isinstance(target, dict):
        raise SchemaValidationError(f"schema: reference target {reference!r} is not an object")
    return target


def _validate_schema_shape(rule: dict[str, Any], path: str) -> None:
    for keyword in rule:
        if keyword not in _SUPPORTED_KEYWORDS:
            raise SchemaValidationError(f"{path}: unsupported schema keyword {keyword!r}")
    definitions = rule.get("$defs", {})
    if definitions:
        if not isinstance(definitions, dict):
            raise SchemaValidationError(f"{path}.$defs: expected object")
        for name, child in definitions.items():
            if not isinstance(child, dict):
                raise SchemaValidationError(f"{path}.$defs.{name}: expected object")
            _validate_schema_shape(child, f"{path}.$defs.{name}")
    properties = rule.get("properties", {})
    if properties:
        if not isinstance(properties, dict):
            raise SchemaValidationError(f"{path}.properties: expected object")
        for name, child in properties.items():
            if not isinstance(child, dict):
                raise SchemaValidationError(f"{path}.properties.{name}: expected object")
            _validate_schema_shape(child, f"{path}.properties.{name}")
    for index, child in enumerate(rule.get("oneOf", [])):
        if not isinstance(child, dict):
            raise SchemaValidationError(f"{path}.oneOf[{index}]: expected object")
        _validate_schema_shape(child, f"{path}.oneOf[{index}]")
    items = rule.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise SchemaValidationError(f"{path}.items: expected object")
        _validate_schema_shape(items, f"{path}.items")
    additional = rule.get("additionalProperties")
    if isinstance(additional, dict):
        _validate_schema_shape(additional, f"{path}.additionalProperties")


def _matches_type(value: object, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise SchemaValidationError(f"schema: unsupported type {expected!r}")


def _json_equal(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


def _child_path(path: str, name: object) -> str:
    return f"{path}.{name}" if isinstance(name, str) and name.isidentifier() else f"{path}[{name!r}]"
