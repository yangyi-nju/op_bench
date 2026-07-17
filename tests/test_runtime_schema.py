from __future__ import annotations

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from op_bench.runtime.contracts import ContentIdentity
from op_bench.runtime.schema import (
    SchemaValidationError,
    load_runtime_schema,
    validate_schema_instance,
)
from tests.test_runtime_contracts import (
    agent_spec,
    agent_task_view,
    budget_policy,
    capability_policy,
    full_task_spec,
    identity,
    public_test,
    runtime_profile,
)
from tests.test_runtime_manifest import manifest
from tests.test_runtime_wire_contracts import (
    action_observation,
    action_request,
    budget_delta,
    evaluation_result,
    evaluation_spec,
    event_record,
    integrity_report,
    session_result,
    session_spec,
    test_summary,
)


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "runtime_contracts.schema.json"


def contract_fixtures():
    run = manifest()
    report = integrity_report()
    return (
        identity("dataset", "pytorch_v0.5"),
        capability_policy(),
        budget_policy(),
        runtime_profile(),
        public_test(),
        full_task_spec(),
        agent_task_view(),
        agent_spec(),
        action_request(),
        budget_delta(),
        action_observation(),
        event_record(),
        session_spec(),
        evaluation_spec(),
        session_result(),
        test_summary(),
        evaluation_result(),
        report.checks[0],
        report,
        run.expected_attempts[0],
        run,
    )


class RuntimeSchemaTests(unittest.TestCase):
    def test_schema_has_and_independently_validates_every_m1_contract(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)
        expected = {value.contract_type for value in contract_fixtures()}

        self.assertEqual(set(schema["$defs"]) - {"JsonValue", "JsonObject"}, expected)
        for value in contract_fixtures():
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                validate_schema_instance(encoded, schema)
                validate_schema_instance(encoded, schema, definition=value.contract_type)

    def test_schema_validation_does_not_call_python_contract_parsers(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)

        with patch.object(ContentIdentity, "from_dict", side_effect=AssertionError("parser called")):
            validate_schema_instance(
                identity("task", "task-a").to_dict(),
                schema,
                definition="content_identity",
            )

    def test_every_contract_rejects_unknown_fields_and_bad_or_missing_versions(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)

        for value in contract_fixtures():
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                unknown = copy.deepcopy(encoded)
                unknown["future_field"] = "must fail closed"
                with self.assertRaises(SchemaValidationError):
                    validate_schema_instance(unknown, schema, definition=value.contract_type)

                missing = copy.deepcopy(encoded)
                del missing["schema_version"]
                with self.assertRaises(SchemaValidationError):
                    validate_schema_instance(missing, schema, definition=value.contract_type)

                wrong_version = copy.deepcopy(encoded)
                wrong_version["schema_version"] = "v999"
                with self.assertRaises(SchemaValidationError):
                    validate_schema_instance(wrong_version, schema, definition=value.contract_type)

    def test_schema_rejects_missing_identity_invalid_enum_and_bool_integer(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)

        missing_identity = agent_spec().to_dict()
        del missing_identity["model"]
        with self.assertRaisesRegex(SchemaValidationError, "missing required property 'model'"):
            validate_schema_instance(missing_identity, schema, definition="agent_spec")

        invalid_enum = action_request().to_dict()
        invalid_enum["action_name"] = "network_probe"
        with self.assertRaisesRegex(SchemaValidationError, "not in enum"):
            validate_schema_instance(invalid_enum, schema, definition="action_request")

        invalid_integer = budget_policy().to_dict()
        invalid_integer["max_actions"] = True
        with self.assertRaisesRegex(SchemaValidationError, "expected integer"):
            validate_schema_instance(invalid_integer, schema, definition="budget_policy")

    def test_schema_rejects_nested_identity_role_swaps(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)
        cases = []

        runtime = runtime_profile().to_dict()
        runtime["image"]["identity_type"] = "agent"
        cases.append(("runtime_profile", runtime))

        task = full_task_spec().to_dict()
        task["task"]["identity_type"] = "model"
        cases.append(("full_task_spec", task))

        session = session_spec().to_dict()
        session["workspace"]["identity_type"] = "source"
        cases.append(("session_spec", session))

        for definition, encoded in cases:
            with self.subTest(definition=definition):
                with self.assertRaises(SchemaValidationError):
                    validate_schema_instance(encoded, schema, definition=definition)

    def test_schema_accepts_nullable_patch_and_agent_terminal_axes(self) -> None:
        schema = load_runtime_schema(SCHEMA_PATH)
        spec = evaluation_spec().to_dict()
        spec["frozen_patch"] = None
        validate_schema_instance(spec, schema, definition="evaluation_spec")

        result = evaluation_result().to_dict()
        result["attempt_validity"] = "infrastructure_invalid"
        result["agent_terminal"] = None
        result["evaluation_outcome"] = "not_evaluated"
        result["invalid_reason"] = "session_platform_error"
        result["patch"] = None
        validate_schema_instance(result, schema, definition="evaluation_result")

    def test_validator_rejects_unsupported_schema_keywords(self) -> None:
        schema = {"type": "string", "format": "hostname"}

        with self.assertRaisesRegex(SchemaValidationError, "unsupported schema keyword 'format'"):
            validate_schema_instance("example", schema)


if __name__ == "__main__":
    unittest.main()
