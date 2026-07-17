from __future__ import annotations

from dataclasses import replace
import unittest

from op_bench.runtime.contracts import FullTaskSpec
from op_bench.runtime.schema import load_runtime_schema, validate_schema_instance
from op_bench.runtime.task_view import (
    AgentLaunchInput,
    TaskViewPolicy,
    agent_task_view_identity,
    assert_public_artifact_safe,
    project_agent_task_view,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_contracts import (
    SHA_B,
    budget_policy,
    capability_policy,
    full_task_spec,
    identity,
)


class AgentTaskViewProjectionTests(unittest.TestCase):
    def project(self, task: FullTaskSpec | None = None):
        return project_agent_task_view(
            task or full_task_spec(),
            capability_policy(),
            budget_policy(),
            policy=TaskViewPolicy(
                termination_notes="Finish after exporting a minimal patch.",
                attachments=(identity("attachment", "public-note", SHA_B),),
            ),
        )

    def test_projection_is_an_explicit_public_whitelist(self) -> None:
        task = full_task_spec()
        object.__setattr__(task, "future_answer", "return 42")

        view = self.project(task)
        encoded = view.to_dict()

        self.assertEqual(
            tuple(encoded),
            (
                "contract_type",
                "schema_version",
                "task",
                "statement_title",
                "statement_body",
                "framework",
                "operator_name",
                "runtime_hint",
                "public_tests",
                "capability_policy",
                "budget_policy",
                "termination_notes",
                "attachments",
            ),
        )
        self.assertNotIn("source", encoded)
        self.assertNotIn("environment", encoded)
        self.assertNotIn("hidden_tests", encoded)
        self.assertNotIn("fail_to_pass", encoded)
        self.assertNotIn("pass_to_pass", encoded)
        self.assertNotIn("patch_scope", encoded)
        self.assertNotIn("gold_patch", encoded)
        self.assertNotIn("hidden_test_asset", encoded)
        self.assertNotIn("admission", encoded)
        self.assertNotIn("future_answer", encoded)
        self.assertEqual(
            view.runtime_hint,
            "tier=cpu_python_overlay; platform=linux/amd64; gpu=no",
        )
        self.assertNotIn(task.runtime.image.identifier, view.runtime_hint)

    def test_projection_and_nested_public_contracts_validate_against_schema(self) -> None:
        view = self.project()
        schema = load_runtime_schema()

        validate_schema_instance(view.to_dict(), schema, definition="agent_task_view")
        validate_schema_instance(
            view.capability_policy.to_dict(), schema, definition="capability_policy"
        )
        validate_schema_instance(view.budget_policy.to_dict(), schema, definition="budget_policy")
        validate_schema_instance(
            view.public_tests[0].to_dict(), schema, definition="test_selector"
        )

    def test_task_view_identity_is_stable_and_content_sensitive(self) -> None:
        first = self.project()
        same = self.project()
        changed = self.project(replace(full_task_spec(), statement_body="Different public issue"))

        first_identity = agent_task_view_identity(first)

        self.assertEqual(first_identity, agent_task_view_identity(same))
        self.assertEqual(first_identity.identity_type, "task_view")
        self.assertEqual(first_identity.digest, first.content_hash)
        self.assertNotEqual(first_identity, agent_task_view_identity(changed))

    def test_adapter_launch_input_can_only_hold_the_public_view(self) -> None:
        view = self.project()
        launch = AgentLaunchInput(
            task_view=view,
            task_view_identity=agent_task_view_identity(view),
        )

        self.assertEqual(set(vars(launch)), {"task_view", "task_view_identity"})
        self.assertFalse(any(isinstance(value, FullTaskSpec) for value in vars(launch).values()))

        with self.assertRaisesRegex(ContractError, "task_view: expected AgentTaskView"):
            AgentLaunchInput(
                task_view=full_task_spec(),  # type: ignore[arg-type]
                task_view_identity=agent_task_view_identity(view),
            )

    def test_launch_input_rejects_a_mismatched_view_identity(self) -> None:
        view = self.project()
        other = self.project(replace(full_task_spec(), statement_body="Different public issue"))

        with self.assertRaisesRegex(ContractError, "task_view_identity: does not match"):
            AgentLaunchInput(
                task_view=view,
                task_view_identity=agent_task_view_identity(other),
            )

    def test_launch_input_rescans_directly_constructed_task_view(self) -> None:
        unsafe = replace(self.project(), statement_body="Copy the attached gold.patch")

        with self.assertRaisesRegex(ContractError, "public artifact"):
            AgentLaunchInput(
                task_view=unsafe,
                task_view_identity=agent_task_view_identity(unsafe),
            )


class PublicArtifactBoundaryTests(unittest.TestCase):
    def project_with_body(self, body: str) -> None:
        project_agent_task_view(
            replace(full_task_spec(), statement_body=body),
            capability_policy(),
            budget_policy(),
            policy=TaskViewPolicy(termination_notes="Finish with session_finish."),
        )

    def test_projection_rejects_direct_answer_source_clues(self) -> None:
        clues = (
            "The fix is in https://github.com/pytorch/pytorch/pull/123/files",
            "Copy https://github.com/pytorch/pytorch/commit/" + "a" * 40,
            "Apply https://patch-diff.githubusercontent.com/raw/org/repo/pull/2.patch",
            "Read https://github.com/org/repo/issues/7?focusedCommentId=12345",
            "Use the attached gold.patch as the answer",
            "Read hidden_test_asset for the expected behavior",
            "Admission evidence says to return 42",
        )

        for clue in clues:
            with self.subTest(clue=clue):
                with self.assertRaisesRegex(ContractError, "public artifact"):
                    self.project_with_body(clue)

    def test_projection_rejects_credentials_local_paths_and_private_output(self) -> None:
        unsafe = (
            "Authorization: Bearer secret-token-value",
            "Use api_key=sk-example-secret",
            "Open /Users/alice/private/result.json",
            "Open /opt/opbench/private/result.json",
            'Open "/Users/alice/private/result.json"',
            "Open file:///home/runner/private/result.json",
            r"Open C:\\Users\\alice\\private\\result.json",
            r"Open D:\\work\\private\\result.json",
            'Open "D:\\work\\private\\result.json"',
            "Use ghp_abcdefghijklmnopqrstuvwxyz123456",
            "-----BEGIN OPENSSH PRIVATE KEY-----\nredacted-fixture",
            "Use AWS access key AKIAIOSFODNN7EXAMPLE",
            "Use temporary AWS access key ASIAIOSFODNN7EXAMPLE",
            "The private_output contains the evaluator result",
        )

        for value in unsafe:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContractError, "public artifact"):
                    self.project_with_body(value)

    def test_recursive_scanner_rejects_sensitive_keys_and_values(self) -> None:
        fixtures = (
            {"nested": [{"gold_patch": "sha256:abc"}]},
            {"nested": {"goldPatch": "sha256:abc"}},
            {"metadata": {"hidden_tests": ["secret::case"]}},
            {"metadata": {"hiddenTestAsset": "secret::case"}},
            {"credentials": {"password": "secret"}},
            {"log": "saved at /home/runner/private.txt"},
            {"output": "https://github.com/org/repo/pull/99.diff"},
            {"opaque": b"not-json-and-cannot-be-scanned"},
        )

        for fixture in fixtures:
            with self.subTest(fixture=fixture):
                with self.assertRaisesRegex(ContractError, "public artifact"):
                    assert_public_artifact_safe(fixture)

    def test_recursive_scanner_accepts_the_projected_view(self) -> None:
        view = project_agent_task_view(
            full_task_spec(),
            capability_policy(),
            budget_policy(),
            policy=TaskViewPolicy(termination_notes="Finish with session_finish."),
        )

        assert_public_artifact_safe(view.to_dict())

    def test_projection_rejects_wrong_input_contracts_and_non_attachment_identity(self) -> None:
        with self.assertRaisesRegex(ContractError, "full_task: expected FullTaskSpec"):
            project_agent_task_view(  # type: ignore[arg-type]
                object(), capability_policy(), budget_policy()
            )

        with self.assertRaisesRegex(ContractError, "attachments: expected attachment identity"):
            TaskViewPolicy(
                termination_notes="Finish.",
                attachments=(identity("patch", "gold.patch"),),
            )


if __name__ == "__main__":
    unittest.main()
