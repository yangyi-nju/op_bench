from __future__ import annotations

import copy
import unittest

from op_bench.runtime.contracts import (
    AgentSpec,
    AgentTaskView,
    BudgetPolicy,
    CapabilityPolicy,
    ContentIdentity,
    FullTaskSpec,
    RuntimeProfile,
    TestSelector,
)
from op_bench.runtime.validation import ContractError


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64


def identity(identity_type: str, identifier: str, digest: str = SHA_A) -> ContentIdentity:
    return ContentIdentity(
        identity_type=identity_type,
        identifier=identifier,
        digest=digest,
        digest_kind="content_sha256",
    )


def capability_policy() -> CapabilityPolicy:
    return CapabilityPolicy(
        policy_id="controlled-v1",
        allowed_actions=("workspace_read", "workspace_write", "test_run", "session_finish"),
        writable_paths=("torch/",),
        allowed_command_prefixes=("python", "rg"),
        registered_tests=("public::smoke",),
        max_read_bytes=1_000_000,
        max_write_bytes=1_000_000,
        max_output_bytes=2_000_000,
        network_access="provider_only",
    )


def budget_policy() -> BudgetPolicy:
    return BudgetPolicy(
        policy_id="standard-v1",
        wall_clock_ms=1_200_000,
        max_actions=200,
        max_tests=20,
        max_commands=50,
        max_output_bytes=10_000_000,
        provider_token_limit=None,
    )


def runtime_profile() -> RuntimeProfile:
    return RuntimeProfile(
        profile_id="cpu-overlay-v1",
        backend="remote_docker",
        runtime_tier="cpu_python_overlay",
        source_loading_mode="python_overlay",
        platform="linux/amd64",
        image=identity("image", "op-bench/pytorch-cpu:torch2.6", SHA_B),
        requires_gpu=False,
        network_policy="denied",
        timeout_ms=900_000,
    )


def public_test() -> TestSelector:
    return TestSelector(
        selector_id="public::smoke",
        visibility="public",
        command_template="{python} test/test_nn.py {test}",
        description="Public smoke selector",
    )


def full_task_spec() -> FullTaskSpec:
    return FullTaskSpec(
        task=identity("task", "pytorch__example", SHA_A),
        source=identity("source", "pytorch@abc", SHA_B),
        environment=identity("environment", "pytorch-cpu", SHA_C),
        runtime=runtime_profile(),
        statement_title="Operator returns the wrong value",
        statement_body="Repair the operator behavior.",
        framework="pytorch",
        operator_name="aten.example.default",
        public_tests=(public_test(),),
        hidden_tests=(
            TestSelector(
                selector_id="hidden::f2p",
                visibility="hidden",
                command_template="{python} test/test_hidden.py {test}",
                description="Evaluator-only selector",
            ),
        ),
        fail_to_pass=("hidden::f2p",),
        pass_to_pass=("public::smoke",),
        patch_scope=("torch/example.py",),
        gold_patch=identity("patch", "gold.patch", SHA_A),
        hidden_test_asset=identity("test", "hidden.patch", SHA_B),
        admission=identity("admission", "evidence.json", SHA_C),
    )


def agent_task_view() -> AgentTaskView:
    return AgentTaskView(
        task=identity("task", "pytorch__example", SHA_A),
        statement_title="Operator returns the wrong value",
        statement_body="Repair the operator behavior.",
        framework="pytorch",
        operator_name="aten.example.default",
        runtime_hint="CPU Python overlay",
        public_tests=(public_test(),),
        capability_policy=capability_policy(),
        budget_policy=budget_policy(),
        termination_notes="Finish after exporting a minimal patch.",
        attachments=(identity("attachment", "public-note", SHA_B),),
    )


def agent_spec() -> AgentSpec:
    return AgentSpec(
        agent=identity("agent", "codex", SHA_A),
        model=identity("model", "gpt-example", SHA_B),
        adapter=identity("adapter", "canonical-cli", SHA_C),
        system_prompt=identity("prompt", "system", SHA_A),
        task_prompt=identity("prompt", "task-template", SHA_B),
        config=identity("agent_config", "codex-config", SHA_C),
        feedback_policy="visible",
    )


class RuntimeContractRoundTripTests(unittest.TestCase):
    def test_every_core_contract_round_trips_with_a_stable_hash(self) -> None:
        values = (
            identity("dataset", "pytorch_v0.5"),
            capability_policy(),
            budget_policy(),
            runtime_profile(),
            public_test(),
            full_task_spec(),
            agent_task_view(),
            agent_spec(),
        )

        for value in values:
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                reordered = dict(reversed(list(encoded.items())))
                decoded = type(value).from_dict(reordered)
                self.assertEqual(decoded, value)
                self.assertEqual(decoded.content_hash, value.content_hash)
                self.assertEqual(encoded["schema_version"], "v1")
                self.assertEqual(encoded["contract_type"], value.contract_type)

    def test_tuple_fields_are_encoded_as_json_arrays(self) -> None:
        encoded = capability_policy().to_dict()

        self.assertIsInstance(encoded["allowed_actions"], list)
        self.assertIsInstance(encoded["writable_paths"], list)


class RuntimeContractNegativeTests(unittest.TestCase):
    def test_every_core_contract_rejects_an_unknown_version(self) -> None:
        values = (
            identity("dataset", "pytorch_v0.5"),
            capability_policy(),
            budget_policy(),
            runtime_profile(),
            public_test(),
            full_task_spec(),
            agent_task_view(),
            agent_spec(),
        )

        for value in values:
            with self.subTest(contract=value.contract_type):
                encoded = value.to_dict()
                encoded["schema_version"] = "v999"
                with self.assertRaisesRegex(ContractError, "schema_version: expected 'v1'"):
                    type(value).from_dict(encoded)

    def test_rejects_unknown_fields(self) -> None:
        encoded = identity("task", "task-a").to_dict()
        encoded["future_answer"] = "must not be silently accepted"

        with self.assertRaisesRegex(ContractError, r"content_identity: unknown fields \['future_answer'\]"):
            ContentIdentity.from_dict(encoded)

    def test_rejects_missing_identity(self) -> None:
        encoded = agent_spec().to_dict()
        del encoded["model"]

        with self.assertRaisesRegex(ContractError, r"agent_spec: missing fields \['model'\]"):
            AgentSpec.from_dict(encoded)

    def test_rejects_invalid_identity_and_digest_kinds(self) -> None:
        with self.assertRaisesRegex(ContractError, "identity_type: unsupported value"):
            identity("credential", "secret")
        with self.assertRaisesRegex(ContractError, "digest_kind: unsupported value"):
            ContentIdentity("task", "task-a", SHA_A, "unverified_guess")

    def test_rejects_invalid_runtime_enums(self) -> None:
        encoded = runtime_profile().to_dict()
        encoded["backend"] = "network_probe"

        with self.assertRaisesRegex(ContractError, "backend: unsupported value"):
            RuntimeProfile.from_dict(encoded)

    def test_rejects_invalid_action_and_network_policy(self) -> None:
        encoded = capability_policy().to_dict()
        encoded["allowed_actions"].append("host_scan")

        with self.assertRaisesRegex(ContractError, r"allowed_actions\[4\]: unsupported value"):
            CapabilityPolicy.from_dict(encoded)

        encoded = capability_policy().to_dict()
        encoded["network_access"] = "probe"
        with self.assertRaisesRegex(ContractError, "network_access: unsupported value"):
            CapabilityPolicy.from_dict(encoded)

    def test_rejects_bool_used_as_budget_integer(self) -> None:
        encoded = budget_policy().to_dict()
        encoded["max_actions"] = True

        with self.assertRaisesRegex(ContractError, "max_actions: expected integer"):
            BudgetPolicy.from_dict(encoded)

    def test_rejects_duplicate_policy_entries(self) -> None:
        encoded = capability_policy().to_dict()
        encoded["writable_paths"] = ["torch/", "torch/"]

        with self.assertRaisesRegex(ContractError, "writable_paths: duplicate value 'torch/'"):
            CapabilityPolicy.from_dict(encoded)

    def test_direct_constructor_is_also_validated(self) -> None:
        with self.assertRaisesRegex(ContractError, "wall_clock_ms: must be >= 1"):
            BudgetPolicy(
                policy_id="bad",
                wall_clock_ms=0,
                max_actions=1,
                max_tests=1,
                max_commands=1,
                max_output_bytes=1,
                provider_token_limit=None,
            )

    def test_nested_contract_type_cannot_be_swapped(self) -> None:
        encoded = copy.deepcopy(full_task_spec().to_dict())
        encoded["runtime"]["contract_type"] = "budget_policy"

        with self.assertRaisesRegex(ContractError, "runtime_profile.contract_type: expected 'runtime_profile'"):
            FullTaskSpec.from_dict(encoded)


if __name__ == "__main__":
    unittest.main()
