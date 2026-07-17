from __future__ import annotations

from dataclasses import replace
import unittest

from op_bench.runtime.manifest import (
    ExpectedAttempt,
    RunManifest,
    attempt_id,
    build_run_manifest,
    cohort_id,
    comparability_key,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_contracts import (
    SHA_A,
    SHA_B,
    SHA_C,
    agent_spec,
    budget_policy,
    capability_policy,
    full_task_spec,
    identity,
)
from op_bench.runtime.task_view import agent_task_view_identity, project_agent_task_view


def manifest(
    *,
    tasks=None,
    agents=None,
    dataset=None,
    capability=None,
    budget=None,
    retry=None,
    termination=None,
    scoring=None,
    repeat_count: int = 2,
    created_at: str = "2026-07-17T10:00:00Z",
    platform_version: str = "opbench-v0.6.0",
    action_protocol: str = "action-v1",
    evaluation_protocol: str = "evaluation-v1",
    scoring_protocol: str = "scoring-v1",
    task_views=None,
) -> RunManifest:
    arguments = dict(
        platform_version=platform_version,
        action_protocol=action_protocol,
        evaluation_protocol=evaluation_protocol,
        scoring_protocol=scoring_protocol,
        dataset=dataset if dataset is not None else identity("dataset", "pytorch_v0.5", SHA_A),
        tasks=tasks if tasks is not None else (full_task_spec(),),
        agents=agents if agents is not None else (agent_spec(),),
        capability_policy=capability if capability is not None else capability_policy(),
        budget_policy=budget if budget is not None else budget_policy(),
        retry_policy=retry if retry is not None else identity("policy", "retry-v1", SHA_B),
        termination_policy=(
            termination
            if termination is not None
            else identity("policy", "termination-v1", SHA_C)
        ),
        scoring=scoring if scoring is not None else identity("scoring", "opbench-scoring-v1", SHA_A),
        repeat_count=repeat_count,
        created_at=created_at,
    )
    if task_views is not None:
        arguments["task_views"] = task_views
    return build_run_manifest(**arguments)


class RunManifestTests(unittest.TestCase):
    def test_expected_matrix_is_frozen_in_deterministic_order(self) -> None:
        task_b = replace(
            full_task_spec(),
            task=identity("task", "pytorch__b", SHA_B),
        )
        task_a = replace(
            full_task_spec(),
            task=identity("task", "pytorch__a", SHA_A),
        )
        agent_b = replace(
            agent_spec(),
            agent=identity("agent", "agent-b", SHA_B),
        )
        agent_a = replace(
            agent_spec(),
            agent=identity("agent", "agent-a", SHA_A),
        )

        result = manifest(tasks=(task_b, task_a), agents=(agent_b, agent_a), repeat_count=2)

        observed = [
            (item.task.identifier, item.agent.identifier, item.repeat)
            for item in result.expected_attempts
        ]
        self.assertEqual(
            observed,
            [
                ("pytorch__a", "agent-a", 1),
                ("pytorch__a", "agent-a", 2),
                ("pytorch__a", "agent-b", 1),
                ("pytorch__a", "agent-b", 2),
                ("pytorch__b", "agent-a", 1),
                ("pytorch__b", "agent-a", 2),
                ("pytorch__b", "agent-b", 1),
                ("pytorch__b", "agent-b", 2),
            ],
        )
        self.assertEqual(len({item.attempt_id for item in result.expected_attempts}), 8)

    def test_manifest_round_trips_and_revalidates_derived_identity(self) -> None:
        value = manifest()

        decoded = RunManifest.from_dict(value.to_dict())

        self.assertEqual(decoded, value)
        self.assertRegex(value.comparability_key, r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(value.cohort_id, r"^cohort:v1:[0-9a-f]{64}$")
        self.assertEqual(len(value.expected_attempts), 2)

    def test_public_comparability_and_cohort_apis_rebuild_manifest_identity(self) -> None:
        value = manifest()

        self.assertEqual(comparability_key(value), value.comparability_key)
        self.assertEqual(cohort_id(value.comparability_key), value.cohort_id)

        with self.assertRaisesRegex(ContractError, "manifest: expected RunManifest"):
            comparability_key(object())
        with self.assertRaisesRegex(ContractError, "comparability_key: does not match"):
            cohort_id("not-a-hash")

    def test_created_at_does_not_change_comparability_or_attempt_identity(self) -> None:
        first = manifest(created_at="2026-07-17T10:00:00Z")
        second = manifest(created_at="2026-07-18T10:00:00Z")

        self.assertEqual(first.comparability_key, second.comparability_key)
        self.assertEqual(first.cohort_id, second.cohort_id)
        self.assertEqual(first.expected_attempts, second.expected_attempts)
        self.assertNotEqual(first.content_hash, second.content_hash)

    def test_comparability_key_changes_for_every_comparability_axis(self) -> None:
        base = manifest()
        task = full_task_spec()
        agent = agent_spec()
        mutations = {
            "dataset": {"dataset": identity("dataset", "pytorch_v0.5", SHA_B)},
            "task": {"tasks": (replace(task, statement_body="Different task statement"),)},
            "source": {"tasks": (replace(task, source=identity("source", "pytorch@other", SHA_C)),)},
            "environment": {
                "tasks": (replace(task, environment=identity("environment", "other-env", SHA_B)),)
            },
            "image": {
                "tasks": (
                    replace(
                        task,
                        runtime=replace(
                            task.runtime,
                            image=identity("image", "other-image", SHA_C),
                        ),
                    ),
                )
            },
            "agent": {"agents": (replace(agent, agent=identity("agent", "other-agent", SHA_B)),)},
            "model": {"agents": (replace(agent, model=identity("model", "other-model", SHA_C)),)},
            "adapter": {
                "agents": (replace(agent, adapter=identity("adapter", "other-adapter", SHA_B)),)
            },
            "prompt": {
                "agents": (
                    replace(agent, task_prompt=identity("prompt", "other-template", SHA_C)),
                )
            },
            "capability": {
                "capability": replace(capability_policy(), max_read_bytes=999_999)
            },
            "budget": {"budget": replace(budget_policy(), max_actions=199)},
            "runtime": {
                "tasks": (replace(task, runtime=replace(task.runtime, timeout_ms=899_000)),)
            },
            "action_protocol": {"action_protocol": "action-v2"},
            "evaluation_protocol": {"evaluation_protocol": "evaluation-v2"},
            "scoring_protocol": {"scoring_protocol": "scoring-v2"},
            "scoring": {"scoring": identity("scoring", "opbench-scoring-v2", SHA_C)},
            "retry": {"retry": identity("policy", "retry-v2", SHA_A)},
            "termination": {"termination": identity("policy", "termination-v2", SHA_A)},
        }

        for name, kwargs in mutations.items():
            with self.subTest(axis=name):
                changed = manifest(**kwargs)
                self.assertNotEqual(changed.comparability_key, base.comparability_key)
                self.assertNotEqual(changed.cohort_id, base.cohort_id)
                self.assertNotEqual(
                    changed.expected_attempts[0].attempt_id,
                    base.expected_attempts[0].attempt_id,
                )

    def test_attempt_identity_uses_cohort_task_agent_repeat_and_effective_config(self) -> None:
        task = identity("task", "task-a", SHA_A)
        agent = identity("agent", "agent-a", SHA_B)
        baseline = attempt_id("cohort:v1:" + "c" * 64, task, agent, 1, SHA_C)

        variants = (
            attempt_id("cohort:v1:" + "d" * 64, task, agent, 1, SHA_C),
            attempt_id("cohort:v1:" + "c" * 64, identity("task", "task-b", SHA_A), agent, 1, SHA_C),
            attempt_id("cohort:v1:" + "c" * 64, task, identity("agent", "agent-b", SHA_B), 1, SHA_C),
            attempt_id("cohort:v1:" + "c" * 64, task, agent, 2, SHA_C),
            attempt_id("cohort:v1:" + "c" * 64, task, agent, 1, SHA_A),
        )

        self.assertRegex(baseline, r"^attempt:v1:[0-9a-f]{64}$")
        self.assertEqual(len(set((baseline, *variants))), 6)

    def test_agent_task_view_is_frozen_into_manifest_and_attempt_identity(self) -> None:
        base = manifest()
        view = base.task_views[0]
        changed_view = replace(
            view,
            termination_notes="Finish only after one final registered test.",
        )
        changed = manifest(task_views=(changed_view,))

        self.assertEqual(view, project_agent_task_view(full_task_spec(), capability_policy(), budget_policy()))
        self.assertEqual(
            base.expected_attempts[0].task_view,
            agent_task_view_identity(view),
        )
        self.assertNotEqual(base.comparability_key, changed.comparability_key)
        self.assertNotEqual(base.cohort_id, changed.cohort_id)
        self.assertNotEqual(
            base.expected_attempts[0].attempt_id,
            changed.expected_attempts[0].attempt_id,
        )

    def test_manifest_rejects_task_view_not_matching_full_task_or_public_policy(self) -> None:
        projected = project_agent_task_view(full_task_spec(), capability_policy(), budget_policy())
        cases = (
            replace(projected, statement_body="Unrelated issue"),
            replace(projected, capability_policy=replace(capability_policy(), max_read_bytes=99)),
            replace(projected, task=identity("task", "other-task", SHA_B)),
        )

        for task_view in cases:
            with self.subTest(task_view=task_view):
                with self.assertRaisesRegex(ContractError, "task_views"):
                    manifest(task_views=(task_view,))


class RunManifestNegativeTests(unittest.TestCase):
    def test_rejects_empty_or_duplicate_inputs_and_invalid_repeat(self) -> None:
        with self.assertRaisesRegex(ContractError, "tasks: must contain at least one value"):
            manifest(tasks=())
        with self.assertRaisesRegex(ContractError, "agents: must contain at least one value"):
            manifest(agents=())
        with self.assertRaisesRegex(ContractError, "tasks: duplicate identifier"):
            manifest(tasks=(full_task_spec(), full_task_spec()))
        with self.assertRaisesRegex(ContractError, "agents: duplicate identifier"):
            manifest(agents=(agent_spec(), agent_spec()))
        with self.assertRaisesRegex(ContractError, "repeat_count: must be >= 1"):
            manifest(repeat_count=0)
        with self.assertRaisesRegex(ContractError, "repeat_count: expected integer"):
            manifest(repeat_count=True)

    def test_rejects_noncanonical_timestamp(self) -> None:
        for value in ("2026-07-17", "2026-07-17T10:00:00+00:00", "yesterday"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ContractError, "created_at: expected UTC RFC3339 seconds"):
                    manifest(created_at=value)

    def test_builder_rejects_wrong_contract_types_without_raw_attribute_errors(self) -> None:
        with self.assertRaisesRegex(ContractError, r"tasks\[0\]: expected FullTaskSpec"):
            manifest(tasks=(object(),))
        with self.assertRaisesRegex(ContractError, r"agents\[0\]: expected AgentSpec"):
            manifest(agents=(object(),))

    def test_from_dict_rejects_tampered_derived_fields(self) -> None:
        encoded = manifest().to_dict()
        encoded["comparability_key"] = SHA_B
        with self.assertRaisesRegex(ContractError, "comparability_key: does not match manifest content"):
            RunManifest.from_dict(encoded)

        encoded = manifest().to_dict()
        encoded["expected_attempts"][0]["attempt_id"] = "attempt:v1:" + "0" * 64
        with self.assertRaisesRegex(ContractError, "expected_attempts: does not match frozen matrix"):
            RunManifest.from_dict(encoded)

    def test_expected_attempt_rejects_zero_repeat(self) -> None:
        with self.assertRaisesRegex(ContractError, "repeat: must be >= 1"):
            ExpectedAttempt(
                attempt_id="attempt:v1:" + "0" * 64,
                task=identity("task", "task-a", SHA_A),
                task_view=identity("task_view", "task-a:view", SHA_A),
                agent=identity("agent", "agent-a", SHA_B),
                repeat=0,
                effective_config_hash=SHA_C,
            )


if __name__ == "__main__":
    unittest.main()
