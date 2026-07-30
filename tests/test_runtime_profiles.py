from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.schema import load_runtime_schema, validate_schema_instance
from op_bench.runtime.validation import ContractError


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "configs" / "runtime_profiles.v1.json"
REGISTRY_SCHEMA_PATH = REPO_ROOT / "schemas" / "runtime_profile_registry.schema.json"
RUNTIME_SCHEMA_PATH = REPO_ROOT / "schemas" / "runtime_contracts.schema.json"
EXPECTED_PROFILE_IDS = (
    "local-cpu-process-v1",
    "remote-cpu-boundary-torch2.2-py311-v1",
    "remote-cpu-boundary-torch2.3-py311-v1",
    "remote-cpu-boundary-torch2.4-py311-v1",
    "remote-cpu-compile-pytorch-2.6-py311-v1",
    "remote-cpu-expansion-nightly-torch2.12.0dev20260407-py311-v1",
    "remote-cpu-matched-torch2.7-py311-v1",
    "remote-cpu-pytorch-2.6-py311-v1",
    "remote-cpu-source-boundary-py311-v1",
    "remote-cuda-boundary-torch2.6-cu124-v1",
    "remote-cuda-expansion-nightly-torch2.12.0dev20260407-cu126-py311-v1",
    "remote-cuda-kernel-pytorch-2.6-cu124-v1",
    "remote-cuda-matched-torch2.4-cu124-py311-v1",
    "remote-cuda-overlay-pytorch-2.6-cu124-v1",
)
BOUNDARY_PROFILE_BY_ENVIRONMENT = {
    "pytorch-boundary-cpu-source-build-py311": (
        "remote-cpu-source-boundary-py311-v1",
        900_000,
    ),
    "pytorch-matched-boundary-torch2.2.0-cpu": (
        "remote-cpu-boundary-torch2.2-py311-v1",
        300_000,
    ),
    "pytorch-matched-boundary-torch2.3.0-cpu": (
        "remote-cpu-boundary-torch2.3-py311-v1",
        300_000,
    ),
    "pytorch-matched-boundary-torch2.4.0-cpu": (
        "remote-cpu-boundary-torch2.4-py311-v1",
        900_000,
    ),
    "pytorch-matched-boundary-torch2.6.0-cu124": (
        "remote-cuda-boundary-torch2.6-cu124-v1",
        300_000,
    ),
}


class RuntimeProfileRegistryTests(unittest.TestCase):
    def test_registry_module_and_checked_in_artifacts_exist(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("op_bench.runtime.profiles"))
        self.assertTrue(REGISTRY_PATH.is_file())
        self.assertTrue(REGISTRY_SCHEMA_PATH.is_file())

    def test_loads_sorted_complete_profiles_deterministically(self) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")

        first = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)
        second = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)

        self.assertEqual(first, second)
        self.assertEqual(first.version, "v1")
        self.assertEqual(tuple(item.profile_id for item in first.profiles), EXPECTED_PROFILE_IDS)
        self.assertEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            first.canonical_bytes,
            (canonical_json(first.to_dict()) + "\n").encode("utf-8"),
        )
        self.assertEqual(len({item.content_hash for item in first.profiles}), 14)
        for profile in first.profiles:
            with self.subTest(profile=profile.profile_id):
                self.assertEqual(profile.hardware.identity_type, "hardware")
                self.assertEqual(profile.mount_policy.artifact_access, "controller_only")
                self.assertEqual(profile.cleanup_policy.scope, "attempt_owned_only")
                self.assertEqual(profile.network_policy, "denied")

    def test_registry_and_every_profile_validate_against_checked_in_schemas(self) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")
        registry = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)

        validate_schema_instance(
            registry.to_dict(),
            load_runtime_schema(REGISTRY_SCHEMA_PATH),
        )
        runtime_schema = load_runtime_schema(RUNTIME_SCHEMA_PATH)
        for profile in registry.profiles:
            validate_schema_instance(
                profile.to_dict(),
                runtime_schema,
                definition="runtime_profile",
            )

    def test_public_registry_contains_no_private_target_or_host_path(self) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")
        registry = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)
        flattened = canonical_json(registry.to_dict())

        for forbidden in (
            "gpu-a10",
            "hostname",
            "identity_file",
            "remote_user",
            "/Users/",
            "/home/",
            "~/.ssh",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, flattened)

    def test_gpu_profiles_and_resource_counts_are_consistent(self) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")
        registry = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)

        for profile in registry.profiles:
            with self.subTest(profile=profile.profile_id):
                if profile.runtime_tier.startswith("cuda_"):
                    self.assertTrue(profile.requires_gpu)
                    self.assertEqual(profile.resource_policy.gpu_count, 1)
                else:
                    self.assertFalse(profile.requires_gpu)
                    self.assertEqual(profile.resource_policy.gpu_count, 0)

    def test_boundary_profiles_bind_public_environment_and_task_contracts(
        self,
    ) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")
        registry = profiles_module.load_runtime_profile_registry(REGISTRY_PATH)
        profiles = {profile.profile_id: profile for profile in registry.profiles}
        environments = {
            item["id"]: item
            for item in json.loads(
                (REPO_ROOT / "environments" / "registry.json").read_text(
                    encoding="utf-8"
                )
            )["environments"]
        }

        for environment_id, (
            profile_id,
            timeout_ms,
        ) in BOUNDARY_PROFILE_BY_ENVIRONMENT.items():
            with self.subTest(environment=environment_id):
                environment = environments[environment_id]
                profile = profiles[profile_id]
                self.assertEqual(profile.backend, environment["backend"])
                self.assertEqual(profile.runtime_tier, environment["runtime_tier"])
                self.assertIn(
                    profile.source_loading_mode,
                    environment["source_loading_modes"],
                )
                self.assertEqual(
                    profile.image.identifier,
                    environment["docker"]["image"],
                )
                self.assertEqual(
                    profile.image.digest,
                    environment["docker"]["digest"],
                )
                self.assertEqual(profile.image.digest_kind, "image_id")
                self.assertEqual(
                    profile.platform,
                    environment["docker"]["platform"],
                )
                self.assertEqual(
                    profile.requires_gpu,
                    environment["hardware"]["requires_gpu"],
                )
                self.assertEqual(
                    profile.resource_policy.gpu_count,
                    1 if profile.requires_gpu else 0,
                )
                self.assertGreaterEqual(
                    profile.resource_policy.memory_bytes,
                    environment["hardware"]["min_memory_gb"] * 1024**3,
                )
                self.assertEqual(profile.timeout_ms, timeout_ms)

        matched_tasks = 0
        for task_path in sorted(
            (REPO_ROOT / "tasks" / "pytorch").glob("*/task.json")
        ):
            task = json.loads(task_path.read_text(encoding="utf-8"))
            environment_id = task.get("environment_ref")
            if environment_id not in BOUNDARY_PROFILE_BY_ENVIRONMENT:
                continue
            matched_tasks += 1
            profile_id, timeout_ms = BOUNDARY_PROFILE_BY_ENVIRONMENT[
                environment_id
            ]
            profile = profiles[profile_id]
            with self.subTest(task=task["task_id"]):
                self.assertEqual(profile.runtime_tier, task["runtime_tier"])
                self.assertEqual(
                    profile.requires_gpu,
                    environments[environment_id]["hardware"]["requires_gpu"],
                )
                self.assertEqual(
                    timeout_ms,
                    task["evaluation"]["timeout_sec"] * 1_000,
                )
        self.assertEqual(matched_tasks, 6)

    def test_loader_rejects_unsorted_duplicate_unknown_and_symlinked_input(self) -> None:
        profiles_module = importlib.import_module("op_bench.runtime.profiles")
        encoded = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        mutations = []

        unsorted = copy.deepcopy(encoded)
        unsorted["profiles"] = list(reversed(unsorted["profiles"]))
        mutations.append((unsorted, "profiles: expected sorted profile_id order"))

        duplicate = copy.deepcopy(encoded)
        duplicate["profiles"][1] = copy.deepcopy(duplicate["profiles"][0])
        mutations.append((duplicate, "profiles: duplicate profile_id"))

        unknown = copy.deepcopy(encoded)
        unknown["profiles"][0]["remote_host"] = "private-host"
        mutations.append((unknown, "unknown properties"))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (value, message) in enumerate(mutations):
                path = root / f"mutation-{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(index=index), self.assertRaisesRegex(ContractError, message):
                    profiles_module.load_runtime_profile_registry(path)

            link = root / "registry-link.json"
            link.symlink_to(REGISTRY_PATH)
            with self.assertRaisesRegex(ContractError, "registry_path: symlink is denied"):
                profiles_module.load_runtime_profile_registry(link)


if __name__ == "__main__":
    unittest.main()
