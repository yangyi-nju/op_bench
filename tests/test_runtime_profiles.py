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
    "remote-cpu-compile-pytorch-2.6-py311-v1",
    "remote-cpu-pytorch-2.6-py311-v1",
    "remote-cuda-kernel-pytorch-2.6-cu124-v1",
    "remote-cuda-overlay-pytorch-2.6-cu124-v1",
)


class RuntimeProfileRegistryTests(unittest.TestCase):
    def test_registry_module_and_checked_in_artifacts_exist(self) -> None:
        self.assertIsNotNone(importlib.util.find_spec("op_bench.runtime.profiles"))
        self.assertTrue(REGISTRY_PATH.is_file())
        self.assertTrue(REGISTRY_SCHEMA_PATH.is_file())

    def test_loads_five_sorted_complete_profiles_deterministically(self) -> None:
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
        self.assertEqual(len({item.content_hash for item in first.profiles}), 5)
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
