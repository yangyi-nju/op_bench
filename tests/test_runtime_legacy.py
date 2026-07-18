from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from op_bench.dataset import DatasetManifest
from op_bench.integrity import replay_spec_hash
from op_bench.runtime.legacy import (
    LegacyV05Defaults,
    full_task_spec_from_v05,
    run_manifest_from_v05_dataset,
    runtime_bundle_from_v05_dataset,
)
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest
from tests.test_runtime_contracts import agent_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "pytorch_v0.5" / "dataset.json"
PROFILE_REGISTRY_PATH = REPO_ROOT / "configs" / "runtime_profiles.v1.json"
PROFILE_BY_ENVIRONMENT = {
    "pytorch-cpu-torch2.6.0-py311": "remote-cpu-pytorch-2.6-py311-v1",
    "pytorch-cpu-compile-torch2.6.0-py311": "remote-cpu-compile-pytorch-2.6-py311-v1",
    "pytorch-cuda-torch2.6.0-py311-cu124": "remote-cuda-overlay-pytorch-2.6-cu124-v1",
    "pytorch-cuda-devel-torch2.6.0-py311-cu124": "remote-cuda-kernel-pytorch-2.6-cu124-v1",
}


class LegacyV05ProjectionTests(unittest.TestCase):
    def test_private_runtime_bindings_preserve_frozen_source_overlay_paths(self) -> None:
        dataset = DatasetManifest.load(DATASET_PATH)
        tasks = dataset.load_tasks(verified_only=True)
        bundle = runtime_bundle_from_v05_dataset(
            DATASET_PATH,
            agents=(agent_spec(),),
            repeat=1,
            created_at="2026-07-18T00:00:00Z",
        )

        bindings = {binding.task_id: binding for binding in bundle.private_tasks}
        for task in tasks:
            with self.subTest(task=task.task_id):
                self.assertEqual(
                    bindings[task.task_id].source_overlay_paths,
                    tuple(task.source_loading_overlay_paths),
                )
        set_submodule = bindings["pytorch__143455__set_submodule"]
        self.assertIn(
            "torch/testing/_internal/common_nn.py",
            set_submodule.source_overlay_paths,
        )

    def test_private_runtime_bindings_use_resolvable_executable_commits(self) -> None:
        bundle = runtime_bundle_from_v05_dataset(
            DATASET_PATH,
            agents=(agent_spec(),),
            repeat=1,
            created_at="2026-07-18T00:00:00Z",
        )

        self.assertEqual(len(bundle.private_tasks), 17)
        for binding in bundle.private_tasks:
            with self.subTest(task=binding.task_id):
                resolved = subprocess.run(
                    (
                        "git",
                        "-C",
                        str(binding.source.repository),
                        "rev-parse",
                        "--verify",
                        "--end-of-options",
                        f"{binding.source.revision}^{{commit}}",
                    ),
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr.decode())

    def test_projects_all_17_verified_tasks_with_explicit_identity_kinds(self) -> None:
        dataset = DatasetManifest.load(DATASET_PATH)
        tasks = dataset.load_tasks(verified_only=True)
        registry = load_runtime_profile_registry(PROFILE_REGISTRY_PATH)
        profiles = {profile.profile_id: profile for profile in registry.profiles}

        specs = tuple(full_task_spec_from_v05(task) for task in tasks)

        self.assertEqual(len(specs), 17)
        self.assertEqual(
            {spec.task.identifier for spec in specs},
            {task.task_id for task in tasks},
        )
        for task, spec in zip(tasks, specs):
            with self.subTest(task=task.task_id):
                self.assertEqual(spec.task.digest, replay_spec_hash(task))
                self.assertEqual(spec.task.digest_kind, "replay_spec_v1")
                self.assertEqual(spec.source.identity_type, "source")
                self.assertIn(spec.source.digest_kind, {"content_sha256", "canonical_config"})
                self.assertEqual(spec.environment.identity_type, "environment")
                self.assertEqual(spec.environment.digest_kind, "canonical_config")
                self.assertEqual(spec.runtime.image.identity_type, "image")
                self.assertIn(
                    spec.runtime.image.digest_kind,
                    {"image_id", "content_sha256", "canonical_config", "declared"},
                )
                self.assertEqual(spec.runtime.backend, "remote_docker")
                expected_profile_id = PROFILE_BY_ENVIRONMENT[task.environment_ref]
                self.assertEqual(spec.runtime, profiles[expected_profile_id])
                self.assertEqual(spec.runtime.hardware.identity_type, "hardware")
                self.assertEqual(spec.runtime.mount_policy.source_access, "remote_sync")
                self.assertEqual(spec.runtime.cleanup_policy.scope, "attempt_owned_only")
                self.assertTrue(spec.fail_to_pass)
                self.assertTrue(spec.pass_to_pass)
                self.assertTrue(spec.patch_scope)

        self.assertTrue(any(spec.runtime.image.digest_kind == "image_id" for spec in specs))
        self.assertTrue(
            any(spec.runtime.image.digest_kind == "declared" for spec in specs)
        )

    def test_builds_a_deterministic_17_by_agent_by_repeat_manifest_offline(self) -> None:
        defaults = LegacyV05Defaults.standard()
        agents = (agent_spec(),)

        with patch.object(subprocess, "run") as run, patch(
            "socket.create_connection"
        ) as connect, patch.dict(os.environ, {"OP_BENCH_FORCE_LOCAL_DOCKER": "1"}):
            first = run_manifest_from_v05_dataset(
                DATASET_PATH,
                agents=agents,
                repeat=2,
                created_at="2026-07-17T10:00:00Z",
                defaults=defaults,
            )

        second = run_manifest_from_v05_dataset(
            DATASET_PATH,
            agents=agents,
            repeat=2,
            created_at="2026-07-17T10:00:00Z",
            defaults=defaults,
        )

        run.assert_not_called()
        connect.assert_not_called()
        self.assertEqual(first, second)
        self.assertEqual(first.dataset.identifier, "pytorch_v0.5")
        self.assertEqual(first.dataset.digest_kind, "canonical_config")
        self.assertEqual(len(first.tasks), 17)
        self.assertEqual(len(first.expected_attempts), 34)
        self.assertEqual(
            [(item.task.identifier, item.repeat) for item in first.expected_attempts],
            sorted(
                [(item.task.identifier, item.repeat) for item in first.expected_attempts]
            ),
        )
        self.assertEqual(first.platform_version, defaults.platform_version)
        self.assertEqual(first.action_protocol, defaults.action_protocol)

    def test_projection_does_not_embed_remote_host_or_absolute_paths(self) -> None:
        task = DatasetManifest.load(DATASET_PATH).load_tasks(verified_only=True)[0]

        encoded = full_task_spec_from_v05(task).to_dict()
        flattened = repr(encoded)

        self.assertNotIn("gpu-a10", flattened)
        self.assertNotIn(str(REPO_ROOT), flattened)

    def test_projection_rejects_artifacts_outside_the_task_root(self) -> None:
        original = DatasetManifest.load(DATASET_PATH).load_tasks(verified_only=True)[0]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_dir = root / "task"
            task_dir.mkdir()
            (root / "outside.patch").write_text("private local bytes", encoding="utf-8")
            data = copy.deepcopy(original.data)
            data["artifacts"]["gold_patch"] = "../outside.patch"
            task = TaskManifest(task_dir=task_dir, data=data)

            with self.assertRaisesRegex(ContractError, "gold_patch: path escapes task root"):
                full_task_spec_from_v05(task)

    def test_direct_projection_rejects_wrong_typed_legacy_fields(self) -> None:
        original = DatasetManifest.load(DATASET_PATH).load_tasks(verified_only=True)[0]
        mutations = (
            (("statement", "title"), 123, "statement.title: expected string"),
            (("evaluation", "timeout_sec"), True, "evaluation.timeout_sec: expected integer"),
            (
                ("environment", "hardware", "requires_gpu"),
                "false",
                "environment.hardware.requires_gpu: expected boolean",
            ),
            (
                ("evaluation", "fail_to_pass"),
                [1],
                r"evaluation.fail_to_pass\[0\]: expected string",
            ),
        )

        for path, replacement, message in mutations:
            with self.subTest(path=".".join(path)):
                data = copy.deepcopy(original.data)
                target = data
                for name in path[:-1]:
                    target = target[name]
                target[path[-1]] = replacement
                task = TaskManifest(task_dir=original.task_dir, data=data)

                with self.assertRaisesRegex(ContractError, message):
                    full_task_spec_from_v05(task)


if __name__ == "__main__":
    unittest.main()
