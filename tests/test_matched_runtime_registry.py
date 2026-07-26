from __future__ import annotations

import json
from pathlib import Path
import re
import unittest

from op_bench.registry import EnvironmentRegistry, load_resolved_task
from scripts.validate_task import validate_manifest


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENTS = ROOT / "environments" / "registry.json"
SOURCES = ROOT / "sources" / "registry.json"
TASKS = {
    "pytorch__129154__exp_decomp_numerics": {
        "task": ROOT / "tasks/pytorch/129154_exp_decomp_numerics/task.json",
        "environment_id": "pytorch-matched-ff89ebc-torch2.4.0-py311-cu124",
        "torch_version": "2.4.0+cu124",
        "artifact_kind": "official_wheel",
        "target_module": "torch/_refs/__init__.py",
        "target_import": "torch._refs",
        "selector_module": "test/test_decomp.py",
        "image": "op-bench/pytorch-matched-ff89ebc:torch2.4.0-cu124-py311",
        "base_image": "op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
    },
    "pytorch__144073__vector_norm_scalar_overflow": {
        "task": ROOT / "tasks/pytorch/144073_vector_norm_scalar_overflow/task.json",
        "environment_id": "pytorch-matched-06e9dea-torch2.7.0-py311-cpu",
        "torch_version": "2.7.0+cpu",
        "artifact_kind": "official_wheel",
        "target_module": "torch/_refs/linalg/__init__.py",
        "target_import": "torch._refs.linalg",
        "selector_module": "test/inductor/test_torchinductor.py",
        "image": "op-bench/pytorch-matched-06e9dea:torch2.7.0-cpu-py311",
        "base_image": "op-bench/pytorch-cpu-compile:torch2.6.0-py311",
    },
}


class MatchedRuntimeRegistryTests(unittest.TestCase):
    def test_registry_binds_pinned_wheel_and_observed_image_identity(self) -> None:
        raw = json.loads(ENVIRONMENTS.read_text(encoding="utf-8"))
        entries = {entry["id"]: entry for entry in raw["environments"]}

        for expected in TASKS.values():
            with self.subTest(environment=expected["environment_id"]):
                entry = entries[expected["environment_id"]]
                artifact = entry["runtime_artifact"]
                self.assertEqual(entry["docker"]["image"], expected["image"])
                self.assertRegex(
                    entry["docker"]["digest"],
                    r"^sha256:[0-9a-f]{64}$",
                )
                self.assertEqual(
                    entry["docker"]["digest_kind"],
                    "local_image_id",
                )
                self.assertEqual(artifact["strategy"], "matched_wheel")
                self.assertEqual(
                    artifact["artifact_kind"],
                    expected["artifact_kind"],
                )
                self.assertEqual(
                    artifact["torch_version"],
                    expected["torch_version"],
                )
                self.assertRegex(
                    artifact["artifact_digest"],
                    r"^sha256:[0-9a-f]{64}$",
                )
                self.assertEqual(
                    artifact["artifact_digest_kind"],
                    "wheel_sha256",
                )
                self.assertEqual(artifact["python_abi"], "cp311-cp311")
                self.assertEqual(
                    entry["source_loading_modes"],
                    ["python_overlay"],
                )

    def test_registry_asset_exposes_runtime_artifact_metadata(self) -> None:
        registry = EnvironmentRegistry.load(ENVIRONMENTS)

        for expected in TASKS.values():
            with self.subTest(environment=expected["environment_id"]):
                asset = registry.get(expected["environment_id"])
                self.assertEqual(
                    asset.runtime_artifact["torch_version"],
                    expected["torch_version"],
                )
                self.assertTrue(
                    asset.runtime_artifact["artifact_digest"].startswith(
                        "sha256:"
                    )
                )

    def test_tasks_resolve_to_matched_images_and_probe_configuration(self) -> None:
        for task_id, expected in TASKS.items():
            with self.subTest(task_id=task_id):
                task = load_resolved_task(
                    expected["task"],
                    environment_registry_path=ENVIRONMENTS,
                    source_registry_path=SOURCES,
                )
                compatibility = task.data["compatibility"]
                self.assertEqual(task.task_id, task_id)
                self.assertEqual(
                    task.environment_ref,
                    expected["environment_id"],
                )
                self.assertEqual(task.environment_image, expected["image"])
                self.assertRegex(
                    task.environment_image_digest,
                    r"^sha256:[0-9a-f]{64}$",
                )
                self.assertEqual(
                    compatibility["target_module"],
                    expected["target_module"],
                )
                self.assertEqual(
                    compatibility["target_import"],
                    expected["target_import"],
                )
                self.assertEqual(
                    compatibility["selector_module"],
                    expected["selector_module"],
                )
                self.assertTrue(
                    compatibility["minimal_operation"].startswith(
                        "import torch;"
                    )
                )
                self.assertEqual(
                    compatibility["evidence"],
                    "compatibility/evidence.json",
                )
                self.assertEqual(validate_manifest(task.data), [])

    def test_dockerfiles_pin_the_registry_wheel_digest(self) -> None:
        raw = json.loads(ENVIRONMENTS.read_text(encoding="utf-8"))
        entries = {entry["id"]: entry for entry in raw["environments"]}

        for expected in TASKS.values():
            with self.subTest(environment=expected["environment_id"]):
                entry = entries[expected["environment_id"]]
                dockerfile = (
                    ENVIRONMENTS.parent / entry["docker"]["dockerfile"]
                ).read_text(encoding="utf-8")
                digest = entry["runtime_artifact"][
                    "artifact_digest"
                ].removeprefix("sha256:")
                self.assertIn(digest, dockerfile)
                self.assertIn("sha256sum --check --strict", dockerfile)
                self.assertIn("python -m pip install", dockerfile)
                self.assertIn(
                    f"FROM {expected['base_image']}",
                    dockerfile,
                )
                self.assertNotRegex(
                    dockerfile,
                    re.compile(r"torch(?:==|/)(?:latest|nightly)"),
                )


if __name__ == "__main__":
    unittest.main()
