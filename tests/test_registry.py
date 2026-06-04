from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.registry import EnvironmentRegistry, RegistryError, SourceRegistry


class RegistryTests(unittest.TestCase):
    def test_environment_registry_loads_asset_and_resolves_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "environments": [
                            {
                                "id": "pytorch-cpu",
                                "framework": "pytorch",
                                "runtime_tier": "cpu_python_overlay",
                                "docker": {
                                    "image": "op-bench/pytorch-cpu:test",
                                    "dockerfile": "pytorch-cpu/Dockerfile",
                                    "build_context": "pytorch-cpu",
                                },
                                "preflight": {"workdir": "/tmp", "commands": ["python --version"]},
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            asset = EnvironmentRegistry.load(registry_path).get("pytorch-cpu")

            self.assertEqual(asset.asset_id, "pytorch-cpu")
            self.assertEqual(asset.runtime_tier, "cpu_python_overlay")
            self.assertEqual(asset.image, "op-bench/pytorch-cpu:test")
            self.assertEqual(asset.dockerfile_path, root / "pytorch-cpu/Dockerfile")
            self.assertEqual(asset.build_context_path, root / "pytorch-cpu")
            self.assertEqual(asset.preflight_commands, ["python --version"])

    def test_source_registry_loads_snapshot_and_resolves_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            registry_path = root / "registry.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "sources": [
                            {
                                "id": "pytorch-base",
                                "repo_url": "https://github.com/pytorch/pytorch.git",
                                "commit": "abc1234",
                                "local_path": "pytorch/abc1234/source",
                                "submodules": {"policy": "none_required", "status": "not_initialized"},
                                "source_loading_modes": ["python_overlay"],
                                "related_tasks": ["pytorch__1"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            asset = SourceRegistry.load(registry_path).get("pytorch-base")

            self.assertEqual(asset.asset_id, "pytorch-base")
            self.assertEqual(asset.commit, "abc1234")
            self.assertEqual(asset.local_path, root / "pytorch/abc1234/source")
            self.assertEqual(asset.submodule_policy, "none_required")
            self.assertEqual(asset.related_tasks, ["pytorch__1"])

    def test_registry_rejects_duplicate_asset_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            asset = {
                "id": "duplicate",
                "framework": "pytorch",
                "runtime_tier": "cpu_python_overlay",
                "docker": {"image": "image"},
                "preflight": {"workdir": "/tmp", "commands": ["python --version"]},
            }
            path.write_text(json.dumps({"version": "v1", "environments": [asset, asset]}), encoding="utf-8")

            with self.assertRaisesRegex(RegistryError, "duplicate asset id: duplicate"):
                EnvironmentRegistry.load(path)

    def test_registry_reports_unknown_asset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps({"version": "v1", "sources": []}), encoding="utf-8")

            with self.assertRaisesRegex(RegistryError, "unknown source asset: missing"):
                SourceRegistry.load(path).get("missing")


if __name__ == "__main__":
    unittest.main()
