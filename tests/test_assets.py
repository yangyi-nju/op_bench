from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.assets import AssetManager
from op_bench.registry import EnvironmentRegistry, SourceRegistry


class AssetManagerTests(unittest.TestCase):
    def test_inventory_reports_ready_assets_and_deduplicated_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            snapshot = root / "snapshot"
            snapshot.mkdir()
            environment_registry, source_registry = self._registries(root, snapshot)
            manager = AssetManager(
                environment_registry,
                source_registry,
                image_inspector=lambda image: {
                    "available": True,
                    "image_id": "sha256:image",
                    "error": None,
                },
            )

            report = manager.inspect(check_docker=True)

            self.assertEqual(report["summary"], {
                "environments": {"total": 1, "ready": 1, "unavailable": 0},
                "sources": {"total": 1, "ready": 1, "unavailable": 0},
            })
            self.assertEqual(report["environments"][0]["status"], "ready")
            self.assertEqual(report["sources"][0]["status"], "ready")
            self.assertEqual(report["sources"][0]["related_tasks"], ["fixture", "fixture-2"])

    def test_inventory_classifies_missing_source_and_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            environment_registry, source_registry = self._registries(root, root / "missing")
            manager = AssetManager(
                environment_registry,
                source_registry,
                image_inspector=lambda image: {
                    "available": True,
                    "image_id": "sha256:different",
                    "error": None,
                },
            )

            report = manager.inspect(check_docker=True)

            self.assertEqual(report["environments"][0]["status"], "digest_mismatch")
            self.assertEqual(report["sources"][0]["status"], "missing")
            self.assertEqual(report["summary"]["environments"]["unavailable"], 1)
            self.assertEqual(report["summary"]["sources"]["unavailable"], 1)

    def test_inventory_without_docker_check_marks_environment_declared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            snapshot = root / "snapshot"
            snapshot.mkdir()
            environment_registry, source_registry = self._registries(root, snapshot)

            report = AssetManager(environment_registry, source_registry).inspect(check_docker=False)

            self.assertEqual(report["environments"][0]["status"], "declared")
            self.assertEqual(report["summary"]["environments"]["ready"], 0)
            self.assertEqual(report["summary"]["environments"]["unavailable"], 0)

    def _registries(
        self,
        root: Path,
        snapshot: Path,
    ) -> tuple[EnvironmentRegistry, SourceRegistry]:
        environment_path = root / "environments.json"
        environment_path.write_text(
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
                                "digest": "sha256:image",
                            },
                            "preflight": {"workdir": "/tmp", "commands": ["python --version"]},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        source_path = root / "sources.json"
        source_path.write_text(
            json.dumps(
                {
                    "version": "v1",
                    "sources": [
                        {
                            "id": "pytorch-base",
                            "repo_url": "https://github.com/pytorch/pytorch.git",
                            "commit": "abc1234",
                            "local_path": str(snapshot),
                            "submodules": {"policy": "none_required", "status": "not_initialized"},
                            "source_loading_modes": ["python_overlay"],
                            "related_tasks": ["fixture", "fixture-2"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return EnvironmentRegistry.load(environment_path), SourceRegistry.load(source_path)


if __name__ == "__main__":
    unittest.main()
