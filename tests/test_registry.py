from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.dataset import DatasetManifest
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

    def test_dataset_load_resolves_registry_assets_and_keeps_task_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            task_dir = root / "tasks/fixture"
            task_dir.mkdir(parents=True)
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "fixture",
                        "environment_ref": "pytorch-cpu",
                        "source_ref": "pytorch-base",
                        "runtime_tier": "cpu_python_overlay",
                        "source": {
                            "repo": "pytorch/pytorch",
                            "base_commit": "abc1234",
                            "snapshot_path": "task-local-snapshot",
                        },
                        "environment": {
                            "workspace_dir": "/task-workspace",
                            "source_loading": {"mode": "python_overlay"},
                        },
                        "artifacts": {"gold_patch": "gold.patch", "test_patch": "test.patch"},
                        "evaluation": {
                            "fail_to_pass": ["fail"],
                            "pass_to_pass": ["pass"],
                            "test_command": "{python} -m unittest {test}",
                            "timeout_sec": 30,
                        },
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )
            (root / "environments.json").write_text(
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
                                    "digest_kind": "local_image_id",
                                    "dockerfile": "Dockerfile",
                                    "build_context": ".",
                                },
                                "runtime": {
                                    "backend": "docker",
                                    "workspace_dir": "/registry-workspace",
                                    "python_executable": "python",
                                },
                                "preflight": {"workdir": "/tmp", "commands": ["python --version"]},
                                "source_loading_modes": ["python_overlay"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (root / "sources.json").write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "sources": [
                            {
                                "id": "pytorch-base",
                                "repo_url": "https://github.com/pytorch/pytorch.git",
                                "commit": "abc1234",
                                "local_path": "registry-snapshot",
                                "checksum": "sha256:source",
                                "submodules": {"policy": "none_required", "status": "not_initialized"},
                                "source_loading_modes": ["python_overlay"],
                                "related_tasks": ["fixture"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "dataset_id": "fixture",
                        "version": "v1",
                        "status": "draft",
                        "registries": {
                            "environments": "environments.json",
                            "sources": "sources.json",
                        },
                        "tasks": [{"task_id": "fixture", "task_path": "tasks/fixture"}],
                    }
                ),
                encoding="utf-8",
            )

            task = DatasetManifest.load(dataset_path).load_tasks()[0]

            self.assertEqual(task.environment_backend, "docker")
            self.assertEqual(task.environment_image, "op-bench/pytorch-cpu:test")
            self.assertEqual(task.environment_image_digest, "sha256:image")
            self.assertEqual(task.environment_workspace_dir, "/task-workspace")
            self.assertEqual(task.environment_preflight_commands, ["python --version"])
            self.assertEqual(task.source_snapshot_path, task_dir / "task-local-snapshot")
            self.assertEqual(task.source_snapshot_hash, "sha256:source")

    def test_asset_resolver_rejects_source_commit_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source_path = root / "sources.json"
            source_path.write_text(
                json.dumps(
                    {
                        "version": "v1",
                        "sources": [
                            {
                                "id": "pytorch-base",
                                "repo_url": "https://github.com/pytorch/pytorch.git",
                                "commit": "different",
                                "local_path": "snapshot",
                                "submodules": {"policy": "none_required", "status": "not_initialized"},
                                "source_loading_modes": ["python_overlay"],
                                "related_tasks": ["fixture"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "fixture",
                        "source_ref": "pytorch-base",
                        "source": {"repo": "pytorch/pytorch", "base_commit": "expected"},
                        "environment": {},
                        "artifacts": {"gold_patch": "gold.patch", "test_patch": "test.patch"},
                        "evaluation": {
                            "fail_to_pass": ["fail"],
                            "pass_to_pass": ["pass"],
                            "test_command": "{python} -m unittest {test}",
                            "timeout_sec": 30,
                        },
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )

            from op_bench.registry import resolve_task_assets
            from op_bench.task import TaskManifest

            with self.assertRaisesRegex(RegistryError, "base_commit expected does not match source asset commit different"):
                resolve_task_assets(
                    TaskManifest.load(task_dir / "task.json"),
                    source_registry=SourceRegistry.load(source_path),
                )


if __name__ == "__main__":
    unittest.main()
