from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from typing import Any

from op_bench.registry import EnvironmentAsset, EnvironmentRegistry, SourceAsset, SourceRegistry


ImageInspector = Callable[[str], dict[str, object]]


class AssetManager:
    def __init__(
        self,
        environment_registry: EnvironmentRegistry | None = None,
        source_registry: SourceRegistry | None = None,
        image_inspector: ImageInspector | None = None,
    ) -> None:
        self.environment_registry = environment_registry
        self.source_registry = source_registry
        self.image_inspector = image_inspector or inspect_docker_image

    def inspect(self, check_docker: bool = False) -> dict[str, object]:
        environments = [
            self._inspect_environment(asset, check_docker)
            for asset in self._environment_assets()
        ]
        sources = [self._inspect_source(asset) for asset in self._source_assets()]
        return {
            "environment_registry": str(self.environment_registry.path) if self.environment_registry else None,
            "source_registry": str(self.source_registry.path) if self.source_registry else None,
            "environments": environments,
            "sources": sources,
            "summary": {
                "environments": self._summary(environments),
                "sources": self._summary(sources),
            },
        }

    def _environment_assets(self) -> list[EnvironmentAsset]:
        if self.environment_registry is None:
            return []
        return list(self.environment_registry.assets.values())

    def _source_assets(self) -> list[SourceAsset]:
        if self.source_registry is None:
            return []
        return list(self.source_registry.assets.values())

    def _inspect_environment(self, asset: EnvironmentAsset, check_docker: bool) -> dict[str, object]:
        record: dict[str, object] = {
            "id": asset.asset_id,
            "framework": asset.framework,
            "runtime_tier": asset.runtime_tier,
            "image": asset.image,
            "expected_digest": asset.image_digest,
            "status": "declared",
        }
        if not check_docker:
            return record
        inspection = self.image_inspector(asset.image)
        record["image_id"] = inspection.get("image_id")
        record["error"] = inspection.get("error")
        if inspection.get("available") is not True:
            record["status"] = "missing"
        elif asset.image_digest and inspection.get("image_id") != asset.image_digest:
            record["status"] = "digest_mismatch"
        else:
            record["status"] = "ready"
        return record

    def _inspect_source(self, asset: SourceAsset) -> dict[str, object]:
        local_path = asset.local_path
        return {
            "id": asset.asset_id,
            "repo_url": asset.repo_url,
            "commit": asset.commit,
            "local_path": str(local_path) if local_path else None,
            "submodule_policy": asset.submodule_policy,
            "submodule_status": asset.submodule_status,
            "source_loading_modes": asset.source_loading_modes,
            "related_tasks": asset.related_tasks,
            "status": "ready" if local_path is not None and local_path.exists() else "missing",
        }

    def _summary(self, records: list[dict[str, object]]) -> dict[str, int]:
        return {
            "total": len(records),
            "ready": sum(1 for record in records if record["status"] == "ready"),
            "unavailable": sum(1 for record in records if record["status"] in {"missing", "digest_mismatch"}),
        }


def inspect_docker_image(image: str) -> dict[str, object]:
    if shutil.which("docker") is None:
        return {"available": False, "image_id": None, "error": "docker command not found"}
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=False,
        )
        return {
            "available": completed.returncode == 0,
            "image_id": completed.stdout.strip() or None,
            "error": completed.stderr.strip() or None,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "image_id": None, "error": str(exc)}
