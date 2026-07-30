from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from op_bench.task import TaskManifest, is_opaque_host_alias


class RegistryError(ValueError):
    """Raised when a registry is malformed or cannot resolve an asset."""


def _resolve_path(registry_dir: Path, value: object | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (registry_dir / path).resolve()


@dataclass(frozen=True)
class EnvironmentAsset:
    registry_dir: Path
    data: dict[str, Any]

    @property
    def asset_id(self) -> str:
        return str(self.data["id"])

    @property
    def framework(self) -> str:
        return str(self.data["framework"])

    @property
    def runtime_tier(self) -> str:
        return str(self.data["runtime_tier"])

    @property
    def image(self) -> str:
        return str(self.data["docker"]["image"])

    @property
    def image_digest(self) -> str | None:
        value = self.data["docker"].get("digest")
        return str(value) if value else None

    @property
    def dockerfile_path(self) -> Path | None:
        return _resolve_path(self.registry_dir, self.data["docker"].get("dockerfile"))

    @property
    def build_context_path(self) -> Path | None:
        return _resolve_path(self.registry_dir, self.data["docker"].get("build_context"))

    @property
    def preflight_workdir(self) -> str:
        return str(self.data["preflight"].get("workdir", "/tmp"))

    @property
    def preflight_commands(self) -> list[str]:
        return [str(command) for command in self.data["preflight"].get("commands", [])]

    @property
    def source_loading_modes(self) -> list[str]:
        return [str(mode) for mode in self.data.get("source_loading_modes", [])]

    @property
    def runtime_artifact(self) -> dict[str, object]:
        value = self.data.get("runtime_artifact")
        return deepcopy(value) if isinstance(value, dict) else {}

    def task_environment_defaults(self) -> dict[str, Any]:
        docker = self.data["docker"]
        runtime = self.data.get("runtime", {})
        defaults: dict[str, Any] = {
            "backend": self.data.get("backend", "docker"),
            "tier": self.runtime_tier,
            "image": self.image,
            "preflight_workdir": self.preflight_workdir,
            "preflight_commands": self.preflight_commands,
        }
        if isinstance(runtime, dict):
            defaults.update(deepcopy(runtime))
        for source, target in (
            ("digest", "image_digest"),
            ("digest_kind", "digest_kind"),
            ("platform", "platform"),
        ):
            if docker.get(source):
                defaults[target] = docker[source]
        if self.dockerfile_path:
            defaults["dockerfile"] = str(self.dockerfile_path)
        if self.build_context_path:
            defaults["build_context"] = str(self.build_context_path)
        for field in (
            "python_version", "os", "build_mode", "hardware",
            "dependencies", "resource_requirements", "gpus", "host",
            "remote_execution_config_hash",
        ):
            if field in self.data:
                defaults[field] = deepcopy(self.data[field])
        return defaults


@dataclass(frozen=True)
class SourceAsset:
    registry_dir: Path
    data: dict[str, Any]

    @property
    def asset_id(self) -> str:
        return str(self.data["id"])

    @property
    def repo_url(self) -> str:
        return str(self.data["repo_url"])

    @property
    def commit(self) -> str:
        return str(self.data["commit"])

    @property
    def local_path(self) -> Path | None:
        return _resolve_path(self.registry_dir, self.data.get("local_path"))

    @property
    def submodule_policy(self) -> str:
        return str(self.data["submodules"]["policy"])

    @property
    def submodule_status(self) -> str:
        return str(self.data["submodules"]["status"])

    @property
    def source_loading_modes(self) -> list[str]:
        return [str(mode) for mode in self.data.get("source_loading_modes", [])]

    @property
    def related_tasks(self) -> list[str]:
        return [str(task_id) for task_id in self.data.get("related_tasks", [])]

    def task_source_defaults(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {
            "repo_url": self.repo_url,
            "base_commit": self.commit,
        }
        if self.local_path:
            defaults["snapshot_path"] = str(self.local_path)
        if self.data.get("checksum"):
            defaults["snapshot_hash"] = str(self.data["checksum"])
        return defaults


Asset = TypeVar("Asset", EnvironmentAsset, SourceAsset)


class _Registry(Generic[Asset]):
    collection_name: str
    asset_name: str
    asset_type: type[Asset]

    def __init__(self, path: Path, version: str, assets: dict[str, Asset]) -> None:
        self.path = path
        self.version = version
        self.assets = assets

    @classmethod
    def load(cls, path: Path | str) -> "_Registry[Asset]":
        registry_path = Path(path).resolve()
        try:
            with registry_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryError(f"cannot load registry {registry_path}: {exc}") from exc

        collection = data.get(cls.collection_name)
        if not isinstance(collection, list):
            raise RegistryError(f"registry field {cls.collection_name!r} must be a list")

        assets: dict[str, Asset] = {}
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                raise RegistryError(f"{cls.collection_name}[{index}] must be an object")
            asset_id = str(item.get("id", ""))
            if not asset_id:
                raise RegistryError(f"{cls.collection_name}[{index}].id is required")
            if asset_id in assets:
                raise RegistryError(f"duplicate asset id: {asset_id}")
            cls._validate_item(item, asset_id)
            assets[asset_id] = cls.asset_type(registry_path.parent, item)
        return cls(registry_path, str(data.get("version", "")), assets)

    @classmethod
    def _validate_item(cls, item: dict[str, Any], asset_id: str) -> None:
        raise NotImplementedError

    def get(self, asset_id: str) -> Asset:
        try:
            return self.assets[asset_id]
        except KeyError as exc:
            raise RegistryError(f"unknown {self.asset_name} asset: {asset_id}") from exc


class EnvironmentRegistry(_Registry[EnvironmentAsset]):
    collection_name = "environments"
    asset_name = "environment"
    asset_type = EnvironmentAsset

    @classmethod
    def _validate_item(cls, item: dict[str, Any], asset_id: str) -> None:
        for field in ("framework", "runtime_tier", "docker", "preflight"):
            if not item.get(field):
                raise RegistryError(f"environment asset {asset_id}: {field} is required")
        if not isinstance(item["docker"], dict) or not item["docker"].get("image"):
            raise RegistryError(f"environment asset {asset_id}: docker.image is required")
        if not isinstance(item["preflight"], dict):
            raise RegistryError(f"environment asset {asset_id}: preflight must be an object")
        host = item.get("host")
        if host is not None and not is_opaque_host_alias(host):
            raise RegistryError(
                f"environment asset {asset_id}: host alias must be an "
                "opaque lowercase identifier"
            )
        if item.get("backend") == "remote_docker":
            remote_hash = item.get("remote_execution_config_hash")
            if (
                not isinstance(remote_hash, str)
                or re.fullmatch(r"sha256:[0-9a-f]{64}", remote_hash)
                is None
            ):
                raise RegistryError(
                    f"environment asset {asset_id}: "
                    "remote_execution_config_hash is required"
                )
        runtime_artifact = item.get("runtime_artifact")
        if runtime_artifact is not None:
            if not isinstance(runtime_artifact, dict):
                raise RegistryError(
                    f"environment asset {asset_id}: runtime_artifact must be an object"
                )
            for field in (
                "strategy",
                "artifact_kind",
                "artifact_id",
                "artifact_digest",
                "artifact_digest_kind",
                "torch_version",
                "python_abi",
            ):
                if not runtime_artifact.get(field):
                    raise RegistryError(
                        f"environment asset {asset_id}: "
                        f"runtime_artifact.{field} is required"
                    )
            companions = runtime_artifact.get("companion_artifacts", [])
            if not isinstance(companions, list):
                raise RegistryError(
                    f"environment asset {asset_id}: "
                    "runtime_artifact.companion_artifacts must be a list"
                )
            for index, companion in enumerate(companions):
                if not isinstance(companion, dict):
                    raise RegistryError(
                        f"environment asset {asset_id}: "
                        f"runtime_artifact.companion_artifacts[{index}] "
                        "must be an object"
                    )
                for field in (
                    "artifact_kind",
                    "artifact_id",
                    "artifact_digest",
                    "artifact_digest_kind",
                ):
                    if not companion.get(field):
                        raise RegistryError(
                            f"environment asset {asset_id}: "
                            "runtime_artifact.companion_artifacts"
                            f"[{index}].{field} is required"
                        )


class SourceRegistry(_Registry[SourceAsset]):
    collection_name = "sources"
    asset_name = "source"
    asset_type = SourceAsset

    @classmethod
    def _validate_item(cls, item: dict[str, Any], asset_id: str) -> None:
        for field in ("repo_url", "commit", "submodules"):
            if not item.get(field):
                raise RegistryError(f"source asset {asset_id}: {field} is required")
        submodules = item["submodules"]
        if not isinstance(submodules, dict) or not submodules.get("policy") or not submodules.get("status"):
            raise RegistryError(f"source asset {asset_id}: submodules.policy and submodules.status are required")


def resolve_task_assets(
    task: TaskManifest,
    environment_registry: EnvironmentRegistry | None = None,
    source_registry: SourceRegistry | None = None,
) -> TaskManifest:
    data = deepcopy(task.data)
    if task.environment_ref:
        if environment_registry is None:
            raise RegistryError(f"task {task.task_id}: environment_ref requires an environment registry")
        environment_asset = environment_registry.get(task.environment_ref)
        task_environment = data.get("environment", {})
        environment_defaults = environment_asset.task_environment_defaults()
        asset_backend = str(environment_defaults["backend"])
        configured_backend = (
            task_environment.get("backend")
            if isinstance(task_environment, dict)
            else None
        )
        if (
            configured_backend is not None
            and configured_backend != asset_backend
        ):
            raise RegistryError(
                f"task {task.task_id}: cannot override registry backend "
                f"{asset_backend} with {configured_backend}"
            )
        configured_remote_hash = (
            task_environment.get("remote_execution_config_hash")
            if isinstance(task_environment, dict)
            else None
        )
        asset_remote_hash = environment_asset.data.get(
            "remote_execution_config_hash"
        )
        if (
            configured_remote_hash is not None
            and configured_remote_hash != asset_remote_hash
        ):
            raise RegistryError(
                f"task {task.task_id}: cannot override "
                "remote_execution_config_hash"
            )
        configured_host = (
            task_environment.get("host")
            if isinstance(task_environment, dict)
            else None
        )
        asset_host = environment_defaults.get("host")
        if configured_host is not None and configured_host != asset_host:
            raise RegistryError(
                f"task {task.task_id}: cannot override registry host alias"
            )
        data["environment"] = _deep_merge(
            environment_defaults,
            data.get("environment", {}),
        )
        resolved_host = data["environment"].get("host")
        if resolved_host is not None and not is_opaque_host_alias(resolved_host):
            raise RegistryError(
                f"task {task.task_id}: resolved host alias must be an "
                "opaque lowercase identifier"
            )
        data.setdefault("runtime_tier", environment_asset.runtime_tier)
    if task.source_ref:
        if source_registry is None:
            raise RegistryError(f"task {task.task_id}: source_ref requires a source registry")
        source_asset = source_registry.get(task.source_ref)
        if task.base_commit != source_asset.commit:
            raise RegistryError(
                f"task {task.task_id}: base_commit {task.base_commit} "
                f"does not match source asset commit {source_asset.commit}"
            )
        data["source"] = _deep_merge(source_asset.task_source_defaults(), data.get("source", {}))
    return TaskManifest(task_dir=task.task_dir, data=data)


def load_resolved_task(
    task_path: Path | str,
    *,
    environment_registry_path: Path | str | None = None,
    source_registry_path: Path | str | None = None,
) -> TaskManifest:
    task = TaskManifest.load(task_path)
    environment_registry = None
    source_registry = None
    if task.environment_ref:
        if environment_registry_path is None:
            raise RegistryError(f"task {task.task_id}: environment_ref requires an environment registry path")
        environment_registry = EnvironmentRegistry.load(environment_registry_path)
    if task.source_ref:
        if source_registry_path is None:
            raise RegistryError(f"task {task.task_id}: source_ref requires a source registry path")
        source_registry = SourceRegistry.load(source_registry_path)
    return resolve_task_assets(task, environment_registry=environment_registry, source_registry=source_registry)


def _deep_merge(defaults: dict[str, Any], overrides: object) -> dict[str, Any]:
    result = deepcopy(defaults)
    if not isinstance(overrides, dict):
        return result
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
