from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import RuntimeProfile
from op_bench.runtime.schema import (
    SchemaValidationError,
    load_runtime_schema,
    validate_schema_instance,
)
from op_bench.runtime.validation import ContractError, require_exact_fields, require_list, require_str


@dataclass(frozen=True)
class RuntimeProfileRegistry:
    version: str
    profiles: tuple[RuntimeProfile, ...]

    def __post_init__(self) -> None:
        version = require_str(self.version, "version")
        if version != "v1":
            raise ContractError("version: expected 'v1'")
        if not isinstance(self.profiles, tuple) or not self.profiles:
            raise ContractError("profiles: expected non-empty tuple")
        identifiers: list[str] = []
        hashes: set[str] = set()
        for index, profile in enumerate(self.profiles):
            if not isinstance(profile, RuntimeProfile):
                raise ContractError(f"profiles[{index}]: expected RuntimeProfile")
            if profile.profile_id in identifiers:
                raise ContractError(
                    f"profiles: duplicate profile_id {profile.profile_id!r}"
                )
            if profile.content_hash in hashes:
                raise ContractError(f"profiles: duplicate profile at index {index}")
            identifiers.append(profile.profile_id)
            hashes.add(profile.content_hash)
        if identifiers != sorted(identifiers):
            raise ContractError("profiles: expected sorted profile_id order")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "profiles": [profile.to_dict() for profile in self.profiles],
        }

    @property
    def content_hash(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def canonical_bytes(self) -> bytes:
        return (canonical_json(self.to_dict()) + "\n").encode("utf-8")

    def get(self, profile_id: str) -> RuntimeProfile:
        requested = require_str(profile_id, "profile_id")
        for profile in self.profiles:
            if profile.profile_id == requested:
                return profile
        raise ContractError(f"profile_id: unknown profile {requested!r}")


def load_runtime_profile_registry(path: Path | str) -> RuntimeProfileRegistry:
    registry_path = Path(path)
    if registry_path.is_symlink():
        raise ContractError("registry_path: symlink is denied")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(registry_path, flags)
    except OSError as exc:
        raise ContractError(f"registry_path: cannot open regular file: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("registry_path: expected regular file")
        with os.fdopen(os.dup(descriptor), "r", encoding="utf-8") as handle:
            encoded = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ContractError(f"registry_path: invalid JSON: {exc}") from exc
    finally:
        os.close(descriptor)

    schema_root = Path(__file__).resolve().parents[3] / "schemas"
    try:
        validate_schema_instance(
            encoded,
            load_runtime_schema(schema_root / "runtime_profile_registry.schema.json"),
        )
    except SchemaValidationError as exc:
        raise ContractError(str(exc)) from exc
    data = require_exact_fields(encoded, "runtime_profile_registry", ("version", "profiles"))
    items = require_list(data["profiles"], "profiles")
    runtime_schema = load_runtime_schema(schema_root / "runtime_contracts.schema.json")
    profiles: list[RuntimeProfile] = []
    for index, item in enumerate(items):
        try:
            validate_schema_instance(item, runtime_schema, definition="runtime_profile")
            profile = RuntimeProfile.from_dict(item, path=f"profiles[{index}]")
        except (ContractError, SchemaValidationError) as exc:
            raise ContractError(f"profiles[{index}]: {exc}") from exc
        profiles.append(profile)
    return RuntimeProfileRegistry(
        version=require_str(data["version"], "version"),
        profiles=tuple(profiles),
    )


__all__ = ["RuntimeProfileRegistry", "load_runtime_profile_registry"]
