from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from op_bench.factory.contracts import (
    FactoryArtifactReference,
    SHA256_PATTERN,
    _validate_relative_path,
    _validate_utc_seconds,
    factory_content_hash,
)
from op_bench.runtime.canonical import JsonValue, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_exact_fields,
    require_list,
    require_str,
)


RELEASE_INPUT_ROLES = (
    "cumulative_base",
    "precision_base",
    "boundary_freeze",
)
RELEASE_SLICE_ROLES = (
    "cumulative",
    "boundary",
    "precision",
)
RELEASE_ORIGIN_SLICES = {
    "inherited_cumulative": ("cumulative",),
    "inherited_precision": ("cumulative", "precision"),
    "restored_precision": ("cumulative", "precision"),
    "factory_boundary": ("cumulative", "boundary"),
}
RELEASE_FAILURE_CONTRACTS = (
    "wrong-result",
    "exception",
    "crash-oob",
    "silent-acceptance",
    "unclassified",
)


@dataclass(frozen=True)
class DatasetReleaseInput:
    role: str
    dataset: FactoryArtifactReference
    provenance: FactoryArtifactReference | None

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return ("role", "dataset", "provenance")

    def __post_init__(self) -> None:
        require_enum(self.role, "role", RELEASE_INPUT_ROLES)
        if not isinstance(self.dataset, FactoryArtifactReference):
            raise ContractError("dataset: expected FactoryArtifactReference")
        if self.dataset.artifact_type != "dataset_manifest":
            raise ContractError(
                "dataset.artifact_type: expected 'dataset_manifest'"
            )
        if self.role == "boundary_freeze":
            if not isinstance(self.provenance, FactoryArtifactReference):
                raise ContractError(
                    "provenance: boundary_freeze requires Dataset Freeze"
                )
            if self.provenance.artifact_type != "dataset_freeze":
                raise ContractError(
                    "provenance.artifact_type: expected 'dataset_freeze'"
                )
        elif self.provenance is not None:
            raise ContractError(
                "provenance: base Dataset inputs must not declare provenance"
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "dataset": self.dataset.to_dict(),
            "provenance": (
                None if self.provenance is None else self.provenance.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "dataset_release_input",
    ) -> "DatasetReleaseInput":
        data = require_exact_fields(value, path, cls.wire_fields())
        provenance = data["provenance"]
        return cls(
            role=require_str(data["role"], f"{path}.role"),
            dataset=FactoryArtifactReference.from_dict(
                data["dataset"],
                path=f"{path}.dataset",
            ),
            provenance=(
                None
                if provenance is None
                else FactoryArtifactReference.from_dict(
                    provenance,
                    path=f"{path}.provenance",
                )
            ),
        )


@dataclass(frozen=True)
class VerifiedReleaseEntry:
    task: FactoryArtifactReference
    admission_evidence: FactoryArtifactReference
    task_id: str
    task_path: str
    admission_evidence_path: str
    runtime_tier: str
    problem_dimension: str
    problem_subclass: str
    failure_contract: str
    origin: str
    slices: tuple[str, ...]
    admission_state: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "task",
            "admission_evidence",
            "task_id",
            "task_path",
            "admission_evidence_path",
            "runtime_tier",
            "problem_dimension",
            "problem_subclass",
            "failure_contract",
            "origin",
            "slices",
            "admission_state",
        )

    def __post_init__(self) -> None:
        for value, path, expected_type in (
            (self.task, "task", "task_bundle"),
            (
                self.admission_evidence,
                "admission_evidence",
                "admission_evidence",
            ),
        ):
            if not isinstance(value, FactoryArtifactReference):
                raise ContractError(
                    f"{path}: expected FactoryArtifactReference"
                )
            if value.artifact_type != expected_type:
                raise ContractError(
                    f"{path}.artifact_type: expected {expected_type!r}"
                )
        require_str(self.task_id, "task_id")
        expected_task_id = f"task:{self.task_id}"
        if self.task.artifact_id != expected_task_id:
            raise ContractError(
                f"task.artifact_id: expected {expected_task_id!r}"
            )
        _validate_relative_path(self.task_path, "task_path")
        expected_task_path = f"{self.task_path}/task.json"
        if self.task.relative_path != expected_task_path:
            raise ContractError(
                f"task.relative_path: expected {expected_task_path!r}"
            )
        _validate_relative_path(
            self.admission_evidence_path,
            "admission_evidence_path",
        )
        if (
            self.admission_evidence.relative_path
            != self.admission_evidence_path
        ):
            raise ContractError(
                "admission_evidence_path: must match evidence reference"
            )
        require_str(self.runtime_tier, "runtime_tier")
        require_enum(
            self.problem_dimension,
            "problem_dimension",
            ("boundary", "precision", "unclassified"),
        )
        subclasses = {
            "boundary": ("B1", "B2", "B3", "B4", "B5"),
            "precision": ("P1", "P2", "P3", "P4", "P5"),
            "unclassified": ("unclassified",),
        }[self.problem_dimension]
        require_enum(
            self.problem_subclass,
            "problem_subclass",
            subclasses,
        )
        require_enum(
            self.failure_contract,
            "failure_contract",
            RELEASE_FAILURE_CONTRACTS,
        )
        require_enum(
            self.origin,
            "origin",
            RELEASE_ORIGIN_SLICES,
        )
        if not isinstance(self.slices, tuple):
            raise ContractError("slices: expected tuple")
        expected_slices = RELEASE_ORIGIN_SLICES[self.origin]
        if self.slices != expected_slices:
            raise ContractError(
                f"slices: origin {self.origin!r} requires {expected_slices!r}"
            )
        expected_dimension = {
            "inherited_cumulative": "unclassified",
            "inherited_precision": "precision",
            "restored_precision": "precision",
            "factory_boundary": "boundary",
        }[self.origin]
        if self.problem_dimension != expected_dimension:
            raise ContractError(
                f"problem_dimension: origin {self.origin!r} requires "
                f"{expected_dimension!r}"
            )
        if (
            self.origin == "factory_boundary"
            and self.failure_contract == "unclassified"
        ):
            raise ContractError(
                "failure_contract: factory_boundary must be classified"
            )
        if self.admission_state != "verified":
            raise ContractError("admission_state: expected 'verified'")

    def dataset_entry(self) -> dict[str, JsonValue]:
        return {
            "task_id": self.task_id,
            "task_path": self.task_path,
            "admission_status": "verified",
            "admission_evidence": self.admission_evidence_path,
            "environment_status": "ready",
            "runtime_tier": self.runtime_tier,
            "source_status": "ready",
            "replay_status": "verified",
        }

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "task": self.task.to_dict(),
            "admission_evidence": self.admission_evidence.to_dict(),
            "task_id": self.task_id,
            "task_path": self.task_path,
            "admission_evidence_path": self.admission_evidence_path,
            "runtime_tier": self.runtime_tier,
            "problem_dimension": self.problem_dimension,
            "problem_subclass": self.problem_subclass,
            "failure_contract": self.failure_contract,
            "origin": self.origin,
            "slices": list(self.slices),
            "admission_state": self.admission_state,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "verified_release_entry",
    ) -> "VerifiedReleaseEntry":
        data = require_exact_fields(value, path, cls.wire_fields())
        slices = require_list(data["slices"], f"{path}.slices")
        return cls(
            task=FactoryArtifactReference.from_dict(
                data["task"],
                path=f"{path}.task",
            ),
            admission_evidence=FactoryArtifactReference.from_dict(
                data["admission_evidence"],
                path=f"{path}.admission_evidence",
            ),
            task_id=require_str(data["task_id"], f"{path}.task_id"),
            task_path=_validate_relative_path(
                data["task_path"],
                f"{path}.task_path",
            ),
            admission_evidence_path=_validate_relative_path(
                data["admission_evidence_path"],
                f"{path}.admission_evidence_path",
            ),
            runtime_tier=require_str(
                data["runtime_tier"],
                f"{path}.runtime_tier",
            ),
            problem_dimension=require_str(
                data["problem_dimension"],
                f"{path}.problem_dimension",
            ),
            problem_subclass=require_str(
                data["problem_subclass"],
                f"{path}.problem_subclass",
            ),
            failure_contract=require_str(
                data["failure_contract"],
                f"{path}.failure_contract",
            ),
            origin=require_str(data["origin"], f"{path}.origin"),
            slices=tuple(
                require_str(item, f"{path}.slices[{index}]")
                for index, item in enumerate(slices)
            ),
            admission_state=require_str(
                data["admission_state"],
                f"{path}.admission_state",
            ),
        )


@dataclass(frozen=True)
class DatasetReleaseOutput:
    role: str
    dataset_id: str
    dataset_version: str
    task_ids: tuple[str, ...]
    generated_dataset_hash: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "role",
            "dataset_id",
            "dataset_version",
            "task_ids",
            "generated_dataset_hash",
        )

    def __post_init__(self) -> None:
        require_enum(self.role, "role", RELEASE_SLICE_ROLES)
        require_str(self.dataset_id, "dataset_id")
        require_str(self.dataset_version, "dataset_version")
        if not isinstance(self.task_ids, tuple) or not self.task_ids:
            raise ContractError("task_ids: expected non-empty tuple")
        normalized: list[str] = []
        for index, task_id in enumerate(self.task_ids):
            normalized.append(require_str(task_id, f"task_ids[{index}]"))
        if len(normalized) != len(set(normalized)):
            raise ContractError("task_ids: duplicate Task identity")
        require_str(
            self.generated_dataset_hash,
            "generated_dataset_hash",
            pattern=SHA256_PATTERN,
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "task_ids": list(self.task_ids),
            "generated_dataset_hash": self.generated_dataset_hash,
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "dataset_release_output",
    ) -> "DatasetReleaseOutput":
        data = require_exact_fields(value, path, cls.wire_fields())
        task_ids = require_list(data["task_ids"], f"{path}.task_ids")
        return cls(
            role=require_str(data["role"], f"{path}.role"),
            dataset_id=require_str(
                data["dataset_id"],
                f"{path}.dataset_id",
            ),
            dataset_version=require_str(
                data["dataset_version"],
                f"{path}.dataset_version",
            ),
            task_ids=tuple(
                require_str(item, f"{path}.task_ids[{index}]")
                for index, item in enumerate(task_ids)
            ),
            generated_dataset_hash=require_str(
                data["generated_dataset_hash"],
                f"{path}.generated_dataset_hash",
            ),
        )


def _dataset_payload(
    output: DatasetReleaseOutput,
    *,
    registries: Mapping[str, FactoryArtifactReference],
    entries: Sequence[VerifiedReleaseEntry],
) -> dict[str, JsonValue]:
    selected = [
        entry.dataset_entry()
        for entry in entries
        if output.role in entry.slices
    ]
    return {
        "dataset_id": output.dataset_id,
        "version": output.dataset_version,
        "status": "verified",
        "registries": {
            key: value.relative_path
            for key, value in registries.items()
        },
        "tasks": selected,
    }


@dataclass(frozen=True)
class DatasetReleaseManifest:
    contract_type = "dataset_release"
    schema_version = "v1"

    release_id: str
    release_version: str
    inputs: tuple[DatasetReleaseInput, ...]
    registries: Mapping[str, FactoryArtifactReference]
    entries: tuple[VerifiedReleaseEntry, ...]
    outputs: tuple[DatasetReleaseOutput, ...]
    created_at: str

    @classmethod
    def wire_fields(cls) -> tuple[str, ...]:
        return (
            "contract_type",
            "schema_version",
            "release_id",
            "release_version",
            "inputs",
            "registries",
            "entries",
            "outputs",
            "created_at",
            "content_hash",
        )

    @classmethod
    def release_id_for(
        cls,
        *,
        release_version: str,
        inputs: tuple[DatasetReleaseInput, ...],
        registries: Mapping[str, FactoryArtifactReference],
        entries: tuple[VerifiedReleaseEntry, ...],
        outputs: tuple[DatasetReleaseOutput, ...],
    ) -> str:
        digest = canonical_sha256(
            {
                "release_version": release_version,
                "inputs": [item.to_dict() for item in inputs],
                "registries": {
                    key: value.to_dict()
                    for key, value in registries.items()
                },
                "entries": [item.to_dict() for item in entries],
                "outputs": [item.to_dict() for item in outputs],
            }
        )
        return "release:v1:" + digest.removeprefix("sha256:")

    def __post_init__(self) -> None:
        require_str(
            self.release_id,
            "release_id",
            pattern=r"release:v1:[0-9a-f]{64}",
        )
        require_str(self.release_version, "release_version")
        if not isinstance(self.inputs, tuple):
            raise ContractError("inputs: expected tuple")
        if not all(
            isinstance(item, DatasetReleaseInput) for item in self.inputs
        ):
            raise ContractError(
                "inputs: expected DatasetReleaseInput entries"
            )
        input_roles = tuple(item.role for item in self.inputs)
        if input_roles != RELEASE_INPUT_ROLES:
            raise ContractError(
                f"input roles: expected {RELEASE_INPUT_ROLES!r}"
            )
        if not isinstance(self.registries, Mapping):
            raise ContractError("registries: expected object")
        normalized_registries: dict[
            str,
            FactoryArtifactReference,
        ] = {}
        for name, value in self.registries.items():
            if name not in ("environments", "sources"):
                raise ContractError(
                    f"registries: unsupported registry {name!r}"
                )
            if not isinstance(value, FactoryArtifactReference):
                raise ContractError(
                    f"registries.{name}: expected FactoryArtifactReference"
                )
            expected_type = (
                "environment_registry"
                if name == "environments"
                else "source_registry"
            )
            if value.artifact_type != expected_type:
                raise ContractError(
                    f"registries.{name}.artifact_type: "
                    f"expected {expected_type!r}"
                )
            normalized_registries[name] = value
        if tuple(normalized_registries) != tuple(
            sorted(normalized_registries)
        ):
            raise ContractError("registries: expected sorted keys")
        if tuple(normalized_registries) != ("environments", "sources"):
            raise ContractError(
                "registries: environments and sources are required"
            )
        object.__setattr__(
            self,
            "registries",
            MappingProxyType(normalized_registries),
        )
        if not isinstance(self.entries, tuple) or not self.entries:
            raise ContractError("entries: expected non-empty tuple")
        if not all(
            isinstance(item, VerifiedReleaseEntry) for item in self.entries
        ):
            raise ContractError(
                "entries: expected VerifiedReleaseEntry entries"
            )
        task_ids = tuple(item.task_id for item in self.entries)
        if len(task_ids) != len(set(task_ids)):
            raise ContractError("entries: duplicate Task identity")
        if not isinstance(self.outputs, tuple):
            raise ContractError("outputs: expected tuple")
        if not all(
            isinstance(item, DatasetReleaseOutput) for item in self.outputs
        ):
            raise ContractError(
                "outputs: expected DatasetReleaseOutput entries"
            )
        output_roles = tuple(item.role for item in self.outputs)
        if output_roles != RELEASE_SLICE_ROLES:
            raise ContractError(
                f"output roles: expected {RELEASE_SLICE_ROLES!r}"
            )
        output_dataset_ids = tuple(
            item.dataset_id for item in self.outputs
        )
        if len(output_dataset_ids) != len(set(output_dataset_ids)):
            raise ContractError(
                "outputs: duplicate Dataset identity"
            )
        for output in self.outputs:
            selected_ids = tuple(
                entry.task_id
                for entry in self.entries
                if output.role in entry.slices
            )
            if output.task_ids != selected_ids:
                raise ContractError(
                    f"outputs.{output.role}.task_ids: membership drift"
                )
            expected_hash = canonical_sha256(
                _dataset_payload(
                    output,
                    registries=self.registries,
                    entries=self.entries,
                )
            )
            if output.generated_dataset_hash != expected_hash:
                raise ContractError(
                    f"outputs.{output.role}.generated_dataset_hash: "
                    "does not match rebuilt Dataset"
                )
        _validate_utc_seconds(self.created_at, "created_at")
        expected_id = self.release_id_for(
            release_version=self.release_version,
            inputs=self.inputs,
            registries=self.registries,
            entries=self.entries,
            outputs=self.outputs,
        )
        if self.release_id != expected_id:
            raise ContractError(
                f"release_id: expected derived identity {expected_id!r}"
            )

    @property
    def content_hash(self) -> str:
        return factory_content_hash(self.to_dict(include_hash=False))

    def to_dict(
        self,
        *,
        include_hash: bool = True,
    ) -> dict[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "contract_type": self.contract_type,
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "release_version": self.release_version,
            "inputs": [item.to_dict() for item in self.inputs],
            "registries": {
                key: value.to_dict()
                for key, value in self.registries.items()
            },
            "entries": [item.to_dict() for item in self.entries],
            "outputs": [item.to_dict() for item in self.outputs],
            "created_at": self.created_at,
        }
        if include_hash:
            payload["content_hash"] = factory_content_hash(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        path: str = "dataset_release",
    ) -> "DatasetReleaseManifest":
        data = require_exact_fields(value, path, cls.wire_fields())
        if data["contract_type"] != cls.contract_type:
            raise ContractError(
                f"{path}.contract_type: expected {cls.contract_type!r}"
            )
        if data["schema_version"] != cls.schema_version:
            raise ContractError(
                f"{path}.schema_version: expected {cls.schema_version!r}"
            )
        inputs = require_list(data["inputs"], f"{path}.inputs")
        registries_value = data["registries"]
        if not isinstance(registries_value, Mapping):
            raise ContractError(f"{path}.registries: expected object")
        entries = require_list(data["entries"], f"{path}.entries")
        outputs = require_list(data["outputs"], f"{path}.outputs")
        manifest = cls(
            release_id=require_str(
                data["release_id"],
                f"{path}.release_id",
            ),
            release_version=require_str(
                data["release_version"],
                f"{path}.release_version",
            ),
            inputs=tuple(
                DatasetReleaseInput.from_dict(
                    item,
                    path=f"{path}.inputs[{index}]",
                )
                for index, item in enumerate(inputs)
            ),
            registries={
                key: FactoryArtifactReference.from_dict(
                    item,
                    path=f"{path}.registries.{key}",
                )
                for key, item in registries_value.items()
            },
            entries=tuple(
                VerifiedReleaseEntry.from_dict(
                    item,
                    path=f"{path}.entries[{index}]",
                )
                for index, item in enumerate(entries)
            ),
            outputs=tuple(
                DatasetReleaseOutput.from_dict(
                    item,
                    path=f"{path}.outputs[{index}]",
                )
                for index, item in enumerate(outputs)
            ),
            created_at=_validate_utc_seconds(
                data["created_at"],
                f"{path}.created_at",
            ),
        )
        stored_hash = require_str(
            data["content_hash"],
            f"{path}.content_hash",
            pattern=SHA256_PATTERN,
        )
        if stored_hash != manifest.content_hash:
            raise ContractError(
                f"{path}.content_hash: expected {manifest.content_hash!r}"
            )
        return manifest


def build_dataset_release(
    *,
    release_version: str,
    inputs: Sequence[DatasetReleaseInput],
    registries: Mapping[str, FactoryArtifactReference],
    entries: Sequence[VerifiedReleaseEntry],
    dataset_ids: Mapping[str, str],
    created_at: str,
) -> DatasetReleaseManifest:
    selected_inputs = tuple(inputs)
    if not all(
        isinstance(item, DatasetReleaseInput) for item in selected_inputs
    ):
        raise ContractError("inputs: expected DatasetReleaseInput entries")
    indexed_inputs = {item.role: item for item in selected_inputs}
    if (
        len(selected_inputs) != len(RELEASE_INPUT_ROLES)
        or len(indexed_inputs) != len(selected_inputs)
        or set(indexed_inputs) != set(RELEASE_INPUT_ROLES)
    ):
        raise ContractError(
            f"input roles: expected {RELEASE_INPUT_ROLES!r}"
        )
    ordered_inputs = tuple(
        indexed_inputs[role] for role in RELEASE_INPUT_ROLES
    )
    selected_entries = tuple(entries)
    if not all(
        isinstance(item, VerifiedReleaseEntry) for item in selected_entries
    ):
        raise ContractError(
            "entries: expected VerifiedReleaseEntry entries"
        )
    if not isinstance(dataset_ids, Mapping):
        raise ContractError("dataset_ids: expected object")
    if set(dataset_ids) != set(RELEASE_SLICE_ROLES):
        raise ContractError(
            f"dataset_ids: expected roles {RELEASE_SLICE_ROLES!r}"
        )
    selected_dataset_ids = tuple(
        dataset_ids[role] for role in RELEASE_SLICE_ROLES
    )
    if len(selected_dataset_ids) != len(set(selected_dataset_ids)):
        raise ContractError("dataset_ids: duplicate Dataset identity")
    if not isinstance(registries, Mapping):
        raise ContractError("registries: expected object")
    normalized_registries = {
        key: registries[key] for key in sorted(registries)
    }
    outputs: list[DatasetReleaseOutput] = []
    for role in RELEASE_SLICE_ROLES:
        task_ids = tuple(
            entry.task_id
            for entry in selected_entries
            if role in entry.slices
        )
        provisional = DatasetReleaseOutput(
            role=role,
            dataset_id=require_str(
                dataset_ids[role],
                f"dataset_ids.{role}",
            ),
            dataset_version=require_str(
                release_version,
                "release_version",
            ),
            task_ids=task_ids,
            generated_dataset_hash="sha256:" + "0" * 64,
        )
        outputs.append(
            replace_output_hash(
                provisional,
                canonical_sha256(
                    _dataset_payload(
                        provisional,
                        registries=normalized_registries,
                        entries=selected_entries,
                    )
                ),
            )
        )
    ordered_outputs = tuple(outputs)
    release_id = DatasetReleaseManifest.release_id_for(
        release_version=release_version,
        inputs=ordered_inputs,
        registries=normalized_registries,
        entries=selected_entries,
        outputs=ordered_outputs,
    )
    return DatasetReleaseManifest(
        release_id=release_id,
        release_version=release_version,
        inputs=ordered_inputs,
        registries=normalized_registries,
        entries=selected_entries,
        outputs=ordered_outputs,
        created_at=created_at,
    )


def replace_output_hash(
    output: DatasetReleaseOutput,
    generated_dataset_hash: str,
) -> DatasetReleaseOutput:
    return DatasetReleaseOutput(
        role=output.role,
        dataset_id=output.dataset_id,
        dataset_version=output.dataset_version,
        task_ids=output.task_ids,
        generated_dataset_hash=generated_dataset_hash,
    )


def rebuild_release_datasets(
    manifest: DatasetReleaseManifest,
) -> dict[str, dict[str, JsonValue]]:
    if not isinstance(manifest, DatasetReleaseManifest):
        raise ContractError(
            "manifest: expected DatasetReleaseManifest"
        )
    rebuilt: dict[str, dict[str, JsonValue]] = {}
    for output in manifest.outputs:
        payload = _dataset_payload(
            output,
            registries=manifest.registries,
            entries=manifest.entries,
        )
        if canonical_sha256(payload) != output.generated_dataset_hash:
            raise ContractError(
                f"release_not_rebuildable: {output.role} hash drift"
            )
        rebuilt[output.role] = payload
    return rebuilt


__all__ = [
    "DatasetReleaseInput",
    "DatasetReleaseManifest",
    "DatasetReleaseOutput",
    "RELEASE_INPUT_ROLES",
    "RELEASE_FAILURE_CONTRACTS",
    "RELEASE_ORIGIN_SLICES",
    "RELEASE_SLICE_ROLES",
    "VerifiedReleaseEntry",
    "build_dataset_release",
    "rebuild_release_datasets",
]
