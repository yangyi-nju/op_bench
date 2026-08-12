from __future__ import annotations

from collections.abc import Mapping, Sequence

from op_bench.factory.contracts import (
    DatasetFreezeEntry,
    DatasetFreezeManifest,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
    freeze_dataset_payload,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError, require_str


def _verify_reference(
    reference: FactoryArtifactReference,
    reference_hashes: Mapping[str, str],
) -> None:
    actual = reference_hashes.get(reference.relative_path)
    if actual is None:
        raise ContractError(
            f"reference missing: {reference.relative_path!r}"
        )
    if actual != reference.content_hash:
        raise ContractError(
            f"reference hash drift: {reference.relative_path!r}"
        )


def build_freeze_manifest(
    *,
    dataset_id: str,
    dataset_version: str,
    base_dataset: FactoryArtifactReference,
    factory_protocol: FactoryArtifactReference,
    screening_rule_set: FactoryArtifactReference,
    exclusion_index: FactoryArtifactReference,
    registries: Mapping[str, str],
    entries: Sequence[DatasetFreezeEntry],
    admissions: Mapping[str, FactoryAdmissionRecord],
    reference_hashes: Mapping[str, str],
    created_at: str,
) -> DatasetFreezeManifest:
    require_str(dataset_id, "dataset_id")
    require_str(dataset_version, "dataset_version")
    require_str(created_at, "created_at")
    if not isinstance(registries, Mapping):
        raise ContractError("registries: expected object")
    if not isinstance(admissions, Mapping):
        raise ContractError("admissions: expected object")
    if not isinstance(reference_hashes, Mapping):
        raise ContractError("reference_hashes: expected object")

    selected_entries = tuple(entries)
    if not selected_entries:
        raise ContractError("entries: expected at least one verified selection")
    for index, entry in enumerate(selected_entries):
        if not isinstance(entry, DatasetFreezeEntry):
            raise ContractError(
                f"entries[{index}]: expected DatasetFreezeEntry"
            )
    candidate_ids = tuple(
        entry.candidate.artifact_id for entry in selected_entries
    )
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ContractError("entries: duplicate Candidate identity")
    task_ids = tuple(entry.task_id for entry in selected_entries)
    if len(set(task_ids)) != len(task_ids):
        raise ContractError("entries: duplicate Task identity")

    for reference in (
        base_dataset,
        factory_protocol,
        screening_rule_set,
        exclusion_index,
    ):
        _verify_reference(reference, reference_hashes)

    for entry in selected_entries:
        for reference in entry.references():
            _verify_reference(reference, reference_hashes)
        admission = admissions.get(entry.admission.artifact_id)
        if admission is None:
            raise ContractError(
                f"Admission missing: {entry.admission.artifact_id!r}"
            )
        if not isinstance(admission, FactoryAdmissionRecord):
            raise ContractError("Admission index: expected FactoryAdmissionRecord")
        if admission.state != "verified":
            raise ContractError(
                f"Admission {admission.admission_id!r} is not verified"
            )
        if (
            entry.admission.content_hash != admission.content_hash
            or entry.admission.artifact_id != admission.admission_id
        ):
            raise ContractError("Admission reference hash drift")
        if entry.candidate != admission.candidate:
            raise ContractError("Candidate reference does not match Admission")
        if entry.decision != admission.decision:
            raise ContractError("Decision reference does not match Admission")
        if entry.task != admission.task:
            raise ContractError("Task reference does not match Admission")

    ordered_entries = tuple(sorted(selected_entries, key=lambda item: item.task_id))
    normalized_registries = {
        key: registries[key] for key in sorted(registries)
    }
    generated_dataset_hash = canonical_sha256(
        freeze_dataset_payload(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            registries=normalized_registries,
            entries=ordered_entries,
        )
    )
    freeze_id = DatasetFreezeManifest.freeze_id_for(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        base_dataset=base_dataset,
        factory_protocol=factory_protocol,
        screening_rule_set=screening_rule_set,
        exclusion_index=exclusion_index,
        registries=normalized_registries,
        entries=ordered_entries,
        generated_dataset_hash=generated_dataset_hash,
    )
    return DatasetFreezeManifest(
        freeze_id=freeze_id,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        base_dataset=base_dataset,
        factory_protocol=factory_protocol,
        screening_rule_set=screening_rule_set,
        exclusion_index=exclusion_index,
        registries=normalized_registries,
        entries=ordered_entries,
        generated_dataset_hash=generated_dataset_hash,
        created_at=created_at,
    )


def rebuild_dataset_manifest(
    freeze: DatasetFreezeManifest,
) -> dict[str, object]:
    if not isinstance(freeze, DatasetFreezeManifest):
        raise ContractError("freeze: expected DatasetFreezeManifest")
    rebuilt = freeze_dataset_payload(
        dataset_id=freeze.dataset_id,
        dataset_version=freeze.dataset_version,
        registries=freeze.registries,
        entries=freeze.entries,
    )
    if canonical_sha256(rebuilt) != freeze.generated_dataset_hash:
        raise ContractError("freeze_not_rebuildable: Dataset identity mismatch")
    return rebuilt


def freeze_dataset_bytes(freeze: DatasetFreezeManifest) -> bytes:
    return canonical_json(rebuild_dataset_manifest(freeze)).encode("utf-8")
