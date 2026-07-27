from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import unittest

from op_bench.factory.contracts import FactoryArtifactReference
from op_bench.factory.release import (
    DatasetReleaseInput,
    DatasetReleaseManifest,
    VerifiedReleaseEntry,
    build_dataset_release,
    rebuild_release_datasets,
)
from op_bench.runtime.validation import ContractError
from op_bench.runtime.schema import (
    load_runtime_schema,
    validate_schema_instance,
)


ROOT = Path(__file__).resolve().parents[1]


def reference(
    artifact_type: str,
    artifact_id: str,
    relative_path: str,
    digit: str,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash="sha256:" + digit * 64,
        relative_path=relative_path,
    )


def release_input(
    role: str,
    digit: str,
    *,
    freeze: bool = False,
) -> DatasetReleaseInput:
    dataset_id = {
        "cumulative_base": "pytorch_v0.5",
        "precision_base": "pytorch_v0.5_precision",
        "boundary_freeze": "pytorch_v0.7_boundary_factory",
    }[role]
    return DatasetReleaseInput(
        role=role,
        dataset=reference(
            "dataset_manifest",
            f"dataset:{dataset_id}",
            f"datasets/{dataset_id}/dataset.json",
            digit,
        ),
        provenance=(
            reference(
                "dataset_freeze",
                "freeze:v1:" + "f" * 64,
                "factory/v0.7/p4/boundary_freeze/freeze_manifest.json",
                "f",
            )
            if freeze
            else None
        ),
    )


def entry(
    task_id: str,
    digit: str,
    *,
    origin: str,
    slices: tuple[str, ...],
    runtime_tier: str = "cpu_python_overlay",
    problem_subclass: str = "unclassified",
    failure_contract: str = "unclassified",
) -> VerifiedReleaseEntry:
    problem_dimension = {
        "inherited_cumulative": "unclassified",
        "inherited_precision": "precision",
        "restored_precision": "precision",
        "factory_boundary": "boundary",
    }[origin]
    task_dir = f"tasks/pytorch/{task_id}"
    return VerifiedReleaseEntry(
        task=reference(
            "task_bundle",
            f"task:{task_id}",
            f"{task_dir}/task.json",
            digit,
        ),
        admission_evidence=reference(
            "admission_evidence",
            f"evidence:{task_id}",
            f"{task_dir}/admission/evidence.json",
            digit,
        ),
        task_id=task_id,
        task_path=task_dir,
        admission_evidence_path=f"{task_dir}/admission/evidence.json",
        runtime_tier=runtime_tier,
        problem_dimension=problem_dimension,
        problem_subclass=problem_subclass,
        failure_contract=failure_contract,
        origin=origin,
        slices=slices,
        admission_state="verified",
    )


def fixture_entries() -> tuple[VerifiedReleaseEntry, ...]:
    return (
        entry(
            "pytorch__100001__base",
            "1",
            origin="inherited_cumulative",
            slices=("cumulative",),
        ),
        entry(
            "pytorch__100002__precision",
            "2",
            origin="inherited_precision",
            slices=("cumulative", "precision"),
            problem_subclass="P3",
        ),
        entry(
            "pytorch__100003__restored",
            "3",
            origin="restored_precision",
            slices=("cumulative", "precision"),
            problem_subclass="P4",
        ),
        entry(
            "pytorch__100004__boundary",
            "4",
            origin="factory_boundary",
            slices=("cumulative", "boundary"),
            runtime_tier="cpu_source_snapshot_fuller",
            problem_subclass="B3",
            failure_contract="exception",
        ),
    )


def build_fixture(
    *,
    created_at: str = "2026-07-27T05:00:00Z",
) -> DatasetReleaseManifest:
    return build_dataset_release(
        release_version="v0.7",
        inputs=(
            release_input("cumulative_base", "a"),
            release_input("precision_base", "b"),
            release_input("boundary_freeze", "c", freeze=True),
        ),
        registries={
            "environments": reference(
                "environment_registry",
                "registry:environments:v1",
                "environments/registry.json",
                "d",
            ),
            "sources": reference(
                "source_registry",
                "registry:sources:v1",
                "sources/registry.json",
                "e",
            ),
        },
        entries=fixture_entries(),
        dataset_ids={
            "cumulative": "pytorch_v0.7",
            "boundary": "pytorch_v0.7_boundary",
            "precision": "pytorch_v0.7_precision",
        },
        created_at=created_at,
    )


class DatasetReleaseContractTests(unittest.TestCase):
    def test_release_round_trip_and_exact_slice_membership(self) -> None:
        release = build_fixture()

        rebuilt = rebuild_release_datasets(release)

        self.assertEqual(
            DatasetReleaseManifest.from_dict(release.to_dict()),
            release,
        )
        self.assertEqual(
            {
                role: [item["task_id"] for item in dataset["tasks"]]
                for role, dataset in rebuilt.items()
            },
            {
                "cumulative": [
                    "pytorch__100001__base",
                    "pytorch__100002__precision",
                    "pytorch__100003__restored",
                    "pytorch__100004__boundary",
                ],
                "boundary": ["pytorch__100004__boundary"],
                "precision": [
                    "pytorch__100002__precision",
                    "pytorch__100003__restored",
                ],
            },
        )
        self.assertEqual(
            {role: dataset["status"] for role, dataset in rebuilt.items()},
            {
                "cumulative": "verified",
                "boundary": "verified",
                "precision": "verified",
            },
        )

    def test_timestamp_changes_record_hash_but_not_release_identity(self) -> None:
        first = build_fixture(created_at="2026-07-27T05:00:00Z")
        second = build_fixture(created_at="2026-07-27T05:00:01Z")

        self.assertEqual(first.release_id, second.release_id)
        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            rebuild_release_datasets(first),
            rebuild_release_datasets(second),
        )

    def test_origin_membership_and_input_roles_are_fail_closed(self) -> None:
        cases = (
            (
                "slices",
                fixture_entries()[0],
                {"origin": "factory_boundary"},
            ),
            (
                "slices",
                fixture_entries()[2],
                {"slices": ("cumulative", "boundary")},
            ),
            (
                "admission_state",
                fixture_entries()[3],
                {"admission_state": "draft"},
            ),
        )
        for expected, selected, changes in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    replace(selected, **changes)

        with self.assertRaisesRegex(ContractError, "input roles"):
            build_dataset_release(
                release_version="v0.7",
                inputs=build_fixture().inputs[:2],
                registries=dict(build_fixture().registries),
                entries=fixture_entries(),
                dataset_ids={
                    "cumulative": "pytorch_v0.7",
                    "boundary": "pytorch_v0.7_boundary",
                    "precision": "pytorch_v0.7_precision",
                },
                created_at="2026-07-27T05:00:00Z",
            )
        with self.assertRaisesRegex(ContractError, "input roles"):
            build_dataset_release(
                release_version="v0.7",
                inputs=build_fixture().inputs
                + (build_fixture().inputs[0],),
                registries=dict(build_fixture().registries),
                entries=fixture_entries(),
                dataset_ids={
                    "cumulative": "pytorch_v0.7",
                    "boundary": "pytorch_v0.7_boundary",
                    "precision": "pytorch_v0.7_precision",
                },
                created_at="2026-07-27T05:00:00Z",
            )

    def test_duplicate_task_and_hash_drift_are_rejected(self) -> None:
        with self.assertRaisesRegex(ContractError, "duplicate Task"):
            build_dataset_release(
                release_version="v0.7",
                inputs=build_fixture().inputs,
                registries=dict(build_fixture().registries),
                entries=fixture_entries() + (fixture_entries()[0],),
                dataset_ids={
                    "cumulative": "pytorch_v0.7",
                    "boundary": "pytorch_v0.7_boundary",
                    "precision": "pytorch_v0.7_precision",
                },
                created_at="2026-07-27T05:00:00Z",
            )

        payload = build_fixture().to_dict()
        payload["outputs"][0]["generated_dataset_hash"] = "sha256:" + "0" * 64
        from op_bench.factory.contracts import factory_content_hash

        payload["content_hash"] = factory_content_hash(payload)
        with self.assertRaisesRegex(
            ContractError,
            "generated_dataset_hash",
        ):
            DatasetReleaseManifest.from_dict(payload)

        release = build_fixture()
        with self.assertRaisesRegex(ContractError, "release_id"):
            replace(
                release,
                registries={
                    **dict(release.registries),
                    "sources": replace(
                        release.registries["sources"],
                        content_hash="sha256:" + "9" * 64,
                    ),
                },
            )

    def test_entry_rejects_identity_and_path_drift(self) -> None:
        selected = fixture_entries()[0]
        with self.assertRaisesRegex(ContractError, "task.artifact_id"):
            replace(
                selected,
                task=replace(
                    selected.task,
                    artifact_id="task:wrong",
                ),
            )
        with self.assertRaisesRegex(
            ContractError,
            "admission_evidence_path",
        ):
            replace(
                selected,
                admission_evidence_path="tasks/wrong/evidence.json",
            )
        with self.assertRaisesRegex(ContractError, "relative"):
            replace(selected, task_path="../private")

    def test_schema_required_fields_match_wire_contracts(self) -> None:
        schema_path = ROOT / "schemas" / "dataset_release.schema.json"
        schema = json.loads(
            schema_path.read_text(encoding="utf-8")
        )

        validate_schema_instance(
            build_fixture().to_dict(),
            load_runtime_schema(schema_path),
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            set(DatasetReleaseManifest.wire_fields()),
        )
        self.assertEqual(
            set(schema["properties"]),
            set(DatasetReleaseManifest.wire_fields()),
        )
        self.assertEqual(
            set(schema["$defs"]["entry"]["required"]),
            set(VerifiedReleaseEntry.wire_fields()),
        )
        self.assertEqual(
            set(schema["$defs"]["input"]["required"]),
            set(DatasetReleaseInput.wire_fields()),
        )


if __name__ == "__main__":
    unittest.main()
