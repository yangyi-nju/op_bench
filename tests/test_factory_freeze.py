from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from op_bench.factory.contracts import (
    CandidateRecord,
    DatasetFreezeEntry,
    DatasetFreezeManifest,
    FactoryAdmissionRecord,
    FactoryArtifactReference,
    factory_content_hash,
)
from op_bench.factory.freeze import (
    build_freeze_manifest,
    freeze_dataset_bytes,
    rebuild_dataset_manifest,
)
from op_bench.factory.lifecycle import advance_admission
from op_bench.factory.screening import (
    V07_BOUNDARY_SCREENING_V1,
    screen_candidate,
)
from op_bench.runtime.validation import ContractError
from tests.test_factory_contracts import SHA_A, candidate
from tests.test_factory_lifecycle import VALID_PATH, transition


ROOT = Path(__file__).resolve().parents[1]
FREEZE_CLI = ROOT / "scripts" / "freeze_dataset.py"


def reference(
    artifact_type: str,
    artifact_id: str,
    relative_path: str,
    *,
    content_hash: str = SHA_A,
) -> FactoryArtifactReference:
    return FactoryArtifactReference(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash=content_hash,
        relative_path=relative_path,
    )


def candidate_for(pr_number: int, subclass: str) -> CandidateRecord:
    selected = candidate()
    digit = str(pr_number % 10)
    base_commit = digit * 40
    merge_commit = str((pr_number + 1) % 10) * 40
    return replace(
        selected,
        candidate_id=CandidateRecord.candidate_id_for(
            repository=selected.repository,
            pr_number=pr_number,
            base_commit=base_commit,
            merge_commit=merge_commit,
        ),
        pr_number=pr_number,
        pr_url=f"https://github.com/pytorch/pytorch/pull/{pr_number}",
        base_commit=base_commit,
        merge_commit=merge_commit,
        proposed_subclass=subclass,
        keyword_pack_id=f"boundary-{subclass.lower()}-v1",
        raw_metadata=replace(
            selected.raw_metadata,
            artifact_id=f"pr:pytorch/pytorch#{pr_number}",
            relative_path=f"raw/pr-{pr_number}.json",
        ),
    )


def verified_admission(
    pr_number: int,
    subclass: str,
    task_id: str,
) -> tuple[CandidateRecord, FactoryAdmissionRecord]:
    selected = candidate_for(pr_number, subclass)
    decision = screen_candidate(selected)
    task = reference(
        "task_bundle",
        f"task:{task_id}",
        f"tasks/{task_id}/task.json",
        content_hash="sha256:" + str(pr_number % 10) * 64,
    )
    previous = None
    for state in VALID_PATH:
        previous = advance_admission(
            previous,
            transition(
                state,
                previous=previous,
                selected_candidate=selected,
                decision=None if state == "discovered" else decision,
                task=task if state in VALID_PATH[2:] else None,
            ),
        )
    assert previous is not None
    return selected, previous


def freeze_entry(
    selected: CandidateRecord,
    admission: FactoryAdmissionRecord,
    task_id: str,
    *,
    subclass: str,
) -> DatasetFreezeEntry:
    assert admission.decision is not None
    assert admission.task is not None
    return DatasetFreezeEntry(
        candidate=admission.candidate,
        decision=admission.decision,
        admission=reference(
            "factory_admission",
            admission.admission_id,
            f"admissions/{admission.admission_id}.json",
            content_hash=admission.content_hash,
        ),
        task=admission.task,
        admission_evidence=reference(
            "admission_evidence",
            f"evidence:{task_id}",
            f"tasks/{task_id}/admission/evidence.json",
            content_hash="sha256:" + "d" * 64,
        ),
        source=reference(
            "source",
            f"source:{task_id}",
            f"sources/{task_id}.json",
            content_hash="sha256:" + "e" * 64,
        ),
        environment=reference(
            "environment",
            f"environment:{task_id}",
            f"environments/{task_id}.json",
            content_hash="sha256:" + "f" * 64,
        ),
        task_id=task_id,
        task_path=f"tasks/{task_id}",
        admission_evidence_path=f"tasks/{task_id}/admission/evidence.json",
        runtime_tier="cpu_python_overlay",
        problem_dimension="boundary",
        problem_subclass=subclass,
        failure_contract="wrong-result",
        admission_state="verified",
    )


def common_references() -> tuple[
    FactoryArtifactReference,
    FactoryArtifactReference,
    FactoryArtifactReference,
    FactoryArtifactReference,
]:
    return (
        reference(
            "dataset_manifest",
            "dataset:pytorch_v0.5",
            "datasets/pytorch_v0.5/dataset.json",
        ),
        reference(
            "factory_protocol",
            "factory-protocol:v1",
            "protocol/factory-v1.json",
        ),
        reference(
            "screening_rule_set",
            V07_BOUNDARY_SCREENING_V1.rule_set_id,
            "protocol/screening-v1.json",
            content_hash=V07_BOUNDARY_SCREENING_V1.rule_set_hash,
        ),
        reference(
            "exclusion_index",
            "exclusions:pytorch_v0.7",
            "screening/exclusions.json",
        ),
    )


def reference_hashes(
    entries: tuple[DatasetFreezeEntry, ...],
    admissions: tuple[FactoryAdmissionRecord, ...],
) -> dict[str, str]:
    base, protocol, rules, exclusions = common_references()
    result = {
        item.relative_path: item.content_hash
        for item in (base, protocol, rules, exclusions)
    }
    for entry in entries:
        for item in entry.references():
            result[item.relative_path] = item.content_hash
    for admission in admissions:
        result[
            f"admissions/{admission.admission_id}.json"
        ] = admission.content_hash
    return result


def build_fixture(
    *,
    created_at: str = "2026-07-26T00:00:00Z",
    reverse: bool = False,
) -> tuple[
    DatasetFreezeManifest,
    tuple[DatasetFreezeEntry, ...],
    tuple[FactoryAdmissionRecord, ...],
    dict[str, str],
]:
    first_candidate, first_admission = verified_admission(
        170001,
        "B1",
        "pytorch__170001__empty_reduction",
    )
    second_candidate, second_admission = verified_admission(
        170002,
        "B3",
        "pytorch__170002__numel_overflow",
    )
    entries = (
        freeze_entry(
            first_candidate,
            first_admission,
            "pytorch__170001__empty_reduction",
            subclass="B1",
        ),
        freeze_entry(
            second_candidate,
            second_admission,
            "pytorch__170002__numel_overflow",
            subclass="B3",
        ),
    )
    if reverse:
        entries = tuple(reversed(entries))
    admissions = (first_admission, second_admission)
    hashes = reference_hashes(entries, admissions)
    base, protocol, rules, exclusions = common_references()
    freeze = build_freeze_manifest(
        dataset_id="pytorch_v0.7_fixture",
        dataset_version="v0.7-fixture",
        base_dataset=base,
        factory_protocol=protocol,
        screening_rule_set=rules,
        exclusion_index=exclusions,
        registries={
            "environments": "environments/registry.json",
            "sources": "sources/registry.json",
        },
        entries=entries,
        admissions={
            admission.admission_id: admission for admission in admissions
        },
        reference_hashes=hashes,
        created_at=created_at,
    )
    return freeze, entries, admissions, hashes


class DatasetFreezeTests(unittest.TestCase):
    def test_entries_sort_by_task_id_and_rebuild_is_input_order_independent(self) -> None:
        first, _, _, _ = build_fixture(reverse=False)
        second, _, _, _ = build_fixture(reverse=True)

        self.assertEqual(
            tuple(entry.task_id for entry in second.entries),
            tuple(sorted(entry.task_id for entry in second.entries)),
        )
        self.assertEqual(freeze_dataset_bytes(first), freeze_dataset_bytes(second))
        self.assertEqual(
            first.generated_dataset_hash,
            second.generated_dataset_hash,
        )

    def test_timestamp_changes_freeze_record_not_generated_dataset(self) -> None:
        first, _, _, _ = build_fixture(
            created_at="2026-07-26T00:00:00Z"
        )
        second, _, _, _ = build_fixture(
            created_at="2026-07-26T00:00:01Z"
        )

        self.assertNotEqual(first.content_hash, second.content_hash)
        self.assertEqual(
            first.generated_dataset_hash,
            second.generated_dataset_hash,
        )
        self.assertEqual(freeze_dataset_bytes(first), freeze_dataset_bytes(second))

    def test_rebuilt_dataset_uses_verified_projection_shape(self) -> None:
        freeze, _, _, _ = build_fixture()

        rebuilt = rebuild_dataset_manifest(freeze)

        self.assertEqual(
            set(rebuilt),
            {"dataset_id", "version", "status", "registries", "tasks"},
        )
        self.assertEqual(rebuilt["status"], "verified")
        self.assertTrue(
            all(
                entry["admission_status"] == "verified"
                for entry in rebuilt["tasks"]
            )
        )

    def test_duplicate_candidate_and_task_are_rejected(self) -> None:
        _, entries, admissions, hashes = build_fixture()
        base, protocol, rules, exclusions = common_references()

        with self.assertRaisesRegex(ContractError, "duplicate Candidate"):
            build_freeze_manifest(
                dataset_id="fixture",
                dataset_version="v1",
                base_dataset=base,
                factory_protocol=protocol,
                screening_rule_set=rules,
                exclusion_index=exclusions,
                registries={},
                entries=(entries[0], entries[0]),
                admissions={item.admission_id: item for item in admissions},
                reference_hashes=hashes,
                created_at="2026-07-26T00:00:00Z",
            )

        other_candidate, other_admission = verified_admission(
            170003,
            "B5",
            entries[0].task_id,
        )
        duplicate_task = freeze_entry(
            other_candidate,
            other_admission,
            entries[0].task_id,
            subclass="B5",
        )
        combined_hashes = reference_hashes(
            (entries[0], duplicate_task),
            (admissions[0], other_admission),
        )
        with self.assertRaisesRegex(ContractError, "duplicate Task"):
            build_freeze_manifest(
                dataset_id="fixture",
                dataset_version="v1",
                base_dataset=base,
                factory_protocol=protocol,
                screening_rule_set=rules,
                exclusion_index=exclusions,
                registries={},
                entries=(entries[0], duplicate_task),
                admissions={
                    admissions[0].admission_id: admissions[0],
                    other_admission.admission_id: other_admission,
                },
                reference_hashes=combined_hashes,
                created_at="2026-07-26T00:00:00Z",
            )

    def test_nonverified_admission_and_taxonomy_mismatch_are_rejected(self) -> None:
        _, entries, admissions, hashes = build_fixture()
        reviewed = replace(
            admissions[0],
            admission_id=FactoryAdmissionRecord.admission_id_for(
                candidate=admissions[0].candidate,
                decision=admissions[0].decision,
                task=admissions[0].task,
                state="reviewed",
                previous_record_hash=admissions[0].previous_record_hash,
                evidence=tuple(
                    item
                    for item in admissions[0].evidence
                    if item.evidence_type != "integrity"
                ),
            ),
            state="reviewed",
            evidence=tuple(
                item
                for item in admissions[0].evidence
                if item.evidence_type != "integrity"
            ),
        )
        entry = replace(
            entries[0],
            admission=replace(
                entries[0].admission,
                artifact_id=reviewed.admission_id,
                content_hash=reviewed.content_hash,
                relative_path=f"admissions/{reviewed.admission_id}.json",
            ),
        )
        base, protocol, rules, exclusions = common_references()
        reviewed_hashes = reference_hashes((entry,), (reviewed,))

        with self.assertRaisesRegex(ContractError, "verified"):
            build_freeze_manifest(
                dataset_id="fixture",
                dataset_version="v1",
                base_dataset=base,
                factory_protocol=protocol,
                screening_rule_set=rules,
                exclusion_index=exclusions,
                registries={},
                entries=(entry,),
                admissions={reviewed.admission_id: reviewed},
                reference_hashes=reviewed_hashes,
                created_at="2026-07-26T00:00:00Z",
            )

        with self.assertRaisesRegex(ContractError, "problem_subclass"):
            replace(entries[0], problem_subclass="P1")

    def test_missing_and_drifted_references_are_rejected(self) -> None:
        _, entries, admissions, hashes = build_fixture()
        base, protocol, rules, exclusions = common_references()
        missing = dict(hashes)
        del missing[entries[0].source.relative_path]
        drifted = dict(hashes)
        drifted[entries[0].source.relative_path] = SHA_A

        for expected, selected_hashes in (
            ("missing", missing),
            ("hash drift", drifted),
        ):
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ContractError, expected):
                    build_freeze_manifest(
                        dataset_id="fixture",
                        dataset_version="v1",
                        base_dataset=base,
                        factory_protocol=protocol,
                        screening_rule_set=rules,
                        exclusion_index=exclusions,
                        registries={},
                        entries=entries,
                        admissions={
                            item.admission_id: item for item in admissions
                        },
                        reference_hashes=selected_hashes,
                        created_at="2026-07-26T00:00:00Z",
                    )

    def test_generated_hash_drift_and_noncanonical_order_are_rejected(self) -> None:
        freeze, _, _, _ = build_fixture()
        payload = freeze.to_dict()
        payload["generated_dataset_hash"] = SHA_A
        payload["content_hash"] = factory_content_hash(payload)

        with self.assertRaisesRegex(ContractError, "generated_dataset_hash"):
            DatasetFreezeManifest.from_dict(payload)
        with self.assertRaisesRegex(ContractError, "sorted"):
            replace(freeze, entries=tuple(reversed(freeze.entries)))

    def test_freeze_round_trip_and_schema_field_parity(self) -> None:
        freeze, _, _, _ = build_fixture()

        self.assertEqual(
            DatasetFreezeManifest.from_dict(freeze.to_dict()),
            freeze,
        )
        schema = json.loads(
            (ROOT / "schemas" / "dataset_freeze.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            set(schema["required"]),
            set(DatasetFreezeManifest.wire_fields()),
        )
        self.assertEqual(
            set(schema["properties"]),
            set(DatasetFreezeManifest.wire_fields()),
        )


class FreezeDatasetCliTests(unittest.TestCase):
    def test_cli_writes_freeze_and_dataset_atomically(self) -> None:
        freeze, entries, admissions, hashes = build_fixture()
        base, protocol, rules, exclusions = common_references()
        request = {
            "dataset_id": freeze.dataset_id,
            "dataset_version": freeze.dataset_version,
            "base_dataset": base.to_dict(),
            "factory_protocol": protocol.to_dict(),
            "screening_rule_set": rules.to_dict(),
            "exclusion_index": exclusions.to_dict(),
            "registries": dict(freeze.registries),
            "entries": [entry.to_dict() for entry in entries],
            "admissions": [item.to_dict() for item in admissions],
            "reference_hashes": hashes,
            "created_at": freeze.created_at,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            input_path = root / "request.json"
            input_path.write_text(json.dumps(request), encoding="utf-8")
            output = root / "output"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(FREEZE_CLI),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads((output / "freeze_manifest.json").read_text()),
                freeze.to_dict(),
            )
            self.assertEqual(
                json.loads((output / "dataset.json").read_text()),
                rebuild_dataset_manifest(freeze),
            )

    def test_malformed_cli_input_leaves_no_output_tree(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            input_path = root / "request.json"
            input_path.write_text("{", encoding="utf-8")
            output = root / "output"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(FREEZE_CLI),
                    "--input",
                    str(input_path),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
