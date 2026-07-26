from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from op_bench.factory.contracts import (
    CandidateRecord,
    ChangedFile,
    FactoryArtifactReference,
    factory_content_hash,
)
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64
BASE_COMMIT = "1" * 40
MERGE_COMMIT = "2" * 40


def candidate(*, created_at: str = "2026-07-26T00:00:00Z") -> CandidateRecord:
    repository = "pytorch/pytorch"
    pr_number = 170001
    return CandidateRecord(
        candidate_id=CandidateRecord.candidate_id_for(
            repository=repository,
            pr_number=pr_number,
            base_commit=BASE_COMMIT,
            merge_commit=MERGE_COMMIT,
        ),
        framework="pytorch",
        repository=repository,
        pr_number=pr_number,
        pr_url="https://github.com/pytorch/pytorch/pull/170001",
        base_commit=BASE_COMMIT,
        merge_commit=MERGE_COMMIT,
        author_date="2025-04-01T12:30:00Z",
        merge_date="2025-04-03T08:15:00Z",
        title="Handle empty reduction without launching a kernel",
        description="An empty reduction must return the documented identity value.",
        changed_files=(
            ChangedFile(
                path="aten/src/ATen/native/ReduceOps.cpp",
                additions=12,
                deletions=3,
                is_test=False,
            ),
            ChangedFile(
                path="test/test_reductions.py",
                additions=17,
                deletions=2,
                is_test=True,
            ),
        ),
        total_files=2,
        total_changed_lines=34,
        discovery_source="fixture",
        keyword_pack_id="boundary-b1-v1",
        matched_keyword_ids=("empty-reduction", "zero-size"),
        proposed_dimension="boundary",
        proposed_subclass="B1",
        raw_metadata=FactoryArtifactReference(
            artifact_type="candidate_raw_metadata",
            artifact_id="pr:pytorch/pytorch#170001",
            content_hash=SHA_A,
            relative_path="raw/pr-170001.json",
        ),
        created_at=created_at,
    )


class CandidateContractTests(unittest.TestCase):
    def test_candidate_identity_ignores_collection_time_but_record_hash_does_not(self) -> None:
        first = candidate(created_at="2026-07-26T00:00:00Z")
        second = candidate(created_at="2026-07-26T00:00:01Z")

        self.assertEqual(first.candidate_id, second.candidate_id)
        self.assertNotEqual(first.content_hash, second.content_hash)

    def test_candidate_round_trip_is_exact(self) -> None:
        selected = candidate()

        encoded = selected.to_dict()

        self.assertEqual(CandidateRecord.from_dict(encoded), selected)
        self.assertEqual(CandidateRecord.from_dict(encoded).to_dict(), encoded)

    def test_candidate_rejects_unknown_field_and_hash_drift(self) -> None:
        payload = candidate().to_dict()
        payload["unexpected"] = True
        with self.assertRaisesRegex(ContractError, "unknown fields"):
            CandidateRecord.from_dict(payload)

        payload = candidate().to_dict()
        payload["title"] = "tampered"
        with self.assertRaisesRegex(ContractError, "content_hash"):
            CandidateRecord.from_dict(payload)

    def test_candidate_rejects_derived_identity_drift(self) -> None:
        payload = candidate().to_dict()
        payload["candidate_id"] = "candidate:v1:" + "f" * 64
        payload["content_hash"] = factory_content_hash(payload)

        with self.assertRaisesRegex(ContractError, "candidate_id"):
            CandidateRecord.from_dict(payload)

    def test_candidate_rejects_noncanonical_candidate_inputs(self) -> None:
        cases: tuple[tuple[str, object], ...] = (
            ("changed_files.0.path", "/private/source.cpp"),
            (
                "changed_files",
                [
                    candidate().to_dict()["changed_files"][0],
                    candidate().to_dict()["changed_files"][0],
                ],
            ),
            ("changed_files.0.additions", -1),
            ("matched_keyword_ids", ["zero-size", "empty-reduction"]),
            ("description", "x" * 4001),
            ("discovery_source", "live_scrape"),
        )

        for field_path, replacement in cases:
            with self.subTest(field_path=field_path):
                payload = copy.deepcopy(candidate().to_dict())
                parts = field_path.split(".")
                target: object = payload
                for part in parts[:-1]:
                    target = target[int(part)] if isinstance(target, list) else target[part]
                final = parts[-1]
                if isinstance(target, list):
                    target[int(final)] = replacement
                else:
                    target[final] = replacement
                payload["content_hash"] = factory_content_hash(payload)

                with self.assertRaises(ContractError):
                    CandidateRecord.from_dict(payload)

    def test_candidate_requires_declared_totals_to_match_changed_files(self) -> None:
        for field, value in (("total_files", 3), ("total_changed_lines", 35)):
            with self.subTest(field=field):
                payload = candidate().to_dict()
                payload[field] = value
                payload["content_hash"] = factory_content_hash(payload)

                with self.assertRaisesRegex(ContractError, field):
                    CandidateRecord.from_dict(payload)

    def test_candidate_round_trips_incomplete_discovery_facts_for_screening(self) -> None:
        selected = candidate()
        incomplete = replace(
            selected,
            candidate_id=CandidateRecord.candidate_id_for(
                repository=selected.repository,
                pr_number=selected.pr_number,
                base_commit=None,
                merge_commit=None,
            ),
            base_commit=None,
            merge_commit=None,
            author_date=None,
            merge_date=None,
        )

        self.assertEqual(CandidateRecord.from_dict(incomplete.to_dict()), incomplete)

    def test_candidate_round_trips_normalized_screening_facts(self) -> None:
        selected = candidate()
        enriched = replace(
            selected,
            change_kind="feature",
            external_test=FactoryArtifactReference(
                artifact_type="external_test",
                artifact_id="test:pytorch/pytorch#170001",
                content_hash=SHA_A,
                relative_path="raw/external-test.json",
            ),
            environment_freeze=FactoryArtifactReference(
                artifact_type="environment_freeze",
                artifact_id="runtime:pytorch-vintage",
                content_hash=SHA_A,
                relative_path="raw/environment-freeze.json",
            ),
            source_available=False,
            runtime_supported=False,
        )

        self.assertEqual(CandidateRecord.from_dict(enriched.to_dict()), enriched)

    def test_candidate_value_objects_reject_unsafe_paths_and_invalid_hashes(self) -> None:
        with self.assertRaisesRegex(ContractError, "relative"):
            FactoryArtifactReference(
                artifact_type="candidate_raw_metadata",
                artifact_id="pr:pytorch/pytorch#170001",
                content_hash=SHA_A,
                relative_path="../private.json",
            )
        with self.assertRaisesRegex(ContractError, "content_hash"):
            FactoryArtifactReference(
                artifact_type="candidate_raw_metadata",
                artifact_id="pr:pytorch/pytorch#170001",
                content_hash="not-a-hash",
                relative_path="raw/pr-170001.json",
            )

    def test_candidate_schema_required_fields_match_wire_contract(self) -> None:
        schema_path = ROOT / "schemas" / "factory_candidate.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(CandidateRecord.wire_fields()))
        self.assertEqual(
            set(schema["properties"]),
            set(CandidateRecord.wire_fields()),
        )

    def test_replace_remains_valid_for_immutable_candidate_snapshots(self) -> None:
        selected = candidate()

        updated = replace(selected, title="Handle empty reductions safely")

        self.assertEqual(updated.candidate_id, selected.candidate_id)
        self.assertNotEqual(updated.content_hash, selected.content_hash)


if __name__ == "__main__":
    unittest.main()
