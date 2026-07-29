from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from op_bench.factory.quality_admission import (
    QualityAcceptedTaskIndex,
    QualityAcceptedTaskRecord,
    QualityCandidateReasonResolution,
    QualityCandidateReassessment,
    load_quality_accepted_task_index,
    validate_quality_accepted_task_index,
)
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.validation import ContractError
from op_bench.integrity import replay_spec_hash
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/v07_quality_release.schema.json"
HISTORICAL = ROOT / "factory/v0.7/p7/historical_readmission.json"
SCREENING = ROOT / "factory/v0.7/p8/screening/screening_index.json"
TASK_TEMPLATE = ROOT / "tasks/pytorch/124385_load_state_dict_prefix"


def _write_canonical(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _rehash(payload: dict[str, object]) -> None:
    payload["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "content_hash"
        }
    )


def _screening_entries() -> list[dict[str, object]]:
    screening = json.loads(SCREENING.read_text(encoding="utf-8"))
    return list(screening["records"])


def _first_screening_index(disposition: str) -> int:
    return next(
        index
        for index, record in enumerate(_screening_entries())
        if record["disposition"] == disposition
    )


def _accepted_fixture(
    root: Path,
    *,
    disposition: str = "accepted_for_build",
    status: str = "building",
) -> tuple[Path, dict[str, object], dict[str, object]]:
    historical_path = root / "factory/v0.7/p7/historical_readmission.json"
    screening_path = (
        root / "factory/v0.7/p8/screening/screening_index.json"
    )
    historical_path.parent.mkdir(parents=True, exist_ok=True)
    screening_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HISTORICAL, historical_path)
    shutil.copy2(SCREENING, screening_path)

    screening_index = json.loads(SCREENING.read_text(encoding="utf-8"))
    screening_record_index = _first_screening_index(disposition)
    screening_record = screening_index["records"][screening_record_index]
    pr_number = screening_record["pr_number"]
    for field in ("candidate", "decision"):
        relative = screening_record[field]["relative_path"]
        source = SCREENING.parent / relative
        destination = screening_path.parent / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    candidate = json.loads(
        (
            screening_path.parent
            / screening_record["candidate"]["relative_path"]
        ).read_text(encoding="utf-8")
    )
    decision = json.loads(
        (
            screening_path.parent
            / screening_record["decision"]["relative_path"]
        ).read_text(encoding="utf-8")
    )

    task_name = f"{pr_number}_synthetic_quality"
    task_path = root / "tasks/pytorch" / task_name
    shutil.copytree(TASK_TEMPLATE, task_path)
    manifest_path = task_path / "task.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["task_id"] = f"pytorch__{pr_number}__synthetic_quality"
    manifest["agent_visible"]["public_task_id"] = "opbench-v07-t9000"
    manifest["source"]["pr_number"] = pr_number
    manifest["source"]["pr_url"] = (
        f"https://github.com/pytorch/pytorch/pull/{pr_number}"
    )
    manifest["source"]["base_commit"] = candidate["base_commit"]
    manifest["source"]["merge_commit"] = candidate["merge_commit"]
    manifest["quality"]["origin"] = "new"
    _write_canonical(manifest_path, manifest)
    task = TaskManifest.load(manifest_path)

    reassessment_reference: dict[str, object] | None = None
    if disposition == "deferred_for_review":
        reasons = decision["preliminary_review_reasons"]
        reassessment_payload: dict[str, object] = {
            "contract_type": "quality_candidate_reassessment",
            "schema_version": "v1",
            "pr_number": pr_number,
            "candidate_id": candidate["candidate_id"],
            "candidate_hash": candidate["content_hash"],
            "decision_id": decision["decision_id"],
            "decision_hash": decision["content_hash"],
            "deferred_reasons": reasons,
            "reason_resolutions": [
                {
                    "reason": reason,
                    "resolution": "resolved",
                    "evidence": f"Independent evidence resolves {reason}.",
                }
                for reason in reasons
            ],
            "reviewer": "independent-reviewer",
            "reviewed_at": "2026-07-30T01:02:03Z",
            "decision": "accepted_for_build",
            "rationale": "Full-context independent review accepted the Task.",
        }
        _rehash(reassessment_payload)
        reassessment_relative = (
            f"factory/v0.7/p8/reassessments/pr-{pr_number}.json"
        )
        _write_canonical(root / reassessment_relative, reassessment_payload)
        reassessment_reference = {
            "artifact_type": "quality_candidate_reassessment",
            "artifact_id": (
                "quality-reassessment:v1:"
                + reassessment_payload["content_hash"].removeprefix("sha256:")
            ),
            "content_hash": reassessment_payload["content_hash"],
            "relative_path": reassessment_relative,
        }

    record: dict[str, object] = {
        "screening_record_index": screening_record_index,
        "pr_number": pr_number,
        "candidate_id": candidate["candidate_id"],
        "candidate_hash": candidate["content_hash"],
        "decision_id": decision["decision_id"],
        "decision_hash": decision["content_hash"],
        "screening_disposition": disposition,
        "reassessment": reassessment_reference,
        "task_id": manifest["task_id"],
        "public_task_id": manifest["agent_visible"]["public_task_id"],
        "origin": "new",
        "task_path": f"tasks/pytorch/{task_name}",
        "task_manifest_hash": canonical_sha256(manifest),
        "replay_spec_hash": replay_spec_hash(task),
    }
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    accepted: dict[str, object] = {
        "contract_type": "quality_accepted_task_index",
        "schema_version": "v1",
        "created_at": "2026-07-30T02:03:04Z",
        "historical_index_path": (
            "factory/v0.7/p7/historical_readmission.json"
        ),
        "historical_index_hash": historical["content_hash"],
        "retained_count": historical["k"],
        "required_task_count": 50 - historical["k"],
        "candidate_index_path": (
            "factory/v0.7/p8/screening/screening_index.json"
        ),
        "candidate_index_hash": screening_index["content_hash"],
        "status": status,
        "task_count": 1,
        "tasks": [record],
    }
    _rehash(accepted)
    accepted_path = root / "factory/v0.7/p8/accepted_tasks.json"
    _write_canonical(accepted_path, accepted)
    return accepted_path, accepted, record


def _reassessment() -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_type": "quality_candidate_reassessment",
        "schema_version": "v1",
        "pr_number": 190777,
        "candidate_id": "quality-candidate:v1:" + "a" * 64,
        "candidate_hash": "sha256:" + "b" * 64,
        "decision_id": "quality-decision:v1:" + "c" * 64,
        "decision_hash": "sha256:" + "d" * 64,
        "deferred_reasons": [
            "review.large_changed_surface",
            "review.title_needs_behavior_confirmation",
        ],
        "reason_resolutions": [
            {
                "reason": "review.large_changed_surface",
                "resolution": "resolved",
                "evidence": (
                    "The changed surface is one behavioral path with a "
                    "single replay contract."
                ),
            },
            {
                "reason": "review.title_needs_behavior_confirmation",
                "resolution": "resolved",
                "evidence": (
                    "The public reproducer and regression test confirm the "
                    "behavioral failure."
                ),
            },
        ],
        "reviewer": "independent-reviewer",
        "reviewed_at": "2026-07-30T01:02:03Z",
        "decision": "accepted_for_build",
        "rationale": (
            "Independent full-context review confirmed a distinct, "
            "reproducible operator contract."
        ),
    }
    payload["content_hash"] = canonical_sha256(payload)
    return payload


class V07ExpansionArtifactContractTests(unittest.TestCase):
    def test_reassessment_round_trips_exact_source_bound_contract(self) -> None:
        payload = _reassessment()

        reassessment = QualityCandidateReassessment.from_dict(payload)

        self.assertEqual(reassessment.to_dict(), payload)
        self.assertEqual(
            tuple(
                resolution.reason
                for resolution in reassessment.reason_resolutions
            ),
            tuple(payload["deferred_reasons"]),
        )

    def test_reassessment_requires_exact_reason_resolution(self) -> None:
        mutations = {
            "missing resolution": lambda payload: payload[
                "reason_resolutions"
            ].pop(),
            "duplicate resolution": lambda payload: payload[
                "reason_resolutions"
            ].append(copy.deepcopy(payload["reason_resolutions"][0])),
            "unknown resolution": lambda payload: payload[
                "reason_resolutions"
            ][0].__setitem__("reason", "review.unknown"),
            "accepted blocker": lambda payload: payload[
                "reason_resolutions"
            ][0].__setitem__("resolution", "confirmed_blocker"),
            "empty evidence": lambda payload: payload[
                "reason_resolutions"
            ][0].__setitem__("evidence", ""),
            "empty reviewer": lambda payload: payload.__setitem__(
                "reviewer", ""
            ),
            "invalid timestamp": lambda payload: payload.__setitem__(
                "reviewed_at", "2026-99-99T01:02:03Z"
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                payload = _reassessment()
                mutate(payload)
                payload["content_hash"] = canonical_sha256(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "content_hash"
                    }
                )

                with self.assertRaises(ContractError):
                    QualityCandidateReassessment.from_dict(payload)

    def test_reassessment_rejects_field_and_hash_tampering(self) -> None:
        payload = _reassessment()
        payload["candidate_hash"] = "sha256:" + "e" * 64
        with self.assertRaisesRegex(ContractError, "content_hash"):
            QualityCandidateReassessment.from_dict(payload)

        payload = _reassessment()
        payload["unreviewed"] = True
        with self.assertRaisesRegex(ContractError, "fields"):
            QualityCandidateReassessment.from_dict(payload)

    def test_schema_tracks_reassessment_wire_contracts(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        resolution = schema["$defs"][
            "quality_candidate_reason_resolution"
        ]
        reassessment = schema["$defs"]["quality_candidate_reassessment"]

        self.assertEqual(
            set(resolution["required"]),
            set(QualityCandidateReasonResolution.wire_fields()),
        )
        self.assertEqual(
            set(resolution["properties"]),
            set(QualityCandidateReasonResolution.wire_fields()),
        )
        self.assertEqual(
            set(reassessment["required"]),
            set(QualityCandidateReassessment.wire_fields()),
        )
        self.assertEqual(
            set(reassessment["properties"]),
            set(QualityCandidateReassessment.wire_fields()),
        )

    def test_accepted_index_loads_direct_and_reassessed_candidates(self) -> None:
        for disposition in (
            "accepted_for_build",
            "deferred_for_review",
        ):
            with self.subTest(disposition=disposition):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, payload, _ = _accepted_fixture(
                        root,
                        disposition=disposition,
                    )

                    loaded = load_quality_accepted_task_index(root, path)

                    self.assertEqual(loaded.to_dict(), payload)
                    self.assertEqual(
                        loaded.tasks[0].screening_disposition,
                        disposition,
                    )

    def test_deferred_candidate_requires_exact_bound_reassessment(self) -> None:
        mutations = {
            "missing": lambda root, accepted, record: record.__setitem__(
                "reassessment", None
            ),
            "candidate hash": lambda root, accepted, record: _mutate_json(
                root / record["reassessment"]["relative_path"],
                "candidate_hash",
                "sha256:" + "f" * 64,
                reference=record["reassessment"],
            ),
            "decision hash": lambda root, accepted, record: _mutate_json(
                root / record["reassessment"]["relative_path"],
                "decision_hash",
                "sha256:" + "e" * 64,
                reference=record["reassessment"],
            ),
            "blocker": lambda root, accepted, record: _mutate_resolution(
                root / record["reassessment"]["relative_path"],
                reference=record["reassessment"],
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, accepted, record = _accepted_fixture(
                        root,
                        disposition="deferred_for_review",
                    )
                    mutate(root, accepted, record)
                    _rehash(accepted)
                    _write_canonical(path, accepted)

                    with self.assertRaises(ContractError):
                        load_quality_accepted_task_index(root, path)

    def test_hard_rejected_candidate_cannot_enter_accepted_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = _accepted_fixture(
                root,
                disposition="hard_rejected",
            )

            with self.assertRaisesRegex(ContractError, "hard_rejected"):
                load_quality_accepted_task_index(root, path)

    def test_accepted_index_requires_unique_strict_screening_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, record = _accepted_fixture(root)
            accepted["tasks"] = [copy.deepcopy(record), copy.deepcopy(record)]
            accepted["task_count"] = 2
            accepted["tasks"][0]["screening_record_index"] += 1
            _rehash(accepted)
            _write_canonical(path, accepted)

            with self.assertRaisesRegex(
                ContractError, "screening_record_index"
            ):
                load_quality_accepted_task_index(root, path)

    def test_accepted_index_rejects_unsafe_task_paths(self) -> None:
        unsafe_paths = (
            "/tasks/pytorch/task",
            "../tasks/pytorch/task",
            r"tasks\\pytorch\\task",
            "other/pytorch/task",
            "tasks/pytorch/../task",
        )
        for unsafe in unsafe_paths:
            with self.subTest(path=unsafe):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, accepted, record = _accepted_fixture(root)
                    record["task_path"] = unsafe
                    _rehash(accepted)
                    _write_canonical(path, accepted)
                    with self.assertRaises(ContractError):
                        load_quality_accepted_task_index(root, path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, record = _accepted_fixture(root)
            target = root / record["task_path"]
            shutil.rmtree(target)
            target.symlink_to(TASK_TEMPLATE, target_is_directory=True)
            with self.assertRaisesRegex(ContractError, "symlink"):
                load_quality_accepted_task_index(root, path)

    def test_accepted_index_binds_task_source_identity_and_hashes(self) -> None:
        mutations = {
            "base commit": lambda manifest, record: manifest["source"].__setitem__(
                "base_commit", "f" * 40
            ),
            "merge commit": lambda manifest, record: manifest[
                "source"
            ].__setitem__("merge_commit", "e" * 40),
            "task id": lambda manifest, record: record.__setitem__(
                "task_id", "pytorch__999999__substitution"
            ),
            "public id": lambda manifest, record: record.__setitem__(
                "public_task_id", "opbench-v07-t9999"
            ),
            "manifest hash": lambda manifest, record: record.__setitem__(
                "task_manifest_hash", "sha256:" + "d" * 64
            ),
            "replay hash": lambda manifest, record: record.__setitem__(
                "replay_spec_hash", "sha256:" + "c" * 64
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    path, accepted, record = _accepted_fixture(root)
                    manifest_path = (
                        root / record["task_path"] / "task.json"
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    mutate(manifest, record)
                    _write_canonical(manifest_path, manifest)
                    if name in {"base commit", "merge commit"}:
                        task = TaskManifest.load(manifest_path)
                        record["task_manifest_hash"] = canonical_sha256(
                            manifest
                        )
                        record["replay_spec_hash"] = replay_spec_hash(task)
                    _rehash(accepted)
                    _write_canonical(path, accepted)

                    with self.assertRaises(ContractError):
                        load_quality_accepted_task_index(root, path)

    def test_accepted_index_rejects_canonical_and_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _ = _accepted_fixture(root)
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ContractError, "canonical"):
                load_quality_accepted_task_index(root, path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, _ = _accepted_fixture(root)
            accepted["created_at"] = "2026-07-30T02:03:05Z"
            _write_canonical(path, accepted)
            with self.assertRaisesRegex(ContractError, "content_hash"):
                load_quality_accepted_task_index(root, path)

    def test_candidate_index_must_bind_the_exact_historical_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, _ = _accepted_fixture(root)
            screening_path = (
                root / accepted["candidate_index_path"]
            )
            screening = json.loads(
                screening_path.read_text(encoding="utf-8")
            )
            screening["historical_index_hash"] = "sha256:" + "f" * 64
            _rehash(screening)
            _write_canonical(screening_path, screening)
            accepted["candidate_index_hash"] = screening["content_hash"]
            _rehash(accepted)
            _write_canonical(path, accepted)

            with self.assertRaisesRegex(
                ContractError, "historical_index_hash"
            ):
                load_quality_accepted_task_index(root, path)

    def test_building_and_complete_exact_count_gates_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, accepted, record = _accepted_fixture(root)
            accepted["tasks"] = [
                copy.deepcopy(record)
                for _ in range(accepted["required_task_count"])
            ]
            accepted["task_count"] = len(accepted["tasks"])
            _rehash(accepted)
            with self.assertRaisesRegex(ContractError, "building"):
                QualityAcceptedTaskIndex.from_dict(accepted)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, _ = _accepted_fixture(root, status="complete")
            with self.assertRaisesRegex(ContractError, "complete"):
                load_quality_accepted_task_index(root, path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, accepted, record = _accepted_fixture(root)
            records: list[dict[str, object]] = []
            for index in range(35):
                selected = copy.deepcopy(record)
                selected["screening_record_index"] = index
                selected["pr_number"] = 200000 + index
                selected["candidate_id"] = (
                    "quality-candidate:v1:" + f"{index + 1:064x}"
                )
                selected["decision_id"] = (
                    "quality-decision:v1:" + f"{index + 1:064x}"
                )
                selected["task_id"] = f"synthetic__{index:02d}"
                selected["public_task_id"] = (
                    f"opbench-v07-t{8000 + index:04d}"
                )
                selected["task_path"] = (
                    f"tasks/pytorch/synthetic_{index:02d}"
                )
                records.append(selected)
            accepted["status"] = "complete"
            accepted["tasks"] = records
            accepted["task_count"] = 35
            _rehash(accepted)
            with self.assertRaisesRegex(ContractError, "complete"):
                QualityAcceptedTaskIndex.from_dict(accepted)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, accepted, _ = _accepted_fixture(root)
            accepted["retained_count"] = 26
            accepted["required_task_count"] = 24
            _rehash(accepted)
            with self.assertRaisesRegex(ContractError, "retained_count"):
                QualityAcceptedTaskIndex.from_dict(accepted)

    def test_public_complete_validator_requires_official_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, accepted, _ = _accepted_fixture(root)
            other = root / "scratch/accepted.json"
            _write_canonical(other, accepted)

            errors = validate_quality_accepted_task_index(root, other)

            self.assertTrue(errors)
            self.assertTrue(
                any("official" in error for error in errors),
                errors,
            )

    def test_schema_tracks_accepted_index_wire_contracts(self) -> None:
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        record = schema["$defs"]["quality_accepted_task_record"]
        index = schema["$defs"]["quality_accepted_task_index"]

        self.assertEqual(
            set(record["required"]),
            set(QualityAcceptedTaskRecord.wire_fields()),
        )
        self.assertEqual(
            set(record["properties"]),
            set(QualityAcceptedTaskRecord.wire_fields()),
        )
        self.assertEqual(
            set(index["required"]),
            set(QualityAcceptedTaskIndex.wire_fields()),
        )
        self.assertEqual(
            set(index["properties"]),
            set(QualityAcceptedTaskIndex.wire_fields()),
        )


def _mutate_json(
    path: Path,
    field: str,
    value: object,
    *,
    reference: dict[str, object],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    _rehash(payload)
    _write_canonical(path, payload)
    reference["content_hash"] = payload["content_hash"]
    reference["artifact_id"] = (
        "quality-reassessment:v1:"
        + payload["content_hash"].removeprefix("sha256:")
    )


def _mutate_resolution(
    path: Path,
    *,
    reference: dict[str, object],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["reason_resolutions"][0]["resolution"] = "confirmed_blocker"
    payload["decision"] = "rejected"
    _rehash(payload)
    _write_canonical(path, payload)
    reference["content_hash"] = payload["content_hash"]
    reference["artifact_id"] = (
        "quality-reassessment:v1:"
        + payload["content_hash"].removeprefix("sha256:")
    )


if __name__ == "__main__":
    unittest.main()
