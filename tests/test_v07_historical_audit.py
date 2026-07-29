from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from op_bench.factory.quality_release import (
    QualityTaskRecord,
    build_historical_dispositions,
    write_historical_dispositions,
)
from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.validation import ContractError


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/pytorch_v0.7/dataset.json"
SCRIPT = ROOT / "scripts/audit_v07_historical.py"
CREATED_AT = "2026-07-29T00:00:00Z"


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _run_audit(review_root: Path, output_root: Path) -> None:
    review_root = review_root.resolve()
    output_root = output_root.resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        (
            str(ROOT / ".venv/bin/python"),
            str(SCRIPT),
            "--dataset",
            str(DATASET),
            "--review-root",
            str(review_root),
            "--output",
            str(output_root / "historical_readmission.json"),
            "--created-at",
            CREATED_AT,
        ),
        cwd=ROOT,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"audit failed ({completed.returncode}):\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def _write_review(path: Path, payload: dict[str, object]) -> None:
    payload["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "content_hash"
        }
    )
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class V07HistoricalAuditTests(unittest.TestCase):
    def test_historical_dataset_requires_exact_25_frozen_identities(self) -> None:
        original = json.loads(DATASET.read_text(encoding="utf-8"))
        cases = {
            "24": original["tasks"][:-1],
            "26": [
                *original["tasks"],
                {
                    **original["tasks"][-1],
                    "task_id": "pytorch__extra",
                },
            ],
            "missing": [
                *original["tasks"][:-1],
                {
                    **original["tasks"][-1],
                    "task_id": "pytorch__replacement",
                },
            ],
            "duplicate": [
                *original["tasks"][:-1],
                dict(original["tasks"][0]),
            ],
        }
        for name, entries in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                dataset = Path(directory).resolve() / "dataset.json"
                payload = dict(original)
                payload["tasks"] = entries
                dataset.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(
                    ContractError,
                    "exact historical 25",
                ):
                    build_historical_dispositions(
                        ROOT,
                        dataset,
                        ROOT / "factory/v0.7/p7/reviews",
                        CREATED_AT,
                    )

    def test_frozen_public_ids_are_independent_of_dataset_order(self) -> None:
        original = json.loads(DATASET.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            temporary = Path(directory).resolve()
            payload = dict(original)
            payload["tasks"] = [
                {
                    **entry,
                    "task_path": str(ROOT / entry["task_path"]),
                }
                for entry in reversed(original["tasks"])
            ]
            dataset = temporary / "permuted-dataset.json"
            dataset.write_text(json.dumps(payload), encoding="utf-8")

            records = build_historical_dispositions(
                ROOT,
                dataset,
                temporary / "missing-reviews",
                CREATED_AT,
            )

        self.assertEqual(
            tuple((record.task_id, record.public_task_id) for record in records),
            tuple(
                (
                    task_id,
                    f"opbench-v07-t{index:04d}",
                )
                for index, task_id in enumerate(
                    sorted(entry["task_id"] for entry in original["tasks"]),
                    start=1,
                )
            ),
        )

    def test_tampered_or_permuted_public_id_mapping_fails_closed(self) -> None:
        original = json.loads(
            (ROOT / "factory/v0.7/p6/public_task_ids.json").read_text(
                encoding="utf-8"
            )
        )
        mutations = {
            "permuted": lambda payload: payload["tasks"].reverse(),
            "wrong-pair": lambda payload: payload["tasks"][0].__setitem__(
                "public_task_id",
                "opbench-v07-t0025",
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                payload = json.loads(json.dumps(original))
                mutate(payload)
                mapping = Path(directory).resolve() / "public-task-ids.json"
                mapping.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ContractError, "public Task ID mapping"):
                    build_historical_dispositions(
                        ROOT,
                        DATASET,
                        Path(directory).resolve() / "missing-reviews",
                        CREATED_AT,
                        public_task_ids_path=mapping,
                    )

    def test_historical_audit_is_complete_and_unique(self) -> None:
        records = build_historical_dispositions(
            ROOT,
            DATASET,
            ROOT / "factory/v0.7/p7/reviews",
            CREATED_AT,
        )
        self.assertEqual(len(records), 25)
        self.assertEqual(len({item.task_id for item in records}), 25)
        self.assertTrue(
            {item.disposition for item in records}
            <= {"retained", "deferred", "retired"}
        )
        self.assertTrue({item.disposition for item in records})

    def test_missing_reviews_defer_every_task_instead_of_accepting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            records = build_historical_dispositions(
                ROOT,
                DATASET,
                Path(directory).resolve() / "missing-reviews",
                CREATED_AT,
            )
        self.assertEqual({item.disposition for item in records}, {"deferred"})

    def test_cli_tree_is_deterministic_and_sensitive_to_review_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            reviews = temporary / "reviews"
            first = temporary / "first"
            second = temporary / "second"
            changed = temporary / "changed"
            _run_audit(reviews, first)
            _run_audit(reviews, second)
            first_hash = _tree_hash(first)
            self.assertEqual(first_hash, _tree_hash(second))

            task_id = json.loads(DATASET.read_text(encoding="utf-8"))["tasks"][0][
                "task_id"
            ]
            review_dir = reviews / task_id
            review_dir.mkdir(parents=True)
            (review_dir / "prompt.json").write_text(
                json.dumps({"decision": "rejected"}),
                encoding="utf-8",
            )
            _run_audit(reviews, changed)
            self.assertNotEqual(first_hash, _tree_hash(changed))

            index = json.loads(
                (changed / "historical_readmission.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(index["k"], 0)
            self.assertEqual(index["records"][0]["disposition"], "deferred")
            self.assertEqual(index["required_candidate_count"], 150)
            self.assertEqual(len(index["records"]), 25)
            self.assertEqual(
                index["content_hash"],
                canonical_sha256(
                    {
                        key: value
                        for key, value in index.items()
                        if key != "content_hash"
                    }
                ),
            )
            for record_payload in index["records"]:
                record = QualityTaskRecord.from_dict(record_payload)
                for reference in (
                    record.prompt_evidence,
                    record.complexity_evidence,
                    record.admission_evidence,
                ):
                    artifact = json.loads(
                        (changed / reference.relative_path).read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(
                        reference.content_hash,
                        artifact["content_hash"],
                    )

    def test_malformed_claimed_rejections_always_defer(self) -> None:
        dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        task_id = dataset["tasks"][0]["task_id"]
        for kind in ("prompt", "complexity"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                reviews = Path(directory).resolve() / "reviews"
                review_dir = reviews / task_id
                review_dir.mkdir(parents=True)
                (review_dir / f"{kind}.json").write_text(
                    json.dumps(
                        {
                            "contract_type": (
                                "prompt_quality"
                                if kind == "prompt"
                                else "complexity_evidence"
                            ),
                            "decision": "rejected",
                        }
                    ),
                    encoding="utf-8",
                )
                records = build_historical_dispositions(
                    ROOT,
                    DATASET,
                    reviews,
                    CREATED_AT,
                )
                self.assertEqual(records[0].disposition, "deferred")

    def test_score_four_review_cannot_self_report_support(self) -> None:
        review_root = ROOT / "factory/v0.7/p7/reviews"
        selected = review_root / "pytorch__124385__load_state_dict_prefix.json"
        mutations = {
            "minimal-pilot-reviewer-reproduction": lambda payload: payload[
                "complexity"
            ].__setitem__(
                "blind_pilot",
                {
                    "decision": "accepted",
                    "counts_toward_final": False,
                },
            ),
            "boolean-second-review-self-report": lambda payload: payload[
                "source_evidence"
            ].__setitem__(
                "second_complexity_review",
                {
                    "second_review": True,
                    "pilot_decision": "accepted",
                },
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                reviews = Path(directory).resolve() / "reviews"
                shutil.copytree(review_root, reviews)
                path = reviews / selected.name
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutate(payload)
                _write_review(path, payload)
                with self.assertRaisesRegex(
                    ContractError,
                    "review-derived dispositions",
                ):
                    build_historical_dispositions(
                        ROOT,
                        DATASET,
                        reviews,
                        CREATED_AT,
                    )

    def test_score_four_review_requires_every_support_binding(self) -> None:
        review_root = ROOT / "factory/v0.7/p7/reviews"
        selected = review_root / "pytorch__124385__load_state_dict_prefix.json"
        invalid_values: dict[str, object] = {
            "public_task_id": "opbench-v07-t0022",
            "expected_attempt_id": "attempt:v1:" + "0" * 64,
            "task_view_sha256": "sha256:" + "0" * 64,
            "validity": "invalid",
            "evaluation_outcome": "unknown",
            "terminal_reason": "timed_out",
            "duration_ms": 0,
            "counts_toward_final": True,
            "decision": "rejected",
            "complexity_evidence_decision": "rejected",
            "complexity_evidence_severity": "high",
            "factual_evidence_hash": "sha256:" + "0" * 64,
            "factual_record_hash": "sha256:" + "0" * 64,
            "second_review_artifact_hash": "sha256:" + "0" * 64,
            "second_review_content_hash": "sha256:" + "0" * 64,
            "second_review_record_hash": "sha256:" + "0" * 64,
            "second_review_source_hash": "sha256:" + "0" * 64,
            "reviewer": "semantic-reviewer-v07-independent-01",
        }
        for field, invalid in invalid_values.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                reviews = Path(directory).resolve() / "reviews"
                shutil.copytree(review_root, reviews)
                path = reviews / selected.name
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["complexity"]["blind_pilot"][field] = invalid
                _write_review(path, payload)
                with self.assertRaises(ContractError):
                    build_historical_dispositions(
                        ROOT,
                        DATASET,
                        reviews,
                        CREATED_AT,
                    )

    def test_score_four_source_evidence_is_exact_and_task_bound(self) -> None:
        review_root = ROOT / "factory/v0.7/p7/reviews"
        selected = review_root / "pytorch__124385__load_state_dict_prefix.json"
        other = review_root / "pytorch__161488__lbfgs_wolfe.json"
        original_other = json.loads(other.read_text(encoding="utf-8"))

        def swap_task(payload: dict[str, object]) -> None:
            payload["complexity"]["blind_pilot"] = original_other["complexity"][
                "blind_pilot"
            ]
            payload["source_evidence"]["blind_pilot"] = original_other[
                "source_evidence"
            ]["blind_pilot"]
            payload["source_evidence"]["second_complexity_review"] = original_other[
                "source_evidence"
            ]["second_complexity_review"]

        mutations = {
            "cross-task-swap": swap_task,
            "source-pilot-mismatch": lambda payload: payload["source_evidence"][
                "blind_pilot"
            ].__setitem__("duration_ms", 1),
            "second-artifact-mismatch": lambda payload: payload[
                "source_evidence"
            ]["second_complexity_review"].__setitem__(
                "artifact_hash",
                "sha256:" + "0" * 64,
            ),
            "non-independent-reviewer": lambda payload: (
                payload["complexity"]["blind_pilot"].__setitem__(
                    "reviewer",
                    "semantic-reviewer-v07-independent-01",
                ),
                payload["source_evidence"]["blind_pilot"].__setitem__(
                    "reviewer",
                    "semantic-reviewer-v07-independent-01",
                ),
                payload["source_evidence"][
                    "second_complexity_review"
                ].__setitem__(
                    "reviewer",
                    "semantic-reviewer-v07-independent-01",
                ),
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                reviews = Path(directory).resolve() / "reviews"
                shutil.copytree(review_root, reviews)
                path = reviews / selected.name
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutate(payload)
                _write_review(path, payload)
                with self.assertRaises(ContractError):
                    build_historical_dispositions(
                        ROOT,
                        DATASET,
                        reviews,
                        CREATED_AT,
                    )

    def test_official_historical_tree_is_exact_review_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            output = temporary / "historical_readmission.json"
            write_historical_dispositions(
                ROOT,
                DATASET,
                ROOT / "factory/v0.7/p7/reviews",
                output,
                CREATED_AT,
            )
            rebuilt = json.loads(output.read_text(encoding="utf-8"))
            official = json.loads(
                (ROOT / "factory/v0.7/p7/historical_readmission.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(output.read_bytes(), canonical_json(rebuilt).encode())
            self.assertEqual(output.read_bytes(), canonical_json(official).encode())
            for record in rebuilt["records"]:
                for field in (
                    "prompt_evidence",
                    "complexity_evidence",
                    "admission_evidence",
                ):
                    relative = record[field]["relative_path"]
                    self.assertEqual(
                        (temporary / relative).read_bytes(),
                        (ROOT / relative).read_bytes(),
                        relative,
                    )

    def test_draft_admission_and_advisory_scope_are_explicitly_deferred(
        self,
    ) -> None:
        original_dataset = json.loads(DATASET.read_text(encoding="utf-8"))
        for gate in ("admission", "patch_scope"):
            with self.subTest(gate=gate), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as directory:
                temporary = Path(directory).resolve()
                entry = dict(original_dataset["tasks"][0])
                source_task = ROOT / entry["task_path"]
                task_dir = temporary / "task"
                shutil.copytree(source_task, task_dir)
                manifest_path = task_dir / "task.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if gate == "admission":
                    manifest["admission"]["status"] = "draft"
                    manifest["metadata"]["admission_status"] = "draft"
                else:
                    manifest["patch_scope"]["mode"] = "advisory"
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                dataset_payload = json.loads(
                    DATASET.read_text(encoding="utf-8")
                )
                dataset_payload["tasks"][0] = {
                    **entry,
                    "task_path": str(task_dir),
                }
                dataset_path = temporary / "dataset.json"
                dataset_path.write_text(
                    json.dumps(dataset_payload),
                    encoding="utf-8",
                )
                output = temporary / "audit/historical_readmission.json"
                write_historical_dispositions(
                    ROOT,
                    dataset_path,
                    temporary / "missing-reviews",
                    output,
                    CREATED_AT,
                )
                index = json.loads(output.read_text(encoding="utf-8"))
                selected_record = next(
                    record
                    for record in index["records"]
                    if record["task_id"] == entry["task_id"]
                )
                admission_reference = selected_record["admission_evidence"][
                    "relative_path"
                ]
                readmission = json.loads(
                    (output.parent / admission_reference).read_text(
                        encoding="utf-8"
                    )
                )
                expected = (
                    "admission.status: verified required"
                    if gate == "admission"
                    else "patch_scope: enforced required"
                )
                self.assertIn(expected, readmission["errors"])
                self.assertEqual(readmission["disposition"], "deferred")

    def test_symlinked_review_and_output_ancestors_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory).resolve()
            external_reviews = temporary / "external-reviews"
            external_reviews.mkdir()
            review_link = temporary / "review-link"
            review_link.symlink_to(external_reviews, target_is_directory=True)
            with self.assertRaisesRegex(ContractError, "symlink"):
                build_historical_dispositions(
                    ROOT,
                    DATASET,
                    review_link / "nested",
                    CREATED_AT,
                )

            external_output = temporary / "external-output"
            external_output.mkdir()
            output_link = temporary / "output-link"
            output_link.symlink_to(external_output, target_is_directory=True)
            with self.assertRaisesRegex(ContractError, "symlink"):
                write_historical_dispositions(
                    ROOT,
                    DATASET,
                    temporary / "missing-reviews",
                    output_link / "nested/historical_readmission.json",
                    CREATED_AT,
                )
            self.assertEqual(list(external_output.iterdir()), [])

    def test_schema_tracks_index_and_record_wire_fields(self) -> None:
        schema = json.loads(
            (ROOT / "schemas/v07_quality_release.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            set(schema["properties"]),
        )
        self.assertEqual(
            set(schema["$defs"]["record"]["required"]),
            set(QualityTaskRecord.wire_fields()),
        )
        self.assertEqual(
            set(schema["$defs"]["record"]["properties"]),
            set(QualityTaskRecord.wire_fields()),
        )
        self.assertEqual(schema["properties"]["records"]["minItems"], 25)
        self.assertEqual(schema["properties"]["records"]["maxItems"], 25)
        self.assertIs(schema["properties"]["records"]["uniqueItems"], True)
        reference_pattern = schema["$defs"]["reference"]["properties"][
            "relative_path"
        ]["pattern"]
        task_pattern = schema["$defs"]["record"]["properties"]["task_path"][
            "pattern"
        ]
        for pattern, valid, invalid in (
            (
                reference_pattern,
                "tasks/opbench-v07-t0001/quality/prompt.json",
                ("/absolute.json", "../escape.json", "a/../escape.json", r"a\b.json"),
            ),
            (
                task_pattern,
                "tasks/pytorch/example",
                ("/absolute", "../escape", "a/../escape", r"a\b"),
            ),
        ):
            self.assertIsNotNone(re.fullmatch(pattern, valid))
            for value in invalid:
                self.assertIsNone(re.fullmatch(pattern, value), value)


if __name__ == "__main__":
    unittest.main()
