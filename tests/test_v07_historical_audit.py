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
from op_bench.runtime.canonical import canonical_sha256
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
                readmission = json.loads(
                    (
                        output.parent
                        / "tasks/opbench-v07-t0001/quality/readmission.json"
                    ).read_text(encoding="utf-8")
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
