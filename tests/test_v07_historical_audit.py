from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from op_bench.factory.quality_release import (
    QualityTaskRecord,
    build_historical_dispositions,
)
from op_bench.runtime.canonical import canonical_sha256


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
                Path(directory) / "missing-reviews",
                CREATED_AT,
            )
        self.assertEqual({item.disposition for item in records}, {"deferred"})

    def test_cli_tree_is_deterministic_and_sensitive_to_review_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
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


if __name__ == "__main__":
    unittest.main()
