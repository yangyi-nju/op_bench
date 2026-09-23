"""Dataset metadata is descriptive, independent from execution and scoring."""
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from _support import make_task
from test_report import identified_row
from op_bench.data.dataset import Dataset
from op_bench.results.report import build_report


class DatasetReportTests(unittest.TestCase):
    def test_candidate_status_and_old_admission_metadata_do_not_change_resolved_rate(self):
        for metadata in ({"release_status": "candidate"},
                         {"release_status": "withdrawn", "ineligible_task_ids": ["x"]},
                         {"admission": {"status": "candidate", "members": []}}):
            records = [identified_row("x"), identified_row("y", resolved=False)]
            for record in records:
                record["dataset_identity"].update(metadata)
            before = deepcopy(records)
            with self.subTest(metadata=metadata):
                report = build_report(records, planned=records)
                self.assertTrue(report["complete"])
                self.assertEqual(.5, report["models"]["a"]["resolved_rate"])
                self.assertEqual(before, records)

    def test_scope_is_provenance_not_a_second_score_gate(self):
        records = [identified_row("x"), identified_row("y")]
        records[1]["task_identity"]["scope"] = "framework_support"
        records[1]["evaluation"]["task_identity"] = deepcopy(records[1]["task_identity"])
        report = build_report(records, planned=records)
        self.assertTrue(report["complete"])
        self.assertEqual(1, report["models"]["a"]["resolved_rate"])

    def test_loader_keeps_paths_and_version_without_enforcing_release_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_task(root / "bundle")
            path = root / "bundle/dataset.json"
            data = json.loads(path.read_text())
            data.update(status="candidate", metadata={"formal_benchmark_member": False},
                        admission={"status": "admitted", "members": [{"task_id": "obsolete"}]})
            path.write_text(json.dumps(data))
            dataset = Dataset.load(path)
            self.assertEqual({"dataset_id": "fixture", "dataset_version": "1", "release_status": "candidate"},
                             dataset.identity_dict())
            self.assertEqual({"formal_benchmark_member": False}, dataset.metadata)
            self.assertEqual("row-sum", dataset.select()[0].task_id)
            with self.assertRaisesRegex(ValueError, "duplicate task selection"):
                dataset.select(["row-sum", "row-sum"])
            with self.assertRaisesRegex(ValueError, "unknown tasks"):
                dataset.select(["not-a-task"])
            for selection in ("row-sum", [1], [[]], [""]):
                with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, "task selection must be a list"):
                    dataset.select(selection)
            for version in (True, 1.0, "1"):
                path.write_text(json.dumps({**data, "schema_version": version}))
                with self.subTest(version=version), self.assertRaisesRegex(ValueError, "schema_version"):
                    Dataset.load(path)
            data["tasks"] = ["../outside/task.json"]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "escapes bundle"):
                Dataset.load(path)


if __name__ == "__main__":
    unittest.main()
