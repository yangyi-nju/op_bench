"""Timing contracts remain explicit when aggregating or comparing attempts."""
from copy import deepcopy
import unittest

from op_bench.results.report import build_report
from test_report import row


class TimingReportTests(unittest.TestCase):
    def test_complete_legacy_and_declared_reports_preserve_their_timing_contract(self):
        for revision in (None, "1", "2"):
            with self.subTest(revision=revision):
                records = [row("x", 1, True), row("x", 1, False, model="b")]
                if revision is not None:
                    for record in records:
                        record["timing_protocol_revision"] = revision
                original = deepcopy(records)
                report = build_report(records, planned=records)
                self.assertTrue(report["complete"])
                self.assertEqual(revision, report["timing_protocol_revision"])
                self.assertEqual(original, records)

    def test_missing_and_explicit_null_remain_the_same_unknown_contract(self):
        records = [row("x", 1, True), dict(row("y", 1, False), timing_protocol_revision=None)]
        report = build_report(records, planned=records)
        self.assertIsNone(report["timing_protocol_revision"])
        self.assertEqual(.5, report["models"]["a"]["resolved_rate"])
        self.assertIsNone(build_report([])["timing_protocol_revision"])

    def test_distinct_known_or_unknown_timing_contracts_cannot_be_pooled(self):
        for first, second in ((None, "2"), ("1", "2"), ("2", None)):
            for same_agent in (True, False):
                with self.subTest(first=first, second=second, same_agent=same_agent):
                    records = [row("x", 1, True), row("y", 1, False, model="a" if same_agent else "b")]
                    for record, revision in zip(records, (first, second)):
                        if revision is not None:
                            record["timing_protocol_revision"] = revision
                    with self.assertRaisesRegex(ValueError, "different timing_protocol_revision"):
                        build_report(records)

    def test_planned_and_started_timing_contracts_must_agree(self):
        for planned_revision, started_revision in ((None, "2"), ("2", None), ("1", "2")):
            with self.subTest(planned=planned_revision, started=started_revision):
                planned = dict(row("x", 1, True), timing_protocol_revision=planned_revision)
                started = dict(planned, timing_protocol_revision=started_revision)
                with self.assertRaisesRegex(ValueError, "different timing_protocol_revision"):
                    build_report([started], planned=[planned])

    def test_unstarted_planned_cells_also_participate_in_contract_validation(self):
        planned = [dict(row("x", 1, True), timing_protocol_revision="2"), row("y", 1, False)]
        with self.assertRaisesRegex(ValueError, "different timing_protocol_revision"):
            build_report([], planned=planned)
        report = build_report([], planned=planned[:1])
        self.assertFalse(report["complete"])
        self.assertEqual("2", report["timing_protocol_revision"])

    def test_malformed_timing_revisions_are_rejected_in_records_and_plans(self):
        for value in ("", "   ", 2, True, [], {}):
            invalid = dict(row("x", 1, True), timing_protocol_revision=value)
            with self.subTest(value=value, source="records"), self.assertRaisesRegex(ValueError, "timing_protocol_revision"):
                build_report([invalid])
            with self.subTest(value=value, source="planned"), self.assertRaisesRegex(ValueError, "timing_protocol_revision"):
                build_report([], planned=[invalid])


if __name__ == "__main__":
    unittest.main()
