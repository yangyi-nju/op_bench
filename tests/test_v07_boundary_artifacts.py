from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from op_bench.factory.artifacts import load_factory_contract
from op_bench.factory.contracts import (
    CandidateRecord,
    DecisionRecord,
    factory_content_hash,
)


ROOT = Path(__file__).resolve().parents[1]
P3 = ROOT / "factory" / "v0.7" / "p3"
CAPTURE_PRS = {
    143792,
    117065,
    126461,
    147352,
    118762,
    139751,
    143461,
    147433,
    127448,
    139502,
}
SELECTED_SUBCLASSES = {
    143792: "B1",
    117065: "B2",
    126461: "B2",
    147352: "B3",
    118762: "B4",
    139751: "B5",
}
SCREEN_CLI = ROOT / "scripts" / "factory_screen_candidates.py"


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*.json"))
    }


class BoundaryArtifactTests(unittest.TestCase):
    def test_real_screening_funnel(self) -> None:
        captures = json.loads(
            (P3 / "captures.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {item["pr_number"] for item in captures},
            CAPTURE_PRS,
        )

        screening = P3 / "screening"
        index = json.loads(
            (screening / "screening_index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            index["counts"],
            {"accepted": 6, "deferred": 2, "rejected": 2},
        )
        self.assertEqual(
            index["content_hash"],
            factory_content_hash(index),
        )

        candidates: dict[int, CandidateRecord] = {}
        decisions: dict[int, DecisionRecord] = {}
        for entry in index["decisions"]:
            candidate = load_factory_contract(
                screening / entry["candidate"]["relative_path"]
            )
            decision = load_factory_contract(
                screening / entry["decision"]["relative_path"]
            )
            self.assertIsInstance(candidate, CandidateRecord)
            self.assertIsInstance(decision, DecisionRecord)
            assert isinstance(candidate, CandidateRecord)
            assert isinstance(decision, DecisionRecord)
            self.assertEqual(
                candidate.content_hash,
                entry["candidate"]["content_hash"],
            )
            self.assertEqual(
                decision.content_hash,
                entry["decision"]["content_hash"],
            )
            self.assertEqual(candidate.pr_number, entry["pr_number"])
            candidates[candidate.pr_number] = candidate
            decisions[candidate.pr_number] = decision

        self.assertEqual(set(candidates), CAPTURE_PRS)
        self.assertEqual(
            {
                pr_number: candidates[pr_number].proposed_subclass
                for pr_number in SELECTED_SUBCLASSES
            },
            SELECTED_SUBCLASSES,
        )
        self.assertEqual(
            tuple(
                finding.code
                for finding in decisions[143461].findings
                if finding.severity == "reject"
            ),
            ("taxonomy.not_boundary",),
        )

        human = load_factory_contract(
            P3 / "human_decisions" / "pr-147433.json"
        )
        self.assertIsInstance(human, DecisionRecord)
        assert isinstance(human, DecisionRecord)
        automated = decisions[147433]
        self.assertEqual(automated.disposition, "deferred")
        self.assertEqual(human.decision_source, "human_review")
        self.assertEqual(human.disposition, "rejected")
        self.assertEqual(
            tuple(finding.code for finding in human.findings),
            ("review.upstream_reverted",),
        )
        self.assertIsNotNone(human.prior_decision)
        assert human.prior_decision is not None
        self.assertEqual(
            human.prior_decision.artifact_id,
            automated.decision_id,
        )
        self.assertEqual(
            human.prior_decision.content_hash,
            automated.content_hash,
        )

        review = json.loads(
            (P3 / "reviews" / "pr-147433.json").read_text(encoding="utf-8")
        )
        self.assertEqual(review["pr_number"], 147433)
        self.assertEqual(
            review["landed_commit"],
            "1d7397a2d04a4d636559f41511a20f7dadbe5777",
        )
        self.assertEqual(
            review["revert_commit"],
            "841451af9f8c4a49d720a0132532c5468963f643",
        )
        self.assertEqual(review["reason_code"], "review.upstream_reverted")
        self.assertEqual(review["content_hash"], factory_content_hash(review))

    def test_real_screening_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            regenerated = Path(raw) / "screening"
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(ROOT / "src")

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCREEN_CLI),
                    "--input",
                    str(P3 / "captures.json"),
                    "--output-dir",
                    str(regenerated),
                    "--created-at",
                    "2026-07-27T03:00:00Z",
                ],
                cwd=ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                tree_hashes(regenerated),
                tree_hashes(P3 / "screening"),
            )


if __name__ == "__main__":
    unittest.main()
