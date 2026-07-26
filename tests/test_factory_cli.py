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


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures" / "factory" / "v0.7" / "candidates.json"
EXPECTED = (
    ROOT
    / "fixtures"
    / "factory"
    / "v0.7"
    / "expected_decisions.json"
)
SCREEN_CLI = ROOT / "scripts" / "factory_screen_candidates.py"
FROZEN_TIME = "2026-07-26T00:00:00Z"


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class FactoryScreeningCliTests(unittest.TestCase):
    def _run(
        self,
        source: Path,
        output: Path,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(ROOT / "src")
        return subprocess.run(
            [
                sys.executable,
                str(SCREEN_CLI),
                "--input",
                str(source),
                "--output-dir",
                str(output),
                "--created-at",
                FROZEN_TIME,
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_fixture_screening_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            first_output = root / "first"
            second_output = root / "second"

            first = self._run(FIXTURE, first_output)
            second = self._run(FIXTURE, second_output)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                tree_hashes(first_output),
                tree_hashes(second_output),
            )
            self.assertEqual(
                json.loads(
                    (first_output / "screening_index.json").read_text(
                        encoding="utf-8"
                    )
                ),
                json.loads(EXPECTED.read_text(encoding="utf-8")),
            )

    def test_fixture_covers_b1_through_b5_and_all_dispositions(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output = Path(raw) / "screened"

            result = self._run(FIXTURE, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            index = json.loads(
                (output / "screening_index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                index["counts"],
                {"accepted": 5, "deferred": 2, "rejected": 2},
            )
            accepted_subclasses = {
                item["target_subclass"]
                for item in index["decisions"]
                if item["disposition"] == "accepted"
            }
            self.assertEqual(accepted_subclasses, {"B1", "B2", "B3", "B4", "B5"})
            for item in index["decisions"]:
                candidate_path = output / item["candidate"]["relative_path"]
                decision_path = output / item["decision"]["relative_path"]
                self.assertEqual(
                    load_factory_contract(candidate_path).content_hash,
                    item["candidate"]["content_hash"],
                )
                self.assertEqual(
                    load_factory_contract(decision_path).content_hash,
                    item["decision"]["content_hash"],
                )

    def test_malformed_list_entry_returns_two_and_leaves_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            malformed = root / "malformed.json"
            malformed.write_text(json.dumps([{"title": "partial"}, "bad"]), encoding="utf-8")
            output = root / "output"

            result = self._run(malformed, output)

            self.assertEqual(result.returncode, 2)
            self.assertIn("[contract_invalid]", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
