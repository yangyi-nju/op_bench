from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from scripts.verify_runtime_resources import main
from tests.test_runtime_integrity import build_complete_run


class RuntimeResourceVerifierCliTests(unittest.TestCase):
    def test_valid_run_passes_and_cleanup_failed_run_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            complete = build_complete_run(Path(temporary) / "valid")
            self.assertEqual(main(["--run-root", str(complete.root)]), 0)

        with tempfile.TemporaryDirectory() as temporary:
            complete = build_complete_run(Path(temporary) / "failed")
            retry = (
                complete.root
                / "attempts"
                / complete.attempt_id
                / "retries"
                / "retry-0001"
            )
            ledger_path = retry / "runtime_resources.jsonl"
            records = [
                json.loads(line)
                for line in ledger_path.read_text(encoding="utf-8").splitlines()
            ]
            records[-1]["transition"] = "cleanup_failed"
            unhashed = dict(records[-1])
            del unhashed["record_hash"]
            records[-1]["record_hash"] = canonical_sha256(unhashed)
            ledger_path.write_text(
                "".join(canonical_json(record) + "\n" for record in records),
                encoding="utf-8",
            )
            cleanup_path = retry / "runtime_cleanup.json"
            cleanup = json.loads(cleanup_path.read_text(encoding="utf-8"))
            cleanup["entries"][0].update(
                {
                    "status": "cleanup_failed",
                    "error_code": "workspace_remove_failed",
                }
            )
            cleanup["all_released"] = False
            cleanup_path.write_text(
                canonical_json(cleanup) + "\n",
                encoding="utf-8",
            )

            self.assertEqual(main(["--run-root", str(complete.root)]), 1)

    def test_missing_run_root_is_usage_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing"
            self.assertEqual(main(["--run-root", str(missing)]), 2)


if __name__ == "__main__":
    unittest.main()
