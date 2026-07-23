from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr
import io

from op_bench.runtime.canonical import canonical_json
from scripts.run_legacy_replay import main


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = tuple(
    ROOT / run / name
    for run in (
        "runs/v0.5_codex_legacy_cpu",
        "runs/v0.5_codex_legacy_cuda",
        "runs/v0.5_precision_codex_cpu",
        "runs/v0.5_precision_codex_gpu",
    )
    for name in ("results.jsonl", "summary.json")
)


def identity(path: Path):
    metadata = path.stat()
    return (
        hashlib.sha256(path.read_bytes()).hexdigest(),
        metadata.st_mtime_ns,
        metadata.st_size,
    )


class RuntimeReplayCliTests(unittest.TestCase):
    def test_private_target_path_is_not_printed_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private_marker = "do-not-print-private-target"
            target = root / private_marker
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                exit_code = main(
                    [
                        "--repository-root",
                        str(ROOT),
                        "--output-root",
                        str(root / "output"),
                        "--target-config",
                        str(target),
                    ]
                )

            self.assertEqual(exit_code, 2)
            self.assertNotIn(private_marker, stderr.getvalue())

    def test_target_config_constructs_and_uses_exact_runtime_observer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            target = root / "private-target.json"
            target.write_text("private fixture", encoding="utf-8")

            with mock.patch(
                "scripts.run_legacy_replay.ExactReplayObserver"
            ) as observer_type:
                observer = observer_type.return_value.__enter__.return_value
                observer.side_effect = lambda case: case.expected_outcome
                exit_code = main(
                    [
                        "--repository-root",
                        str(ROOT),
                        "--output-root",
                        str(output),
                        "--target-config",
                        str(target),
                    ]
                )

            observer_type.assert_called_once_with(ROOT, target)
            self.assertEqual(observer.call_count, 85)
            self.assertEqual(exit_code, 0)
            summary = json.loads(
                (output / "replay" / "replay_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["passed"], 85)
            self.assertEqual(summary["blocked"], 0)

    def test_inventory_only_cli_writes_85_blocked_cases_without_mutating_v05(self) -> None:
        before = {path: identity(path) for path in HISTORICAL}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            exit_code = main(
                [
                    "--repository-root",
                    str(ROOT),
                    "--output-root",
                    str(output),
                ]
            )

            replay = output / "replay"
            expected_files = {
                "replay_manifest.json",
                "replay_results.jsonl",
                "replay_differences.jsonl",
                "replay_summary.json",
            }
            self.assertEqual(exit_code, 0)
            self.assertEqual({path.name for path in replay.iterdir()}, expected_files)
            manifest_raw = (replay / "replay_manifest.json").read_bytes()
            manifest = json.loads(manifest_raw)
            summary = json.loads(
                (replay / "replay_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(manifest["cases"]), 85)
            self.assertEqual(summary["total"], 85)
            self.assertEqual(summary["blocked"], 85)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(
                manifest_raw,
                (canonical_json(manifest) + "\n").encode("utf-8"),
            )
        self.assertEqual({path: identity(path) for path in HISTORICAL}, before)


if __name__ == "__main__":
    unittest.main()
