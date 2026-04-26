from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RunExperimentTests(unittest.TestCase):
    def test_cli_runs_noop_and_gold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            completed = subprocess.run(
                [
                    "python",
                    "scripts/run_experiment.py",
                    "--task",
                    "tasks/smoke/expit_nan_cpu",
                    "--agent",
                    "noop",
                    "--agent",
                    "gold",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["agents"]["gold"]["resolved_rate"], 1.0)
            self.assertEqual(summary["agents"]["noop"]["resolved_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
