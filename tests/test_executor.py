from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from op_bench.executor import LocalExecutor


class LocalExecutorTests(unittest.TestCase):
    def test_run_captures_success_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = LocalExecutor().run(["python3", "-c", "print('ok')"], cwd=Path(tmp), timeout_sec=5)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("ok", result.stdout)
            self.assertFalse(result.timed_out)

    def test_run_marks_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = LocalExecutor().run(
                ["python3", "-c", "import time; time.sleep(2)"],
                cwd=Path(tmp),
                timeout_sec=1,
            )
            self.assertNotEqual(result.exit_code, 0)
            self.assertTrue(result.timed_out)

    def test_collect_environment_includes_python(self) -> None:
        evidence = LocalExecutor().collect_environment()
        self.assertEqual(evidence.executor, "local")
        self.assertTrue(evidence.python_version)
        self.assertTrue(evidence.platform)


if __name__ == "__main__":
    unittest.main()
