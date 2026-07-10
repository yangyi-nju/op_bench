from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.preflight_task import _task_dirs_from_dataset


class PreflightDatasetTests(unittest.TestCase):
    def test_task_dirs_resolve_relative_to_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets" / "slice"
            dataset_dir.mkdir(parents=True)
            manifest = dataset_dir / "dataset.json"
            manifest.write_text(json.dumps({
                "tasks": [{"task_path": "../../tasks/example"}],
            }))

            self.assertEqual(
                _task_dirs_from_dataset(manifest),
                [(root / "tasks" / "example").resolve()],
            )


if __name__ == "__main__":
    unittest.main()
