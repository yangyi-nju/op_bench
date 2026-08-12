from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from op_bench.runtime.validation import ContractError
from scripts.promote_v07_quality_admission import (
    _promoted_source_registry,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class V07QualityPromotionTests(unittest.TestCase):
    def _registries(
        self, root: Path
    ) -> tuple[Path, Path, list[dict[str, object]]]:
        official_path = root / "sources/registry.json"
        staging_path = root / "runs/staging/source_registry.json"
        staging_sources: list[dict[str, object]] = []
        for index in range(72):
            source_dir = root / ".cache" / f"source-{index:02d}"
            source_dir.mkdir(parents=True)
            staging_sources.append(
                {
                    "id": f"source-{index:02d}",
                    "commit": f"{index:040x}",
                    "local_path": Path(
                        os.path.relpath(source_dir, staging_path.parent)
                    ).as_posix(),
                }
            )
        official_sources = []
        for source in staging_sources[:43]:
            selected = dict(source)
            absolute = (staging_path.parent / selected["local_path"]).resolve()
            selected["local_path"] = Path(
                os.path.relpath(absolute, official_path.parent.resolve())
            ).as_posix()
            official_sources.append(selected)
        _write_json(
            official_path,
            {"version": "v1", "sources": official_sources},
        )
        _write_json(
            staging_path,
            {"version": "v1", "sources": staging_sources},
        )
        return official_path, staging_path, staging_sources

    def test_rebases_all_sources_and_preserves_official_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official_path, staging_path, staging_sources = self._registries(
                root
            )

            promoted = _promoted_source_registry(
                official_path=official_path,
                staging_path=staging_path,
            )

            self.assertEqual(len(promoted["sources"]), 72)
            for expected, actual in zip(
                staging_sources, promoted["sources"]
            ):
                self.assertEqual(actual["id"], expected["id"])
                self.assertEqual(actual["commit"], expected["commit"])
                self.assertEqual(
                    (official_path.parent / actual["local_path"]).resolve(),
                    (staging_path.parent / expected["local_path"]).resolve(),
                )

    def test_rejects_semantic_drift_in_an_official_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            official_path, staging_path, _ = self._registries(root)
            official = json.loads(official_path.read_text(encoding="utf-8"))
            official["sources"][0]["commit"] = "f" * 40
            _write_json(official_path, official)

            with self.assertRaisesRegex(ContractError, "official source drift"):
                _promoted_source_registry(
                    official_path=official_path,
                    staging_path=staging_path,
                )


if __name__ == "__main__":
    unittest.main()
