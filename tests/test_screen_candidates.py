from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "screen_candidates", ROOT / "scripts" / "screen_candidates.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


screen_mod = _load_module()


def _pr(**overrides) -> dict:
    """Baseline PR that passes all filters. Overrides mutate individual fields."""
    base = {
        "number": 12345,
        "title": "Fix log_softmax accumulation precision on float16",
        "url": "https://github.com/pytorch/pytorch/pull/12345",
        "mergedAt": "2024-06-01T12:00:00Z",
        "body": "Corrects overflow in the accumulator for log_softmax.",
        "files": [
            {"path": "aten/src/ATen/native/cuda/SoftMax.cu", "additions": 20, "deletions": 5},
            {"path": "test/test_nn.py", "additions": 15, "deletions": 0},
        ],
    }
    base.update(overrides)
    return base


class ScreenCandidatesTests(unittest.TestCase):
    def test_baseline_passes(self) -> None:
        ok, reasons = screen_mod.screen(_pr(), "P1")
        self.assertTrue(ok, reasons)
        self.assertEqual(reasons, [])

    def test_reject_revert(self) -> None:
        ok, reasons = screen_mod.screen(_pr(title="Revert 'Fix log_softmax'"), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("revert" in r for r in reasons))

    def test_reject_reland(self) -> None:
        ok, reasons = screen_mod.screen(_pr(title="Reland fix for log_softmax"), "P1")
        self.assertFalse(ok)

    def test_reject_merge_before_window(self) -> None:
        ok, reasons = screen_mod.screen(_pr(mergedAt="2023-06-01T00:00:00Z"), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("rule2" in r for r in reasons))

    def test_reject_merge_after_window(self) -> None:
        ok, reasons = screen_mod.screen(_pr(mergedAt="2025-12-01T00:00:00Z"), "P1")
        self.assertFalse(ok)

    def test_reject_too_many_files(self) -> None:
        files = [
            {"path": f"src/f{i}.py", "additions": 5, "deletions": 5}
            for i in range(5)
        ] + [{"path": "test/test_foo.py", "additions": 10, "deletions": 0}]
        ok, reasons = screen_mod.screen(_pr(files=files), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("rule3" in r for r in reasons))

    def test_reject_too_few_lines(self) -> None:
        files = [
            {"path": "src/f.py", "additions": 2, "deletions": 1},
            {"path": "test/test_f.py", "additions": 3, "deletions": 0},
        ]
        ok, reasons = screen_mod.screen(_pr(files=files), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("rule4" in r for r in reasons))

    def test_reject_too_many_lines(self) -> None:
        files = [
            {"path": "src/f.py", "additions": 150, "deletions": 150},
            {"path": "test/test_f.py", "additions": 20, "deletions": 0},
        ]
        ok, reasons = screen_mod.screen(_pr(files=files), "P1")
        self.assertFalse(ok)

    def test_reject_no_test_file(self) -> None:
        files = [
            {"path": "src/f.py", "additions": 30, "deletions": 5},
            {"path": "src/g.py", "additions": 15, "deletions": 5},
        ]
        ok, reasons = screen_mod.screen(_pr(files=files), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("rule5" in r for r in reasons))

    def test_reject_feature_add_title(self) -> None:
        ok, reasons = screen_mod.screen(_pr(title="Add support for float8_e4m3 in log_softmax"), "P1")
        self.assertFalse(ok)
        self.assertTrue(any("rule6" in r for r in reasons))

    def test_reject_feature_add_body(self) -> None:
        body = "This PR adds support for a new dtype in log_softmax kernel."
        ok, reasons = screen_mod.screen(_pr(title="log_softmax dtype extension", body=body), "P1")
        self.assertFalse(ok)

    def test_fix_prefix_overrides_feature_add_pattern(self) -> None:
        # "Fix" prefix is a strong signal it's a correctness fix, even if body mentions "add support"
        body = "Fixes a bug where the code path added support for float16 didn't propagate dtype."
        ok, reasons = screen_mod.screen(_pr(title="Fix dtype propagation in log_softmax", body=body), "P1")
        self.assertTrue(ok, reasons)

    def test_cli_end_to_end(self) -> None:
        # Run the screener via stdin-ish JSON, then check output shape
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "candidates.json"
            output_path = tmp_path / "screened.json"
            input_path.write_text(json.dumps([
                _pr(number=1),
                _pr(number=2, title="Revert Fix"),
                _pr(number=3, files=[{"path": "src/x.py", "additions": 300, "deletions": 0}, {"path": "test/x.py", "additions": 10, "deletions": 0}]),
            ]))
            rc = screen_mod.main(["--input", str(input_path), "--subclass", "P1", "--output", str(output_path)])
            self.assertEqual(rc, 0)
            out = json.loads(output_path.read_text())
            self.assertEqual(out["total"], 3)
            self.assertEqual(out["passed_count"], 1)
            self.assertEqual(out["rejected_count"], 2)
            self.assertEqual(out["passed"][0]["number"], 1)


if __name__ == "__main__":
    unittest.main()
