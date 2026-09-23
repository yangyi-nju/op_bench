from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from _support import make_task
from op_bench.evaluation.controls import check_task
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.data.task import TaskSpec
from op_bench.io import read_json


class ControlInputTests(unittest.TestCase):
    def test_controls_keep_their_inputs_when_original_source_and_grader_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = TaskSpec.load(make_task(root / "bundle"))
            source = original.source.path / "operator_impl.py"
            grader = original.grader_dir / "check.py"
            real = PatchEvaluator()

            class EditOriginalAfterBaseline:
                def evaluate(self, task, patch_text, output):
                    result = real.evaluate(task, patch_text, output)
                    if output.name == "baseline":
                        # Simulate the author's independent edits, not changes
                        # to the input snapshot owned by the controls.
                        source.write_text(source.read_text().replace("sum(rows[0])", "sum(row)"))
                        grader.write_text("raise RuntimeError('grader changed during controls')\n")
                    return result

            with patch("op_bench.evaluation.controls.PatchEvaluator", EditOriginalAfterBaseline):
                result = check_task(original, "", root / "controls")
            self.assertFalse(result["execution_checks_passed"])
            self.assertEqual(result["baseline"]["status"], "test_failed")
            self.assertEqual(result["reference"]["status"], "test_failed")
            reference = read_json(root / "controls" / result["reference"]["evaluation_path"])
            self.assertEqual(reference["groups"]["pass_to_pass"]["passed"], 1)
            self.assertNotIn("cases", result["reference"])
            self.assertTrue(result["checks"]["baseline_fail_to_pass_fail"])
            self.assertFalse(result["checks"]["reference_passes"])
            self.assertEqual(result["phase"], "complete")
            saved = TaskSpec.load(root / "controls" / result["task_file"])
            self.assertNotEqual(saved.source.path, original.source.path)
            self.assertNotEqual(saved.grader_dir, original.grader_dir)


if __name__ == "__main__":
    unittest.main()
