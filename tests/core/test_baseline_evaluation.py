"""Executable cache semantics through the public baseline/evaluator APIs.

These tests compile and run a small C++ operator. They observe binary output and
actual link events, rather than artifact hashes or evaluator implementation shape.
"""

import difflib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from op_bench.runtime.baseline import build_baseline
from op_bench.evaluation.evaluator import PatchEvaluator
from op_bench.data.task import TaskSpec
from _support import make_task


BUGGY_SOURCE = """#include <iostream>
#include <string>
#include <vector>
#ifndef INCLUDE_LAST
#define INCLUDE_LAST 0
#endif
int row_sum(const std::vector<int>& row) {
    int result = 0;
    for (size_t i = 0; i + (INCLUDE_LAST ? 0 : 1) < row.size(); ++i) {
        result += row[i];
    }
    return result;
}
int main(int argc, char** argv) {
    std::vector<int> row;
    for (int i = 1; i < argc; ++i) row.push_back(std::stoi(argv[i]));
    std::cout << row_sum(row) << '\\n';
}
"""

FIXED_SOURCE = BUGGY_SOURCE.replace(
    "i + (INCLUDE_LAST ? 0 : 1) < row.size()", "i < row.size()")

MAKEFILE = """CXX := c++
CPPFLAGS := -DINCLUDE_LAST=0

app: main.o Makefile
	$(CXX) main.o -o app
	@printf 'linked\\n' >> link-events.txt

main.o: main.cpp Makefile
	$(CXX) -std=c++17 $(CPPFLAGS) -c main.cpp -o main.o
"""


def candidate_patch(name, before, after):
    return "diff --git a/{0} b/{0}\n".format(name) + "".join(
        difflib.unified_diff(before.splitlines(keepends=True),
                             after.splitlines(keepends=True),
                             fromfile="a/" + name,
                             tofile="b/" + name if after else "/dev/null"))


@unittest.skipUnless(shutil.which("c++") and shutil.which("make"), "C++ and make are required")
class BaselineEvaluationBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="opbench-baseline-eval-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.task_file = make_task(self.root / "task")
        self.source = self.task_file.parent / "source"
        self.grader_visits = self.root / "grader-visits.jsonl"
        (self.source / "main.cpp").write_text(BUGGY_SOURCE)
        (self.source / "Makefile").write_text(MAKEFILE)
        # This trusted development-only marker observes whether the grader ran;
        # neither baseline preparation nor a failed build may invoke it.
        (self.task_file.parent / "grader/check.py").write_text(
            "import json, pathlib, subprocess, sys\n"
            f"visits = pathlib.Path({str(self.grader_visits)!r})\n"
            "workspace, case = pathlib.Path(sys.argv[1]), sys.argv[2]\n"
            "with visits.open('a') as stream:\n"
            "    stream.write(json.dumps({'case': case}) + '\\n')\n"
            "arguments = ['4', '5'] if case == 'rows' else []\n"
            "value = int(subprocess.check_output([str(workspace / 'app'), *arguments], text=True, timeout=5))\n"
            "links = len((workspace / 'link-events.txt').read_text().splitlines())\n"
            "print(json.dumps({'case': case, 'value': value, 'links': links}), flush=True)\n"
            "assert value == (9 if case == 'rows' else 0), 'incorrect compiled operator result'\n")
        raw = json.loads(self.task_file.read_text())
        raw["statement"] = "Sum all row entries and preserve an empty row's zero result."
        raw["environment"].update(build=[[shutil.which("make")]], build_timeout_sec=30)
        self.task_file.write_text(json.dumps(raw))
        self.task = TaskSpec.load(self.task_file)
        self.artifact = self.root / "baseline"

    def build(self):
        result = build_baseline(self.task, self.artifact)
        self.assertEqual("ready", result["status"], result)
        self.assertFalse(self.grader_visits.exists())
        return result

    def evaluate(self, patch_text, name):
        output = self.root / name
        result = PatchEvaluator().evaluate(
            self.task, patch_text, output, baseline_artifact=self.artifact)
        self.assertEqual(result, json.loads((output / "result.json").read_text()))
        return result, output

    def observed(self, result, output, case="rows"):
        record, = [record for record in result["cases"] if record["id"] == case]
        lines = (output / record["log"]).read_text().splitlines()
        # Command logs may also contain diagnostics or an assertion traceback.
        values = [json.loads(line) for line in lines if line.startswith('{"case":')]
        self.assertEqual(1, len(values), lines)
        return values[0]

    def artifact_value(self):
        return int(subprocess.check_output(
            [str(self.artifact / "workspace/app"), "4", "5"], text=True, timeout=5))

    def test_cached_baseline_runs_without_relink_and_each_candidate_rebuilds_independently(self):
        self.build()
        self.assertEqual(4, self.artifact_value())
        baseline, output = self.evaluate("", "unchanged")
        self.assertEqual("test_failed", baseline["status"], baseline)
        self.assertEqual({"passed": 1, "total": 1}, baseline["groups"]["pass_to_pass"])
        self.assertEqual({"case": "rows", "value": 4, "links": 1}, self.observed(baseline, output))

        reference = candidate_patch("main.cpp", BUGGY_SOURCE, FIXED_SOURCE)
        alternative = candidate_patch("Makefile", MAKEFILE,
                                      MAKEFILE.replace("-DINCLUDE_LAST=0", "-DINCLUDE_LAST=1"))
        for name, patch_text in (("reference", reference), ("alternative", alternative)):
            with self.subTest(candidate=name):
                result, output = self.evaluate(patch_text, name)
                self.assertEqual("resolved", result["status"], result)
                self.assertEqual({"case": "rows", "value": 9, "links": 2}, self.observed(result, output))
                self.assertEqual(4, self.artifact_value())
                # A later empty candidate must still start from the buggy
                # compiled baseline, with no source/flag/binary contamination.
                unchanged, output = self.evaluate("", name + "-then-empty")
                self.assertEqual("test_failed", unchanged["status"], unchanged)
                self.assertEqual({"case": "rows", "value": 4, "links": 1}, self.observed(unchanged, output))

    def test_deleting_required_source_fails_build_instead_of_grading_cached_binary(self):
        self.build()
        result, output = self.evaluate(candidate_patch("main.cpp", BUGGY_SOURCE, ""), "deleted-source")
        self.assertEqual("build_failed", result["status"], result)
        self.assertEqual("build", result["error"]["stage"])
        self.assertFalse(result["resolved"])
        self.assertTrue(all(case["status"] == "not_run" for case in result["cases"]))
        self.assertFalse(self.grader_visits.exists())
        build_logs = [output / stage["log"] for stage in result["stages"] if stage["stage"] == "build"]
        self.assertTrue(build_logs)
        self.assertIn("main.cpp", "\n".join(path.read_text() for path in build_logs))
        self.assertEqual(4, self.artifact_value())

    def test_failed_baseline_cannot_be_reused_or_silently_fall_back_to_grading(self):
        broken = BUGGY_SOURCE + "\n#error deliberate baseline compilation failure\n"
        (self.source / "main.cpp").write_text(broken)
        baseline = build_baseline(self.task, self.artifact)
        self.assertEqual("failed", baseline["status"], baseline)
        self.assertEqual("build", baseline["error"]["stage"])
        self.assertFalse(self.grader_visits.exists())
        # This patch could repair a fresh build. Explicitly selecting an invalid
        # artifact must still refuse reuse, rather than hiding baseline failure.
        result, _ = self.evaluate(candidate_patch("main.cpp", broken, FIXED_SOURCE), "rejected-baseline")
        self.assertFalse(result["resolved"])
        self.assertNotIn(result["status"], {"resolved", "test_failed"})
        self.assertIsNotNone(result["error"])
        self.assertIn("baseline", result["error"]["message"].lower())
        self.assertTrue(all(case["status"] == "not_run" for case in result["cases"]))
        self.assertFalse(self.grader_visits.exists())


if __name__ == "__main__":
    unittest.main()
