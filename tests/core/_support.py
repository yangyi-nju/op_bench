import json
from pathlib import Path
import sys


def make_task(root: Path) -> Path:
    source, grader = root / "source", root / "grader"
    source.mkdir(parents=True)
    grader.mkdir()
    (source / "operator_impl.py").write_text("def row_sum(rows):\n    return [sum(rows[0]) for row in rows]\n")
    (grader / "check.py").write_text(
        "import sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from operator_impl import row_sum\n"
        "if sys.argv[2] == 'rows':\n"
        "    assert row_sum([[1, 2], [4, 5]]) == [3, 9]\n"
        "elif sys.argv[2] == 'single':\n"
        "    assert row_sum([[2, 3]]) == [5]\n"
        "else:\n"
        "    raise ValueError('unknown case')\n")
    task = {"schema_version": 2, "task_id": "row-sum", "statement": "Compute each row's own sum. Preserve single-row behavior.",
            "task_revision": "1", "scoring_revision": "1", "scope": "operator",
            "defect_group": "fixture:row-indexing",
            "source": {"path": "source"}, "environment": {"backend": "local", "python": sys.executable,
                "environment_id": "fixture-stdlib-local", "revision": "1"},
            "grader_dir": "grader", "public_commands": [],
            "tests": [{"id": case, "group": group, "argv": ["{python}", "{grader}/check.py", "{workspace}", case]}
                      for case, group in [("rows", "fail_to_pass"), ("single", "pass_to_pass")]]}
    (root / "task.json").write_text(json.dumps(task))
    (root / "dataset.json").write_text(json.dumps({"schema_version": 1, "dataset_id": "fixture", "version": "1", "tasks": ["task.json"]}))
    return root / "task.json"


CORRECT_PATCH = """diff --git a/operator_impl.py b/operator_impl.py
--- a/operator_impl.py
+++ b/operator_impl.py
@@ -1,2 +1,2 @@
 def row_sum(rows):
-    return [sum(rows[0]) for row in rows]
+    return [sum(row) for row in rows]
"""


def decide(name, **arguments):
    """A deterministic model decision; the real harness still executes the tool."""
    return {"content": "", "tool_calls": [{"id": "call-1", "name": name, "arguments": arguments}]}


class FakeModel:
    def __init__(self, *decisions):
        self.decisions = iter(decisions)

    def complete(self, messages, tools, timeout_sec):
        return next(self.decisions)
