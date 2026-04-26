# op_bench MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable MVP that evaluates `noop` and `gold` agents on a verified CPU-only operator smoke task, while preserving the environment-first architecture needed for real PyTorch tasks.

**Architecture:** Keep the existing PR builder intact and add focused runtime modules for task loading, local execution, patch evaluation, agent adapters, and reporting. The first executor uses local workspaces and local fixture repos; Docker/GPU execution stays behind the same executor boundary.

**Tech Stack:** Python standard library, `git apply`, `unittest`, JSON/JSONL, existing `src/op_bench` package layout.

---

## File Structure

- Create `tests/`: standard-library unit tests run with `python -m unittest discover tests`.
- Create `src/op_bench/task.py`: manifest loading, artifact path resolution, test command expansion.
- Create `src/op_bench/executor.py`: command execution, timeout handling, environment evidence collection.
- Create `src/op_bench/evaluator.py`: baseline/gold/agent patch evaluation and failure classification.
- Create `src/op_bench/agents.py`: `noop` and `gold` agent adapters.
- Create `src/op_bench/reporter.py`: JSONL writing and aggregate summaries.
- Create `scripts/run_experiment.py`: CLI to run agents on task directories.
- Create `fixtures/smoke_repo/`: small operator-like repository with an intentional NaN bug.
- Create `tasks/smoke/expit_nan_cpu/`: verified smoke task using hidden test and gold patch artifacts.
- Modify `schemas/task_manifest.schema.json`: allow local-copy source fields for smoke and offline MVP execution.
- Modify `scripts/validate_task.py`: validate the new local source fields and reject unresolved draft tests.
- Modify `README.md`: document the smoke experiment command and expected result.

User instruction: do not commit until the first runnable version passes. Each task ends with a checkpoint status command instead of a commit.

## Task 1: Task Model

**Files:**
- Create: `src/op_bench/task.py`
- Create: `tests/test_task_model.py`

- [ ] **Step 1: Write tests for manifest loading and command expansion**

```python
# tests/test_task_model.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from op_bench.task import TaskManifest


class TaskManifestTests(unittest.TestCase):
    def test_load_resolves_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifacts").mkdir()
            data = {
                "task_id": "smoke",
                "version": "v1",
                "source": {
                    "pr_url": "https://github.com/local/op-smoke/pull/1",
                    "issue_url": "https://github.com/local/op-smoke/issues/1",
                    "repo": "local/op-smoke",
                    "issue_number": 1,
                    "pr_number": 1,
                    "base_commit": "localbase",
                    "merge_commit": "localmerge",
                    "checkout_mode": "local-copy",
                    "local_path": "../../fixtures/smoke_repo",
                },
                "statement": {"title": "bug", "body": "body", "labels": []},
                "operator": {
                    "framework": "pytorch",
                    "component": "torch.special",
                    "operator_name": "torch.special.expit",
                    "problem_type": "numerical-semantics",
                    "tags": ["cpu"],
                },
                "environment": {
                    "tier": "cpu-deterministic",
                    "image": "local",
                    "python_version": "3",
                    "os": "local",
                    "build_mode": "editable-python",
                    "hardware": {"device": "cpu", "min_memory_gb": 1},
                    "dependencies": [],
                },
                "agent_visible": {
                    "repo_setup_commands": [],
                    "known_constraints": [],
                    "allowed_test_commands": ["python -m unittest {test}"],
                },
                "evaluation": {
                    "setup_commands": [],
                    "fail_to_pass": ["tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
                    "pass_to_pass": ["tests.test_special.TestSpecialExpit.test_regular_value"],
                    "test_command": "python -m unittest {test}",
                    "timeout_sec": 30,
                },
                "artifacts": {
                    "gold_patch": "artifacts/gold.patch",
                    "test_patch": "artifacts/test.patch",
                },
                "metadata": {
                    "difficulty": "easy",
                    "curation_status": "verified",
                    "deterministic": True,
                    "estimated_runtime_min": 1,
                },
            }
            (root / "task.json").write_text(json.dumps(data), encoding="utf-8")

            task = TaskManifest.load(root / "task.json")

            self.assertEqual(task.task_id, "smoke")
            self.assertEqual(task.task_dir, root)
            self.assertEqual(task.gold_patch_path, root / "artifacts/gold.patch")
            self.assertEqual(
                task.command_for_test("tests.test_special.TestSpecialExpit.test_nan_is_preserved"),
                ["python", "-m", "unittest", "tests.test_special.TestSpecialExpit.test_nan_is_preserved"],
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails because `op_bench.task` does not exist**

Run: `PYTHONPATH=src python3 -m unittest tests.test_task_model -v`

Expected: failure with `ModuleNotFoundError: No module named 'op_bench.task'`.

- [ ] **Step 3: Implement `TaskManifest`**

```python
# src/op_bench/task.py
from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskManifest:
    task_dir: Path
    data: dict[str, Any]

    @classmethod
    def load(cls, path: Path | str) -> "TaskManifest":
        manifest_path = Path(path).resolve()
        with manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(task_dir=manifest_path.parent, data=data)

    @property
    def task_id(self) -> str:
        return str(self.data["task_id"])

    @property
    def checkout_mode(self) -> str:
        return str(self.data["source"].get("checkout_mode", "git"))

    @property
    def local_source_path(self) -> Path | None:
        value = self.data["source"].get("local_path")
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.task_dir / path).resolve()
        return path

    @property
    def gold_patch_path(self) -> Path:
        return self.task_dir / self.data["artifacts"]["gold_patch"]

    @property
    def test_patch_path(self) -> Path:
        return self.task_dir / self.data["artifacts"]["test_patch"]

    @property
    def fail_to_pass_tests(self) -> list[str]:
        return list(self.data["evaluation"]["fail_to_pass"])

    @property
    def pass_to_pass_tests(self) -> list[str]:
        return list(self.data["evaluation"]["pass_to_pass"])

    @property
    def setup_commands(self) -> list[str]:
        return list(self.data["evaluation"].get("setup_commands", []))

    @property
    def timeout_sec(self) -> int:
        return int(self.data["evaluation"]["timeout_sec"])

    def command_for_test(self, test_name: str) -> list[str]:
        template = str(self.data["evaluation"]["test_command"])
        if "{test}" in template:
            rendered = template.replace("{test}", test_name)
        else:
            rendered = f"{template} {test_name}"
        return shlex.split(rendered)
```

- [ ] **Step 4: Run the test and verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_task_model -v`

Expected: `OK`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: new `src/op_bench/task.py` and `tests/test_task_model.py` are uncommitted.

## Task 2: Local Executor

**Files:**
- Create: `src/op_bench/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write executor tests**

```python
# tests/test_executor.py
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
```

- [ ] **Step 2: Run the test and verify it fails because `op_bench.executor` does not exist**

Run: `PYTHONPATH=src python3 -m unittest tests.test_executor -v`

Expected: failure with `ModuleNotFoundError: No module named 'op_bench.executor'`.

- [ ] **Step 3: Implement command execution and environment evidence**

```python
# src/op_bench/executor.py
from __future__ import annotations

import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    cwd: str
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnvironmentEvidence:
    executor: str
    python_executable: str
    python_version: str
    platform: str
    machine: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class LocalExecutor:
    name = "local"

    def run(self, command: list[str], cwd: Path, timeout_sec: int) -> CommandResult:
        start = time.monotonic()
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                check=False,
            )
            return CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                duration_sec=time.monotonic() - start,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                command=command,
                cwd=str(cwd),
                exit_code=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                duration_sec=time.monotonic() - start,
                timed_out=True,
            )

    def collect_environment(self) -> EnvironmentEvidence:
        return EnvironmentEvidence(
            executor=self.name,
            python_executable=sys.executable,
            python_version=sys.version.replace("\n", " "),
            platform=platform.platform(),
            machine=platform.machine(),
        )
```

- [ ] **Step 4: Run executor tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_executor -v`

Expected: `OK`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: executor files are uncommitted.

## Task 3: Smoke Task Bundle

**Files:**
- Create: `fixtures/smoke_repo/op_lib/__init__.py`
- Create: `fixtures/smoke_repo/op_lib/special.py`
- Create: `tasks/smoke/expit_nan_cpu/task.json`
- Create: `tasks/smoke/expit_nan_cpu/issue.md`
- Create: `tasks/smoke/expit_nan_cpu/artifacts/gold.patch`
- Create: `tasks/smoke/expit_nan_cpu/artifacts/test.patch`

- [ ] **Step 1: Create the intentionally broken operator fixture**

```python
# fixtures/smoke_repo/op_lib/__init__.py
from .special import expit

__all__ = ["expit"]
```

```python
# fixtures/smoke_repo/op_lib/special.py
from __future__ import annotations

import math


def expit(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))
```

- [ ] **Step 2: Create hidden test patch**

```diff
diff --git a/tests/test_special.py b/tests/test_special.py
new file mode 100644
index 0000000..4af1891
--- /dev/null
+++ b/tests/test_special.py
@@ -0,0 +1,18 @@
+from __future__ import annotations
+
+import math
+import unittest
+
+from op_lib import expit
+
+
+class TestSpecialExpit(unittest.TestCase):
+    def test_nan_is_preserved(self) -> None:
+        self.assertTrue(math.isnan(expit(float("nan"))))
+
+    def test_regular_value(self) -> None:
+        self.assertAlmostEqual(expit(0.0), 0.5)
+
+
+if __name__ == "__main__":
+    unittest.main()
```

- [ ] **Step 3: Create gold patch**

```diff
diff --git a/op_lib/special.py b/op_lib/special.py
index 96329ed..6bbda2a 100644
--- a/op_lib/special.py
+++ b/op_lib/special.py
@@ -5,6 +5,6 @@ import math
 
 
 def expit(value: float) -> float:
     if math.isnan(value):
-        return 0.0
+        return float("nan")
     return 1.0 / (1.0 + math.exp(-value))
```

- [ ] **Step 4: Create smoke task manifest**

Use `source.checkout_mode` as `local-copy` and `source.local_path` as `../../../fixtures/smoke_repo` from the task directory.

Run after writing: `python3 scripts/validate_task.py tasks/smoke/expit_nan_cpu/task.json`

Expected before schema updates: invalid because `checkout_mode` and `local_path` are not allowed.

- [ ] **Step 5: Checkpoint**

Run: `find tasks/smoke/expit_nan_cpu -maxdepth 3 -type f -print`

Expected: task manifest, issue, gold patch, and test patch are present.

## Task 4: Schema And Validator Updates

**Files:**
- Modify: `schemas/task_manifest.schema.json`
- Modify: `scripts/validate_task.py`

- [ ] **Step 1: Add failing validation coverage through command checks**

Run: `python3 scripts/validate_task.py tasks/smoke/expit_nan_cpu/task.json`

Expected: invalid before schema and validator accept local-copy source fields.

- [ ] **Step 2: Extend schema source properties**

In `schemas/task_manifest.schema.json`, add optional fields under `source.properties`:

```json
"checkout_mode": {
  "type": "string",
  "enum": ["git", "local-copy"]
},
"local_path": {
  "type": "string"
}
```

Keep existing required fields unchanged so current examples remain valid.

- [ ] **Step 3: Extend `validate_task.py`**

Add `ALLOWED_CHECKOUT_MODES = {"git", "local-copy"}`.

In `validate_manifest`, validate:

```python
    source = data.get("source", {})
    checkout_mode = source.get("checkout_mode", "git")
    if checkout_mode not in ALLOWED_CHECKOUT_MODES:
        errors.append(
            f"invalid source.checkout_mode: {checkout_mode!r}; expected one of {sorted(ALLOWED_CHECKOUT_MODES)}"
        )
    if checkout_mode == "local-copy" and not source.get("local_path"):
        errors.append("source.local_path is required when source.checkout_mode is 'local-copy'")
```

Reject unresolved generated test placeholders:

```python
    for field_name in ("fail_to_pass", "pass_to_pass"):
        try:
            tests = lookup(data, ("evaluation", field_name))
        except KeyError:
            continue
        draft_prefix = "TO" + "DO:"
        if any(str(test).startswith(draft_prefix) for test in tests):
            errors.append(f"evaluation.{field_name} contains unresolved draft test entries")
```

- [ ] **Step 4: Validate sample and smoke manifests**

Run: `python3 scripts/validate_task.py tasks/examples/sample_task.json`

Expected: `manifest looks valid`.

Run: `python3 scripts/validate_task.py tasks/smoke/expit_nan_cpu/task.json`

Expected: `manifest looks valid`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: schema, validator, and smoke task files are uncommitted.

## Task 5: Evaluator

**Files:**
- Create: `src/op_bench/evaluator.py`
- Create: `tests/test_evaluator.py`

- [ ] **Step 1: Write evaluator tests against the smoke task**

```python
# tests/test_evaluator.py
from __future__ import annotations

import unittest
from pathlib import Path

from op_bench.evaluator import Evaluator
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "tasks" / "smoke" / "expit_nan_cpu" / "task.json"


class EvaluatorTests(unittest.TestCase):
    def test_baseline_reproduces_failure(self) -> None:
        result = Evaluator().evaluate_baseline(TaskManifest.load(TASK_PATH))
        self.assertEqual(result.status, "baseline_reproduced")
        self.assertEqual(result.fail_to_pass_passed, 0)
        self.assertEqual(result.pass_to_pass_passed, 1)

    def test_gold_resolves_task(self) -> None:
        result = Evaluator().evaluate_gold(TaskManifest.load(TASK_PATH))
        self.assertEqual(result.status, "resolved")
        self.assertEqual(result.fail_to_pass_passed, 1)
        self.assertEqual(result.pass_to_pass_passed, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify failure because evaluator does not exist**

Run: `PYTHONPATH=src python3 -m unittest tests.test_evaluator -v`

Expected: failure with `ModuleNotFoundError: No module named 'op_bench.evaluator'`.

- [ ] **Step 3: Implement workspace preparation, patch application, and test execution**

Create `EvaluationResult` with:

- `task_id`
- `mode`
- `status`
- `fail_to_pass_total`
- `fail_to_pass_passed`
- `pass_to_pass_total`
- `pass_to_pass_passed`
- `duration_sec`
- `environment`
- `commands`

Implement `Evaluator` methods:

- `evaluate_baseline(task)`
- `evaluate_gold(task)`
- `evaluate_patch(task, patch_path, agent_name)`

Implementation rules:

- copy `task.local_source_path` into a temporary workspace
- apply `task.test_patch_path` before tests
- apply gold or agent patch after hidden tests are added
- run each fail-to-pass test independently with `task.command_for_test(test_name)`
- run each pass-to-pass test independently with `task.command_for_test(test_name)`
- classify baseline as `baseline_reproduced` when at least one fail-to-pass test fails and all pass-to-pass tests pass
- classify gold/agent as `resolved` only when all tests pass
- return `patch_apply_failed` if `git apply` returns non-zero
- return `timeout` if any command times out

- [ ] **Step 4: Run evaluator tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_evaluator -v`

Expected: `OK`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: evaluator files are uncommitted.

## Task 6: Agent Adapters And Reporter

**Files:**
- Create: `src/op_bench/agents.py`
- Create: `src/op_bench/reporter.py`
- Create: `tests/test_agents_reporter.py`

- [ ] **Step 1: Write tests for `noop`, `gold`, and summary aggregation**

```python
# tests/test_agents_reporter.py
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from op_bench.agents import GoldAgent, NoopAgent
from op_bench.reporter import summarize_results
from op_bench.task import TaskManifest


ROOT = Path(__file__).resolve().parents[1]
TASK = TaskManifest.load(ROOT / "tasks" / "smoke" / "expit_nan_cpu" / "task.json")


class AgentReporterTests(unittest.TestCase):
    def test_noop_agent_returns_empty_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = NoopAgent().produce_patch(TASK, Path(tmp))
            self.assertEqual(output.agent_name, "noop")
            self.assertEqual(output.patch_path.read_text(encoding="utf-8"), "")

    def test_gold_agent_returns_gold_patch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = GoldAgent().produce_patch(TASK, Path(tmp))
            self.assertEqual(output.agent_name, "gold")
            self.assertEqual(output.patch_path.read_text(encoding="utf-8"), TASK.gold_patch_path.read_text(encoding="utf-8"))

    def test_summarize_results_counts_resolved_rate(self) -> None:
        records = [
            {"agent": "gold", "status": "resolved", "duration_sec": 1.0},
            {"agent": "noop", "status": "fail_to_pass_failed", "duration_sec": 1.0},
        ]
        summary = summarize_results(records)
        self.assertEqual(summary["agents"]["gold"]["resolved_rate"], 1.0)
        self.assertEqual(summary["agents"]["noop"]["resolved_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run and verify failure because modules do not exist**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agents_reporter -v`

Expected: failure with missing modules.

- [ ] **Step 3: Implement adapters and reporter**

`NoopAgent` writes an empty patch to the provided output directory.

`GoldAgent` copies `task.gold_patch_path` to the output directory.

`summarize_results(records)` groups by `agent`, counts total, resolved count, failure reasons, and computes `resolved_rate`.

- [ ] **Step 4: Run tests**

Run: `PYTHONPATH=src python3 -m unittest tests.test_agents_reporter -v`

Expected: `OK`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: agent and reporter files are uncommitted.

## Task 7: Experiment CLI

**Files:**
- Create: `scripts/run_experiment.py`
- Create: `tests/test_run_experiment.py`

- [ ] **Step 1: Write CLI smoke test**

```python
# tests/test_run_experiment.py
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
                    "python3",
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
```

- [ ] **Step 2: Run and verify failure because CLI does not exist**

Run: `PYTHONPATH=src python3 -m unittest tests.test_run_experiment -v`

Expected: failure showing `scripts/run_experiment.py` cannot be opened.

- [ ] **Step 3: Implement `scripts/run_experiment.py`**

CLI behavior:

- accepts repeated `--task` directory arguments
- accepts repeated `--agent` names from `noop` and `gold`
- accepts `--output-dir`
- validates each task manifest first through in-process manifest loading
- runs `Evaluator.evaluate_baseline` once per task and records baseline result
- runs each selected agent and evaluates its patch
- writes `results.jsonl`
- writes `summary.json`
- prints the summary path

- [ ] **Step 4: Run CLI test**

Run: `PYTHONPATH=src python3 -m unittest tests.test_run_experiment -v`

Expected: `OK`.

- [ ] **Step 5: Checkpoint**

Run: `git status --short`

Expected: CLI and test files are uncommitted.

## Task 8: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document smoke experiment command**

Add a section:

```markdown
## MVP Smoke Experiment

Run the local smoke benchmark:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --task tasks/smoke/expit_nan_cpu \
  --agent noop \
  --agent gold \
  --output-dir /tmp/op_bench_smoke
```

Expected result:

- `gold` resolves the task
- `noop` fails the fail-to-pass test
- `/tmp/op_bench_smoke/results.jsonl` contains per-run records
- `/tmp/op_bench_smoke/summary.json` contains aggregate resolved rates
```

- [ ] **Step 2: Run all unit tests**

Run: `PYTHONPATH=src python3 -m unittest discover tests -v`

Expected: all tests pass.

- [ ] **Step 3: Validate manifests**

Run: `python3 scripts/validate_task.py tasks/examples/sample_task.json`

Expected: `manifest looks valid`.

Run: `python3 scripts/validate_task.py tasks/smoke/expit_nan_cpu/task.json`

Expected: `manifest looks valid`.

- [ ] **Step 4: Run the MVP smoke experiment manually**

Run:

```bash
PYTHONPATH=src python3 scripts/run_experiment.py \
  --task tasks/smoke/expit_nan_cpu \
  --agent noop \
  --agent gold \
  --output-dir /tmp/op_bench_smoke
```

Expected:

- command exits `0`
- `/tmp/op_bench_smoke/summary.json` exists
- `gold.resolved_rate` is `1.0`
- `noop.resolved_rate` is `0.0`

- [ ] **Step 5: Inspect final git status without committing**

Run: `git status --short`

Expected: all MVP files are uncommitted and ready for review. Do not commit until the user approves the first runnable version.

## Self-Review

Spec coverage:

- Task bundle: covered by Task 3 and Task 4.
- Environment evidence: covered by Task 2 and included in evaluator results in Task 5.
- Local executor boundary: covered by Task 2.
- Baseline/gold/agent evaluation modes: covered by Task 5 and Task 7.
- Failure taxonomy: covered by Task 5 result status rules.
- Agent adapters: covered by Task 6.
- Reporter: covered by Task 6 and Task 7.
- First experiment: covered by Task 7 and Task 8.

Implementation constraints:

- No commit steps are included because the user requested committing only after the first version runs.
- The first closed loop uses a local smoke task so the benchmark mechanics can be validated without network or PyTorch source-build risk.
- Real PyTorch issue replay remains the next dataset-expansion task after this MVP passes.
