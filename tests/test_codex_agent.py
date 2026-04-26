from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from op_bench.agents import CodexAgent
from op_bench.task import TaskManifest


class CodexAgentTests(unittest.TestCase):
    def test_codex_agent_runs_cli_and_exports_git_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            workspace = root / "workspace"
            workspace.mkdir()
            self._run_git(["init"], workspace)
            self._run_git(["config", "user.name", "Test"], workspace)
            self._run_git(["config", "user.email", "test@example.com"], workspace)
            (workspace / "bug.py").write_text("value = 'bug'\n", encoding="utf-8")
            self._run_git(["add", "bug.py"], workspace)
            self._run_git(["commit", "-m", "base"], workspace)

            task_dir = root / "task"
            task_dir.mkdir()
            (task_dir / "issue.md").write_text("Fix bug.py", encoding="utf-8")
            (task_dir / "task.json").write_text(
                json.dumps(
                    {
                        "task_id": "codex_agent_fixture",
                        "version": "v1",
                        "source": {
                            "pr_url": "https://github.com/local/repo/pull/1",
                            "issue_url": "https://github.com/local/repo/issues/1",
                            "repo": "local/repo",
                            "issue_number": 1,
                            "pr_number": 1,
                            "base_commit": "localbase",
                            "merge_commit": "localmerge",
                            "checkout_mode": "local-copy",
                            "local_path": "../workspace",
                        },
                        "statement": {"title": "Fix bug", "body": "Fix bug.py", "labels": []},
                        "operator": {
                            "framework": "pytorch",
                            "component": "test",
                            "operator_name": "bug",
                            "problem_type": "operator-behavior",
                            "tags": [],
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
                            "allowed_test_commands": [],
                        },
                        "evaluation": {
                            "setup_commands": [],
                            "fail_to_pass": ["unused"],
                            "pass_to_pass": ["unused"],
                            "test_command": "{python} -m unittest {test}",
                            "timeout_sec": 30,
                        },
                        "artifacts": {"gold_patch": "artifacts/gold.patch", "test_patch": "artifacts/test.patch"},
                        "metadata": {
                            "difficulty": "easy",
                            "curation_status": "verified",
                            "deterministic": True,
                            "estimated_runtime_min": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_codex = fake_bin / "codex"
            fake_codex.write_text(
                "#!/bin/sh\n"
                "while [ \"$1\" != \"\" ]; do\n"
                "  if [ \"$1\" = \"--cd\" ]; then shift; cd \"$1\"; fi\n"
                "  shift || true\n"
                "done\n"
                "printf \"value = 'fixed'\\n\" > bug.py\n",
                encoding="utf-8",
            )
            fake_codex.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
            try:
                task = TaskManifest.load(task_dir / "task.json")
                output = CodexAgent().produce_patch(task, root / "agent-output", workspace=workspace)
            finally:
                os.environ["PATH"] = old_path

            patch = output.patch_path.read_text(encoding="utf-8")
            self.assertEqual(output.agent_name, "codex")
            self.assertIn("-value = 'bug'", patch)
            self.assertIn("+value = 'fixed'", patch)

    def _run_git(self, args: list[str], cwd: Path) -> None:
        import subprocess

        subprocess.run(["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


if __name__ == "__main__":
    unittest.main()
