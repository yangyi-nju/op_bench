from __future__ import annotations

import json
import shlex
import sys
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
    def repo_url(self) -> str:
        explicit = self.data["source"].get("repo_url")
        if explicit:
            return str(explicit)
        repo = str(self.data["source"]["repo"])
        if repo.startswith(("http://", "https://", "ssh://", "git@")):
            return repo
        return f"https://github.com/{repo}.git"

    @property
    def base_commit(self) -> str:
        return str(self.data["source"]["base_commit"])

    @property
    def issue_markdown_path(self) -> Path:
        return self.task_dir / "issue.md"

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
        rendered = rendered.replace("{python}", shlex.quote(sys.executable))
        return shlex.split(rendered)
