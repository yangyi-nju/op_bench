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
    def task_json_path(self) -> Path:
        return self.task_dir / "task.json"

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
    def source_snapshot_path(self) -> Path | None:
        value = self.data["source"].get("snapshot_path")
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.task_dir / path).resolve()
        return path

    @property
    def source_snapshot_hash(self) -> str | None:
        value = self.data["source"].get("snapshot_hash")
        return str(value) if value else None

    @property
    def source_snapshot_method(self) -> str | None:
        value = self.data["source"].get("snapshot_method")
        return str(value) if value else None

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

    @property
    def environment_backend(self) -> str:
        return str(self.data["environment"].get("backend", "local"))

    @property
    def environment_image(self) -> str:
        return str(self.data["environment"].get("image", ""))

    @property
    def environment_image_digest(self) -> str | None:
        value = self.data["environment"].get("image_digest")
        return str(value) if value else None

    @property
    def environment_digest_kind(self) -> str | None:
        value = self.data["environment"].get("digest_kind")
        return str(value) if value else None

    @property
    def environment_platform(self) -> str | None:
        value = self.data["environment"].get("platform")
        return str(value) if value else None

    @property
    def environment_workspace_dir(self) -> str:
        return str(self.data["environment"].get("workspace_dir", "/workspace"))

    @property
    def environment_preflight_workdir(self) -> str:
        return str(self.data["environment"].get("preflight_workdir", "/tmp"))

    @property
    def environment_preflight_commands(self) -> list[str]:
        return list(self.data["environment"].get("preflight_commands", []))

    @property
    def environment_python_executable(self) -> str:
        if self.environment_backend == "docker":
            return str(self.data["environment"].get("python_executable", "python"))
        return sys.executable

    @property
    def environment_dockerfile_path(self) -> Path | None:
        value = self.data["environment"].get("dockerfile")
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.task_dir / path).resolve()
        return path

    @property
    def environment_build_context_path(self) -> Path | None:
        value = self.data["environment"].get("build_context")
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (self.task_dir / path).resolve()
        return path

    @property
    def source_loading(self) -> dict[str, Any]:
        value = self.data["environment"].get("source_loading")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def source_loading_mode(self) -> str | None:
        value = self.source_loading.get("mode")
        return str(value) if value else None

    @property
    def source_loading_overlay_paths(self) -> list[str]:
        value = self.source_loading.get("overlay_paths", [])
        return [str(path) for path in value] if isinstance(value, list) else []

    @property
    def metadata_layer(self) -> str | None:
        value = self.data["metadata"].get("layer")
        return str(value) if value else None

    @property
    def metadata_admission_status(self) -> str | None:
        value = self.data["metadata"].get("admission_status")
        return str(value) if value else None

    @property
    def metadata_source_loading_verified(self) -> bool | None:
        value = self.data["metadata"].get("source_loading_verified")
        return value if isinstance(value, bool) else None

    def render_command(self, command: str, python_executable: str | None = None) -> list[str]:
        rendered = command.replace("{python}", shlex.quote(python_executable or self.environment_python_executable))
        return shlex.split(rendered)

    def command_for_test(self, test_name: str, python_executable: str | None = None) -> list[str]:
        template = str(self.data["evaluation"]["test_command"])
        if "{test}" in template:
            rendered = template.replace("{test}", test_name)
        else:
            rendered = f"{template} {test_name}"
        return self.render_command(rendered, python_executable=python_executable)
