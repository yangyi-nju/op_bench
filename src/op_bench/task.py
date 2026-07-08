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
    def source_ref(self) -> str | None:
        value = self.data.get("source_ref")
        return str(value) if value else None

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
        return self.hidden_test_patch_path

    @property
    def hidden_test_patch_path(self) -> Path:
        artifacts = self.data["artifacts"]
        if "hidden_test_patch" in artifacts:
            return self.task_dir / artifacts["hidden_test_patch"]
        return self.task_dir / artifacts["test_patch"]

    @property
    def public_test_patch_path(self) -> Path | None:
        value = self.data["artifacts"].get("public_test_patch")
        if not value:
            return None
        return self.task_dir / value

    @property
    def public_tests(self) -> list[str]:
        return list(self.data["evaluation"].get("public_tests", []))

    @property
    def patch_scope_paths(self) -> list[str]:
        scope = self.data.get("patch_scope")
        if not isinstance(scope, dict):
            return []
        return [str(p) for p in scope.get("allowed_paths", [])]

    @property
    def patch_scope_mode(self) -> str | None:
        scope = self.data.get("patch_scope")
        if not isinstance(scope, dict):
            return None
        return str(scope.get("mode", "enforced"))

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
    def build_timeout_sec(self) -> int:
        """Timeout for the source_loading build step (python_overlay sync or
        inplace_build compile). Falls back to `evaluation.build_timeout_sec`,
        then a tier-aware default: kernel_build gets 6 hours to accommodate
        first-time PyTorch CUDA compiles on modest hardware (V100 + 8 vCPU
        takes 3.5-5 hours; higher-core machines finish in 90-150 minutes).
        Other tiers use timeout_sec unchanged."""
        explicit = self.data.get("evaluation", {}).get("build_timeout_sec")
        if explicit is not None:
            return int(explicit)
        if self.runtime_tier == "cuda_kernel_build" or self.source_loading_mode == "inplace_build":
            return max(self.timeout_sec, 21600)  # 6 hours
        return self.timeout_sec

    @property
    def environment_backend(self) -> str:
        import os
        backend = str(self.data["environment"].get("backend", "local"))
        if backend == "remote_docker" and os.environ.get("OP_BENCH_FORCE_LOCAL_DOCKER") == "1":
            return "docker"
        return backend

    @property
    def environment_host(self) -> str | None:
        value = self.data["environment"].get("host")
        return str(value) if value else None

    @property
    def environment_gpus(self) -> str | None:
        env = self.data["environment"]
        gpus = env.get("gpus")
        if gpus:
            return str(gpus)
        if env.get("hardware", {}).get("requires_gpu"):
            return "all"
        return None

    @property
    def requires_gpu(self) -> bool:
        return bool(self.data["environment"].get("hardware", {}).get("requires_gpu", False))

    @property
    def environment_ref(self) -> str | None:
        value = self.data.get("environment_ref")
        return str(value) if value else None

    @property
    def runtime_tier(self) -> str:
        value = self.data.get("runtime_tier", self.data["environment"].get("tier", ""))
        return str(value)

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
        if self.environment_backend in {"docker", "remote_docker"}:
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

    @property
    def admission_status(self) -> str | None:
        admission = self.data.get("admission")
        if isinstance(admission, dict) and admission.get("status"):
            return str(admission["status"])
        return self.metadata_admission_status

    @property
    def admission_evidence_path(self) -> Path | None:
        admission = self.data.get("admission")
        if not isinstance(admission, dict) or not admission.get("evidence"):
            return None
        path = Path(str(admission["evidence"]))
        if not path.is_absolute():
            path = (self.task_dir / path).resolve()
        return path

    @property
    def admission_verified_at(self) -> str | None:
        admission = self.data.get("admission")
        if not isinstance(admission, dict) or not admission.get("verified_at"):
            return None
        return str(admission["verified_at"])

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
