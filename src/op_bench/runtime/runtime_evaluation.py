from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import subprocess

from op_bench.runtime.backends import (
    RuntimeAttemptContext,
    RuntimeBackend,
    RuntimeBackendUnavailable,
)
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import (
    EvaluationSpec,
    RuntimeProfile,
    TestExecutionSummary,
)
from op_bench.runtime.evaluation import (
    EvaluationInfrastructureError,
    PrivateEvaluationEvidence,
    SelectorExecution,
    StrictPatchApplyError,
)
from op_bench.runtime.local_evaluation import (
    EvaluationOnlyTestAsset,
    LocalGitSource,
    _selector_map,
    structured_unittest_summary,
)
from op_bench.runtime.validation import ContractError, require_str
from op_bench.runtime.workspace import FrozenPatch


_WRITE_BYTES_PROGRAM = (
    "import base64,pathlib,sys;"
    "p=pathlib.Path(sys.argv[1]);"
    "p.parent.mkdir(parents=True,exist_ok=True);"
    "p.write_bytes(base64.b64decode(sys.argv[2],validate=True))"
)
_READ_BYTES_PROGRAM = (
    "import pathlib,sys;"
    "sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
)
_PYTHON_OVERLAY_PROGRAM = r"""
import importlib
import json
import os
import pathlib
import shutil
import sys

cfg = json.loads(sys.argv[1])
workspace = pathlib.Path(".").resolve()
runtime_site = pathlib.Path("/tmp/op_bench_runtime/site-packages")
package = cfg["package"]
paths = cfg["paths"]
runtime_site.mkdir(parents=True, exist_ok=True)
os.chdir("/tmp")
installed = pathlib.Path(importlib.import_module(package).__file__).resolve().parent
destination_package = runtime_site / package
if not destination_package.exists():
    shutil.copytree(installed, destination_package, symlinks=True)
libs = installed.parent / f"{package}.libs"
if libs.exists() and not (runtime_site / libs.name).exists():
    shutil.copytree(libs, runtime_site / libs.name, symlinks=True)
for relative in paths:
    pure = pathlib.PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != package:
        raise ValueError("invalid overlay path")
    source = workspace.joinpath(*pure.parts)
    target = runtime_site.joinpath(*pure.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
print(json.dumps({"mode": "python_overlay", "overlay_count": len(paths)}, sort_keys=True))
""".strip()
_RUN_SCRIPT_WITH_PATH_PROGRAM = (
    "import runpy,sys;"
    "overlay=sys.argv[1];script=sys.argv[2];"
    "sys.path.insert(0,overlay);"
    "sys.argv=sys.argv[2:];"
    "runpy.run_path(script,run_name='__main__')"
)
_INPLACE_BUILD_COMMAND = (
    "set -o pipefail; "
    "test -f setup.py || { echo 'setup.py missing' >&2; exit 2; }; "
    "export MAX_JOBS=${MAX_JOBS:-8}; "
    "{python} setup.py develop --no-deps"
)


class RuntimeFreshEvaluationBackend:
    """Fresh evaluator that executes only through an exact runtime lease."""

    def __init__(
        self,
        *,
        source: LocalGitSource,
        hidden_asset: EvaluationOnlyTestAsset,
        python_executable: str,
        runtime_backend: RuntimeBackend,
        runtime_profile: RuntimeProfile,
        attempt_context: RuntimeAttemptContext,
        source_overlay_paths: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(source, LocalGitSource):
            raise ContractError("source: expected LocalGitSource")
        if not isinstance(hidden_asset, EvaluationOnlyTestAsset):
            raise ContractError("hidden_asset: expected EvaluationOnlyTestAsset")
        self.python_executable = require_str(
            python_executable,
            "python_executable",
        )
        for method in ("prepare", "run", "collect", "cleanup"):
            if not callable(getattr(runtime_backend, method, None)):
                raise ContractError("runtime_backend: expected RuntimeBackend")
        if not isinstance(runtime_profile, RuntimeProfile):
            raise ContractError("runtime_profile: expected RuntimeProfile")
        if not isinstance(attempt_context, RuntimeAttemptContext):
            raise ContractError("attempt_context: expected RuntimeAttemptContext")
        if runtime_profile.content_hash != attempt_context.runtime_profile_hash:
            raise ContractError("runtime profile/context identity mismatch")
        self.source = source
        self.hidden_asset = hidden_asset
        self.runtime_backend = runtime_backend
        self.runtime_profile = runtime_profile
        self.attempt_context = attempt_context
        if not isinstance(source_overlay_paths, tuple):
            raise ContractError("source_overlay_paths: expected tuple")
        normalized_paths: list[str] = []
        for index, value in enumerate(source_overlay_paths):
            selected = require_str(value, f"source_overlay_paths[{index}]")
            pure = PurePosixPath(selected)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise ContractError(
                    f"source_overlay_paths[{index}]: expected safe relative path"
                )
            normalized_paths.append(str(pure))
        self.source_overlay_paths = tuple(normalized_paths)
        if (
            runtime_profile.source_loading_mode == "python_overlay"
            and not self.source_overlay_paths
        ):
            raise ContractError(
                "source_overlay_paths: required for python_overlay Runtime Profile"
            )
        self.last_cleanup_result = None
        self.last_backend_unavailable_reason: str | None = None

    def evaluate(
        self,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch,
    ) -> PrivateEvaluationEvidence:
        if not isinstance(frozen_patch, FrozenPatch):
            raise ContractError("frozen_patch: expected FrozenPatch")
        return self._evaluate(spec, frozen_patch)

    def evaluate_replay(
        self,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
    ) -> PrivateEvaluationEvidence:
        if frozen_patch is not None and not isinstance(frozen_patch, FrozenPatch):
            raise ContractError("frozen_patch: expected FrozenPatch or None")
        return self._evaluate(spec, frozen_patch)

    def _evaluate(
        self,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch | None,
    ) -> PrivateEvaluationEvidence:
        if not isinstance(spec, EvaluationSpec):
            raise ContractError("spec: expected EvaluationSpec")
        if spec.attempt_id != self.attempt_context.attempt_id:
            raise EvaluationInfrastructureError("attempt_binding_mismatch")
        if spec.runtime != self.runtime_profile:
            raise EvaluationInfrastructureError("runtime_profile_binding_mismatch")
        if spec.source != self.source.identity:
            raise EvaluationInfrastructureError("source_binding_mismatch")
        if frozen_patch is not None and frozen_patch.source != self.source.identity:
            raise EvaluationInfrastructureError("source_binding_mismatch")
        if spec.hidden_test_asset != self.hidden_asset.identity:
            raise EvaluationInfrastructureError("hidden_test_binding_mismatch")

        self.last_backend_unavailable_reason = None
        lease = None
        evidence: PrivateEvaluationEvidence | None = None
        try:
            try:
                lease = self.runtime_backend.prepare(
                    self.runtime_profile,
                    self.attempt_context,
                )
                self._verify_source(lease, spec.timeout_ms)
                if frozen_patch is not None:
                    self._apply_patch(
                        lease,
                        ".opbench/agent.patch",
                        frozen_patch.patch_bytes,
                        timeout_ms=spec.timeout_ms,
                        agent_patch=True,
                    )
                self._apply_patch(
                    lease,
                    ".opbench/hidden.patch",
                    self.hidden_asset.patch_bytes,
                    timeout_ms=spec.timeout_ms,
                    agent_patch=False,
                )
                self._write_runtime_file(
                    lease,
                    ".opbench/_unittest_runner.py",
                    Path(__file__).with_name("_unittest_runner.py").read_bytes(),
                    spec.timeout_ms,
                )
                self._prepare_source_loading(lease, spec.timeout_ms)
                selectors = _selector_map(spec, self.hidden_asset)
                executions: list[SelectorExecution] = []
                for group, selector_ids in (
                    ("fail_to_pass", spec.fail_to_pass),
                    ("pass_to_pass", spec.pass_to_pass),
                ):
                    for selector_id in selector_ids:
                        try:
                            selector = selectors[selector_id]
                        except KeyError as exc:
                            raise EvaluationInfrastructureError(
                                "selector_not_registered"
                            ) from exc
                        executions.append(
                            self._run_selector(
                                lease,
                                selector.selector_id,
                                selector.command_template,
                                group,
                                spec.timeout_ms,
                            )
                        )
                evidence = PrivateEvaluationEvidence(
                    source=self.source.identity,
                    patch=(None if frozen_patch is None else frozen_patch.patch),
                    hidden_test_asset=self.hidden_asset.identity,
                    selectors=tuple(executions),
                    cleanup_completed=True,
                )
            except RuntimeBackendUnavailable as exc:
                self.last_backend_unavailable_reason = exc.reason_code
                raise EvaluationInfrastructureError(exc.reason_code) from exc
        finally:
            if lease is not None:
                cleanup = self.runtime_backend.cleanup(lease)
                self.last_cleanup_result = cleanup
                if not cleanup.report.all_released:
                    raise EvaluationInfrastructureError(
                        "evaluation_cleanup_failed"
                    )
        if evidence is None:
            raise EvaluationInfrastructureError("runtime_evaluation_failed")
        return evidence

    def _verify_source(self, lease, timeout_ms: int) -> None:
        workspace = self._controller_workspace(lease)
        try:
            head = _controller_git(
                workspace,
                "rev-parse",
                "--verify",
                "HEAD",
                timeout_ms=timeout_ms,
            )
            archive = _controller_git(
                workspace,
                "archive",
                "--format=tar",
                "HEAD",
                timeout_ms=timeout_ms,
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationInfrastructureError("evaluation_timeout") from exc
        if head.returncode != 0 or archive.returncode != 0:
            raise EvaluationInfrastructureError(
                "source_identity_verification_failed"
            )
        if head.stdout.decode("ascii").strip() != self.attempt_context.frozen_source_revision:
            raise EvaluationInfrastructureError("source_identity_mismatch")
        observed_digest = "sha256:" + hashlib.sha256(archive.stdout).hexdigest()
        if (
            self.source.identity.digest_kind == "content_sha256"
            and observed_digest != self.source.identity.digest
        ):
            raise EvaluationInfrastructureError("source_identity_mismatch")
        if self.source.identity.digest_kind not in {
            "content_sha256",
            "canonical_config",
        }:
            raise EvaluationInfrastructureError("source_identity_kind_unsupported")

    def _controller_workspace(self, lease) -> Path:
        workspaces = tuple(
            handle for handle in lease.handles if handle.resource_type == "workspace"
        )
        if len(workspaces) != 1:
            raise EvaluationInfrastructureError("source_identity_verification_failed")
        workspace = Path(workspaces[0].raw_handle)
        if workspace.is_symlink() or not workspace.is_dir():
            raise EvaluationInfrastructureError("source_identity_verification_failed")
        try:
            boundary = self.attempt_context.target_binding.local_workspace_parent.resolve(
                strict=True
            )
            resolved = workspace.resolve(strict=True)
            resolved.relative_to(boundary)
        except (OSError, ValueError) as exc:
            raise EvaluationInfrastructureError(
                "source_identity_verification_failed"
            ) from exc
        return resolved

    def _apply_patch(
        self,
        lease,
        path: str,
        raw: bytes,
        *,
        timeout_ms: int,
        agent_patch: bool,
    ) -> None:
        workspace = self._controller_workspace(lease)
        for check_only in (True, False):
            arguments = ["apply"]
            if check_only:
                arguments.append("--check")
            arguments.extend(("--whitespace=nowarn", "-"))
            try:
                result = _controller_git(
                    workspace,
                    *arguments,
                    input_bytes=raw,
                    timeout_ms=timeout_ms,
                )
            except subprocess.TimeoutExpired as exc:
                raise EvaluationInfrastructureError("evaluation_timeout") from exc
            if result.returncode == 0:
                continue
            detail = result.stderr.decode("utf-8", errors="replace")
            if agent_patch:
                raise StrictPatchApplyError(detail)
            raise EvaluationInfrastructureError(
                "hidden_test_injection_failed"
            )

    def _write_runtime_file(
        self,
        lease,
        path: str,
        raw: bytes,
        timeout_ms: int,
    ) -> None:
        encoded = base64.b64encode(raw).decode("ascii")
        result = self.runtime_backend.run(
            lease,
            (
                self.python_executable,
                "-I",
                "-c",
                _WRITE_BYTES_PROGRAM,
                path,
                encoded,
            ),
            ".",
            timeout_ms,
        )
        if result.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        if result.exit_code != 0:
            raise EvaluationInfrastructureError("runtime_artifact_write_failed")

    def _run_selector(
        self,
        lease,
        selector_id: str,
        command_template: str,
        group: str,
        timeout_ms: int,
    ) -> SelectorExecution:
        try:
            template = tuple(shlex.split(command_template))
        except ValueError as exc:
            raise EvaluationInfrastructureError(
                "invalid_test_command_template"
            ) from exc
        if template == ("{python}", "-m", "unittest", "{test}"):
            return self._run_unittest_selector(
                lease,
                selector_id,
                group,
                timeout_ms,
            )
        if not template or template[0] != "{python}":
            raise EvaluationInfrastructureError("invalid_test_command_template")
        command = tuple(
            self.python_executable
            if part == "{python}"
            else selector_id
            if part == "{test}"
            else part
            for part in template
        )
        if "{test}" not in template or any("{" in part or "}" in part for part in command):
            raise EvaluationInfrastructureError("invalid_test_command_template")
        command = self._runtime_selector_command(command)
        result = self.runtime_backend.run(lease, command, ".", timeout_ms)
        if result.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        passed = result.exit_code == 0
        return SelectorExecution(
            selector_id=selector_id,
            group=group,
            command_digest=canonical_sha256(
                {"runner": "exact-selector-command-v1", "command": list(command)}
            ),
            exit_code=result.exit_code,
            timed_out=False,
            summary=TestExecutionSummary(
                collected=1,
                executed=1,
                passed=1 if passed else 0,
                failed=0 if passed else 1,
                skipped=0,
            ),
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _run_unittest_selector(
        self,
        lease,
        selector_id: str,
        group: str,
        timeout_ms: int,
    ) -> SelectorExecution:
        result_name = hashlib.sha256(selector_id.encode("utf-8")).hexdigest()
        result_path = f".opbench/result-{result_name}.json"
        result = self.runtime_backend.run(
            lease,
            (
                self.python_executable,
                "-I",
                ".opbench/_unittest_runner.py",
                "--workspace",
                ".",
                "--selector",
                selector_id,
                "--result",
                result_path,
            ),
            ".",
            timeout_ms,
        )
        if result.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        evidence = self.runtime_backend.run(
            lease,
            (
                self.python_executable,
                "-I",
                "-c",
                _READ_BYTES_PROGRAM,
                result_path,
            ),
            ".",
            timeout_ms,
        )
        if evidence.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        if evidence.exit_code != 0:
            raise EvaluationInfrastructureError("test_runner_evidence_missing")
        summary = structured_unittest_summary(
            evidence.stdout.encode("utf-8"),
            exit_code=result.exit_code,
        )
        return SelectorExecution(
            selector_id=selector_id,
            group=group,
            command_digest=canonical_sha256(
                {
                    "runner": "opbench-unittest-runner-v1",
                    "selector": selector_id,
                }
            ),
            exit_code=result.exit_code,
            timed_out=False,
            summary=summary,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    def _prepare_source_loading(self, lease, timeout_ms: int) -> None:
        mode = self.runtime_profile.source_loading_mode
        if mode == "none":
            return
        if mode == "python_overlay":
            top_levels = {PurePosixPath(path).parts[0] for path in self.source_overlay_paths}
            if len(top_levels) != 1:
                raise EvaluationInfrastructureError("source_overlay_package_ambiguous")
            command = (
                self.python_executable,
                "-I",
                "-c",
                _PYTHON_OVERLAY_PROGRAM,
                json.dumps(
                    {
                        "package": next(iter(top_levels)),
                        "paths": list(self.source_overlay_paths),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            )
        elif mode == "inplace_build":
            command = (
                "bash",
                "-lc",
                _INPLACE_BUILD_COMMAND.replace(
                    "{python}",
                    shlex.quote(self.python_executable),
                ),
            )
        else:
            raise EvaluationInfrastructureError("source_loading_mode_unsupported")
        result = self.runtime_backend.run(lease, command, ".", timeout_ms)
        if result.timed_out:
            raise EvaluationInfrastructureError("evaluation_timeout")
        if result.exit_code != 0:
            raise EvaluationInfrastructureError("source_loading_failed")

    def _runtime_selector_command(
        self,
        command: tuple[str, ...],
    ) -> tuple[str, ...]:
        if self.runtime_profile.source_loading_mode != "python_overlay":
            return command
        if len(command) < 2 or command[0] != self.python_executable:
            raise EvaluationInfrastructureError("invalid_overlay_test_command")
        script = command[1]
        if script.startswith("-"):
            raise EvaluationInfrastructureError("invalid_overlay_test_command")
        return (
            self.python_executable,
            "-I",
            "-c",
            _RUN_SCRIPT_WITH_PATH_PROGRAM,
            "/tmp/op_bench_runtime/site-packages",
            script,
            *command[2:],
        )


def _controller_git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    timeout_ms: int,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        (
            "git",
            "-c",
            "core.autocrlf=false",
            "-C",
            str(repository),
            *arguments,
        ),
        check=False,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_ms / 1000,
        env={
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "LANG": "C",
        },
    )


__all__ = ["RuntimeFreshEvaluationBackend"]
