from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import tempfile

from op_bench.runtime.canonical import canonical_json, canonical_sha256
from op_bench.runtime.contracts import ContentIdentity, EvaluationSpec, TestExecutionSummary, TestSelector
from op_bench.runtime.evaluation import (
    EvaluationInfrastructureError,
    PrivateEvaluationEvidence,
    SelectorExecution,
    StrictPatchApplyError,
)
from op_bench.runtime.validation import (
    ContractError,
    require_exact_fields,
    require_int,
    require_str,
)
from op_bench.runtime.workspace import FrozenPatch


@dataclass(frozen=True)
class LocalGitSource:
    identity: ContentIdentity
    repository: Path
    revision: str

    def __post_init__(self) -> None:
        _require_identity(self.identity, "source", "source")
        _require_local_git_repository(self.repository, "repository")
        require_str(self.revision, "revision")


@dataclass(frozen=True)
class EvaluationOnlyTestAsset:
    identity: ContentIdentity
    patch_bytes: bytes
    selectors: tuple[TestSelector, ...]

    def __post_init__(self) -> None:
        _require_identity(self.identity, "test", "identity")
        if not isinstance(self.patch_bytes, bytes):
            raise ContractError("patch_bytes: expected bytes")
        expected = "sha256:" + hashlib.sha256(self.patch_bytes).hexdigest()
        if self.identity.digest_kind != "content_sha256" or self.identity.digest != expected:
            raise ContractError("patch_bytes: does not match hidden test identity")
        if not isinstance(self.selectors, tuple):
            raise ContractError("selectors: expected tuple")
        seen: set[str] = set()
        for index, selector in enumerate(self.selectors):
            if not isinstance(selector, TestSelector):
                raise ContractError(f"selectors[{index}]: expected TestSelector")
            if selector.visibility not in {"hidden", "evaluation_only"}:
                raise ContractError(
                    f"selectors[{index}]: expected hidden or evaluation_only visibility"
                )
            if selector.selector_id in seen:
                raise ContractError(
                    f"selectors: duplicate selector {selector.selector_id!r}"
                )
            seen.add(selector.selector_id)


def git_archive_source_identity(
    repository: Path,
    revision: str,
    identifier: str,
    *,
    timeout_seconds: float | None = None,
) -> ContentIdentity:
    _require_local_git_repository(repository, "repository")
    require_str(revision, "revision")
    require_str(identifier, "identifier")
    result = subprocess.run(
        ("git", "-C", str(repository), "archive", "--format=tar", revision),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ContractError(f"source archive failed: {detail or 'git archive failed'}")
    return ContentIdentity(
        identity_type="source",
        identifier=identifier,
        digest="sha256:" + hashlib.sha256(result.stdout).hexdigest(),
        digest_kind="content_sha256",
    )


class LocalGitEvaluationBackend:
    def __init__(
        self,
        *,
        source: LocalGitSource,
        hidden_asset: EvaluationOnlyTestAsset,
        python_executable: str,
        workspace_parent: Path | None = None,
    ) -> None:
        if not isinstance(source, LocalGitSource):
            raise ContractError("source: expected LocalGitSource")
        if not isinstance(hidden_asset, EvaluationOnlyTestAsset):
            raise ContractError(
                "hidden_asset: expected EvaluationOnlyTestAsset"
            )
        require_str(python_executable, "python_executable")
        if workspace_parent is not None:
            if not isinstance(workspace_parent, Path):
                raise ContractError("workspace_parent: expected Path")
            if workspace_parent.is_symlink() or not workspace_parent.is_dir():
                raise ContractError("workspace_parent: expected real directory")
        self.source = source
        self.hidden_asset = hidden_asset
        self.python_executable = python_executable
        self.workspace_parent = workspace_parent

    def evaluate(
        self,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch,
    ) -> PrivateEvaluationEvidence:
        if not isinstance(spec, EvaluationSpec):
            raise ContractError("spec: expected EvaluationSpec")
        if not isinstance(frozen_patch, FrozenPatch):
            raise ContractError("frozen_patch: expected FrozenPatch")
        if spec.source != self.source.identity:
            raise EvaluationInfrastructureError("source_binding_mismatch")
        if spec.hidden_test_asset != self.hidden_asset.identity:
            raise EvaluationInfrastructureError("hidden_test_binding_mismatch")

        temporary = Path(
            tempfile.mkdtemp(
                prefix="opbench-evaluator-",
                dir=(
                    None
                    if self.workspace_parent is None
                    else str(self.workspace_parent)
                ),
            )
        )
        try:
            return self._evaluate_in(temporary, spec, frozen_patch)
        except (StrictPatchApplyError, EvaluationInfrastructureError):
            raise
        except subprocess.TimeoutExpired as exc:
            raise EvaluationInfrastructureError(
                "evaluation_timeout", str(exc)
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvaluationInfrastructureError("evaluator_error", str(exc)) from exc
        finally:
            try:
                shutil.rmtree(temporary)
            except OSError as exc:
                raise EvaluationInfrastructureError(
                    "evaluation_cleanup_failed", str(exc)
                ) from exc

    def _evaluate_in(
        self,
        temporary: Path,
        spec: EvaluationSpec,
        frozen_patch: FrozenPatch,
    ) -> PrivateEvaluationEvidence:
        workspace = temporary / "workspace"
        timeout_seconds = max(1.0, spec.timeout_ms / 1000.0)
        clone = _run(
            (
                "git",
                "clone",
                "--quiet",
                "--local",
                "--no-hardlinks",
                "--no-checkout",
                str(self.source.repository),
                str(workspace),
            ),
            cwd=temporary,
            timeout_seconds=timeout_seconds,
        )
        if clone.returncode != 0:
            raise EvaluationInfrastructureError(
                "source_materialization_failed", clone.stderr
            )
        checkout = _run(
            (
                "git",
                "-c",
                "advice.detachedHead=false",
                "-C",
                str(workspace),
                "checkout",
                "--detach",
                self.source.revision,
            ),
            cwd=temporary,
            timeout_seconds=timeout_seconds,
        )
        if checkout.returncode != 0:
            raise EvaluationInfrastructureError(
                "source_materialization_failed", checkout.stderr
            )
        remove_remote = _run(
            ("git", "-C", str(workspace), "remote", "remove", "origin"),
            cwd=temporary,
            timeout_seconds=timeout_seconds,
        )
        if remove_remote.returncode != 0:
            raise EvaluationInfrastructureError(
                "source_materialization_failed", remove_remote.stderr
            )

        try:
            observed_source = git_archive_source_identity(
                workspace,
                "HEAD",
                self.source.identity.identifier,
                timeout_seconds=timeout_seconds,
            )
        except ContractError as exc:
            raise EvaluationInfrastructureError(
                "source_identity_verification_failed",
                str(exc),
            ) from exc
        if observed_source != self.source.identity:
            raise EvaluationInfrastructureError("source_identity_mismatch")

        agent_patch = temporary / "frozen.patch"
        agent_patch.write_bytes(frozen_patch.patch_bytes)
        _strict_apply(
            workspace,
            agent_patch,
            timeout_seconds,
            agent_patch=True,
        )
        hidden_patch = temporary / "evaluation-only.patch"
        hidden_patch.write_bytes(self.hidden_asset.patch_bytes)
        _strict_apply(
            workspace,
            hidden_patch,
            timeout_seconds,
            agent_patch=False,
        )

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
                        selector,
                        group=group,
                        workspace=workspace,
                        timeout_seconds=timeout_seconds,
                    )
                )
        return PrivateEvaluationEvidence(
            source=observed_source,
            patch=frozen_patch.patch,
            hidden_test_asset=self.hidden_asset.identity,
            selectors=tuple(executions),
            cleanup_completed=True,
        )

    def _run_selector(
        self,
        selector: TestSelector,
        *,
        group: str,
        workspace: Path,
        timeout_seconds: float,
    ) -> SelectorExecution:
        try:
            template = tuple(shlex.split(selector.command_template))
        except ValueError as exc:
            raise EvaluationInfrastructureError(
                "invalid_test_command_template", str(exc)
            ) from exc
        if template != ("{python}", "-m", "unittest", "{test}"):
            raise EvaluationInfrastructureError("invalid_test_command_template")
        result_name = hashlib.sha256(selector.selector_id.encode("utf-8")).hexdigest()
        result_path = workspace.parent / f"unittest-result-{result_name}.json"
        runner = Path(__file__).with_name("_unittest_runner.py").resolve()
        command = (
            self.python_executable,
            "-I",
            str(runner),
            "--workspace",
            str(workspace),
            "--selector",
            selector.selector_id,
            "--result",
            str(result_path),
        )
        try:
            result = _run(
                command,
                cwd=workspace,
                timeout_seconds=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise EvaluationInfrastructureError(
                "evaluation_timeout", str(exc)
            ) from exc
        summary = _structured_unittest_summary(result, result_path)
        return SelectorExecution(
            selector_id=selector.selector_id,
            group=group,
            command_digest=canonical_sha256(
                {
                    "runner": "opbench-unittest-runner-v1",
                    "selector": selector.selector_id,
                }
            ),
            exit_code=result.returncode,
            timed_out=False,
            summary=summary,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def _selector_map(
    spec: EvaluationSpec,
    hidden_asset: EvaluationOnlyTestAsset,
) -> dict[str, TestSelector]:
    result: dict[str, TestSelector] = {}
    for selector in (*spec.public_tests, *hidden_asset.selectors):
        existing = result.get(selector.selector_id)
        if existing is not None and existing != selector:
            raise EvaluationInfrastructureError("duplicate_selector_definition")
        result[selector.selector_id] = selector
    return result


def _strict_apply(
    workspace: Path,
    patch_path: Path,
    timeout_seconds: float,
    *,
    agent_patch: bool,
) -> None:
    check = _run(
        (
            "git",
            "-C",
            str(workspace),
            "apply",
            "--check",
            "--whitespace=nowarn",
            str(patch_path),
        ),
        cwd=workspace,
        timeout_seconds=timeout_seconds,
    )
    if check.returncode != 0:
        if agent_patch:
            raise StrictPatchApplyError(check.stderr)
        raise EvaluationInfrastructureError(
            "hidden_test_injection_failed", check.stderr
        )
    applied = _run(
        (
            "git",
            "-C",
            str(workspace),
            "apply",
            "--whitespace=nowarn",
            str(patch_path),
        ),
        cwd=workspace,
        timeout_seconds=timeout_seconds,
    )
    if applied.returncode != 0:
        if agent_patch:
            raise StrictPatchApplyError(applied.stderr)
        raise EvaluationInfrastructureError(
            "hidden_test_injection_failed", applied.stderr
        )


def _structured_unittest_summary(
    result: subprocess.CompletedProcess[str],
    result_path: Path,
) -> TestExecutionSummary:
    try:
        metadata = result_path.lstat()
    except FileNotFoundError as exc:
        raise EvaluationInfrastructureError(
            "test_runner_evidence_missing"
        ) from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    try:
        raw = result_path.read_bytes()
    except OSError as exc:
        raise EvaluationInfrastructureError(
            "test_runner_evidence_invalid"
        ) from exc
    return structured_unittest_summary(raw, exit_code=result.returncode)


def structured_unittest_summary(
    raw: bytes,
    *,
    exit_code: int,
) -> TestExecutionSummary:
    if not isinstance(raw, bytes) or len(raw) > 4096:
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    require_int(exit_code, "exit_code")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationInfrastructureError(
            "test_runner_evidence_invalid"
        ) from exc
    if canonical_json(payload).encode("utf-8") != raw:
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    try:
        data = require_exact_fields(
            payload,
            "opbench_unittest_result",
            (
                "record_type",
                "schema_version",
                "collected",
                "executed",
                "passed",
                "failed",
                "skipped",
            ),
        )
    except ContractError as exc:
        raise EvaluationInfrastructureError(
            "test_runner_evidence_invalid"
        ) from exc
    if data["record_type"] != "opbench_unittest_result":
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    if data["schema_version"] != "v1":
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    try:
        summary = TestExecutionSummary(
            collected=require_int(data["collected"], "collected", minimum=0),
            executed=require_int(data["executed"], "executed", minimum=0),
            passed=require_int(data["passed"], "passed", minimum=0),
            failed=require_int(data["failed"], "failed", minimum=0),
            skipped=require_int(data["skipped"], "skipped", minimum=0),
        )
    except ContractError as exc:
        raise EvaluationInfrastructureError(
            "test_runner_evidence_invalid"
        ) from exc
    if (exit_code == 0) != (summary.failed == 0):
        raise EvaluationInfrastructureError("test_runner_evidence_invalid")
    return summary


def _run(
    command: tuple[str, ...],
    *,
    cwd: Path,
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        check=False,
    )


def _require_local_git_repository(value: object, path: str) -> Path:
    if not isinstance(value, Path):
        raise ContractError(f"{path}: expected Path")
    if value.is_symlink() or not value.is_dir() or not (value / ".git").is_dir():
        raise ContractError(f"{path}: expected local Git repository")
    return value


def _require_identity(
    value: object,
    identity_type: str,
    path: str,
) -> ContentIdentity:
    if not isinstance(value, ContentIdentity):
        raise ContractError(f"{path}: expected ContentIdentity")
    if value.identity_type != identity_type:
        raise ContractError(f"{path}: expected identity_type {identity_type!r}")
    return value


__all__ = [
    "EvaluationOnlyTestAsset",
    "LocalGitEvaluationBackend",
    "LocalGitSource",
    "git_archive_source_identity",
    "structured_unittest_summary",
]
