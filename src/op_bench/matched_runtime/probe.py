from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import tempfile
from typing import Callable, Protocol
import uuid

from op_bench.environment import EnvironmentManager
from op_bench.evaluator import Evaluator
from op_bench.executor import CommandResult
from op_bench.matched_runtime.contracts import (
    ARTIFACT_DIGEST_KINDS,
    ARTIFACT_KINDS,
    BuildIdentity,
    CompatibilityCheck,
    CompatibilityEvidence,
    CompatibilityFailure,
    FAILURE_CODES,
    MATCH_STRATEGIES,
    REQUIRED_CHECKS,
    RuntimeIdentity,
    SOURCE_LOADING_MODES,
    SourceIdentity,
)
from op_bench.progress import Progress, noop_progress
from op_bench.runtime.canonical import JsonValue, canonical_json, canonical_sha256
from op_bench.runtime.validation import (
    ContractError,
    require_enum,
    require_str,
)
from op_bench.source_loading import build_source_loading_command
from op_bench.task import TaskManifest


SHA256_PATTERN = r"sha256:[0-9a-f]{64}"
_SENSITIVE_TEXT = (
    re.compile(r"(?:^|[\s\"'])/(?:Users|home|private|tmp)/"),
    re.compile(r"(?:^|[\s\"'])[A-Za-z]:\\"),
    re.compile(r"\b(?:ghp_|github_pat_|sk-)[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
)
_FAILURE_BY_CHECK = {
    "source_identity": "source_identity_mismatch",
    "runtime_identity": "python_abi_mismatch",
    "target_module_provenance": "target_module_not_from_snapshot",
    "target_import": "target_import_failed",
    "selector_collection": "selector_not_collected",
    "minimal_operation": "minimal_operation_failed",
}

_RUNTIME_IDENTITY_CODE = """
import ctypes
import json
import platform
import sysconfig
import torch

cuda_available = torch.cuda.is_available()
payload = {
    "torch_version": torch.__version__,
    "python_implementation": platform.python_implementation(),
    "python_abi": sysconfig.get_config_var("SOABI"),
    "platform": f"linux/{platform.machine()}",
    "cuda_build": torch.version.cuda,
    "cuda_runtime": None,
    "device_name": None,
    "compute_capability": None,
}
if cuda_available:
    cuda_major = str(torch.version.cuda).split(".", 1)[0]
    cudart = None
    errors = []
    for library_name in (f"libcudart.so.{cuda_major}", "libcudart.so"):
        try:
            cudart = ctypes.CDLL(library_name)
            break
        except OSError as exc:
            errors.append(str(exc))
    if cudart is None:
        raise RuntimeError("CUDA runtime library was not loadable: " + "; ".join(errors))
    runtime_version = ctypes.c_int()
    runtime_status = cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version))
    if runtime_status != 0:
        raise RuntimeError(
            f"cudaRuntimeGetVersion failed with status {runtime_status}"
        )
    runtime_raw = runtime_version.value
    payload["cuda_runtime"] = (
        f"{runtime_raw // 1000}.{(runtime_raw % 1000) // 10}"
    )
    payload["device_name"] = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    payload["compute_capability"] = f"{major}.{minor}"
print(json.dumps(payload, sort_keys=True))
""".strip()

_TARGET_PROVENANCE_CODE = """
import hashlib
import importlib
import json
from pathlib import Path
import sys

module_name, expected_suffix = sys.argv[1:3]
module = importlib.import_module(module_name)
path = Path(module.__file__).resolve()
normalized = path.as_posix()
if not normalized.endswith(expected_suffix):
    raise SystemExit(3)
print(json.dumps({
    "target_module_path_suffix": expected_suffix,
    "target_module_sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
}, sort_keys=True))
""".strip()

_SELECTOR_COLLECTION_CODE = """
import importlib
import json
from pathlib import Path
import sys
import unittest

module_path = Path(sys.argv[1])
selectors = json.loads(sys.argv[2])
sys.path.insert(0, str(module_path.parent))
module = importlib.import_module(module_path.stem)

def failed_test(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            if failed_test(item):
                return True
        elif item.__class__.__name__ == "_FailedTest":
            return True
    return False

counts = {}
for selector in selectors:
    suite = unittest.defaultTestLoader.loadTestsFromName(selector, module)
    counts[selector] = suite.countTestCases()
    if counts[selector] != 1 or failed_test(suite):
        print(json.dumps({"counts": counts}, sort_keys=True))
        raise SystemExit(4)
print(json.dumps({"counts": counts}, sort_keys=True))
""".strip()


def _validate_relative_path(value: object, path: str) -> str:
    text = require_str(value, path)
    if "\\" in text:
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != text
    ):
        raise ContractError(f"{path}: expected normalized relative POSIX path")
    return text


@dataclass(frozen=True)
class ProbeCommand:
    name: str
    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        require_enum(self.name, "probe_command.name", REQUIRED_CHECKS)
        if not isinstance(self.argv, tuple) or not self.argv:
            raise ContractError("probe_command.argv: expected non-empty tuple")
        for index, value in enumerate(self.argv):
            require_str(value, f"probe_command.argv[{index}]")

    @property
    def command_digest(self) -> str:
        return canonical_sha256(list(self.argv))


@dataclass(frozen=True)
class ProbeSpec:
    task_id: str
    source_id: str
    environment_id: str
    source_commit: str
    strategy: str
    artifact_kind: str
    artifact_id: str
    artifact_digest: str | None
    artifact_digest_kind: str | None
    source_loading_mode: str
    target_module: str
    target_import: str
    selector_module: str
    selectors: tuple[str, ...]
    minimal_operation: str
    build: BuildIdentity
    python_executable: str

    def __post_init__(self) -> None:
        require_enum(self.strategy, "strategy", MATCH_STRATEGIES)
        require_enum(self.artifact_kind, "artifact_kind", ARTIFACT_KINDS)
        if self.artifact_digest_kind is not None:
            require_enum(
                self.artifact_digest_kind,
                "artifact_digest_kind",
                ARTIFACT_DIGEST_KINDS,
            )
        require_enum(
            self.source_loading_mode,
            "source_loading_mode",
            SOURCE_LOADING_MODES,
        )

    @classmethod
    def from_task(
        cls,
        task: TaskManifest,
        *,
        strategy: str,
        artifact_kind: str,
        artifact_id: str,
        artifact_digest: str | None,
        artifact_digest_kind: str | None,
        build: BuildIdentity,
    ) -> "ProbeSpec":
        compatibility = task.data.get("compatibility")
        if not isinstance(compatibility, Mapping):
            raise ContractError("task.compatibility: expected object")
        target_module = _validate_relative_path(
            compatibility.get("target_module"),
            "task.compatibility.target_module",
        )
        selector_module = _validate_relative_path(
            compatibility.get("selector_module"),
            "task.compatibility.selector_module",
        )
        target_import = require_str(
            compatibility.get("target_import"),
            "task.compatibility.target_import",
            pattern=r"[A-Za-z_][A-Za-z0-9_.]*",
        )
        minimal_operation = require_str(
            compatibility.get("minimal_operation"),
            "task.compatibility.minimal_operation",
        )
        if not minimal_operation.startswith("import torch;"):
            raise ContractError(
                "task.compatibility.minimal_operation: must begin with 'import torch;'"
            )
        if not task.fail_to_pass_tests or not task.pass_to_pass_tests:
            raise ContractError(
                "task evaluation selectors: compatibility requires F2P and P2P"
            )
        if task.source_ref is None:
            raise ContractError("task.source_ref: matched runtime requires a source asset")
        if task.environment_ref is None:
            raise ContractError(
                "task.environment_ref: matched runtime requires an environment asset"
            )
        if not isinstance(build, BuildIdentity):
            raise ContractError("build: expected BuildIdentity")
        if artifact_digest is not None:
            require_str(
                artifact_digest,
                "artifact_digest",
                pattern=SHA256_PATTERN,
            )
        if (artifact_digest is None) != (artifact_digest_kind is None):
            raise ContractError(
                "artifact digest and digest kind must both be present or null"
            )
        return cls(
            task_id=task.task_id,
            source_id=task.source_ref,
            environment_id=task.environment_ref,
            source_commit=task.base_commit,
            strategy=strategy,
            artifact_kind=artifact_kind,
            artifact_id=require_str(artifact_id, "artifact_id"),
            artifact_digest=artifact_digest,
            artifact_digest_kind=artifact_digest_kind,
            source_loading_mode=task.source_loading_mode or "python_overlay",
            target_module=target_module,
            target_import=target_import,
            selector_module=selector_module,
            selectors=tuple(task.fail_to_pass_tests + task.pass_to_pass_tests),
            minimal_operation=minimal_operation,
            build=build,
            python_executable=task.environment_python_executable,
        )

    @property
    def commands(self) -> tuple[ProbeCommand, ...]:
        python = self.python_executable
        return (
            ProbeCommand(
                "source_identity",
                (
                    "op-bench",
                    "verify-source",
                    self.source_commit,
                    self.target_module,
                ),
            ),
            ProbeCommand(
                "runtime_identity",
                (python, "-I", "-c", _RUNTIME_IDENTITY_CODE),
            ),
            ProbeCommand(
                "target_module_provenance",
                (
                    python,
                    "-I",
                    "-c",
                    _TARGET_PROVENANCE_CODE,
                    self.target_import,
                    self.target_module,
                ),
            ),
            ProbeCommand(
                "target_import",
                (
                    python,
                    "-I",
                    "-c",
                    f"import {self.target_import}",
                ),
            ),
            ProbeCommand(
                "selector_collection",
                (
                    python,
                    "-I",
                    "-c",
                    _SELECTOR_COLLECTION_CODE,
                    self.selector_module,
                    json.dumps(self.selectors, separators=(",", ":")),
                ),
            ),
            ProbeCommand(
                "minimal_operation",
                (python, "-I", "-c", self.minimal_operation),
            ),
        )


@dataclass(frozen=True)
class ProbeObservation:
    name: str
    exit_code: int | None
    status: str
    summary: str


@dataclass(frozen=True)
class ProbeExecution:
    snapshot_digest: str
    source_target_sha256: str
    runtime_observation: Mapping[str, object]
    observations: tuple[ProbeObservation, ...]
    cleanup_failed: bool
    failure_code: str | None = None


class ProbeBackend(Protocol):
    def execute(self, task: TaskManifest, spec: ProbeSpec) -> ProbeExecution:
        ...


class EnvironmentProbeBackend:
    """Run compatibility checks through the existing isolated environment path."""

    def __init__(
        self,
        *,
        evaluator: Evaluator | None = None,
        environment_manager: EnvironmentManager | None = None,
        progress: Progress | None = None,
    ) -> None:
        self.progress = progress or noop_progress
        self.environment_manager = environment_manager or EnvironmentManager(
            progress=self.progress
        )
        self.evaluator = evaluator or Evaluator(
            environment_manager=self.environment_manager,
            progress=self.progress,
        )

    def execute(self, task: TaskManifest, spec: ProbeSpec) -> ProbeExecution:
        with tempfile.TemporaryDirectory(prefix=f"op-bench-probe-{task.task_id}-") as tmp:
            workspace = Path(tmp) / "workspace"
            commands: list[CommandResult] = []
            prepare_error = self.evaluator.prepare_workspace(
                task,
                workspace,
                commands,
            )
            snapshot_digest, target_hash, source_observation = self._source_identity(
                task,
                spec,
                workspace,
                prepare_error,
            )
            if source_observation.status != "passed":
                return self._stopped_execution(
                    snapshot_digest,
                    target_hash,
                    source_observation,
                    failure_code="source_identity_mismatch",
                )

            test_patch_result = self.evaluator.apply_hidden_test_patch(
                task,
                workspace,
            )
            if (
                test_patch_result is not None
                and test_patch_result.exit_code != 0
            ):
                return ProbeExecution(
                    snapshot_digest=snapshot_digest,
                    source_target_sha256=target_hash,
                    runtime_observation={},
                    observations=(
                        source_observation,
                        ProbeObservation(
                            "runtime_identity",
                            None,
                            "unavailable",
                            "not run after hidden test patch failure",
                        ),
                        ProbeObservation(
                            "target_module_provenance",
                            None,
                            "unavailable",
                            "not run after hidden test patch failure",
                        ),
                        ProbeObservation(
                            "target_import",
                            None,
                            "unavailable",
                            "not run after hidden test patch failure",
                        ),
                        ProbeObservation(
                            "selector_collection",
                            test_patch_result.exit_code,
                            "failed",
                            "hidden test patch could not be applied",
                        ),
                        ProbeObservation(
                            "minimal_operation",
                            None,
                            "unavailable",
                            "not run after hidden test patch failure",
                        ),
                    ),
                    cleanup_failed=False,
                    failure_code="selector_not_collected",
                )

            preparation = self.environment_manager.prepare(task, workspace)
            if not preparation.available:
                observations = [source_observation]
                observations.extend(
                    ProbeObservation(
                        name=name,
                        exit_code=None,
                        status="unavailable",
                        summary="runtime was unavailable",
                    )
                    for name in REQUIRED_CHECKS[1:]
                )
                cleanup = self.environment_manager.cleanup(preparation)
                return ProbeExecution(
                    snapshot_digest=snapshot_digest,
                    source_target_sha256=target_hash,
                    runtime_observation={},
                    observations=tuple(observations),
                    cleanup_failed=cleanup is not None and cleanup.exit_code != 0,
                    failure_code="runtime_unavailable",
                )

            executor = preparation.executor
            observations: list[ProbeObservation] = [source_observation]
            runtime_payload: dict[str, object] = {}
            source_loading = build_source_loading_command(task)
            if source_loading is not None:
                loading_result = executor.run(
                    source_loading,
                    workspace,
                    task.build_timeout_sec,
                )
                if loading_result.exit_code != 0:
                    observations.append(
                        ProbeObservation(
                            name="runtime_identity",
                            exit_code=loading_result.exit_code,
                            status="failed",
                            summary="source loading failed",
                        )
                    )
                    observations.extend(
                        ProbeObservation(
                            name=name,
                            exit_code=None,
                            status="unavailable",
                            summary="not run after source loading failure",
                        )
                        for name in REQUIRED_CHECKS[2:]
                    )
                    cleanup = self.environment_manager.cleanup(preparation)
                    return ProbeExecution(
                        snapshot_digest=snapshot_digest,
                        source_target_sha256=target_hash,
                        runtime_observation={},
                        observations=tuple(observations),
                        cleanup_failed=cleanup is not None and cleanup.exit_code != 0,
                        failure_code=(
                            "build_failed"
                            if spec.strategy != "matched_wheel"
                            else "target_import_failed"
                        ),
                    )

            for command in spec.commands[1:]:
                result = executor.run(
                    list(command.argv),
                    workspace,
                    task.timeout_sec,
                )
                status = "passed" if result.exit_code == 0 else "failed"
                observations.append(
                    ProbeObservation(
                        name=command.name,
                        exit_code=result.exit_code,
                        status=status,
                        summary=(
                            f"{command.name} passed"
                            if status == "passed"
                            else f"{command.name} failed"
                        ),
                    )
                )
                if result.exit_code == 0 and command.name in {
                    "runtime_identity",
                    "target_module_provenance",
                }:
                    runtime_payload.update(_last_json_object(result.stdout))
                if result.exit_code != 0:
                    completed = {item.name for item in observations}
                    observations.extend(
                        ProbeObservation(
                            name=name,
                            exit_code=None,
                            status="unavailable",
                            summary=f"not run after {command.name} failure",
                        )
                        for name in REQUIRED_CHECKS
                        if name not in completed
                    )
                    break
            cleanup = self.environment_manager.cleanup(preparation)
            return ProbeExecution(
                snapshot_digest=snapshot_digest,
                source_target_sha256=target_hash,
                runtime_observation=runtime_payload,
                observations=tuple(observations),
                cleanup_failed=cleanup is not None and cleanup.exit_code != 0,
            )

    def _source_identity(
        self,
        task: TaskManifest,
        spec: ProbeSpec,
        workspace: Path,
        prepare_error: str | None,
    ) -> tuple[str, str, ProbeObservation]:
        fallback_digest = canonical_sha256(
            {
                "source_id": spec.source_id,
                "commit": spec.source_commit,
                "prepare_error": bool(prepare_error),
            }
        )
        fallback_target = canonical_sha256(
            {
                "source_id": spec.source_id,
                "target_module": spec.target_module,
                "present": False,
            }
        )
        if prepare_error is not None:
            return (
                fallback_digest,
                fallback_target,
                ProbeObservation(
                    "source_identity",
                    1,
                    "failed",
                    "source snapshot preparation failed",
                ),
            )
        target = workspace / spec.target_module
        if not target.is_file():
            return (
                fallback_digest,
                fallback_target,
                ProbeObservation(
                    "source_identity",
                    1,
                    "failed",
                    "snapshot target module was missing",
                ),
            )
        archive = subprocess.run(
            ["git", "-C", str(workspace), "archive", "--format=tar", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=task.timeout_sec,
        )
        ancestry = subprocess.run(
            [
                "git",
                "-C",
                str(workspace),
                "merge-base",
                "--is-ancestor",
                spec.source_commit,
                "HEAD",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=task.timeout_sec,
        )
        snapshot_digest = (
            "sha256:" + hashlib.sha256(archive.stdout).hexdigest()
            if archive.returncode == 0
            else fallback_digest
        )
        target_hash = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
        passed = archive.returncode == 0 and ancestry.returncode == 0
        return (
            snapshot_digest,
            target_hash,
            ProbeObservation(
                "source_identity",
                0 if passed else 1,
                "passed" if passed else "failed",
                "source identity passed" if passed else "source identity failed",
            ),
        )

    def _stopped_execution(
        self,
        snapshot_digest: str,
        target_hash: str,
        observation: ProbeObservation,
        *,
        failure_code: str,
    ) -> ProbeExecution:
        observations = [observation]
        observations.extend(
            ProbeObservation(
                name=name,
                exit_code=None,
                status="unavailable",
                summary="not run after source identity failure",
            )
            for name in REQUIRED_CHECKS[1:]
        )
        return ProbeExecution(
            snapshot_digest=snapshot_digest,
            source_target_sha256=target_hash,
            runtime_observation={},
            observations=tuple(observations),
            cleanup_failed=False,
            failure_code=failure_code,
        )


class MatchedRuntimeProbe:
    def __init__(
        self,
        *,
        backend: ProbeBackend | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.backend = backend or EnvironmentProbeBackend()
        self.now = now or (lambda: datetime.now(timezone.utc))

    def run(
        self,
        task: TaskManifest,
        spec: ProbeSpec,
    ) -> CompatibilityEvidence:
        if task.task_id != spec.task_id:
            raise ContractError("task_id: task and probe spec do not match")
        execution = self.backend.execute(task, spec)
        observations = list(execution.observations)
        commands = spec.commands
        if len(observations) != len(commands):
            raise ContractError(
                "probe observations: expected one observation per command"
            )
        if tuple(item.name for item in observations) != REQUIRED_CHECKS:
            raise ContractError(
                "probe observations: expected required checks in canonical order"
            )
        runtime_target = execution.runtime_observation.get(
            "target_module_sha256"
        )
        runtime_path = execution.runtime_observation.get(
            "target_module_path_suffix"
        )
        explicit_failure_code = execution.failure_code
        if (
            observations[2].status == "passed"
            and (
                runtime_target != execution.source_target_sha256
                or runtime_path != spec.target_module
            )
        ):
            observations[2] = ProbeObservation(
                name="target_module_provenance",
                exit_code=5,
                status="failed",
                summary="runtime target module did not match the snapshot",
            )
            explicit_failure_code = "target_module_not_from_snapshot"
        if execution.cleanup_failed and all(
            item.status == "passed" for item in observations
        ):
            observations[1] = ProbeObservation(
                name="runtime_identity",
                exit_code=1,
                status="failed",
                summary="runtime cleanup failed",
            )
            explicit_failure_code = "cleanup_failed"

        if all(item.status == "passed" for item in observations):
            status = "compatible"
            failure = None
        else:
            failed = next(
                (item for item in observations if item.status == "failed"),
                None,
            )
            if failed is not None:
                status = "incompatible"
                failure_code = explicit_failure_code or _FAILURE_BY_CHECK[failed.name]
                failure = CompatibilityFailure(
                    code=require_enum(
                        failure_code,
                        "probe failure code",
                        FAILURE_CODES,
                    ),
                    check=failed.name,
                    summary=failed.summary,
                )
            else:
                status = "unavailable"
                unavailable = next(
                    item for item in observations if item.status == "unavailable"
                )
                failure = CompatibilityFailure(
                    code=require_enum(
                        explicit_failure_code or "runtime_unavailable",
                        "probe failure code",
                        FAILURE_CODES,
                    ),
                    check=unavailable.name,
                    summary=unavailable.summary,
                )

        checks = tuple(
            CompatibilityCheck(
                name=observation.name,
                command_digest=command.command_digest,
                exit_code=observation.exit_code,
                status=observation.status,
                summary=observation.summary,
            )
            for observation, command in zip(observations, commands)
        )
        runtime = self._runtime_identity(spec, execution.runtime_observation)
        timestamp = self.now().astimezone(timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        return CompatibilityEvidence(
            task_id=task.task_id,
            strategy=spec.strategy,
            status=status,
            source=SourceIdentity(
                source_id=spec.source_id,
                commit=spec.source_commit,
                snapshot_digest=execution.snapshot_digest,
                snapshot_digest_kind="git_archive_sha256",
                target_module_path=spec.target_module,
                target_module_sha256=execution.source_target_sha256,
                runtime_path_suffix=spec.target_module,
            ),
            runtime=runtime,
            build=spec.build,
            checks=checks,
            failure=failure,
            created_at=timestamp,
        )

    def _runtime_identity(
        self,
        spec: ProbeSpec,
        observation: Mapping[str, object],
    ) -> RuntimeIdentity:
        def optional(name: str) -> str | None:
            value = observation.get(name)
            return str(value) if value is not None else None

        return RuntimeIdentity(
            environment_id=spec.environment_id,
            artifact_kind=spec.artifact_kind,
            artifact_id=spec.artifact_id,
            artifact_digest=spec.artifact_digest,
            artifact_digest_kind=spec.artifact_digest_kind,
            torch_version=optional("torch_version"),
            python_implementation=optional("python_implementation"),
            python_abi=optional("python_abi"),
            platform=optional("platform"),
            cuda_build=optional("cuda_build"),
            cuda_runtime=optional("cuda_runtime"),
            device_name=optional("device_name"),
            compute_capability=optional("compute_capability"),
            source_loading_mode=spec.source_loading_mode,
            target_module_path_suffix=optional("target_module_path_suffix"),
            target_module_sha256=optional("target_module_sha256"),
        )


def _last_json_object(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return {str(key): item for key, item in value.items()}
    raise ContractError("probe command: expected a JSON object on stdout")


def _assert_public_safe(value: object, *, path: str = "$") -> None:
    if isinstance(value, str):
        if any(pattern.search(value) for pattern in _SENSITIVE_TEXT):
            raise ContractError(
                f"compatibility evidence {path}: sensitive text is denied"
            )
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_public_safe(item, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_public_safe(item, path=f"{path}[{index}]")
        return
    raise ContractError(
        f"compatibility evidence {path}: unsupported public value"
    )


def write_compatibility_evidence(
    path: Path,
    evidence: CompatibilityEvidence,
) -> None:
    if not isinstance(path, Path):
        raise ContractError("evidence path: expected Path")
    if not isinstance(evidence, CompatibilityEvidence):
        raise ContractError("evidence: expected CompatibilityEvidence")
    payload = evidence.to_dict()
    _assert_public_safe(payload)
    encoded = canonical_json(payload).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ContractError("evidence path: already exists")
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ContractError("evidence path: already exists") from exc
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    CompatibilityEvidence.from_dict(decoded)
