from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any

from op_bench.dataset import DatasetManifest
from op_bench.integrity import replay_spec_hash
from op_bench.runtime.canonical import canonical_sha256
from op_bench.runtime.contracts import (
    ACTION_NAMES,
    AgentSpec,
    BudgetPolicy,
    CapabilityPolicy,
    ContentIdentity,
    FullTaskSpec,
    RuntimeProfile,
    SHA256_PATTERN,
    TestSelector,
)
from op_bench.runtime.manifest import RunManifest, build_run_manifest
from op_bench.runtime.validation import ContractError, require_bool, require_int, require_str
from op_bench.task import TaskManifest


@dataclass(frozen=True)
class LegacyV05Defaults:
    platform_version: str
    action_protocol: str
    evaluation_protocol: str
    scoring_protocol: str
    evaluation: ContentIdentity
    capability_policy: CapabilityPolicy
    budget_policy: BudgetPolicy
    retry_policy: ContentIdentity
    termination_policy: ContentIdentity
    scoring: ContentIdentity

    @classmethod
    def standard(cls) -> "LegacyV05Defaults":
        capability = CapabilityPolicy(
            policy_id="legacy-v0.5-controlled-v1",
            allowed_actions=ACTION_NAMES,
            writable_paths=("task-patch-scope",),
            allowed_command_prefixes=("git", "python", "python3", "rg"),
            registered_tests=(),
            max_read_bytes=2_000_000,
            max_write_bytes=2_000_000,
            max_output_bytes=10_000_000,
            network_access="provider_only",
        )
        budget = BudgetPolicy(
            policy_id="legacy-v0.5-standard-v1",
            wall_clock_ms=7_200_000,
            max_actions=500,
            max_tests=100,
            max_commands=200,
            max_output_bytes=50_000_000,
            provider_token_limit=None,
        )
        return cls(
            platform_version="opbench-v0.6.0",
            action_protocol="action-v1",
            evaluation_protocol="evaluation-v1",
            scoring_protocol="scoring-v1",
            evaluation=_config_identity(
                "evaluation",
                "evaluation-v1",
                {"runner": "fresh-evaluation-v1", "patch_apply": "strict"},
            ),
            capability_policy=capability,
            budget_policy=budget,
            retry_policy=_config_identity(
                "policy",
                "legacy-v0.5-retry-v1",
                {"resume_policy": "skip_valid", "infrastructure_retries": 1},
            ),
            termination_policy=_config_identity(
                "policy",
                "legacy-v0.5-termination-v1",
                {"finish": "freeze", "timeout": "freeze-if-safe"},
            ),
            scoring=_config_identity(
                "scoring",
                "legacy-v0.5-resolved-v1",
                {"resolved": "all_f2p_pass_and_all_p2p_pass"},
            ),
        )


def full_task_spec_from_v05(task: TaskManifest) -> FullTaskSpec:
    _require_v05_task_shape(task)
    source = _source_identity(task)
    image = _image_identity(task)
    runtime = _runtime_profile(task, image)
    environment = _environment_identity(task, image, runtime)
    public_tests, hidden_tests = _test_selectors(task)
    statement = _mapping(task.data.get("statement"))
    operator = _mapping(task.data.get("operator"))

    return FullTaskSpec(
        task=ContentIdentity(
            identity_type="task",
            identifier=task.task_id,
            digest=replay_spec_hash(task),
            digest_kind="replay_spec_v1",
        ),
        source=source,
        environment=environment,
        runtime=runtime,
        statement_title=str(statement.get("title", task.task_id)),
        statement_body=str(statement.get("body", "Legacy v0.5 operator repair task.")),
        framework=str(operator.get("framework", "unknown")),
        operator_name=str(operator.get("operator_name", operator.get("component", "unknown"))),
        public_tests=public_tests,
        hidden_tests=hidden_tests,
        fail_to_pass=tuple(str(value) for value in task.fail_to_pass_tests),
        pass_to_pass=tuple(str(value) for value in task.pass_to_pass_tests),
        patch_scope=tuple(str(value) for value in task.patch_scope_paths),
        gold_patch=_file_identity(
            "patch",
            f"{task.task_id}:gold-patch",
            _checked_task_file(
                task,
                _mapping(task.data.get("artifacts")).get("gold_patch"),
                "gold_patch",
            ),
        ),
        hidden_test_asset=_file_identity(
            "test",
            f"{task.task_id}:hidden-test",
            _checked_task_file(
                task,
                _hidden_test_path(_mapping(task.data.get("artifacts"))),
                "hidden_test",
            ),
        ),
        admission=_file_identity(
            "admission",
            f"{task.task_id}:admission",
            _checked_task_file(
                task,
                _mapping(task.data.get("admission")).get("evidence"),
                "admission",
            ),
        ),
    )


def run_manifest_from_v05_dataset(
    dataset_path: Path | str,
    *,
    agents: tuple[AgentSpec, ...],
    repeat: int,
    created_at: str,
    defaults: LegacyV05Defaults | None = None,
) -> RunManifest:
    selected_defaults = defaults or LegacyV05Defaults.standard()
    dataset = DatasetManifest.load(dataset_path)
    _require_verified_dataset(dataset)
    tasks = tuple(full_task_spec_from_v05(task) for task in dataset.load_tasks(verified_only=True))
    return build_run_manifest(
        platform_version=selected_defaults.platform_version,
        action_protocol=selected_defaults.action_protocol,
        evaluation_protocol=selected_defaults.evaluation_protocol,
        scoring_protocol=selected_defaults.scoring_protocol,
        evaluation=selected_defaults.evaluation,
        dataset=_dataset_identity(dataset),
        tasks=tasks,
        agents=agents,
        capability_policy=selected_defaults.capability_policy,
        budget_policy=selected_defaults.budget_policy,
        retry_policy=selected_defaults.retry_policy,
        termination_policy=selected_defaults.termination_policy,
        scoring=selected_defaults.scoring,
        repeat_count=repeat,
        created_at=created_at,
    )


def _dataset_identity(dataset: DatasetManifest) -> ContentIdentity:
    return _config_identity(
        "dataset",
        dataset.dataset_id,
        {
            "dataset_id": dataset.dataset_id,
            "version": dataset.version,
            "manifest": dataset.data,
        },
    )


def _source_identity(task: TaskManifest) -> ContentIdentity:
    identifier = task.source_ref or f"{task.repo_url}@{task.base_commit}"
    snapshot_hash = task.source_snapshot_hash
    if snapshot_hash is not None and re.fullmatch(SHA256_PATTERN, snapshot_hash):
        return ContentIdentity("source", identifier, snapshot_hash, "content_sha256")
    return _config_identity(
        "source",
        identifier,
        {
            "source_ref": task.source_ref,
            "repo_url": task.repo_url,
            "base_commit": task.base_commit,
            "checkout_mode": task.checkout_mode,
            "snapshot_method": task.source_snapshot_method,
            "source_loading_mode": task.source_loading_mode or "none",
        },
    )


def _image_identity(task: TaskManifest) -> ContentIdentity:
    identifier = task.environment_image or f"legacy-inline-image:{task.task_id}"
    digest = task.environment_image_digest
    if digest is not None and re.fullmatch(SHA256_PATTERN, digest):
        digest_kind = "image_id" if task.environment_digest_kind == "local_image_id" else "content_sha256"
        return ContentIdentity("image", identifier, digest, digest_kind)
    return _config_identity(
        "image",
        identifier,
        {
            "environment_ref": task.environment_ref,
            "image": task.environment_image,
            "runtime_tier": task.runtime_tier,
        },
    )


def _runtime_profile(task: TaskManifest, image: ContentIdentity) -> RuntimeProfile:
    environment = _mapping(task.data.get("environment"))
    source_loading = _mapping(environment.get("source_loading"))
    mode = str(source_loading.get("mode", "none"))
    profile_payload = {
        "environment_ref": task.environment_ref,
        "backend": str(environment.get("backend", "local")),
        "runtime_tier": task.runtime_tier,
        "source_loading_mode": mode,
        "platform": task.environment_platform or "unspecified",
        "image": image.to_dict(),
        "requires_gpu": task.requires_gpu,
        "timeout_ms": task.timeout_sec * 1000,
    }
    suffix = canonical_sha256(profile_payload).removeprefix("sha256:")[:16]
    return RuntimeProfile(
        profile_id=f"legacy-v0.5:{task.environment_ref or 'inline'}:{suffix}",
        backend=str(environment.get("backend", "local")),
        runtime_tier=task.runtime_tier,
        source_loading_mode=mode,
        platform=task.environment_platform or "unspecified",
        image=image,
        requires_gpu=task.requires_gpu,
        network_policy="denied",
        timeout_ms=task.timeout_sec * 1000,
    )


def _environment_identity(
    task: TaskManifest,
    image: ContentIdentity,
    runtime: RuntimeProfile,
) -> ContentIdentity:
    environment = _mapping(task.data.get("environment"))
    return _config_identity(
        "environment",
        task.environment_ref or f"legacy-inline-environment:{task.task_id}",
        {
            "environment_ref": task.environment_ref,
            "backend": runtime.backend,
            "runtime_tier": runtime.runtime_tier,
            "source_loading_mode": runtime.source_loading_mode,
            "platform": runtime.platform,
            "image": image.to_dict(),
            "requires_gpu": runtime.requires_gpu,
            "python_version": environment.get("python_version", environment.get("python")),
            "os": environment.get("os"),
            "build_mode": environment.get("build_mode"),
            "hardware": environment.get("hardware", {}),
            "dependencies": environment.get("dependencies", []),
            "preflight_commands": environment.get("preflight_commands", []),
        },
    )


def _test_selectors(
    task: TaskManifest,
) -> tuple[tuple[TestSelector, ...], tuple[TestSelector, ...]]:
    public_ids = tuple(str(value) for value in task.public_tests)
    evaluation_ids = tuple(
        dict.fromkeys(
            [
                *(str(value) for value in task.fail_to_pass_tests),
                *(str(value) for value in task.pass_to_pass_tests),
            ]
        )
    )
    command_template = str(_mapping(task.data.get("evaluation")).get("test_command", "{test}"))
    public = tuple(
        TestSelector(
            selector_id=selector_id,
            visibility="registered",
            command_template=command_template,
            description="Legacy v0.5 registered test selector.",
        )
        for selector_id in public_ids
    )
    hidden = tuple(
        TestSelector(
            selector_id=selector_id,
            visibility="evaluation_only",
            command_template=command_template,
            description="Legacy v0.5 evaluator-only test selector.",
        )
        for selector_id in evaluation_ids
        if selector_id not in public_ids
    )
    return public, hidden


def _file_identity(
    identity_type: str,
    identifier: str,
    path: Path,
) -> ContentIdentity:
    digest = f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"
    return ContentIdentity(identity_type, identifier, digest, "content_sha256")


def _checked_task_file(task: TaskManifest, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label}: expected non-empty task-relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractError(f"{label}: path escapes task root")

    root = task.task_dir.resolve()
    candidate = root / relative
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ContractError(f"{label}: symlinks are not allowed")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{label}: artifact file is unavailable") from exc
    if root != resolved and root not in resolved.parents:
        raise ContractError(f"{label}: path escapes task root")
    if not resolved.is_file():
        raise ContractError(f"{label}: expected regular file")
    return resolved


def _hidden_test_path(artifacts: dict[str, Any]) -> object:
    return artifacts.get("hidden_test_patch", artifacts.get("test_patch"))


def _require_verified_dataset(dataset: DatasetManifest) -> None:
    if not isinstance(dataset.data.get("dataset_id"), str) or not dataset.data["dataset_id"]:
        raise ContractError("dataset.dataset_id: expected non-empty string")
    if dataset.data.get("status") != "verified":
        raise ContractError("dataset.status must be 'verified'")
    require_str(dataset.data.get("version"), "dataset.version")
    registries = dataset.data.get("registries", {})
    if not isinstance(registries, dict):
        raise ContractError("dataset.registries: expected object")
    for name, path in registries.items():
        require_str(name, "dataset.registries key")
        require_str(path, f"dataset.registries.{name}")
    entries = dataset.data.get("tasks")
    if not isinstance(entries, list) or not entries:
        raise ContractError("dataset.tasks: expected non-empty array")
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ContractError(f"dataset.tasks[{index}]: expected object")
        for name in ("task_id", "task_path"):
            require_str(entry.get(name), f"dataset.tasks[{index}].{name}")
        if entry.get("admission_status") != "verified":
            raise ContractError(
                f"dataset.tasks[{index}].admission_status must be 'verified'"
            )
        for name in (
            "environment_status",
            "source_status",
            "replay_status",
            "admission_evidence",
        ):
            require_str(entry.get(name), f"dataset.tasks[{index}].{name}")


def _require_v05_task_shape(task: TaskManifest) -> None:
    if not isinstance(task, TaskManifest):
        raise ContractError("task: expected TaskManifest")
    data = task.data
    if not isinstance(data, dict):
        raise ContractError("task.data: expected object")
    require_str(data.get("task_id"), "task_id")
    for name in ("environment_ref", "source_ref", "runtime_tier"):
        require_str(data.get(name), name)

    statement = _require_object(data.get("statement"), "statement")
    require_str(statement.get("title"), "statement.title")
    require_str(statement.get("body"), "statement.body")

    operator = _require_object(data.get("operator"), "operator")
    require_str(operator.get("framework"), "operator.framework")
    operator_name = operator.get("operator_name", operator.get("component"))
    require_str(operator_name, "operator.operator_name")

    source = _require_object(data.get("source"), "source")
    require_str(source.get("base_commit"), "source.base_commit")
    require_str(source.get("checkout_mode", "git"), "source.checkout_mode")
    source_location = source.get("repo_url", source.get("repo"))
    require_str(source_location, "source.repo_url")
    for name in ("snapshot_hash", "snapshot_method"):
        if source.get(name) is not None:
            require_str(source[name], f"source.{name}")

    environment = _require_object(data.get("environment"), "environment")
    require_str(environment.get("backend", "local"), "environment.backend")
    require_str(environment.get("image"), "environment.image")
    for name in ("platform", "image_digest", "digest_kind", "python_version", "os", "build_mode"):
        if environment.get(name) is not None:
            require_str(environment[name], f"environment.{name}")
    hardware = _require_object(environment.get("hardware", {}), "environment.hardware")
    if "requires_gpu" in hardware:
        require_bool(hardware["requires_gpu"], "environment.hardware.requires_gpu")
    source_loading = _require_object(
        environment.get("source_loading", {}),
        "environment.source_loading",
    )
    if source_loading:
        require_str(source_loading.get("mode"), "environment.source_loading.mode")
    for name in ("dependencies", "preflight_commands"):
        if name in environment:
            _require_str_list(environment[name], f"environment.{name}", allow_empty=True)

    evaluation = _require_object(data.get("evaluation"), "evaluation")
    require_str(evaluation.get("test_command"), "evaluation.test_command")
    require_int(evaluation.get("timeout_sec"), "evaluation.timeout_sec", minimum=1)
    _require_str_list(
        evaluation.get("fail_to_pass"),
        "evaluation.fail_to_pass",
        allow_empty=False,
    )
    _require_str_list(
        evaluation.get("pass_to_pass"),
        "evaluation.pass_to_pass",
        allow_empty=False,
    )
    _require_str_list(
        evaluation.get("public_tests", []),
        "evaluation.public_tests",
        allow_empty=True,
    )

    patch_scope = _require_object(data.get("patch_scope"), "patch_scope")
    _require_str_list(
        patch_scope.get("allowed_paths"),
        "patch_scope.allowed_paths",
        allow_empty=False,
    )
    require_str(patch_scope.get("mode", "enforced"), "patch_scope.mode")

    artifacts = _require_object(data.get("artifacts"), "artifacts")
    require_str(artifacts.get("gold_patch"), "artifacts.gold_patch")
    require_str(_hidden_test_path(artifacts), "artifacts.hidden_test")

    admission = _require_object(data.get("admission"), "admission")
    for name in ("status", "evidence", "verified_at"):
        require_str(admission.get(name), f"admission.{name}")


def _require_object(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _require_str_list(value: object, path: str, *, allow_empty: bool) -> None:
    if not isinstance(value, list):
        raise ContractError(f"{path}: expected array")
    if not allow_empty and not value:
        raise ContractError(f"{path}: must contain at least one value")
    for index, item in enumerate(value):
        require_str(item, f"{path}[{index}]")


def _config_identity(
    identity_type: str,
    identifier: str,
    payload: dict[str, Any],
) -> ContentIdentity:
    return ContentIdentity(
        identity_type=identity_type,
        identifier=identifier,
        digest=canonical_sha256(payload),
        digest_kind="canonical_config",
    )


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    raise ContractError("legacy v0.5 field: expected object")
