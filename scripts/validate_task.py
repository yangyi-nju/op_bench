#!/usr/bin/env python3

"""Minimal validator for op_bench task manifests."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any


REQUIRED_PATHS = [
    ("task_id",),
    ("version",),
    ("source", "repo"),
    ("source", "pr_url"),
    ("source", "issue_url"),
    ("source", "issue_number"),
    ("source", "pr_number"),
    ("source", "base_commit"),
    ("source", "merge_commit"),
    ("statement", "title"),
    ("statement", "body"),
    ("operator", "framework"),
    ("operator", "operator_name"),
    ("environment", "tier"),
    ("environment", "build_mode"),
    ("environment", "hardware", "device"),
    ("agent_visible", "repo_setup_commands"),
    ("evaluation", "fail_to_pass"),
    ("evaluation", "pass_to_pass"),
    ("evaluation", "test_command"),
    ("artifacts", "gold_patch"),
    ("artifacts", "test_patch"),
    ("metadata", "difficulty"),
    ("metadata", "curation_status"),
    ("metadata", "deterministic"),
]

ALLOWED_TIERS = {
    "cpu-deterministic",
    "single-gpu",
    "kernel-build",
    "cpu_python_overlay",
    "cpu_package_runtime",
    "cpu_source_snapshot_fuller",
    "cuda_declared",
    "cuda_python_overlay",
    "cuda_kernel_build",
    "hardware_specific",
}
ALLOWED_BUILD_MODES = {"editable-python", "source-build", "prebuilt-wheel"}
ALLOWED_ENVIRONMENT_BACKENDS = {"local", "docker", "remote_docker"}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}
ALLOWED_CURATION_STATUSES = {"draft", "verified"}
ALLOWED_CHECKOUT_MODES = {"git", "local-copy"}
ALLOWED_SNAPSHOT_METHODS = {"from_local_repo", "github_archive", "git_fetch"}
ALLOWED_DIGEST_KINDS = {"repo_digest", "local_image_id", "build_hash"}
ALLOWED_SOURCE_LOADING_MODES = {"python_overlay", "prebuilt_source_image", "full_source_build", "inplace_build"}
ALLOWED_LAYERS = {"A", "B"}
ALLOWED_ADMISSION_STATUSES = {
    "candidate",
    "draft",
    "environment_ready",
    "source_ready",
    "baseline_reproduced",
    "gold_verified",
    "verified",
    "blocked",
    "blocked_environment",
    "blocked_source",
    "blocked_test",
    "not_reproduced",
    "gold_failed",
    "deprecated",
}
ALLOWED_PATCH_SCOPE_MODES = {"enforced", "filtered"}
ALLOW_EMPTY_REQUIRED_PATHS = {
    ("agent_visible", "repo_setup_commands"),
}


def lookup(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(path))
        current = current[key]
    return current


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    for path in REQUIRED_PATHS:
        if data.get("environment_ref") and path[:1] == ("environment",):
            continue
        if path == ("artifacts", "test_patch") and data.get("artifacts", {}).get("hidden_test_patch"):
            continue
        try:
            value = lookup(data, path)
        except KeyError:
            errors.append(f"missing required field: {'.'.join(path)}")
            continue

        if path not in ALLOW_EMPTY_REQUIRED_PATHS and value in ("", [], {}):
            errors.append(f"empty required field: {'.'.join(path)}")

    source = data.get("source", {})
    for reference in ("environment_ref", "source_ref"):
        value = data.get(reference)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"{reference} must be a non-empty string when provided")

    checkout_mode = source.get("checkout_mode", "git")
    if checkout_mode not in ALLOWED_CHECKOUT_MODES:
        errors.append(
            f"invalid source.checkout_mode: {checkout_mode!r}; expected one of {sorted(ALLOWED_CHECKOUT_MODES)}"
        )
    if checkout_mode == "local-copy" and not source.get("local_path"):
        errors.append("source.local_path is required when source.checkout_mode is 'local-copy'")
    if checkout_mode == "git" and not (source.get("repo_url") or source.get("repo") or source.get("snapshot_path")):
        errors.append("source.repo_url, source.repo, or source.snapshot_path is required when source.checkout_mode is 'git'")
    snapshot_method = source.get("snapshot_method")
    if snapshot_method is not None and snapshot_method not in ALLOWED_SNAPSHOT_METHODS:
        errors.append(
            f"invalid source.snapshot_method: {snapshot_method!r}; expected one of {sorted(ALLOWED_SNAPSHOT_METHODS)}"
        )
    snapshot_hash = source.get("snapshot_hash")
    if snapshot_hash is not None and not str(snapshot_hash).startswith("sha256:"):
        errors.append("source.snapshot_hash must start with 'sha256:' when provided")

    try:
        backend = lookup(data, ("environment", "backend"))
    except KeyError:
        backend = "local"
    if backend not in ALLOWED_ENVIRONMENT_BACKENDS:
        errors.append(
            "invalid environment.backend: "
            f"{backend!r}; expected one of {sorted(ALLOWED_ENVIRONMENT_BACKENDS)}"
        )
    if backend == "docker" and not data.get("environment_ref"):
        environment = data.get("environment", {})
        if not environment.get("image"):
            errors.append("environment.image is required when environment.backend is 'docker'")
        preflight_commands = environment.get("preflight_commands")
        if not isinstance(preflight_commands, list) or not preflight_commands:
            errors.append("environment.preflight_commands must be a non-empty list for docker tasks")
        preflight_workdir = environment.get("preflight_workdir", "/tmp")
        workspace_dir = environment.get("workspace_dir", "/workspace")
        if preflight_workdir == workspace_dir:
            errors.append("environment.preflight_workdir must not equal environment.workspace_dir for docker tasks")
        digest_kind = environment.get("digest_kind")
        if environment.get("image_digest") and digest_kind not in ALLOWED_DIGEST_KINDS:
            errors.append(
                "environment.digest_kind is required with environment.image_digest "
                f"and must be one of {sorted(ALLOWED_DIGEST_KINDS)}"
            )
        if digest_kind is not None and not environment.get("image_digest"):
            errors.append("environment.image_digest is required when environment.digest_kind is provided")
    if backend == "remote_docker":
        environment = data.get("environment", {})
        if not environment.get("host") and not data.get("environment_ref"):
            errors.append("environment.host is required when environment.backend is 'remote_docker' (or set via environment_ref)")

    source_loading = data.get("environment", {}).get("source_loading")
    if source_loading is not None:
        errors.extend(validate_source_loading(source_loading))

    try:
        tier = lookup(data, ("environment", "tier"))
        if tier not in ALLOWED_TIERS:
            errors.append(
                f"invalid environment.tier: {tier!r}; expected one of {sorted(ALLOWED_TIERS)}"
            )
    except KeyError:
        pass
    runtime_tier = data.get("runtime_tier")
    if runtime_tier is not None and runtime_tier not in ALLOWED_TIERS:
        errors.append(f"invalid runtime_tier: {runtime_tier!r}; expected one of {sorted(ALLOWED_TIERS)}")

    try:
        build_mode = lookup(data, ("environment", "build_mode"))
        if build_mode not in ALLOWED_BUILD_MODES:
            errors.append(
                "invalid environment.build_mode: "
                f"{build_mode!r}; expected one of {sorted(ALLOWED_BUILD_MODES)}"
            )
    except KeyError:
        pass

    try:
        difficulty = lookup(data, ("metadata", "difficulty"))
        if difficulty not in ALLOWED_DIFFICULTIES:
            errors.append(
                f"invalid metadata.difficulty: {difficulty!r}; expected one of {sorted(ALLOWED_DIFFICULTIES)}"
            )
    except KeyError:
        pass

    try:
        curation_status = lookup(data, ("metadata", "curation_status"))
        if curation_status not in ALLOWED_CURATION_STATUSES:
            errors.append(
                "invalid metadata.curation_status: "
                f"{curation_status!r}; expected one of {sorted(ALLOWED_CURATION_STATUSES)}"
            )
    except KeyError:
        pass
    metadata = data.get("metadata", {})
    layer = metadata.get("layer")
    if layer is not None and layer not in ALLOWED_LAYERS:
        errors.append(f"invalid metadata.layer: {layer!r}; expected one of {sorted(ALLOWED_LAYERS)}")
    admission_status = metadata.get("admission_status")
    if admission_status is not None and admission_status not in ALLOWED_ADMISSION_STATUSES:
        errors.append(
            f"invalid metadata.admission_status: {admission_status!r}; "
            f"expected one of {sorted(ALLOWED_ADMISSION_STATUSES)}"
        )
    if metadata.get("curation_status") == "verified" and admission_status not in (None, "verified"):
        errors.append("metadata.admission_status must be 'verified' when metadata.curation_status is 'verified'")
    if "source_loading_verified" in metadata and not isinstance(metadata["source_loading_verified"], bool):
        errors.append("metadata.source_loading_verified must be a boolean when provided")

    admission = data.get("admission")
    if admission is not None:
        if not isinstance(admission, dict):
            errors.append("admission must be an object")
        else:
            formal_status = admission.get("status")
            if formal_status not in ALLOWED_ADMISSION_STATUSES:
                errors.append(
                    f"invalid admission.status: {formal_status!r}; "
                    f"expected one of {sorted(ALLOWED_ADMISSION_STATUSES)}"
                )
            if formal_status == "verified":
                if not admission.get("evidence"):
                    errors.append("admission.evidence is required when admission.status is 'verified'")
                if not admission.get("verified_at"):
                    errors.append("admission.verified_at is required when admission.status is 'verified'")
            if admission_status is not None and formal_status != admission_status:
                errors.append("admission.status must match metadata.admission_status when both are provided")

    try:
        fail_to_pass = lookup(data, ("evaluation", "fail_to_pass"))
        if not isinstance(fail_to_pass, list) or not fail_to_pass:
            errors.append("evaluation.fail_to_pass must be a non-empty list")
        elif contains_draft_test_entry(fail_to_pass):
            errors.append("evaluation.fail_to_pass contains unresolved draft test entries")
    except KeyError:
        pass

    try:
        pass_to_pass = lookup(data, ("evaluation", "pass_to_pass"))
        if not isinstance(pass_to_pass, list) or not pass_to_pass:
            errors.append("evaluation.pass_to_pass must be a non-empty list")
        elif contains_draft_test_entry(pass_to_pass):
            errors.append("evaluation.pass_to_pass contains unresolved draft test entries")
    except KeyError:
        pass

    patch_scope = data.get("patch_scope")
    if patch_scope is not None:
        if not isinstance(patch_scope, dict):
            errors.append("patch_scope must be an object")
        else:
            mode = patch_scope.get("mode", "enforced")
            if mode not in ALLOWED_PATCH_SCOPE_MODES:
                errors.append(f"invalid patch_scope.mode: {mode!r}; expected one of {sorted(ALLOWED_PATCH_SCOPE_MODES)}")
            allowed_paths = patch_scope.get("allowed_paths")
            if not isinstance(allowed_paths, list) or not allowed_paths:
                errors.append("patch_scope.allowed_paths must be a non-empty list")
            else:
                for p in allowed_paths:
                    p_path = Path(str(p))
                    if p_path.is_absolute() or ".." in p_path.parts:
                        errors.append(f"patch_scope.allowed_paths entries must be relative without '..': {p!r}")

    artifacts = data.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, value in artifacts.items():
            artifact_path = Path(str(value))
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                errors.append(
                    f"artifacts.{name} must be a task-relative path without '..': {value!r}"
                )

    public_tests = data.get("evaluation", {}).get("public_tests")
    if public_tests is not None:
        if not isinstance(public_tests, list):
            errors.append("evaluation.public_tests must be a list when provided")

    return errors


def validate_source_loading(source_loading: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(source_loading, dict):
        return ["environment.source_loading must be an object"]
    mode = source_loading.get("mode")
    if mode not in ALLOWED_SOURCE_LOADING_MODES:
        errors.append(
            f"invalid environment.source_loading.mode: {mode!r}; "
            f"expected one of {sorted(ALLOWED_SOURCE_LOADING_MODES)}"
        )
        return errors
    if mode == "python_overlay":
        for field in ("installed_package", "runtime_site_packages"):
            if not source_loading.get(field):
                errors.append(f"environment.source_loading.{field} is required for python_overlay")
        overlay_paths = source_loading.get("overlay_paths")
        if not isinstance(overlay_paths, list) or not overlay_paths:
            errors.append("environment.source_loading.overlay_paths must be a non-empty list for python_overlay")
        else:
            for path in overlay_paths:
                path_value = Path(str(path))
                if path_value.is_absolute() or ".." in path_value.parts:
                    errors.append(
                        "environment.source_loading.overlay_paths entries must be workspace-relative paths "
                        f"without '..': {path!r}"
                    )
        if not isinstance(source_loading.get("sync_before_tests"), bool):
            errors.append("environment.source_loading.sync_before_tests must be a boolean for python_overlay")
    if mode == "inplace_build":
        # build_command is optional (default provided); installed_package recommended for diagnostics
        build_command = source_loading.get("build_command")
        if build_command is not None and not isinstance(build_command, str):
            errors.append("environment.source_loading.build_command must be a string when provided")
        build_environment = source_loading.get("build_environment")
        if build_environment is not None:
            if not isinstance(build_environment, dict):
                errors.append("environment.source_loading.build_environment must be an object when provided")
            else:
                for key, value in build_environment.items():
                    if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                        errors.append(
                            "environment.source_loading.build_environment keys must be shell variable names"
                        )
                    if not isinstance(value, (str, int, float, bool)):
                        errors.append(
                            "environment.source_loading.build_environment values must be scalar"
                        )
        if not isinstance(source_loading.get("sync_before_tests", True), bool):
            errors.append("environment.source_loading.sync_before_tests must be a boolean for inplace_build")
    return errors


def contains_draft_test_entry(tests: list[Any]) -> bool:
    draft_prefix = "TO" + "DO:"
    return any(str(test).startswith(draft_prefix) for test in tests)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_task.py <task_manifest.json>", file=sys.stderr)
        return 2

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"file not found: {path}", file=sys.stderr)
        return 2

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    errors = validate_manifest(data)
    if errors:
        print(f"{path}: invalid manifest")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"{path}: manifest looks valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
