"""Explicit, trusted baseline builds for fresh candidate workspaces.

Artifacts are selected by the operator. Metadata compatibility is not content
authentication. No candidate patch, grader or model channel enters this builder.
Every consumer must still apply its patch and run the normal build recipe.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import stat
import time

from op_bench.runtime.workspace import workspace_environment
from op_bench.runtime.execution import ExecutionError, create_session, materialize_source, resolve_image_id, source_identity
from op_bench.io import write_json
from op_bench.provenance import software_identity
from op_bench.data.task import TaskSpec


class BaselineError(ValueError):
    """A baseline artifact cannot be prepared or reused as declared."""


def _compatibility(task: TaskSpec, expected_source: dict | None = None) -> dict:
    environment = task.environment.to_dict()
    if task.environment.backend == "docker":
        try:
            environment["image"] = resolve_image_id(task.environment.image)
        except ExecutionError as exc:
            raise BaselineError(f"Cannot identify baseline image: {exc}") from exc
    if expected_source is not None and not isinstance(expected_source, dict):
        raise BaselineError("expected_source must be explicit source provenance")
    # The baseline has never seen the statement, private tests or scoring
    # rules. Their revision can change without invalidating these build inputs.
    # Keep their producer identity as provenance, outside build compatibility.
    return {"source": source_identity(task) if expected_source is None else expected_source,
            "environment": environment}


def _files(directory: Path) -> list[str]:
    names = []
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if ".git" in relative.parts:
            raise BaselineError("A baseline artifact must not contain Git history")
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            if not path.resolve().is_relative_to(directory.resolve()):
                raise BaselineError(f"External baseline symlink: {relative}")
            names.append(relative.as_posix())
        elif stat.S_ISREG(mode):
            names.append(relative.as_posix())
        elif not stat.S_ISDIR(mode):
            raise BaselineError(f"Unsupported baseline file: {relative}")
    return sorted(names)


def _same_source(before: Path, after: Path, names: list[str]) -> None:
    """Build must preserve the prepared source that subsequent patches target."""
    for name in names:
        left, right = before / name, after / name
        if left.is_symlink():
            same = right.is_symlink() and os.readlink(left) == os.readlink(right)
        else:
            same = right.is_file() and not right.is_symlink()
            if same:
                with left.open("rb") as a, right.open("rb") as b:
                    while True:
                        first, second = a.read(1024 * 1024), b.read(1024 * 1024)
                        if first != second:
                            same = False
                            break
                        if not first:
                            break
        if not same:
            raise BaselineError(f"Build changed or deleted prepared source: {name}; declare intentional source generation in prepare")


def _build_variants(task: TaskSpec, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    variants = task.required_variants()
    record = {"schema_version": 1, "kind": "trusted_baseline_variants", "status": "building",
              "software_identity": software_identity(), "task_id": task.task_id,
              "task_identity": task.identity_dict(),
              "variants": {variant.variant_id: f"variant-{index:03d}" for index, variant in enumerate(variants)},
              "variant_results": {variant.variant_id: None for variant in variants},
              "error": None, "duration_sec": 0.0,
              "assurance": "All declared grading variants require their own ready baseline; no candidate patch, grader or model channel is provided."}
    write_json(output / "baseline.json", record)
    started = time.monotonic()
    try:
        source = source_identity(task)
        record["source_identity"] = source
        if source["kind"] == "git":
            task = replace(task, source=replace(task.source, revision=source["revision"]))
        for variant in variants:
            name = variant.variant_id
            record["variant_results"][name] = build_baseline(
                task.as_variant(name), output / record["variants"][name])
            write_json(output / "baseline.json", record)
        record["status"] = "ready" if all(
            result is not None and result["status"] == "ready" for result in record["variant_results"].values()
        ) else "failed"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if not isinstance(exc, Exception):
            raise
    finally:
        record["duration_sec"] = time.monotonic() - started
        write_json(output / "baseline.json", record)
    return record


def build_baseline(task: TaskSpec, output_dir: str | Path) -> dict:
    """Build one unpatched baseline; only status=ready is reusable.

    source/ is the prepared, pre-build source view; workspace/ includes built
    artifacts. Local outputs may contain nonportable absolute paths and remain
    development-only. Docker consumers use the stable /workspace mount.
    """
    output = task.validate_output_path(output_dir)
    if task.variants:
        return _build_variants(task, output)
    output.mkdir(parents=True, exist_ok=False)
    workspace, prepared_source = output / "workspace", output / "source"
    record = {"schema_version": 1, "kind": "trusted_baseline_build", "status": "building",
              "software_identity": software_identity(), "task_id": task.task_id,
              "task_identity": task.identity_dict(),
              "source": "source", "workspace": "workspace", "workspace_at_build": str(workspace),
              "source_files": [],
              "compatibility": None, "execution_environment": None, "session": None,
              "artifact_contract": {},
              "stages": [], "error": None, "duration_sec": 0.0,
              "patch_applied": False, "grader_provided": False,
              "assurance": "Operator-selected trusted input; metadata matching is not tamper detection. Always build after applying a candidate patch. Incremental dependency correctness requires task-specific validation."}
    write_json(output / "baseline.json", record)
    started = time.monotonic()
    phase, session = "compatibility", None
    try:
        compatibility = _compatibility(task)
        record["compatibility"] = compatibility
        write_json(output / "baseline.json", record)
        if compatibility["source"]["kind"] == "git":
            task = replace(task, source=replace(task.source, revision=compatibility["source"]["revision"]))
        phase = "source"
        record["materialized_source"] = materialize_source(task, workspace)
        pinned = replace(task, environment=replace(task.environment,
                         image=compatibility["environment"]["image"]))
        execution_task = workspace_environment(pinned, workspace, record["artifact_contract"])
        record["execution_environment"] = execution_task.environment.to_dict()
        phase = "environment"
        with create_session(execution_task, workspace, None) as session:
            for stage, commands, timeout in (
                ("prepare", task.environment.prepare, task.environment.prepare_timeout_sec),
                ("build", task.environment.build, task.environment.build_timeout_sec),
            ):
                phase = stage
                record["active_stage"] = stage
                record["duration_sec"] = time.monotonic() - started
                write_json(output / "baseline.json", record)
                if stage == "build":
                    session.sync_from_container()
                    reserved = set(record["artifact_contract"]["runtime_paths"])
                    shutil.copytree(workspace, prepared_source, symlinks=True,
                        ignore=lambda path, names: [name for name in names
                            if ((Path(path) / name).relative_to(workspace)).as_posix() in reserved])
                    record["source_files"] = _files(prepared_source)
                for index, argv in enumerate(commands):
                    log = f"logs/{stage}-{index:03d}.log"
                    result = session.run(argv, timeout, output / log)
                    record["stages"].append({"stage": stage, "argv": list(argv), "log": log, **result.to_dict()})
                    record["duration_sec"] = time.monotonic() - started
                    write_json(output / "baseline.json", record)
                    if result.exit_code or result.timed_out or result.output_limited:
                        raise BaselineError(f"Baseline {stage} failed; see {log}")
            phase = "capture"
        if not session.execution_stopped or not session.workspace_capture_ready:
            raise BaselineError("Baseline execution was not stopped and captured")
        phase = "source_contract"
        _files(workspace)
        _same_source(prepared_source, workspace, record["source_files"])
        record["status"] = "ready"
    except BaseException as exc:
        record["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        record["error"] = {"stage": phase, "type": type(exc).__name__, "message": str(exc)}
        if session is None and getattr(exc, "execution_session_info", None) is not None:
            record["session"] = exc.execution_session_info
        if not isinstance(exc, Exception):
            raise
    finally:
        if session is not None:
            record["session"] = session.info
        record["active_stage"] = None
        record["duration_sec"] = time.monotonic() - started
        write_json(output / "baseline.json", record)
    return record


def fork_baseline(task: TaskSpec, artifact_dir: str | Path, destination: str | Path,
                  *, expected_source: dict | None = None) -> dict:
    """Copy a manually selected artifact, retaining incremental-build mtimes.

    expected_source lets a controller supply preserved origin metadata after
    snapshotting removed Git metadata. It is trusted provenance, not a content
    check. This operation never applies a patch or skips the required build.
    """
    artifact = Path(artifact_dir).expanduser().resolve()
    destination = task.validate_output_path(destination)
    if destination == artifact or destination.is_relative_to(artifact):
        raise BaselineError("A fork must be outside the trusted baseline artifact")
    try:
        record = json.loads((artifact / "baseline.json").read_text())
    except (OSError, ValueError) as exc:
        raise BaselineError(f"Cannot read baseline artifact: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("kind") != "trusted_baseline_build" or record.get("status") != "ready":
        raise BaselineError("Only a completed baseline artifact can be reused")
    compatibility = record.get("compatibility")
    expected = _compatibility(task, expected_source)
    if not isinstance(compatibility, dict) or {key: compatibility.get(key) for key in expected} != expected:
        raise BaselineError("Baseline source, image or build environment is incompatible")
    source_files = record.get("source_files")
    if not isinstance(source_files, list) or any(not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts for name in source_files):
        raise BaselineError("Invalid prepared source inventory")
    workspace = artifact / "workspace"
    if not workspace.is_dir() or not (artifact / "source").is_dir():
        raise BaselineError("Baseline source or built workspace is absent")
    _files(workspace)
    shutil.copytree(workspace, destination, symlinks=True, copy_function=shutil.copy2)
    environment = record["execution_environment"]
    if task.environment.backend == "local":
        # Publication/moving the artifact does not rewrite its build evidence.
        # Resolve runtime directories from where the build actually happened.
        environment = {**environment, "environment": {
            key: value.replace(record.get("workspace_at_build", str(workspace)), str(destination))
            for key, value in environment["environment"].items()}}
    return {"artifact_path": str(artifact), "source_files": list(source_files),
            "artifact_contract": dict(record["artifact_contract"]),
            "compatibility": expected, "execution_environment": environment,
            "producer_task_identity": record.get("task_identity", compatibility.get("task_identity")),
            "assurance": record["assurance"]}
