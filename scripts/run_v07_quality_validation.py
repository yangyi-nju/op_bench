#!/usr/bin/env python3
"""Execute the frozen v0.7 Agent cohorts without exposing private runtime output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.factory.artifacts import load_regular_file_bytes  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.experiment_report import (  # noqa: E402
    McpExperimentCohortContract,
    load_mcp_experiment_contract,
)
from op_bench.runtime.legacy import (  # noqa: E402
    LegacyV05Defaults,
    agent_spec_for_v1_adapter,
    full_task_spec_from_v05,
    run_manifest_from_v05_dataset,
)
from op_bench.runtime.task_view import project_agent_task_view  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402
from op_bench.runtime.integrity import (  # noqa: E402
    load_run_manifest_artifact,
    verify_run_artifacts,
)
from scripts.build_v07_quality_validation_contract import (  # noqa: E402
    CODEX_CLI_VERSION,
    MODEL_ID,
    build_validation_contract,
)
from scripts.run_experiment import detect_codex_cli_version  # noqa: E402


@dataclass(frozen=True)
class CohortExecution:
    index: int
    contract: McpExperimentCohortContract
    canonical_task_ids: tuple[str, ...]
    manifest: object


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(load_regular_file_bytes(path)).hexdigest()


def _bounded_process_failure_code(stderr: bytes) -> str | None:
    """Map owned runner messages to a bounded code without retaining output."""

    known = (
        (b"cannot construct frozen v1 runtime inputs", "input_construction_failed"),
        (
            b"selected Task Runtime Profile does not match --runtime-profile",
            "runtime_profile_mismatch",
        ),
        (b"exact process group recovery evidence is invalid", "recovery_invalid"),
        (
            b"previous exact process group cleanup is still unproven",
            "recovery_unproven",
        ),
        (
            b"v1 orchestration failed before a valid run result",
            "orchestration_failed",
        ),
        (
            b"exact process group cleanup failed without recovery identity",
            "cleanup_identity_missing",
        ),
        (
            b"exact process group cleanup recovery could not be persisted",
            "cleanup_recovery_persist_failed",
        ),
        (
            b"exact process group cleanup is unproven; recovery evidence persisted",
            "cleanup_unproven",
        ),
        (b"v1 run did not pass Integrity", "integrity_failed"),
        (b"v1 summary is unavailable or invalid", "summary_invalid"),
        (
            b"v1 run completed with infrastructure-invalid Attempts",
            "infrastructure_invalid",
        ),
    )
    for message, code in known:
        if message in stderr:
            return code
    return None


def _bounded_process_diagnostic(stderr: bytes) -> dict[str, str]:
    match = re.search(
        rb"\[error_type=([A-Za-z][A-Za-z0-9_]*), "
        rb"error_digest=(sha256:[0-9a-f]{64})\]",
        stderr,
    )
    if match is None:
        return {}
    return {
        "error_type": match.group(1).decode("ascii"),
        "error_digest": match.group(2).decode("ascii"),
    }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _release_inputs(root: Path, release_path: Path) -> tuple[Path, dict[str, dict[str, Any]]]:
    release = _load_json(release_path)
    try:
        dataset_ref = release["datasets"]["cumulative"]["dataset"]
        request_ref = release["request"]
        dataset_path = root / str(dataset_ref["relative_path"])
        request_path = root / str(request_ref["relative_path"])
    except (KeyError, TypeError) as exc:
        raise ContractError("release: missing frozen Dataset inputs") from exc
    if _file_digest(dataset_path) != dataset_ref.get("content_hash"):
        raise ContractError("release: Dataset hash drift")
    if _file_digest(request_path) != request_ref.get("content_hash"):
        raise ContractError("release: request hash drift")
    records = _load_json(request_path).get("records")
    if not isinstance(records, list) or len(records) != 50:
        raise ContractError("release request: expected 50 records")
    by_public: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("public_task_id"), str):
            raise ContractError("release request: invalid Task record")
        public_id = str(record["public_task_id"])
        if public_id in by_public:
            raise ContractError("release request: duplicate public Task ID")
        by_public[public_id] = record
    return dataset_path, by_public


def cohort_executions(
    root: Path = ROOT,
    *,
    release_path: Path | str = "factory/v0.7/p9/release_manifest.json",
    contract_path: Path | str = "factory/v0.7/p9/validation_contract.json",
) -> tuple[CohortExecution, ...]:
    root = root.resolve()
    release = Path(release_path)
    contract_file = Path(contract_path)
    if not release.is_absolute():
        release = root / release
    if not contract_file.is_absolute():
        contract_file = root / contract_file
    contract = load_mcp_experiment_contract(contract_file)
    if contract != build_validation_contract(root, release):
        raise ContractError("validation contract does not match frozen release inputs")
    dataset_path, by_public = _release_inputs(root, release)
    agent = agent_spec_for_v1_adapter(
        "codex_mcp_canonical",
        model_id=MODEL_ID,
        codex_cli_version=CODEX_CLI_VERSION,
    )
    standard = LegacyV05Defaults.standard()
    executions: list[CohortExecution] = []
    for index, cohort in enumerate(contract.cohorts, start=1):
        if cohort.binding is None:
            raise ContractError("validation cohort is missing its frozen binding")
        canonical_ids = tuple(str(by_public[public_id]["task_id"]) for public_id in cohort.task_ids)
        first_task_path = root / str(by_public[cohort.task_ids[0]]["task_path"]) / "task.json"
        from op_bench.task import TaskManifest

        profile = full_task_spec_from_v05(TaskManifest.load(first_task_path)).runtime
        repeats = {item for _, item in cohort.task_repeats}
        if len(repeats) != 1:
            raise ContractError("cohort contains mixed repeat budgets")
        repeat_tuple = next(iter(repeats))
        defaults = replace(
            standard,
            budget_policy=replace(standard.budget_policy, wall_clock_ms=profile.timeout_ms),
        )
        manifest = run_manifest_from_v05_dataset(
            dataset_path,
            agents=(agent,),
            repeat=len(repeat_tuple),
            created_at="1970-01-01T00:00:00Z",
            defaults=defaults,
            selected_task_ids=canonical_ids,
        )
        views = tuple(
            project_agent_task_view(task, manifest.capability_policy, manifest.budget_policy)
            for task in manifest.tasks
        )
        actual_binding = {
            "run_manifest_digest": canonical_sha256(manifest.to_dict()),
            "runtime_profile_digest": canonical_sha256(profile.to_dict()),
            "capability_policy_digest": canonical_sha256(
                manifest.capability_policy.to_dict()
            ),
            "budget_policy_digest": canonical_sha256(manifest.budget_policy.to_dict()),
            "task_view_digests": tuple(
                (view.task.identifier, view.content_hash) for view in views
            ),
        }
        expected_binding = {
            "run_manifest_digest": cohort.binding.run_manifest_digest,
            "runtime_profile_digest": cohort.binding.runtime_profile_digest,
            "capability_policy_digest": cohort.binding.capability_policy_digest,
            "budget_policy_digest": cohort.binding.budget_policy_digest,
            "task_view_digests": cohort.binding.task_view_digests,
        }
        if actual_binding != expected_binding or profile.profile_id != cohort.profile_id:
            raise ContractError("cohort execution inputs do not match frozen binding")
        executions.append(
            CohortExecution(
                index=index,
                contract=cohort,
                canonical_task_ids=canonical_ids,
                manifest=manifest,
            )
        )
    return tuple(executions)


def recommended_canaries(
    executions: tuple[CohortExecution, ...],
    task_records: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    def origin(task_id: str) -> str:
        return str(task_records[task_id]["origin"])

    def modes(task_id: str) -> tuple[str, ...]:
        taxonomy = task_records[task_id].get("taxonomy")
        if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("modes"), list):
            raise ContractError("validation canary Task is missing mode metadata")
        return tuple(str(value) for value in taxonomy["modes"])

    predicates = (
        lambda execution, task_id: (
            origin(task_id) == "retained_historical"
            and execution.contract.profile_id == "remote-cpu-pytorch-2.6-py311-v1"
        ),
        lambda execution, task_id: (
            origin(task_id) != "retained_historical"
            and execution.contract.profile_id.startswith("remote-cpu-")
            and "compile" in modes(task_id)
        ),
        lambda execution, task_id: "cuda-overlay" in execution.contract.profile_id,
        lambda execution, task_id: "cuda-kernel" in execution.contract.profile_id,
    )
    selected: list[str] = []
    for predicate in predicates:
        candidate = next(
            (
                task_id
                for execution in executions
                for task_id in execution.contract.task_ids
                if predicate(execution, task_id)
            ),
            None,
        )
        if candidate is None:
            raise ContractError("required validation canary category is unavailable")
        if candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def _summary_record(
    execution: CohortExecution,
    output_root: Path,
    *,
    returncode: int,
    process_digest: str,
) -> dict[str, object]:
    summary_path = output_root / "summary.json"
    totals: dict[str, Any] = {}
    if summary_path.is_file() and not summary_path.is_symlink():
        summary = _load_json(summary_path)
        raw_totals = summary.get("totals")
        if isinstance(raw_totals, dict):
            totals = raw_totals
    expected_full = len(execution.manifest.expected_attempts)
    valid = totals.get("valid") if isinstance(totals.get("valid"), int) else 0
    infrastructure_invalid = (
        totals.get("infrastructure_invalid")
        if isinstance(totals.get("infrastructure_invalid"), int)
        else 0
    )
    return {
        "cohort_index": execution.index,
        "profile_id": execution.contract.profile_id,
        "public_task_ids": list(execution.contract.task_ids),
        "expected_attempts": expected_full,
        "selected_expected_attempts": (
            totals.get("expected") if isinstance(totals.get("expected"), int) else 0
        ),
        "valid_attempts": valid,
        "infrastructure_invalid": infrastructure_invalid,
        "complete": returncode == 0 and valid == expected_full and infrastructure_invalid == 0,
        "output_root": output_root.relative_to(ROOT).as_posix(),
        "process_returncode": returncode,
        "process_digest": process_digest,
    }


def _write_index(
    output_root: Path,
    *,
    contract_path: Path,
    created_at: str,
    records: list[dict[str, object]],
) -> None:
    payload: dict[str, object] = {
        "contract_type": "quality_validation_execution_index",
        "schema_version": "v1",
        "release_version": "v0.7",
        "created_at": created_at,
        "contract_path": contract_path.relative_to(ROOT).as_posix(),
        "contract_digest": _file_digest(contract_path),
        "cohort_count": 17,
        "expected_attempt_count": 122,
        "completed_cohort_count": sum(record["complete"] is True for record in records),
        "valid_attempt_count": sum(int(record["valid_attempts"]) for record in records),
        "records": sorted(records, key=lambda record: int(record["cohort_index"])),
    }
    payload["content_hash"] = canonical_sha256(payload)
    path = output_root / "index.json"
    temporary = output_root / ".index.json.tmp"
    temporary.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_quality_validation_index(
    root: Path,
    index_path: Path,
    *,
    release_path: Path | str = "factory/v0.7/p9/release_manifest.json",
    contract_path: Path | str = "factory/v0.7/p9/validation_contract.json",
) -> list[str]:
    """Verify final 17-cohort/122-Attempt coverage and every Integrity graph."""

    errors: list[str] = []
    try:
        root = root.resolve()
        selected_contract = Path(contract_path)
        if not selected_contract.is_absolute():
            selected_contract = root / selected_contract
        index = _load_json(index_path)
        core = dict(index)
        observed_hash = core.pop("content_hash", None)
        if observed_hash != canonical_sha256(core):
            raise ContractError("validation execution index content hash drift")
        if (
            index.get("contract_type") != "quality_validation_execution_index"
            or index.get("schema_version") != "v1"
            or index.get("release_version") != "v0.7"
            or index.get("contract_digest") != _file_digest(selected_contract)
            or index.get("cohort_count") != 17
            or index.get("expected_attempt_count") != 122
            or index.get("completed_cohort_count") != 17
            or index.get("valid_attempt_count") != 122
        ):
            raise ContractError("validation execution index final counters mismatch")
        executions = cohort_executions(
            root,
            release_path=release_path,
            contract_path=selected_contract,
        )
        records = index.get("records")
        if not isinstance(records, list) or len(records) != len(executions):
            raise ContractError("validation execution index cohort partition mismatch")
        by_index = {
            int(record["cohort_index"]): record
            for record in records
            if isinstance(record, dict) and isinstance(record.get("cohort_index"), int)
        }
        if set(by_index) != {execution.index for execution in executions}:
            raise ContractError("validation execution index cohort identities mismatch")
        for execution in executions:
            record = by_index[execution.index]
            expected_attempts = len(execution.manifest.expected_attempts)
            if (
                record.get("profile_id") != execution.contract.profile_id
                or record.get("public_task_ids") != list(execution.contract.task_ids)
                or record.get("expected_attempts") != expected_attempts
                or record.get("selected_expected_attempts") != expected_attempts
                or record.get("valid_attempts") != expected_attempts
                or record.get("infrastructure_invalid") != 0
                or record.get("complete") is not True
                or record.get("process_returncode") != 0
            ):
                raise ContractError("validation execution cohort result mismatch")
            invocations = record.get("invocations")
            if not isinstance(invocations, list) or not invocations:
                raise ContractError("validation execution invocation history is missing")
            last_invocation = invocations[-1]
            if (
                not isinstance(last_invocation, dict)
                or last_invocation.get("process_returncode") != record.get("process_returncode")
                or last_invocation.get("process_digest") != record.get("process_digest")
            ):
                raise ContractError("validation execution invocation history mismatch")
            output_value = record.get("output_root")
            if not isinstance(output_value, str):
                raise ContractError("validation execution cohort output is missing")
            output = root / output_value
            manifest = load_run_manifest_artifact(output)
            if canonical_sha256(manifest.to_dict()) != execution.contract.binding.run_manifest_digest:
                raise ContractError("validation execution RunManifest identity mismatch")
            integrity = verify_run_artifacts(output, manifest)
            if integrity.status != "passed" or any(
                check.status != "passed" for check in integrity.checks
            ):
                raise ContractError("validation execution Integrity failed")
    except (ContractError, OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("canary", "full"), required=True)
    parser.add_argument("--release", type=Path, default=Path("factory/v0.7/p9/release_manifest.json"))
    parser.add_argument("--contract", type=Path, default=Path("factory/v0.7/p9/validation_contract.json"))
    parser.add_argument("--dataset", type=Path, default=Path("datasets/pytorch_v0.7/dataset.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/v0.7_quality_validation"))
    parser.add_argument("--runtime-profile-registry", type=Path, default=Path("configs/runtime_profiles.v1.json"))
    parser.add_argument("--target-config", type=Path, default=Path("configs/remote_hosts.json"))
    parser.add_argument("--max-cohorts", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = ROOT.resolve()
    release = args.release if args.release.is_absolute() else root / args.release
    contract_path = args.contract if args.contract.is_absolute() else root / args.contract
    dataset_path = args.dataset if args.dataset.is_absolute() else root / args.dataset
    output_root = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    profile_registry = (
        args.runtime_profile_registry
        if args.runtime_profile_registry.is_absolute()
        else root / args.runtime_profile_registry
    )
    target_config = args.target_config if args.target_config.is_absolute() else root / args.target_config
    try:
        if detect_codex_cli_version() != CODEX_CLI_VERSION:
            raise ContractError("local Codex CLI does not match the frozen contract")
        executions = cohort_executions(
            root,
            release_path=release,
            contract_path=contract_path,
        )
        _, by_public = _release_inputs(root, release)
        canaries = set(recommended_canaries(executions, by_public))
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"[quality_validation_inputs_invalid] {exc}", file=sys.stderr)
        return 2
    selected_executions = (
        tuple(
            execution
            for execution in executions
            if canaries.intersection(execution.contract.task_ids)
        )
        if args.mode == "canary"
        else executions
    )
    if args.max_cohorts is not None:
        if args.max_cohorts < 1:
            print("[quality_validation_inputs_invalid] --max-cohorts must be positive", file=sys.stderr)
            return 2
        selected_executions = selected_executions[: args.max_cohorts]
    output_root.mkdir(parents=True, exist_ok=True)
    created_at = _now()
    index_path = output_root / "index.json"
    existing_records: dict[int, dict[str, object]] = {}
    if index_path.is_file():
        existing = _load_json(index_path)
        if existing.get("contract_digest") != _file_digest(contract_path):
            print("[quality_validation_inputs_invalid] existing index contract drift", file=sys.stderr)
            return 2
        created_at = str(existing.get("created_at"))
        for record in existing.get("records", []):
            if isinstance(record, dict) and isinstance(record.get("cohort_index"), int):
                existing_records[int(record["cohort_index"])] = record

    for execution in selected_executions:
        cohort_root = output_root / f"cohort-{execution.index:02d}"
        command = [
            sys.executable,
            str(root / "scripts/run_experiment.py"),
            "--dataset",
            str(dataset_path),
            "--verified-only",
            "--only-tasks",
            *execution.canonical_task_ids,
            "--agent",
            "codex_mcp_canonical",
            "--agent-repeat",
            str(len(execution.contract.task_repeats[0][1])),
            "--output-dir",
            str(cohort_root),
            "--runtime-protocol",
            "v1",
            "--runtime-profile",
            execution.contract.profile_id,
            "--runtime-profile-registry",
            str(profile_registry),
            "--target-config",
            str(target_config),
            "--codex-model",
            MODEL_ID,
            "--enable-external-canary",
            "--quiet",
        ]
        if args.mode == "canary":
            public_canaries = canaries.intersection(execution.contract.task_ids)
            attempt_ids = [
                attempt.attempt_id
                for attempt in execution.manifest.expected_attempts
                if attempt.task.identifier in public_canaries and attempt.repeat == 1
            ]
            command.extend(("--only-attempt-ids", *attempt_ids))
        execution_environment = dict(os.environ)
        execution_environment["OP_BENCH_REMOTE_HOSTS_PATH"] = str(target_config)
        completed = subprocess.run(
            command,
            check=False,
            cwd=root,
            env=execution_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        process_digest = canonical_sha256(
            {
                "process_returncode": completed.returncode,
                "stdout_bytes": len(completed.stdout),
                "stderr_bytes": len(completed.stderr),
            }
        )
        previous = existing_records.get(execution.index)
        record = _summary_record(
            execution,
            cohort_root,
            returncode=completed.returncode,
            process_digest=process_digest,
        )
        failure_code = _bounded_process_failure_code(completed.stderr)
        process_diagnostic = _bounded_process_diagnostic(completed.stderr)
        if failure_code is not None:
            record["failure_code"] = failure_code
        record.update(process_diagnostic)
        invocation_history = (
            list(previous.get("invocations", []))
            if isinstance(previous, dict) and isinstance(previous.get("invocations"), list)
            else []
        )
        invocation_history.append(
            {
                "recorded_at": _now(),
                "mode": args.mode,
                "process_returncode": completed.returncode,
                "process_digest": process_digest,
                **({"failure_code": failure_code} if failure_code is not None else {}),
                **process_diagnostic,
            }
        )
        record["invocations"] = invocation_history
        existing_records[execution.index] = record
        _write_index(
            output_root,
            contract_path=contract_path,
            created_at=created_at,
            records=list(existing_records.values()),
        )
        record = existing_records[execution.index]
        print(
            canonical_json(
                {
                    "cohort_index": execution.index,
                    "complete": record["complete"],
                    "infrastructure_invalid": record["infrastructure_invalid"],
                    "mode": args.mode,
                    "process_returncode": completed.returncode,
                    "failure_code": failure_code,
                    **process_diagnostic,
                    "valid_attempts": record["valid_attempts"],
                }
            )
        )
        if completed.returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
