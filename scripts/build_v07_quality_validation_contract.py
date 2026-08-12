#!/usr/bin/env python3
"""Build the frozen 122-Attempt Agent validation contract for v0.7."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.dataset import DatasetManifest  # noqa: E402
from op_bench.factory.artifacts import load_regular_file_bytes  # noqa: E402
from op_bench.runtime.canonical import canonical_json, canonical_sha256  # noqa: E402
from op_bench.runtime.codex_mcp_adapter import render_mcp_prompt  # noqa: E402
from op_bench.runtime.experiment_report import (  # noqa: E402
    McpExperimentCohortBinding,
    McpExperimentCohortContract,
    McpExperimentContract,
    McpExperimentFrozenConfig,
)
from op_bench.runtime.legacy import (  # noqa: E402
    LegacyV05Defaults,
    agent_spec_for_v1_adapter,
    full_task_spec_from_v05,
    run_manifest_from_v05_dataset,
)
from op_bench.runtime.task_view import (  # noqa: E402
    assert_public_artifact_safe,
    project_agent_task_view,
)
from op_bench.runtime.validation import ContractError  # noqa: E402


MODEL_ID = "gpt-5.6-sol"
CODEX_CLI_VERSION = "codex-cli 0.147.0-alpha.1.2"
RETAINED_REPEATS = (1,)
EXPANSION_REPEATS = (1, 2, 3)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path}: expected object")
    return value


def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(load_regular_file_bytes(path)).hexdigest()


def _renderer_digest() -> str:
    source = inspect.getsource(render_mcp_prompt).encode("utf-8")
    return "sha256:" + hashlib.sha256(source).hexdigest()


def _release_records(root: Path, release_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    release = _load_json(release_path)
    if (
        release.get("contract_type") != "quality_release_manifest"
        or release.get("release_version") != "v0.7"
        or release.get("composition")
        != {
            "new_or_replacement": 36,
            "retained_historical": 14,
            "total": 50,
        }
    ):
        raise ContractError("release: expected the exact final v0.7 composition")
    try:
        dataset_ref = release["datasets"]["cumulative"]["dataset"]
        request_ref = release["request"]
        dataset_path = root / str(dataset_ref["relative_path"])
        request_path = root / str(request_ref["relative_path"])
    except (KeyError, TypeError) as exc:
        raise ContractError("release: missing cumulative Dataset or request") from exc
    if _file_digest(dataset_path) != dataset_ref.get("content_hash"):
        raise ContractError("release: cumulative Dataset hash drift")
    if _file_digest(request_path) != request_ref.get("content_hash"):
        raise ContractError("release: release request hash drift")
    request = _load_json(request_path)
    records = request.get("records")
    if not isinstance(records, list) or len(records) != 50:
        raise ContractError("release request: expected exactly 50 Task records")
    selected: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ContractError(f"release request records[{index}]: expected object")
        if record.get("origin") not in {"retained_historical", "new", "replacement"}:
            raise ContractError(f"release request records[{index}]: invalid origin")
        selected.append(record)
    return dataset_path, selected


def final_task_origins_by_public_task_id(
    root: Path = ROOT,
    release_path: Path | str = "factory/v0.7/p9/release_manifest.json",
) -> dict[str, str]:
    root = root.resolve()
    selected_release = Path(release_path)
    if not selected_release.is_absolute():
        selected_release = root / selected_release
    _, records = _release_records(root, selected_release)
    result: dict[str, str] = {}
    for record in records:
        public_id = record.get("public_task_id")
        origin = record.get("origin")
        if not isinstance(public_id, str) or not isinstance(origin, str):
            raise ContractError("release request: invalid public Task identity")
        if public_id in result:
            raise ContractError("release request: duplicate public Task identity")
        result[public_id] = origin
    return result


def build_validation_contract(
    root: Path = ROOT,
    release_path: Path | str = "factory/v0.7/p9/release_manifest.json",
) -> McpExperimentContract:
    root = root.resolve()
    selected_release = Path(release_path)
    if not selected_release.is_absolute():
        selected_release = root / selected_release
    dataset_path, records = _release_records(root, selected_release)
    dataset = DatasetManifest.load(dataset_path)
    tasks = dataset.load_tasks(verified_only=True)
    if len(tasks) != 50:
        raise ContractError("validation contract requires exactly 50 verified Tasks")
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != 50:
        raise ContractError("validation contract Dataset contains duplicate Task IDs")

    grouped: dict[tuple[str, tuple[int, ...]], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = record.get("task_id")
        public_id = record.get("public_task_id")
        if not isinstance(task_id, str) or task_id not in tasks_by_id:
            raise ContractError("release request Task is absent from the Dataset")
        if not isinstance(public_id, str):
            raise ContractError("release request public Task identity is invalid")
        task = tasks_by_id[task_id]
        if task.public_task_id != public_id:
            raise ContractError(f"{task_id}: public Task identity drift")
        profile_id = full_task_spec_from_v05(task).runtime.profile_id
        repeats = (
            RETAINED_REPEATS
            if record["origin"] == "retained_historical"
            else EXPANSION_REPEATS
        )
        grouped[(profile_id, repeats)].append(record)

    agent = agent_spec_for_v1_adapter(
        "codex_mcp_canonical",
        model_id=MODEL_ID,
        codex_cli_version=CODEX_CLI_VERSION,
    )
    standard = LegacyV05Defaults.standard()
    cohorts: list[McpExperimentCohortContract] = []
    manifests = []
    for (profile_id, repeats), cohort_records in sorted(
        grouped.items(), key=lambda item: (item[0][0], len(item[0][1]))
    ):
        selected_ids = tuple(sorted(str(record["task_id"]) for record in cohort_records))
        profile = full_task_spec_from_v05(tasks_by_id[selected_ids[0]]).runtime
        if profile.profile_id != profile_id:
            raise ContractError("Runtime Profile grouping drift")
        defaults = replace(
            standard,
            budget_policy=replace(standard.budget_policy, wall_clock_ms=profile.timeout_ms),
        )
        manifest = run_manifest_from_v05_dataset(
            dataset_path,
            agents=(agent,),
            repeat=len(repeats),
            created_at="1970-01-01T00:00:00Z",
            defaults=defaults,
            selected_task_ids=selected_ids,
        )
        if len(manifest.runtime_profiles) != 1 or manifest.runtime_profiles[0] != profile:
            raise ContractError("cohort does not bind one exact Runtime Profile")
        public_repeats = tuple(
            (task.task.identifier, repeats)
            for task in manifest.tasks
        )
        views = tuple(
            project_agent_task_view(
                task,
                manifest.capability_policy,
                manifest.budget_policy,
            )
            for task in manifest.tasks
        )
        binding = McpExperimentCohortBinding(
            run_manifest_digest=canonical_sha256(manifest.to_dict()),
            runtime_profile_digest=canonical_sha256(profile.to_dict()),
            capability_policy_digest=canonical_sha256(
                manifest.capability_policy.to_dict()
            ),
            budget_policy_digest=canonical_sha256(manifest.budget_policy.to_dict()),
            task_view_digests=tuple(
                (view.task.identifier, view.content_hash) for view in views
            ),
        )
        cohorts.append(
            McpExperimentCohortContract(
                profile_id=profile_id,
                task_repeats=public_repeats,
                binding=binding,
            )
        )
        manifests.append(manifest)

    if not manifests:
        raise ContractError("validation contract has no cohorts")
    first = manifests[0]
    invariant_fields = (
        "dataset",
        "platform_version",
        "action_protocol",
        "evaluation_protocol",
        "scoring_protocol",
        "evaluation",
        "retry_policy",
        "termination_policy",
        "scoring",
    )
    for manifest in manifests[1:]:
        for field in invariant_fields:
            if getattr(manifest, field) != getattr(first, field):
                raise ContractError(f"cohort comparability drift: {field}")

    frozen = McpExperimentFrozenConfig(
        release_digest=_file_digest(selected_release),
        adapter_id=agent.adapter.identifier,
        model_id=agent.model.identifier,
        codex_cli_version=CODEX_CLI_VERSION,
        agent_spec_digest=canonical_sha256(agent.to_dict()),
        system_prompt_digest=agent.system_prompt.digest,
        task_prompt_digest=agent.task_prompt.digest,
        prompt_renderer_digest=_renderer_digest(),
        action_protocol=first.action_protocol,
        evaluation_protocol=first.evaluation_protocol,
        scoring_protocol=first.scoring_protocol,
        evaluation_digest=first.evaluation.digest,
        retry_policy_digest=first.retry_policy.digest,
        termination_policy_digest=first.termination_policy.digest,
        scoring_digest=first.scoring.digest,
    )
    contract = McpExperimentContract(
        dataset_identifier=first.dataset.identifier,
        dataset_digest=first.dataset.digest,
        platform_version=first.platform_version,
        cohorts=tuple(cohorts),
        frozen_config=frozen,
    )
    if contract.expected_attempt_count != 122:
        raise ContractError("validation contract must contain exactly 122 Attempts")
    if sum(len(cohort.task_ids) for cohort in contract.cohorts) != 50:
        raise ContractError("validation contract must partition exactly 50 Tasks")
    assert_public_artifact_safe(contract.to_dict())
    return contract


def _write_contract(path: Path, contract: McpExperimentContract) -> str:
    encoded = (canonical_json(contract.to_dict()) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or load_regular_file_bytes(path) != encoded:
            raise ContractError("Output exists with different bytes")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("factory/v0.7/p9/release_manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    try:
        contract = build_validation_contract(root, args.release)
        status = _write_contract(output, contract)
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"[quality_validation_contract_invalid] {exc}", file=sys.stderr)
        return 2
    print(
        canonical_json(
            {
                "cohort_count": len(contract.cohorts),
                "expected_attempts": contract.expected_attempt_count,
                "status": status,
                "task_count": sum(len(cohort.task_ids) for cohort in contract.cohorts),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
