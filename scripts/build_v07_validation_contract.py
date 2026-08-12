#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.experiment_report import (  # noqa: E402
    McpExperimentCohortContract,
    McpExperimentContract,
)
from op_bench.runtime.legacy import (  # noqa: E402
    agent_spec_for_v1_adapter,
    run_manifest_from_v05_dataset,
)
from op_bench.runtime.task_view import (  # noqa: E402
    assert_public_artifact_safe,
)
from op_bench.runtime.validation import ContractError  # noqa: E402


COHORT_TASKS = (
    (
        "remote-cpu-boundary-torch2.2-py311-v1",
        ("pytorch__117065__index_copy_zero_dim",),
    ),
    (
        "remote-cpu-boundary-torch2.3-py311-v1",
        ("pytorch__118762__weight_norm_default_dim",),
    ),
    (
        "remote-cpu-boundary-torch2.4-py311-v1",
        ("pytorch__126461__cummin_rank_zero",),
    ),
    (
        "remote-cpu-source-boundary-py311-v1",
        (
            "pytorch__143792__addmv_empty_matrix",
            "pytorch__147352__storage_offset_overflow",
        ),
    ),
    (
        "remote-cuda-boundary-torch2.6-cu124-v1",
        ("pytorch__139751__triton_ygrid_mask",),
    ),
)
REPEATS = (1, 2, 3)


def build_validation_contract(
    root: Path = ROOT,
) -> McpExperimentContract:
    root = root.resolve()
    historical_dataset_path = (
        root
        / "archives/v0.7-pre-quality/datasets/pytorch_v0.7_boundary/dataset.json"
    )
    dataset_path = (
        historical_dataset_path
        if historical_dataset_path.is_file()
        else root / "datasets/pytorch_v0.7_boundary/dataset.json"
    )
    selected_task_ids = tuple(
        task_id
        for _, task_ids in COHORT_TASKS
        for task_id in task_ids
    )
    manifest = run_manifest_from_v05_dataset(
        dataset_path,
        agents=(agent_spec_for_v1_adapter("scripted_canonical"),),
        repeat=3,
        created_at="1970-01-01T00:00:00Z",
        selected_task_ids=selected_task_ids,
    )
    mapping_payload = json.loads(
        (root / "factory/v0.7/p6/public_task_ids.json").read_text(
            encoding="utf-8"
        )
    )
    public_by_canonical = {
        item["task_id"]: item["public_task_id"]
        for item in mapping_payload["tasks"]
    }
    if set(selected_task_ids) - set(public_by_canonical):
        raise ContractError(
            "Validation partition is missing frozen public Task identities"
        )
    actual_profiles = {
        task.task.identifier: task.runtime.profile_id
        for task in manifest.tasks
    }
    expected_profiles = {
        public_by_canonical[task_id]: profile_id
        for profile_id, task_ids in COHORT_TASKS
        for task_id in task_ids
    }
    if actual_profiles != expected_profiles:
        raise ContractError(
            "Validation partition does not match Task Runtime Profiles"
        )
    contract = McpExperimentContract(
        dataset_identifier=manifest.dataset.identifier,
        dataset_digest=manifest.dataset.digest,
        platform_version=manifest.platform_version,
        cohorts=tuple(
            McpExperimentCohortContract(
                profile_id=profile_id,
                task_repeats=tuple(
                    (public_by_canonical[task_id], REPEATS)
                    for task_id in task_ids
                ),
            )
            for profile_id, task_ids in COHORT_TASKS
        ),
    )
    if contract.expected_attempt_count != 18:
        raise ContractError("Validation contract must contain 18 Attempts")
    assert_public_artifact_safe(contract.to_dict())
    return contract


def _write_contract(
    path: Path,
    contract: McpExperimentContract,
) -> str:
    encoded = (canonical_json(contract.to_dict()) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_symlink() or path.read_bytes() != encoded:
            raise ContractError("Output exists with different bytes")
        return "verified"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return "created"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic OpBench v0.7 validation contract."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        status = _write_contract(
            args.output,
            build_validation_contract(args.repo_root),
        )
    except (ContractError, OSError, ValueError, KeyError) as exc:
        print(f"[validation_contract_invalid] {exc}", file=sys.stderr)
        return 2
    print(
        canonical_json(
            {
                "expected_attempts": 18,
                "status": status,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
