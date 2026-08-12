#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.experiment_report import (
    FORMAL_MCP_EXPERIMENT_CONTRACT,
    McpExperimentContract,
    build_mcp_experiment_report,
    load_mcp_experiment_contract,
    load_public_task_id_aliases,
    write_mcp_experiment_report,
)
from op_bench.runtime.validation import ContractError
from op_bench.factory.artifacts import load_regular_file_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic public MCP validation report."
    )
    parser.add_argument("--run-root", action="append")
    parser.add_argument("--input-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-root")
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-cli-version")
    parser.add_argument("--contract")
    parser.add_argument("--public-task-id-mapping")
    parser.add_argument("--quality-release-request")
    return parser


def _quality_task_metadata(path: Path) -> dict[str, dict[str, object]]:
    value = json.loads(load_regular_file_bytes(path).decode("utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ContractError("quality release request is invalid")
    result: dict[str, dict[str, object]] = {}
    for record in value["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("public_task_id"), str):
            raise ContractError("quality release request contains an invalid Task record")
        task_id = str(record["public_task_id"])
        if task_id in result:
            raise ContractError("quality release request contains duplicate public Task IDs")
        result[task_id] = {
            "origin": record.get("origin"),
            "difficulty": record.get("difficulty"),
            "slices": record.get("slices"),
            "taxonomy": record.get("taxonomy"),
        }
    return result


def main(
    argv: list[str] | None = None,
    *,
    experiment_contract: McpExperimentContract | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        selected_contract = (
            load_mcp_experiment_contract(Path(args.contract))
            if args.contract is not None
            else experiment_contract or FORMAL_MCP_EXPERIMENT_CONTRACT
        )
        if bool(args.run_root) == bool(args.input_root):
            raise ContractError("provide exactly one --run-root set or --input-root")
        if bool(args.output_dir) == bool(args.output_root):
            raise ContractError("provide exactly one --output-dir or --output-root")
        if args.input_root is not None:
            input_root = Path(args.input_root)
            run_roots = tuple(
                input_root / f"cohort-{index:02d}"
                for index in range(1, len(selected_contract.cohorts) + 1)
            )
        else:
            run_roots = tuple(Path(value) for value in args.run_root)
        frozen = selected_contract.frozen_config
        expected_model = args.expected_model or (
            frozen.model_id if frozen is not None else None
        )
        expected_cli = args.expected_cli_version or (
            frozen.codex_cli_version if frozen is not None else None
        )
        if expected_model is None or expected_cli is None:
            raise ContractError(
                "expected model and CLI version are required for an unbound contract"
            )
        quality_request = (
            Path(args.quality_release_request)
            if args.quality_release_request is not None
            else (
                ROOT / "factory/v0.7/p9/release_request.json"
                if frozen is not None
                else None
            )
        )
        index, summary = build_mcp_experiment_report(
            run_roots,
            expected_adapter_id="codex_mcp_canonical",
            expected_model_id=expected_model,
            expected_codex_cli_version=expected_cli,
            experiment_contract=selected_contract,
            task_id_aliases=(
                load_public_task_id_aliases(args.public_task_id_mapping)
                if args.public_task_id_mapping is not None
                else None
            ),
            task_metadata=(
                _quality_task_metadata(quality_request)
                if quality_request is not None
                else None
            ),
        )
        write_mcp_experiment_report(
            Path(args.output_dir or args.output_root), index, summary
        )
    except (ContractError, OSError, ValueError, TypeError) as exc:
        print(f"MCP experiment report failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"MCP experiment report: {summary['totals']['attempts']} Attempts, "
        f"{summary['totals']['trace_complete']} complete traces"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
