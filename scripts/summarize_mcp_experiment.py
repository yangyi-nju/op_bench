#!/usr/bin/env python

from __future__ import annotations

import argparse
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
    write_mcp_experiment_report,
)
from op_bench.runtime.validation import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the deterministic public report for four MCP cohorts."
    )
    parser.add_argument("--run-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-cli-version", required=True)
    return parser


def main(
    argv: list[str] | None = None,
    *,
    experiment_contract: McpExperimentContract = FORMAL_MCP_EXPERIMENT_CONTRACT,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        index, summary = build_mcp_experiment_report(
            tuple(Path(value) for value in args.run_root),
            expected_adapter_id="codex_mcp_canonical",
            expected_model_id=args.expected_model,
            expected_codex_cli_version=args.expected_cli_version,
            experiment_contract=experiment_contract,
        )
        write_mcp_experiment_report(Path(args.output_dir), index, summary)
    except (ContractError, OSError) as exc:
        print(f"MCP experiment report failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"MCP experiment report: {summary['totals']['attempts']} Attempts, "
        f"{summary['totals']['trace_complete']} complete traces"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
