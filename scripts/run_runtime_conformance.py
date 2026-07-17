#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.conformance import RuntimeConformanceRunner
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.validation import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the deterministic OpBench runtime conformance matrix. "
            "External mode uses only the supplied exact target configuration "
            "and never searches for hosts or services."
        )
    )
    parser.add_argument("--fixture", required=True, help="Exact local Git fixture")
    parser.add_argument("--output-dir", required=True, help="New report directory")
    parser.add_argument(
        "--profile-registry",
        default=str(ROOT / "configs" / "runtime_profiles.v1.json"),
    )
    parser.add_argument(
        "--profile-id",
        default="local-cpu-process-v1",
    )
    parser.add_argument("--target-config")
    parser.add_argument(
        "--external-profile-id",
        help="Explicit Runtime Profile for the exact external target",
    )
    parser.add_argument("--include-external", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_runtime_profile_registry(args.profile_registry)
        profile = registry.get(args.profile_id)
        external_profile = (
            None
            if args.external_profile_id is None
            else registry.get(args.external_profile_id)
        )
        report = RuntimeConformanceRunner(
            fixture_source=Path(args.fixture),
            runtime_profile=profile,
        ).run(
            Path(args.output_dir),
            include_external=args.include_external,
            target_config=(
                None if args.target_config is None else Path(args.target_config)
            ),
            external_profile=external_profile,
        )
    except (ContractError, OSError) as exc:
        print(f"runtime conformance failed to start: {exc}", file=sys.stderr)
        return 2
    print(
        f"runtime conformance: {report.status} "
        f"({len(report.entries)} entries)"
    )
    return 1 if report.status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
