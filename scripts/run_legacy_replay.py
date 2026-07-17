#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.replay import (
    ExactReplayObserver,
    ReplayRunner,
    build_replay_inventory,
)
from op_bench.runtime.validation import ContractError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze and report the OpBench v0.5 17+17+51 replay inventory. "
            "Without an explicitly configured executor, cases are recorded as "
            "Blocked; no runtime target is searched or probed."
        )
    )
    parser.add_argument(
        "--repository-root",
        default=str(ROOT),
        help="Repository containing the immutable v0.5 inputs",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--target-config",
        help=(
            "Exact private target config. When supplied, replay executes through "
            "that target only; no target is searched or probed."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository_root = Path(args.repository_root)
    try:
        inventory = build_replay_inventory(repository_root)
        runner = ReplayRunner(repository_root, inventory)
        if args.target_config is None:
            report = runner.run(Path(args.output_root))
        else:
            with ExactReplayObserver(
                repository_root,
                Path(args.target_config),
            ) as observer:
                report = runner.run(
                    Path(args.output_root),
                    observer=observer,
                )
    except (ContractError, OSError, ValueError):
        print("legacy replay failed: invalid input or exact runtime setup", file=sys.stderr)
        return 2
    print(
        "legacy replay: "
        f"total={report.summary.total} "
        f"passed={report.summary.passed} "
        f"failed={report.summary.failed} "
        f"blocked={report.summary.blocked}"
    )
    return 1 if report.summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
