#!/usr/bin/env python

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.integrity import (
    load_run_manifest_artifact,
    verify_run_artifacts,
)
from op_bench.runtime.validation import ContractError


RESOURCE_CHECK_IDS = (
    "runtime_resource_ownership",
    "runtime_cleanup",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact runtime-resource ownership and cleanup artifacts. "
            "This command only reads the supplied run root; it does not inspect "
            "processes, containers, remote hosts, or network state."
        )
    )
    parser.add_argument("--run-root", required=True, help="Completed v0.6 run root")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root)
    try:
        manifest = load_run_manifest_artifact(run_root)
    except (ContractError, OSError) as exc:
        print(f"invalid run root: {exc}", file=sys.stderr)
        return 2

    report = verify_run_artifacts(run_root, manifest)
    by_id = {check.check_id: check for check in report.checks}
    statuses = {
        check_id: by_id[check_id].status
        for check_id in RESOURCE_CHECK_IDS
    }
    passed = all(status == "passed" for status in statuses.values())
    print(
        canonical_json(
            {
                "run_id": report.run_id,
                "status": "passed" if passed else "failed",
                "checks": statuses,
            }
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
