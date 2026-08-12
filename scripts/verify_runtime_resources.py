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
    roots = (
        (run_root,)
        if (run_root / "run_manifest.json").is_file()
        else tuple(
            path
            for path in sorted(run_root.glob("cohort-*"))
            if path.is_dir() and not path.is_symlink()
            and (path / "run_manifest.json").is_file()
        )
    )
    if not roots:
        print("invalid run root: no completed run artifacts", file=sys.stderr)
        return 2
    reports = []
    try:
        for selected_root in roots:
            manifest = load_run_manifest_artifact(selected_root)
            reports.append(verify_run_artifacts(selected_root, manifest))
    except (ContractError, OSError) as exc:
        print(f"invalid run root: {exc}", file=sys.stderr)
        return 2
    if len(reports) == 1:
        report = reports[0]
        by_id = {check.check_id: check for check in report.checks}
        statuses = {
            check_id: by_id[check_id].status
            for check_id in RESOURCE_CHECK_IDS
        }
        passed = all(status == "passed" for status in statuses.values())
        payload = {
            "run_id": report.run_id,
            "status": "passed" if passed else "failed",
            "checks": statuses,
        }
    else:
        counts = {
            check_id: {
                "passed": sum(
                    next(
                        check.status
                        for check in report.checks
                        if check.check_id == check_id
                    )
                    == "passed"
                    for report in reports
                ),
                "total": len(reports),
            }
            for check_id in RESOURCE_CHECK_IDS
        }
        passed = all(
            value["passed"] == value["total"] for value in counts.values()
        )
        payload = {
            "run_count": len(reports),
            "status": "passed" if passed else "failed",
            "checks": counts,
        }
    print(
        canonical_json(payload)
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
