#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.matched_runtime.promotion import promote_matched_runtime_task  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote a matched-runtime task after compatible probe and Admission.",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--compatibility-evidence", required=True)
    parser.add_argument("--admission-evidence", required=True)
    parser.add_argument("--verified-at", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        promoted = promote_matched_runtime_task(
            Path(args.task),
            Path(args.compatibility_evidence),
            Path(args.admission_evidence),
            args.verified_at,
        )
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"matched-runtime promotion rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": promoted["admission"]["status"],
                "task_id": promoted["task_id"],
                "verified_at": promoted["admission"]["verified_at"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
