#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.containers import ContainerManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and clean up Docker containers managed by OpBench.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="List all Docker containers labelled as OpBench-managed.")
    prune = subparsers.add_parser("prune-stopped", help="Preview or remove stopped OpBench-managed containers.")
    prune.add_argument(
        "--execute",
        action="store_true",
        help="Actually remove stopped containers. Without this flag the command is preview-only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = ContainerManager()
    if args.command == "list":
        result: object = [record.to_dict() for record in manager.list_managed()]
    else:
        result = manager.prune_stopped(execute=args.execute)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
