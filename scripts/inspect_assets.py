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

from op_bench.assets import AssetManager
from op_bench.registry import EnvironmentRegistry, SourceRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect OpBench environment and source asset cache state.")
    parser.add_argument(
        "--environment-registry",
        default=str(ROOT / "environments/registry.json"),
        help="Environment registry path.",
    )
    parser.add_argument(
        "--source-registry",
        default=str(ROOT / "sources/registry.json"),
        help="Source registry path.",
    )
    parser.add_argument("--check-docker", action="store_true", help="Inspect local Docker image IDs and digests.")
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    environment_registry = _load_optional(EnvironmentRegistry, args.environment_registry)
    source_registry = _load_optional(SourceRegistry, args.source_registry)
    report = AssetManager(environment_registry, source_registry).inspect(check_docker=args.check_docker)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    unavailable = (
        report["summary"]["environments"]["unavailable"]
        + report["summary"]["sources"]["unavailable"]
    )
    return 1 if unavailable else 0


def _load_optional(registry_type: type, path_value: str | None) -> object | None:
    if not path_value:
        return None
    path = Path(path_value)
    return registry_type.load(path) if path.exists() else None


if __name__ == "__main__":
    raise SystemExit(main())
