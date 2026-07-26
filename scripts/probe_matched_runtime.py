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

from op_bench.matched_runtime.contracts import (  # noqa: E402
    ARTIFACT_DIGEST_KINDS,
    ARTIFACT_KINDS,
    MATCH_STRATEGIES,
    BuildIdentity,
)
from op_bench.matched_runtime.probe import (  # noqa: E402
    EnvironmentProbeBackend,
    MatchedRuntimeProbe,
    ProbeSpec,
    write_compatibility_evidence,
)
from op_bench.progress import ProgressLogger  # noqa: E402
from op_bench.registry import load_resolved_task  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe one OpBench task against a matched runtime.",
    )
    parser.add_argument("--task", required=True, help="Task directory or task.json path.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=MATCH_STRATEGIES,
    )
    parser.add_argument(
        "--artifact-kind",
        required=True,
        choices=ARTIFACT_KINDS,
    )
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-digest", required=True)
    parser.add_argument(
        "--artifact-digest-kind",
        required=True,
        choices=ARTIFACT_DIGEST_KINDS,
    )
    parser.add_argument("--build-flag", action="append", default=[])
    parser.add_argument("--gpu-arch", action="append", default=[])
    parser.add_argument("--ccache-key")
    parser.add_argument("--toolchain", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--environment-registry",
        default=str(ROOT / "environments" / "registry.json"),
    )
    parser.add_argument(
        "--source-registry",
        default=str(ROOT / "sources" / "registry.json"),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() or output.is_symlink():
        print(f"{output}: evidence output already exists", file=sys.stderr)
        return 2
    try:
        task_path = Path(args.task).resolve()
        if task_path.is_dir():
            task_path = task_path / "task.json"
        task = load_resolved_task(
            task_path,
            environment_registry_path=args.environment_registry,
            source_registry_path=args.source_registry,
        )
        source_build = args.strategy != "matched_wheel"
        build = BuildIdentity(
            flags=tuple(sorted(set(args.build_flag))),
            gpu_arches=tuple(sorted(set(args.gpu_arch))),
            ccache_key=args.ccache_key,
            artifact_digest=args.artifact_digest if source_build else None,
            toolchain=tuple(sorted(set(args.toolchain))),
        )
        spec = ProbeSpec.from_task(
            task,
            strategy=args.strategy,
            artifact_kind=args.artifact_kind,
            artifact_id=args.artifact_id,
            artifact_digest=args.artifact_digest,
            artifact_digest_kind=args.artifact_digest_kind,
            build=build,
        )
        progress = ProgressLogger(enabled=not args.quiet)
        probe = MatchedRuntimeProbe(
            backend=EnvironmentProbeBackend(progress=progress)
        )
        evidence = probe.run(task, spec)
        write_compatibility_evidence(output, evidence)
    except (ContractError, OSError, ValueError) as exc:
        print(f"matched-runtime probe invalid: {exc}", file=sys.stderr)
        return 2
    summary = {
        "content_hash": evidence.content_hash,
        "evidence_id": evidence.evidence_id,
        "failure_code": evidence.failure.code if evidence.failure else None,
        "status": evidence.status,
        "task_id": evidence.task_id,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0 if evidence.status == "compatible" else 1


if __name__ == "__main__":
    raise SystemExit(main())
