#!/usr/bin/env python3
"""Recover exact cleanup-failed resources without exposing private handles."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.runtime.backends import (  # noqa: E402
    load_runtime_target_binding,
    recover_remote_cleanup_resource,
)
from op_bench.runtime.canonical import canonical_json  # noqa: E402
from op_bench.runtime.integrity import (  # noqa: E402
    load_run_manifest_artifact,
    persist_integrity_reports,
    verify_run_artifacts,
)
from op_bench.runtime.resources import (  # noqa: E402
    AttemptResourceLedger,
    RuntimeCleanupEntry,
    RuntimeCleanupReport,
    parse_runtime_lease_store,
    parse_runtime_resource_ledger,
)
from op_bench.runtime.resume import parse_attempt_ledger  # noqa: E402
from op_bench.runtime.run_artifacts import AttemptArtifactStore  # noqa: E402
from op_bench.runtime.validation import ContractError  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--target-config", type=Path, required=True)
    return parser


def _clock_ms() -> int:
    return max(1, time.monotonic_ns() // 1_000_000)


def _cleanup_report(records) -> RuntimeCleanupReport:
    final = {record.resource_id: record for record in records}
    if not final:
        raise ContractError("resource recovery: empty resource ledger")
    entries = tuple(
        RuntimeCleanupEntry(
            resource_id=resource_id,
            resource_type=record.resource_type,
            status=record.transition,
            error_code=(
                None
                if record.transition == "released"
                else "resource_create_failed"
                if record.transition == "create_failed"
                else "resource_cleanup_failed"
            ),
        )
        for resource_id, record in sorted(final.items())
    )
    first = records[0]
    return RuntimeCleanupReport(
        attempt_id=first.attempt_id,
        retry_index=first.retry_index,
        runtime_profile_hash=first.runtime_profile_hash,
        entries=entries,
        all_released=all(
            entry.status in {"released", "create_failed"}
            for entry in entries
        ),
    )


def recover_run(run_root: Path, target_config: Path) -> dict[str, object]:
    run_root = run_root.resolve(strict=True)
    target_config = target_config.resolve(strict=True)
    manifest = load_run_manifest_artifact(run_root)
    target = load_runtime_target_binding(
        target_config,
        local_workspace_parent=run_root.parent,
    )
    attempts = parse_attempt_ledger((run_root / "attempts.jsonl").read_bytes())
    expected = {item.attempt_id: item for item in manifest.expected_attempts}
    tasks = {item.task: item for item in manifest.tasks}
    retries: dict[str, set[int]] = defaultdict(set)
    for record in attempts:
        if record.attempt_id not in expected:
            raise ContractError("resource recovery: unknown Attempt in ledger")
        retries[record.attempt_id].add(record.retry_index)

    store = AttemptArtifactStore(run_root, manifest)
    recovered = 0
    remaining = 0
    try:
        for attempt_id in sorted(retries):
            task = tasks[expected[attempt_id].task]
            for retry_index in sorted(retries[attempt_id]):
                resource_path = store.runtime_resources_path(
                    attempt_id,
                    retry_index=retry_index,
                )
                private_path = store.private_runtime_resources_path(
                    attempt_id,
                    retry_index=retry_index,
                )
                records = parse_runtime_resource_ledger(
                    resource_path.read_bytes(),
                    attempt_id=attempt_id,
                    retry_index=retry_index,
                    runtime_profile_hash=task.runtime.content_hash,
                )
                handles = parse_runtime_lease_store(
                    private_path.read_bytes(),
                    attempt_id=attempt_id,
                    retry_index=retry_index,
                    runtime_profile_hash=task.runtime.content_hash,
                )
                handle_by_id = {handle.resource_id: handle for handle in handles}
                final = {record.resource_id: record for record in records}
                failed = [
                    record
                    for record in final.values()
                    if record.transition == "cleanup_failed"
                ]
                if failed:
                    ledger = AttemptResourceLedger(
                        resource_path,
                        attempt_id=attempt_id,
                        retry_index=retry_index,
                        runtime_profile_hash=task.runtime.content_hash,
                        clock_ms=_clock_ms,
                    )
                    try:
                        for record in sorted(failed, key=lambda item: item.resource_id):
                            handle = handle_by_id.get(record.resource_id)
                            if handle is None:
                                raise ContractError(
                                    "resource recovery: failed resource has no private handle"
                                )
                            released = recover_remote_cleanup_resource(
                                target,
                                handle,
                                attempt_id=attempt_id,
                                retry_index=retry_index,
                                timeout_ms=task.runtime.cleanup_policy.timeout_ms,
                            )
                            if released:
                                ledger.recover_released(record.resource_id)
                                recovered += 1
                            else:
                                remaining += 1
                        records = ledger.records
                    finally:
                        ledger.close()
                recovered_report = _cleanup_report(records)
                current_report = store.read_runtime_cleanup(
                    attempt_id,
                    retry_index=retry_index,
                )
                if current_report != recovered_report:
                    store.replace_runtime_cleanup_after_recovery(
                        attempt_id,
                        recovered_report,
                        retry_index=retry_index,
                    )
    finally:
        store.close()

    integrity = verify_run_artifacts(run_root, manifest)
    if integrity.status == "passed":
        persist_integrity_reports(run_root, manifest, integrity)
    return {
        "status": (
            "passed"
            if remaining == 0 and integrity.status == "passed"
            else "failed"
        ),
        "recovered_resource_count": recovered,
        "remaining_cleanup_failed_count": remaining,
        "integrity_status": integrity.status,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = recover_run(args.run_root, args.target_config)
    except (ContractError, OSError, ValueError) as exc:
        print(
            canonical_json(
                {
                    "status": "failed",
                    "failure_code": "resource_recovery_failed",
                    "error_type": type(exc).__name__,
                    "error_digest": "sha256:"
                    + hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                }
            )
        )
        return 1
    print(canonical_json(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
