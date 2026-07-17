"""Evaluator-owned isolated unittest runner.

This file is executed with ``python -I``. It intentionally imports the stdlib
runner before placing the evaluated workspace on ``sys.path`` and writes one
strict structured result outside that workspace. Agent/test stdout is never a
source of score counts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve(strict=True)
    result_path = Path(args.result)
    if not workspace.is_dir() or result_path.exists():
        return 2

    # Do not expose control arguments through the normal test argv surface.
    sys.argv = ["opbench-unittest-runner"]
    sys.path.insert(0, str(workspace))
    suite = unittest.defaultTestLoader.loadTestsFromName(args.selector)
    result = unittest.TestResult()
    suite.run(result)

    skipped = len(result.skipped)
    collected = result.testsRun
    executed = collected - skipped
    failed = min(
        executed,
        len(result.failures) + len(result.errors) + len(result.unexpectedSuccesses),
    )
    payload = {
        "collected": collected,
        "executed": executed,
        "failed": failed,
        "passed": executed - failed,
        "record_type": "opbench_unittest_result",
        "schema_version": "v1",
        "skipped": skipped,
    }
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(result_path, flags, 0o600)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                return 2
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
