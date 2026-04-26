#!/usr/bin/env python3

"""Thin wrapper around the op_bench PR-to-task builder."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_bench.builder import main


if __name__ == "__main__":
    raise SystemExit(main())
