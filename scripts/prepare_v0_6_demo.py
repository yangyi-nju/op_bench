#!/usr/bin/env python3
"""Materialize the deterministic synthetic input for the OpBench v0.6 demo."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = ROOT / "configs" / "examples" / "v0.6_demo"
SOURCE_ID = "opbench-demo-source-v1"
ENVIRONMENT_ID = "opbench-local-cpu-process-v1"
DATASET_ID = "opbench_v0.6_scripted_demo"

_SPEC_KEYS = {
    "schema_version",
    "task_id",
    "statement",
    "operator",
    "patch_scope",
    "public_test",
    "hidden_test",
    "runtime_profile_id",
}
_GIT_ENV = {
    "PATH": os.environ.get("PATH", ""),
    "LC_ALL": "C",
    "GIT_AUTHOR_NAME": "OpBench Demo",
    "GIT_AUTHOR_EMAIL": "opbench-demo@example.invalid",
    "GIT_AUTHOR_DATE": "2026-07-18T00:00:00Z",
    "GIT_COMMITTER_NAME": "OpBench Demo",
    "GIT_COMMITTER_EMAIL": "opbench-demo@example.invalid",
    "GIT_COMMITTER_DATE": "2026-07-18T00:00:00Z",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}

_HIDDEN_TEST_PATCH = b"""diff --git a/test_calc.py b/test_calc.py
new file mode 100644
--- /dev/null
+++ b/test_calc.py
@@ -0,0 +1,9 @@
+import math
+import unittest
+
+from calc import normalize
+
+
+class NormalizeTests(unittest.TestCase):
+    def test_nan_is_preserved(self):
+        self.assertTrue(math.isnan(normalize(float(\"nan\"))))
"""


class DemoPreparationError(ValueError):
    """The requested public Demo input cannot be prepared safely."""


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            (
                "git",
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.filemode=true",
                "-C",
                str(repository),
                *arguments,
            ),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_GIT_ENV,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DemoPreparationError("cannot create deterministic Demo source") from exc


def _load_spec() -> Mapping[str, object]:
    try:
        value = json.loads((TEMPLATE_ROOT / "spec.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DemoPreparationError("cannot load Demo specification") from exc
    if not isinstance(value, dict) or set(value) != _SPEC_KEYS:
        raise DemoPreparationError("Demo specification has an invalid shape")
    if value.get("schema_version") != "opbench.demo_spec.v1":
        raise DemoPreparationError("Demo specification has an invalid version")
    for key in ("task_id", "public_test", "hidden_test", "runtime_profile_id"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise DemoPreparationError(f"Demo specification {key} is invalid")
    if value["runtime_profile_id"] != "local-cpu-process-v1":
        raise DemoPreparationError("Demo specification must select the local profile")
    if not isinstance(value.get("statement"), dict) or not isinstance(
        value.get("operator"), dict
    ):
        raise DemoPreparationError("Demo specification metadata is invalid")
    if value.get("patch_scope") != ["calc.py"]:
        raise DemoPreparationError("Demo specification patch scope is invalid")
    return value


def _prepare_source(output_root: Path) -> tuple[Path, str, bytes]:
    source = output_root / "source"
    try:
        shutil.copytree(TEMPLATE_ROOT / "source", source)
    except OSError as exc:
        raise DemoPreparationError("cannot copy Demo source templates") from exc
    _git(source, "init", "--quiet", "--initial-branch=main")
    _git(source, "add", "--all")
    _git(source, "commit", "--quiet", "-m", "opbench v0.6 demo source")
    revision = _git(source, "rev-parse", "HEAD").stdout.decode("ascii").strip()

    target = source / "calc.py"
    base = target.read_bytes()
    try:
        target.write_text("def normalize(value):\n    return value\n", encoding="utf-8")
        gold_patch = _git(
            source,
            "diff",
            "--binary",
            "--no-ext-diff",
            "--",
            "calc.py",
        ).stdout
    finally:
        target.write_bytes(base)
    if not gold_patch or _git(source, "status", "--short").stdout:
        raise DemoPreparationError("Demo source patch generation is not clean")
    return source, revision, gold_patch


def _write_runtime_inputs(
    output_root: Path,
    spec: Mapping[str, object],
    revision: str,
    gold_patch: bytes,
) -> Path:
    task_id = str(spec["task_id"])
    public_test = str(spec["public_test"])
    hidden_test = str(spec["hidden_test"])
    task_root = output_root / "task"
    artifacts = task_root / "artifacts"
    admission = task_root / "admission"
    artifacts.mkdir(parents=True)
    admission.mkdir()
    (artifacts / "gold.patch").write_bytes(gold_patch)
    (artifacts / "test.patch").write_bytes(_HIDDEN_TEST_PATCH)
    _write_json(
        admission / "evidence.json",
        {
            "decision": "verified",
            "schema_version": "opbench.demo_admission.v1",
            "synthetic": True,
            "task_id": task_id,
        },
    )

    _write_json(
        task_root / "task.json",
        {
            "admission": {
                "evidence": "admission/evidence.json",
                "status": "verified",
                "verified_at": "2026-07-18T00:00:00Z",
            },
            "artifacts": {
                "gold_patch": "artifacts/gold.patch",
                "test_patch": "artifacts/test.patch",
            },
            "environment": {
                "backend": "local",
                "hardware": {"requires_gpu": False},
                "image": "host-python-current-v1",
            },
            "environment_ref": ENVIRONMENT_ID,
            "evaluation": {
                "fail_to_pass": [hidden_test],
                "pass_to_pass": [public_test],
                "public_tests": [public_test],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 300,
            },
            "operator": spec["operator"],
            "patch_scope": {"allowed_paths": list(spec["patch_scope"]), "mode": "enforced"},
            "runtime_tier": "local_fixture",
            "source": {
                "base_commit": revision,
                "checkout_mode": "git",
                "repo_url": "https://example.invalid/opbench-v0.6-demo.git",
            },
            "source_ref": SOURCE_ID,
            "statement": spec["statement"],
            "task_id": task_id,
            "version": "v1",
        },
    )

    _write_json(
        output_root / "environment-registry.json",
        {
            "environments": [
                {
                    "backend": "local",
                    "docker": {"image": "host-python-current-v1"},
                    "framework": "fixture",
                    "hardware": {"requires_gpu": False},
                    "id": ENVIRONMENT_ID,
                    "preflight": {"commands": []},
                    "runtime_tier": "local_fixture",
                }
            ],
            "version": "v1",
        },
    )
    _write_json(
        output_root / "source-registry.json",
        {
            "sources": [
                {
                    "commit": revision,
                    "id": SOURCE_ID,
                    "local_path": "source",
                    "repo_url": "https://example.invalid/opbench-v0.6-demo.git",
                    "submodules": {
                        "policy": "none_required",
                        "status": "not_initialized",
                    },
                }
            ],
            "version": "v1",
        },
    )
    dataset_root = output_root / "dataset"
    dataset_root.mkdir()
    dataset = dataset_root / "dataset.json"
    _write_json(
        dataset,
        {
            "dataset_id": DATASET_ID,
            "registries": {
                "environments": "../environment-registry.json",
                "sources": "../source-registry.json",
            },
            "status": "verified",
            "tasks": [
                {
                    "admission_evidence": "../task/admission/evidence.json",
                    "admission_status": "verified",
                    "environment_status": "ready",
                    "replay_status": "verified",
                    "source_status": "ready",
                    "task_id": task_id,
                    "task_path": "../task",
                }
            ],
            "version": "v1",
        },
    )
    return dataset


def prepare_demo(output_dir: Path) -> Path:
    """Create one deterministic local Demo input and return its Dataset path."""

    if not isinstance(output_dir, Path):
        raise DemoPreparationError("output_dir must be a Path")
    target = output_dir.absolute()
    if target.is_symlink():
        raise DemoPreparationError("output_dir must not be a symlink")
    created = False
    if target.exists():
        if not target.is_dir():
            raise DemoPreparationError("output_dir must be a directory")
        try:
            if any(target.iterdir()):
                raise DemoPreparationError("output_dir must be empty")
        except OSError as exc:
            raise DemoPreparationError("cannot inspect output_dir") from exc
    else:
        try:
            target.mkdir(parents=True)
            created = True
        except OSError as exc:
            raise DemoPreparationError("cannot create output_dir") from exc

    try:
        spec = _load_spec()
        _, revision, gold_patch = _prepare_source(target)
        dataset = _write_runtime_inputs(target, spec, revision, gold_patch)
    except Exception:
        if created:
            shutil.rmtree(target, ignore_errors=True)
        raise
    return dataset.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the deterministic synthetic Dataset used by the OpBench v0.6 "
            "local Demo. This command performs no network or runtime operation."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New or empty directory for the generated Demo input.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = prepare_demo(Path(args.output_dir))
    except (DemoPreparationError, OSError, ValueError):
        print("cannot prepare OpBench v0.6 Demo input", file=sys.stderr)
        return 1
    print(dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
