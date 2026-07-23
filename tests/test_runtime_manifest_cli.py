from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from op_bench.dataset import DatasetManifest
from op_bench.runtime.manifest import RunManifest
from op_bench.runtime.schema import load_runtime_schema, validate_schema_instance
from scripts.build_run_manifest import main as build_main
from scripts.validate_runtime_contract import main as validate_main
from tests.test_runtime_wire_contracts import action_observation


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "datasets" / "pytorch_v0.5" / "dataset.json"
EXAMPLE_PATH = REPO_ROOT / "configs" / "examples" / "v0.6_run_manifest.example.json"


def build_args(output: Path, *, repeat: int = 2) -> list[str]:
    return [
        "--dataset",
        str(DATASET_PATH),
        "--output",
        str(output),
        "--agent",
        "example-agent",
        "--model",
        "example-model",
        "--adapter",
        "canonical-cli-v1",
        "--repeat",
        str(repeat),
        "--created-at",
        "2026-07-17T10:00:00Z",
    ]


class RuntimeManifestCliTests(unittest.TestCase):
    def test_builder_is_deterministic_and_does_not_launch_or_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            with patch.object(subprocess, "run") as run, patch.object(
                socket, "create_connection"
            ) as connect:
                self.assertEqual(build_main(build_args(first)), 0)
                self.assertEqual(build_main(build_args(second)), 0)

            run.assert_not_called()
            connect.assert_not_called()
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertTrue(first.read_bytes().endswith(b"\n"))
            encoded = json.loads(first.read_text(encoding="utf-8"))
            validate_schema_instance(encoded, load_runtime_schema())
            manifest = RunManifest.from_dict(encoded)
            self.assertEqual(len(manifest.tasks), 17)
            self.assertEqual(len(manifest.expected_attempts), 34)

    def test_builder_rejects_invalid_repeat_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, redirect_stderr(io.StringIO()) as stderr:
            output = Path(tmp) / "invalid.json"

            result = build_main(build_args(output, repeat=0))

            self.assertEqual(result, 2)
            self.assertIn("--repeat must be >= 1", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_validator_accepts_manifest_and_detects_tampered_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "manifest.json"
            self.assertEqual(build_main(build_args(output)), 0)

            with redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(validate_main([str(output)]), 0)
            self.assertIn("valid run_manifest", stdout.getvalue())

            encoded = json.loads(output.read_text(encoding="utf-8"))
            encoded["comparability_key"] = "sha256:" + "0" * 64
            output.write_text(json.dumps(encoded), encoding="utf-8")
            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(validate_main([str(output)]), 1)
            self.assertIn("does not match manifest content", stderr.getvalue())

    def test_validator_rejects_semantically_invalid_non_manifest_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "observation.json"
            encoded = action_observation().to_dict()
            encoded["ok"] = True
            encoded["error_code"] = "runtime_error"
            output.write_text(json.dumps(encoded), encoding="utf-8")

            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(validate_main([str(output)]), 1)
            self.assertIn("successful observation must use 'ok'", stderr.getvalue())

    def test_builder_rejects_unverified_dataset_before_writing(self) -> None:
        source = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
        loaded = DatasetManifest.load(DATASET_PATH)
        source["status"] = "draft"
        for raw, entry in zip(source["tasks"], loaded.tasks):
            raw["task_path"] = str(entry.task_path)
            raw["admission_evidence"] = str(entry.admission_evidence_path)
        source["registries"] = {
            "environments": str(REPO_ROOT / "environments" / "registry.json"),
            "sources": str(REPO_ROOT / "sources" / "registry.json"),
        }

        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "dataset.json"
            output = Path(tmp) / "manifest.json"
            dataset.write_text(json.dumps(source), encoding="utf-8")
            args = build_args(output)
            args[args.index(str(DATASET_PATH))] = str(dataset)

            with redirect_stderr(io.StringIO()) as stderr:
                self.assertEqual(build_main(args), 1)
            self.assertIn("dataset.status must be 'verified'", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_checked_in_example_is_valid_and_contains_only_synthetic_agent_identity(self) -> None:
        encoded = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

        validate_schema_instance(encoded, load_runtime_schema())
        manifest = RunManifest.from_dict(encoded)
        flattened = EXAMPLE_PATH.read_text(encoding="utf-8")
        self.assertEqual(manifest.agents[0].agent.identifier, "example-agent")
        self.assertEqual(manifest.agents[0].model.identifier, "example-model")
        self.assertNotIn("gpu-a10", flattened)
        self.assertNotIn(str(REPO_ROOT), flattened)


if __name__ == "__main__":
    unittest.main()
