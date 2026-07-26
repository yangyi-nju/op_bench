from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.probe_matched_runtime import main as probe_main
from scripts.validate_matched_runtime_evidence import main as validate_main
from tests.test_matched_runtime_contracts import (
    compatible_evidence,
    incompatible_evidence,
)
from tests.test_matched_runtime_probe import task_manifest


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "sha256:" + "a" * 64


class ValidateMatchedRuntimeEvidenceCliTests(unittest.TestCase):
    def _write(self, path: Path, payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_valid_compatible_evidence_returns_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write(path, compatible_evidence().to_dict())

            with redirect_stdout(StringIO()):
                exit_code = validate_main([str(path)])

        self.assertEqual(exit_code, 0)

    def test_valid_incompatible_evidence_returns_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            self._write(path, incompatible_evidence().to_dict())

            with redirect_stdout(StringIO()):
                exit_code = validate_main([str(path)])

        self.assertEqual(exit_code, 1)

    def test_tampered_evidence_returns_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            payload = compatible_evidence().to_dict()
            payload["runtime"]["torch_version"] = "tampered"
            self._write(path, payload)

            stderr = StringIO()
            with redirect_stderr(stderr):
                exit_code = validate_main([str(path)])

        self.assertEqual(exit_code, 2)
        self.assertIn("invalid", stderr.getvalue())

    def test_schema_option_rejects_wrong_contract_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "evidence.json"
            schema = root / "schema.json"
            self._write(path, compatible_evidence().to_dict())
            self._write(
                schema,
                {
                    "required": ["different"],
                    "properties": {"different": {"type": "string"}},
                    "additionalProperties": False,
                },
            )

            with redirect_stderr(StringIO()):
                exit_code = validate_main(
                    [str(path), "--schema", str(schema)]
                )

        self.assertEqual(exit_code, 2)


class ProbeMatchedRuntimeCliTests(unittest.TestCase):
    def test_probe_cli_resolves_task_writes_evidence_and_emits_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_manifest(root)
            output = root / "evidence.json"
            evidence = replace(
                compatible_evidence(),
                task_id=task.task_id,
                source=replace(
                    compatible_evidence().source,
                    source_id=task.source_ref,
                    commit=task.base_commit,
                ),
                runtime=replace(
                    compatible_evidence().runtime,
                    environment_id=task.environment_ref,
                ),
            )
            stdout = StringIO()

            with (
                patch(
                    "scripts.probe_matched_runtime.load_resolved_task",
                    return_value=task,
                ),
                patch(
                    "scripts.probe_matched_runtime.MatchedRuntimeProbe.run",
                    return_value=evidence,
                ) as run,
                redirect_stdout(stdout),
            ):
                exit_code = probe_main(
                    [
                        "--task",
                        str(task.task_json_path),
                        "--strategy",
                        "matched_wheel",
                        "--artifact-kind",
                        "official_wheel",
                        "--artifact-id",
                        "torch-2.4.0+cu124-cp311-linux_x86_64",
                        "--artifact-digest",
                        SHA_A,
                        "--artifact-digest-kind",
                        "wheel_sha256",
                        "--output",
                        str(output),
                        "--quiet",
                    ]
                )

        self.assertEqual(exit_code, 0)
        run.assert_called_once()
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["status"], "compatible")
        self.assertTrue(summary["content_hash"].startswith("sha256:"))

    def test_probe_cli_maps_incompatible_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_manifest(root)
            output = root / "evidence.json"
            evidence = replace(
                incompatible_evidence(),
                task_id=task.task_id,
                source=replace(
                    incompatible_evidence().source,
                    source_id=task.source_ref,
                    commit=task.base_commit,
                ),
                runtime=replace(
                    incompatible_evidence().runtime,
                    environment_id=task.environment_ref,
                ),
            )

            with (
                patch(
                    "scripts.probe_matched_runtime.load_resolved_task",
                    return_value=task,
                ),
                patch(
                    "scripts.probe_matched_runtime.MatchedRuntimeProbe.run",
                    return_value=evidence,
                ),
                redirect_stdout(StringIO()),
            ):
                exit_code = probe_main(
                    [
                        "--task",
                        str(task.task_json_path),
                        "--strategy",
                        "matched_wheel",
                        "--artifact-kind",
                        "official_wheel",
                        "--artifact-id",
                        "torch-wheel",
                        "--artifact-digest",
                        SHA_A,
                        "--artifact-digest-kind",
                        "wheel_sha256",
                        "--output",
                        str(output),
                        "--quiet",
                    ]
                )

        self.assertEqual(exit_code, 1)

    def test_probe_cli_refuses_to_overwrite_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = task_manifest(root)
            output = root / "evidence.json"
            output.write_text("existing", encoding="utf-8")

            stderr = StringIO()
            with (
                patch(
                    "scripts.probe_matched_runtime.load_resolved_task",
                    return_value=task,
                ),
                redirect_stderr(stderr),
            ):
                exit_code = probe_main(
                    [
                        "--task",
                        str(task.task_json_path),
                        "--strategy",
                        "matched_wheel",
                        "--artifact-kind",
                        "official_wheel",
                        "--artifact-id",
                        "torch-wheel",
                        "--artifact-digest",
                        SHA_A,
                        "--artifact-digest-kind",
                        "wheel_sha256",
                        "--output",
                        str(output),
                    ]
                )

        self.assertEqual(exit_code, 2)
        self.assertIn("exists", stderr.getvalue())

    def test_probe_cli_requires_artifact_identity_arguments(self) -> None:
        stderr = StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stderr(stderr):
            probe_main(["--task", "task.json"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--strategy", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
