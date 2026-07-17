from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from op_bench.dataset import DatasetManifest
from op_bench.runtime.integrity import (
    load_run_manifest_artifact,
    verify_run_artifacts,
)
from scripts.prepare_v0_6_demo import DemoPreparationError, main, prepare_demo
from scripts.run_experiment import main as run_experiment_main
from scripts.verify_runtime_resources import main as verify_resources_main


ROOT = Path(__file__).resolve().parents[1]


def _git_head(repository: Path) -> str:
    return subprocess.run(
        ("git", "-C", str(repository), "rev-parse", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.decode("ascii").strip()


def _files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class PrepareV06DemoTests(unittest.TestCase):
    def test_prepare_demo_is_deterministic_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(socket, "create_connection") as connect:
                first_dataset = prepare_demo(root / "first")
                second_dataset = prepare_demo(root / "second")
            connect.assert_not_called()

            self.assertEqual(
                _git_head(root / "first" / "source"),
                _git_head(root / "second" / "source"),
            )
            first = DatasetManifest.load(first_dataset)
            tasks = first.load_tasks(verified_only=True)
            self.assertEqual(first.dataset_id, "opbench_v0.6_scripted_demo")
            self.assertEqual([task.task_id for task in tasks], ["opbench_demo__normalize"])
            self.assertEqual(
                tasks[0].base_commit,
                _git_head(root / "first" / "source"),
            )

    def test_prepare_demo_refuses_nonempty_output_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep\n", encoding="utf-8")

            with self.assertRaises(DemoPreparationError):
                prepare_demo(output)

            self.assertEqual(marker.read_text(encoding="utf-8"), "keep\n")

    def test_cli_prints_generated_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(["--output-dir", str(output)]), 0)

            self.assertEqual(
                stdout.getvalue().strip(),
                str((output / "dataset" / "dataset.json").resolve()),
            )

    def test_entrypoint_contains_no_probe_or_resource_discovery_commands(self) -> None:
        source = (ROOT / "scripts" / "prepare_v0_6_demo.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            r"\bping\s",
            r"\bnmap\b",
            r"\bmasscan\b",
            r"\bssh-keyscan\b",
            r"\bdocker\s+(ps|images)\b",
            r"\bps\s+aux\b",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertIsNone(re.search(forbidden, source))

    def test_standard_v1_smoke_resume_integrity_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = prepare_demo(root / "input")
            output = root / "run"
            args = [
                "--dataset",
                str(dataset),
                "--verified-only",
                "--agent",
                "scripted_canonical",
                "--agent-repeat",
                "1",
                "--output-dir",
                str(output),
                "--runtime-protocol",
                "v1",
                "--runtime-profile",
                "local-cpu-process-v1",
                "--quiet",
            ]

            with patch.object(socket, "create_connection") as connect:
                self.assertEqual(run_experiment_main(args), 0)
            connect.assert_not_called()
            manifest = load_run_manifest_artifact(output)
            self.assertEqual(verify_run_artifacts(output, manifest).status, "passed")
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(verify_resources_main(["--run-root", str(output)]), 0)
            first = _files(output)

            with patch.object(socket, "create_connection") as connect:
                self.assertEqual(run_experiment_main(args), 0)
            connect.assert_not_called()
            self.assertEqual(_files(output), first)


if __name__ == "__main__":
    unittest.main()
