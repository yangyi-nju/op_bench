from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import re
import socket
import stat
import tempfile
import unittest
from unittest.mock import patch

from op_bench.runtime.validation import ContractError
from scripts import run_experiment
from op_bench.runtime.integrity import load_run_manifest_artifact, verify_run_artifacts
from scripts.verify_runtime_resources import main as verify_resources_main
from scripts.run_experiment import build_parser, main
from tests.runtime_git_fixture import git, initialize_evaluation_git_fixture


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "pytorch_v0.5" / "dataset.json"
PROFILE_REGISTRY = ROOT / "configs" / "runtime_profiles.v1.json"


def v1_args(output: Path, *extra: str) -> list[str]:
    return [
        "--dataset",
        str(DATASET),
        "--agent",
        "scripted_canonical",
        "--output-dir",
        str(output),
        "--runtime-protocol",
        "v1",
        *extra,
    ]


class RuntimeV1CliTests(unittest.TestCase):
    def test_private_cleanup_recovery_blocks_until_exact_pgid_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            run_experiment._write_process_group_recovery(output, 6767)
            marker = output / "private_process_group_recovery.json"

            self.assertEqual(stat.S_IMODE(marker.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(marker.read_text(encoding="utf-8"))["process_group_id"],
                6767,
            )
            with patch(
                "op_bench.runtime.process_group.exact_process_group_is_absent",
                return_value=False,
            ) as absent:
                self.assertFalse(run_experiment._resolve_process_group_recovery(output))
            absent.assert_called_once_with(6767)
            self.assertTrue(marker.exists())

            with patch(
                "op_bench.runtime.process_group.exact_process_group_is_absent",
                return_value=True,
            ) as absent:
                self.assertTrue(run_experiment._resolve_process_group_recovery(output))
            absent.assert_called_once_with(6767)
            self.assertFalse(marker.exists())

    def test_default_protocol_is_legacy_and_help_exposes_explicit_v1_controls(self) -> None:
        parser = build_parser()
        parsed = parser.parse_args(
            ["--task", "fixture", "--agent", "gold", "--output-dir", "out"]
        )

        self.assertEqual(parsed.runtime_protocol, "legacy")
        help_text = parser.format_help()
        for flag in (
            "--runtime-protocol",
            "--runtime-profile",
            "--runtime-profile-registry",
            "--target-config",
            "--enable-external-canary",
            "--codex-model",
        ):
            self.assertIn(flag, help_text)

    def test_v1_rejects_missing_and_unknown_profile_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_output = root / "missing"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(v1_args(missing_output))
            self.assertEqual(result, 2)
            self.assertIn("--runtime-profile is required", stderr.getvalue())
            self.assertFalse(missing_output.exists())

            unknown_output = root / "unknown"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    v1_args(
                        unknown_output,
                        "--runtime-profile",
                        "unknown-profile-v1",
                    )
                )
            self.assertEqual(result, 2)
            self.assertIn("unknown Runtime Profile", stderr.getvalue())
            self.assertFalse(unknown_output.exists())

    def test_v1_remote_requires_exact_private_target_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "remote"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    v1_args(
                        output,
                        "--runtime-profile",
                        "remote-cpu-pytorch-2.6-py311-v1",
                    )
                )

            self.assertEqual(result, 2)
            self.assertIn("--target-config is required", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_v1_codex_requires_explicit_external_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "codex"
            args = v1_args(
                output,
                "--runtime-profile",
                "remote-cpu-pytorch-2.6-py311-v1",
                "--target-config",
                str(Path(tmp) / "target.json"),
            )
            args[args.index("scripted_canonical")] = "codex_canonical"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(args)

            self.assertEqual(result, 2)
            self.assertIn("--enable-external-canary is required", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_mcp_adapter_requires_model_external_opt_in_and_exact_cli_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "mcp"
            base = v1_args(
                output,
                "--runtime-profile",
                "local-cpu-process-v1",
            )
            base[base.index("scripted_canonical")] = "codex_mcp_canonical"

            with redirect_stderr(io.StringIO()) as stderr:
                result = main(base)
            self.assertEqual(result, 2)
            self.assertIn("--codex-model is required", stderr.getvalue())
            self.assertFalse(output.exists())

            with redirect_stderr(io.StringIO()) as stderr:
                result = main([*base, "--codex-model", "gpt-5.6-sol"])
            self.assertEqual(result, 2)
            self.assertIn("--enable-external-canary is required", stderr.getvalue())
            self.assertFalse(output.exists())

            accepted = [
                *base,
                "--codex-model",
                "gpt-5.6-sol",
                "--enable-external-canary",
            ]
            with (
                patch.object(
                    run_experiment,
                    "detect_codex_cli_version",
                    return_value="codex-cli 0.145.0-alpha.18",
                    create=True,
                ) as detect,
                patch.object(run_experiment, "_execute_v1", return_value=0) as execute,
            ):
                self.assertEqual(main(accepted), 0)

            detect.assert_called_once_with()
            self.assertEqual(
                execute.call_args.kwargs["codex_cli_version"],
                "codex-cli 0.145.0-alpha.18",
            )
            self.assertFalse(output.exists())

    def test_codex_model_is_rejected_outside_the_mcp_v1_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v1_output = root / "v1"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    v1_args(
                        v1_output,
                        "--runtime-profile",
                        "local-cpu-process-v1",
                        "--codex-model",
                        "gpt-5.6-sol",
                    )
                )
            self.assertEqual(result, 2)
            self.assertIn("only supported", stderr.getvalue())
            self.assertFalse(v1_output.exists())

            legacy_output = root / "legacy"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    [
                        "--dataset",
                        str(DATASET),
                        "--agent",
                        "gold",
                        "--output-dir",
                        str(legacy_output),
                        "--codex-model",
                        "gpt-5.6-sol",
                    ]
                )
            self.assertEqual(result, 2)
            self.assertIn("require --runtime-protocol v1", stderr.getvalue())
            self.assertFalse(legacy_output.exists())

    def test_cli_version_detection_uses_one_exact_local_command(self) -> None:
        detector = getattr(run_experiment, "detect_codex_cli_version", None)
        self.assertIsNotNone(detector)
        completed = run_experiment.subprocess.CompletedProcess(
            ("codex", "--version"),
            0,
            stdout="codex-cli 0.145.0-alpha.18\n",
            stderr="",
        )
        with patch.object(
            run_experiment.subprocess,
            "run",
            return_value=completed,
        ) as invoke:
            self.assertEqual(detector(), "codex-cli 0.145.0-alpha.18")

        invoke.assert_called_once_with(
            ("codex", "--version"),
            check=False,
            text=True,
            stdout=run_experiment.subprocess.PIPE,
            stderr=run_experiment.subprocess.PIPE,
        )
        for stdout in ("", "codex 1.0\n", "codex-cli bad version\n"):
            with (
                patch.object(
                    run_experiment.subprocess,
                    "run",
                    return_value=run_experiment.subprocess.CompletedProcess(
                        ("codex", "--version"),
                        0,
                        stdout=stdout,
                        stderr="",
                    ),
                ),
                self.assertRaises(ContractError),
            ):
                detector()

    def test_v1_rejects_legacy_only_inputs_and_adapter_names_before_resources(self) -> None:
        cases = (
            ("task input", ("--task", "fixture")),
            ("fresh output deletion", ("--fresh",)),
            ("legacy adapter", ()),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, (label, extra) in enumerate(cases):
                with self.subTest(label=label):
                    output = root / f"case-{index}"
                    args = v1_args(
                        output,
                        "--runtime-profile",
                        "local-cpu-process-v1",
                        *extra,
                    )
                    if label == "legacy adapter":
                        args[args.index("scripted_canonical")] = "gold"
                    with redirect_stderr(io.StringIO()):
                        result = main(args)
                    self.assertEqual(result, 2)
                    self.assertFalse(output.exists())

    def test_legacy_rejects_v1_only_controls_instead_of_ignoring_them(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    [
                        "--dataset",
                        str(DATASET),
                        "--agent",
                        "gold",
                        "--output-dir",
                        str(output),
                        "--runtime-profile",
                        "local-cpu-process-v1",
                    ]
                )

            self.assertEqual(result, 2)
            self.assertIn("require --runtime-protocol v1", stderr.getvalue())
            self.assertFalse(output.exists())

    def test_private_target_values_are_not_printed_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            secret_alias = "private-hostname-must-not-leak"
            target = root / "target.json"
            target.write_text(
                json.dumps(
                    {
                        "backend": "local",
                        "local_workspace_parent": str(root),
                        "host_alias": secret_alias,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "out"
            with redirect_stderr(io.StringIO()) as stderr:
                result = main(
                    v1_args(
                        output,
                        "--runtime-profile",
                        "remote-cpu-pytorch-2.6-py311-v1",
                        "--target-config",
                        str(target),
                    )
                )

            self.assertEqual(result, 2)
            self.assertNotIn(secret_alias, stderr.getvalue())
            self.assertNotIn(str(target), stderr.getvalue())
            self.assertFalse(output.exists())

    def test_entrypoint_contains_no_network_or_resource_discovery_commands(self) -> None:
        source = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
        for forbidden in (
            r"\bping\s",
            r"\bnmap\b",
            r"\bssh-keyscan\b",
            r"\bdocker\s+ps\b",
            r"\bps\s+aux\b",
        ):
            self.assertIsNone(re.search(forbidden, source))

    def test_local_scripted_v1_smoke_and_resume_are_byte_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = self._local_v1_dataset(root)
            output = root / "run"
            args = [
                "--dataset",
                str(dataset),
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
                self.assertEqual(main(args), 0)
            connect.assert_not_called()
            manifest = load_run_manifest_artifact(output)
            self.assertEqual(verify_run_artifacts(output, manifest).status, "passed")
            with redirect_stderr(io.StringIO()):
                self.assertEqual(verify_resources_main(["--run-root", str(output)]), 0)
            first = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }

            with patch.object(socket, "create_connection") as connect:
                self.assertEqual(main(args), 0)
            connect.assert_not_called()
            second = {
                path.relative_to(output).as_posix(): path.read_bytes()
                for path in sorted(output.rglob("*"))
                if path.is_file()
            }
            self.assertEqual(second, first)

    @staticmethod
    def _local_v1_dataset(root: Path) -> Path:
        fixture = initialize_evaluation_git_fixture(root / "source")
        (fixture.repository / "test_public.py").write_text(
            "import unittest\n\n"
            "from calc import normalize\n\n"
            "class PublicTests(unittest.TestCase):\n"
            "    def test_number_is_preserved(self):\n"
            "        self.assertEqual(normalize(1), 1)\n",
            encoding="utf-8",
        )
        git(fixture.repository, "add", "test_public.py")
        git(fixture.repository, "commit", "--quiet", "-m", "add public test")
        revision = git(fixture.repository, "rev-parse", "HEAD").stdout.decode(
            "ascii"
        ).strip()

        task_dir = root / "task"
        (task_dir / "artifacts").mkdir(parents=True)
        (task_dir / "admission").mkdir()
        (task_dir / "artifacts" / "gold.patch").write_bytes(fixture.gold_patch)
        (task_dir / "artifacts" / "test.patch").write_bytes(
            fixture.hidden_test_patch
        )
        (task_dir / "admission" / "evidence.json").write_text(
            "{}\n", encoding="utf-8"
        )
        task = {
            "task_id": "local__v1_smoke",
            "version": "v1",
            "environment_ref": "opbench-local-cpu-process-v1",
            "runtime_tier": "local_fixture",
            "source_ref": "local-v1-source",
            "admission": {
                "status": "verified",
                "evidence": "admission/evidence.json",
                "verified_at": "2026-07-18T00:00:00Z",
            },
            "patch_scope": {"allowed_paths": ["calc.py"], "mode": "enforced"},
            "source": {
                "repo_url": "https://example.invalid/local-v1-source.git",
                "base_commit": revision,
                "checkout_mode": "git",
            },
            "statement": {
                "title": "Local v1 smoke",
                "body": "Exercise the versioned runtime controller.",
            },
            "operator": {
                "framework": "fixture",
                "operator_name": "normalize",
            },
            "environment": {
                "backend": "local",
                "image": "host-python-current-v1",
                "hardware": {"requires_gpu": False},
            },
            "evaluation": {
                "fail_to_pass": [
                    "test_calc.NormalizeTests.test_nan_is_preserved"
                ],
                "pass_to_pass": [
                    "test_public.PublicTests.test_number_is_preserved"
                ],
                "public_tests": [
                    "test_public.PublicTests.test_number_is_preserved"
                ],
                "test_command": "{python} -m unittest {test}",
                "timeout_sec": 300,
            },
            "artifacts": {
                "gold_patch": "artifacts/gold.patch",
                "test_patch": "artifacts/test.patch",
            },
        }
        (task_dir / "task.json").write_text(
            json.dumps(task, sort_keys=True), encoding="utf-8"
        )

        (root / "environment-registry.json").write_text(
            json.dumps(
                {
                    "version": "v1",
                    "environments": [
                        {
                            "id": "opbench-local-cpu-process-v1",
                            "framework": "fixture",
                            "runtime_tier": "local_fixture",
                            "backend": "local",
                            "docker": {"image": "host-python-current-v1"},
                            "preflight": {"commands": []},
                            "hardware": {"requires_gpu": False},
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (root / "source-registry.json").write_text(
            json.dumps(
                {
                    "version": "v1",
                    "sources": [
                        {
                            "id": "local-v1-source",
                            "repo_url": "https://example.invalid/local-v1-source.git",
                            "commit": revision,
                            "local_path": "source",
                            "submodules": {
                                "policy": "none_required",
                                "status": "not_initialized",
                            },
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        dataset_dir = root / "dataset"
        dataset_dir.mkdir()
        dataset = dataset_dir / "dataset.json"
        dataset.write_text(
            json.dumps(
                {
                    "dataset_id": "local_v1_smoke",
                    "version": "v1",
                    "status": "verified",
                    "registries": {
                        "environments": "../environment-registry.json",
                        "sources": "../source-registry.json",
                    },
                    "tasks": [
                        {
                            "task_id": "local__v1_smoke",
                            "task_path": "../task",
                            "admission_status": "verified",
                            "environment_status": "ready",
                            "source_status": "ready",
                            "replay_status": "verified",
                            "admission_evidence": "../task/admission/evidence.json",
                        }
                    ],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return dataset


if __name__ == "__main__":
    unittest.main()
