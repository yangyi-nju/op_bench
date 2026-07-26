from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest

from op_bench.matched_runtime.contracts import BuildIdentity
from op_bench.matched_runtime.probe import (
    MatchedRuntimeProbe,
    ProbeExecution,
    ProbeObservation,
    ProbeSpec,
    write_compatibility_evidence,
)
from op_bench.runtime.validation import ContractError
from op_bench.task import TaskManifest


SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
SHA_C = "sha256:" + "c" * 64
BASE_COMMIT = "1" * 40
REQUIRED_CHECKS = (
    "source_identity",
    "runtime_identity",
    "target_module_provenance",
    "target_import",
    "selector_collection",
    "minimal_operation",
)


def task_manifest(root: Path) -> TaskManifest:
    task_dir = root / "task"
    task_dir.mkdir()
    manifest = {
        "task_id": "pytorch__fixture",
        "version": "v1",
        "environment_ref": "pytorch-matched-fixture",
        "runtime_tier": "cuda_python_overlay",
        "source_ref": "pytorch-source-fixture",
        "source": {
            "repo": "pytorch/pytorch",
            "base_commit": BASE_COMMIT,
            "snapshot_path": "snapshot/source",
        },
        "environment": {
            "backend": "remote_docker",
            "host": "gpu-a10",
            "tier": "cuda_python_overlay",
            "image": "op-bench/pytorch-matched:test",
            "python_executable": "python",
            "workspace_dir": "/workspace",
            "source_loading": {
                "mode": "python_overlay",
                "installed_package": "torch",
                "overlay_paths": ["torch/_refs/__init__.py"],
                "runtime_site_packages": "/tmp/op_bench_runtime/site-packages",
                "sync_before_tests": True,
            },
        },
        "evaluation": {
            "fail_to_pass": ["DecompTestsCUDA.test_target_cuda"],
            "pass_to_pass": ["DecompTestsCUDA.test_control_cuda"],
            "test_command": "{python} test/test_decomp.py {test}",
            "timeout_sec": 30,
        },
        "artifacts": {
            "gold_patch": "artifacts/gold.patch",
            "hidden_test_patch": "artifacts/hidden.patch",
        },
        "metadata": {"curation_status": "draft"},
        "compatibility": {
            "target_module": "torch/_refs/__init__.py",
            "target_import": "torch._refs",
            "selector_module": "test/test_decomp.py",
            "minimal_operation": (
                "import torch; "
                "torch._refs.exponential(torch.ones(1, device='cuda'))"
            ),
        },
    }
    (task_dir / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
    return TaskManifest.load(task_dir / "task.json")


def spec(task: TaskManifest) -> ProbeSpec:
    return ProbeSpec.from_task(
        task,
        strategy="matched_wheel",
        artifact_kind="official_wheel",
        artifact_id="torch-2.4.0+cu124-cp311-linux_x86_64",
        artifact_digest=SHA_A,
        artifact_digest_kind="wheel_sha256",
        build=BuildIdentity(
            flags=(),
            gpu_arches=(),
            ccache_key=None,
            artifact_digest=None,
            toolchain=(),
        ),
    )


def passed_observations() -> tuple[ProbeObservation, ...]:
    return tuple(
        ProbeObservation(
            name=name,
            exit_code=0,
            status="passed",
            summary=f"{name} passed",
        )
        for name in REQUIRED_CHECKS
    )


def successful_execution() -> ProbeExecution:
    return ProbeExecution(
        snapshot_digest=SHA_A,
        source_target_sha256=SHA_B,
        runtime_observation={
            "torch_version": "2.4.0+cu124",
            "python_implementation": "CPython",
            "python_abi": "cpython-311-x86_64-linux-gnu",
            "platform": "linux/amd64",
            "cuda_build": "12.4",
            "cuda_runtime": "12.4",
            "device_name": "Tesla V100-SXM2-32GB",
            "compute_capability": "7.0",
            "target_module_path_suffix": "torch/_refs/__init__.py",
            "target_module_sha256": SHA_B,
        },
        observations=passed_observations(),
        cleanup_failed=False,
    )


class StubBackend:
    def __init__(self, execution: ProbeExecution) -> None:
        self.execution = execution
        self.calls: list[tuple[str, str]] = []

    def execute(self, task: TaskManifest, selected: ProbeSpec) -> ProbeExecution:
        self.calls.append((task.task_id, selected.strategy))
        return self.execution


class ProbeSpecTests(unittest.TestCase):
    def test_plan_contains_required_checks_in_canonical_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            selected = spec(task_manifest(Path(tmp)))

        self.assertEqual(
            tuple(command.name for command in selected.commands),
            REQUIRED_CHECKS,
        )
        self.assertTrue(
            all(command.command_digest.startswith("sha256:") for command in selected.commands)
        )
        self.assertEqual(len({command.command_digest for command in selected.commands}), 6)

    def test_plan_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))

            first = spec(task)
            second = spec(task)

        self.assertEqual(first, second)

    def test_plan_rejects_unsafe_or_incomplete_probe_configuration(self) -> None:
        cases = (
            ("target_module", "../torch/_refs/__init__.py"),
            ("selector_module", "/test/test_decomp.py"),
            ("minimal_operation", "torch.ones(1)"),
        )
        for field, value in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                task = task_manifest(Path(tmp))
                task.data["compatibility"][field] = value
                with self.assertRaises(ContractError):
                    spec(task)

    def test_plan_requires_declared_f2p_and_p2p_selectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            task.data["evaluation"]["pass_to_pass"] = []

            with self.assertRaisesRegex(ContractError, "selectors"):
                spec(task)

    def test_plan_rejects_an_unknown_strategy_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))

            with self.assertRaisesRegex(ContractError, "strategy"):
                ProbeSpec.from_task(
                    task,
                    strategy="monkeypatch",
                    artifact_kind="official_wheel",
                    artifact_id="invalid",
                    artifact_digest=SHA_A,
                    artifact_digest_kind="wheel_sha256",
                    build=BuildIdentity(
                        flags=(),
                        gpu_arches=(),
                        ccache_key=None,
                        artifact_digest=None,
                        toolchain=(),
                    ),
                )


class MatchedRuntimeProbeTests(unittest.TestCase):
    def _run(self, execution: ProbeExecution):
        with tempfile.TemporaryDirectory() as tmp:
            task = task_manifest(Path(tmp))
            backend = StubBackend(execution)
            evidence = MatchedRuntimeProbe(
                backend=backend,
                now=lambda: datetime(2026, 7, 26, 12, tzinfo=timezone.utc),
            ).run(task, spec(task))
        self.assertEqual(backend.calls, [("pytorch__fixture", "matched_wheel")])
        return evidence

    def test_successful_probe_builds_compatible_evidence(self) -> None:
        evidence = self._run(successful_execution())

        self.assertEqual(evidence.status, "compatible")
        self.assertIsNone(evidence.failure)
        self.assertEqual(evidence.source.target_module_sha256, SHA_B)
        self.assertEqual(evidence.runtime.target_module_sha256, SHA_B)
        self.assertEqual(evidence.created_at, "2026-07-26T12:00:00Z")

    def test_target_hash_mismatch_is_incompatible_even_if_backend_claims_pass(self) -> None:
        execution = replace(
            successful_execution(),
            runtime_observation={
                **successful_execution().runtime_observation,
                "target_module_sha256": SHA_C,
            },
        )

        evidence = self._run(execution)

        self.assertEqual(evidence.status, "incompatible")
        self.assertEqual(evidence.failure.code, "target_module_not_from_snapshot")
        self.assertEqual(evidence.failure.check, "target_module_provenance")

    def test_import_failure_is_classified_without_running_later_checks(self) -> None:
        observations = list(passed_observations())
        observations[3] = ProbeObservation(
            name="target_import",
            exit_code=1,
            status="failed",
            summary="AttributeError while importing torch._refs",
        )
        observations[4] = ProbeObservation(
            name="selector_collection",
            exit_code=None,
            status="unavailable",
            summary="not run after target import failure",
        )
        observations[5] = ProbeObservation(
            name="minimal_operation",
            exit_code=None,
            status="unavailable",
            summary="not run after target import failure",
        )

        evidence = self._run(
            replace(successful_execution(), observations=tuple(observations))
        )

        self.assertEqual(evidence.status, "incompatible")
        self.assertEqual(evidence.failure.code, "target_import_failed")
        self.assertEqual(evidence.failure.check, "target_import")

    def test_unavailable_runtime_records_missing_observations(self) -> None:
        observations = list(passed_observations())
        for index in range(1, len(observations)):
            observations[index] = ProbeObservation(
                name=REQUIRED_CHECKS[index],
                exit_code=None,
                status="unavailable",
                summary="runtime was unavailable",
            )
        execution = replace(
            successful_execution(),
            runtime_observation={},
            observations=tuple(observations),
        )

        evidence = self._run(execution)

        self.assertEqual(evidence.status, "unavailable")
        self.assertEqual(evidence.failure.code, "runtime_unavailable")
        self.assertIsNone(evidence.runtime.torch_version)
        self.assertIsNone(evidence.runtime.target_module_sha256)

    def test_cleanup_failure_prevents_compatible_result(self) -> None:
        evidence = self._run(
            replace(successful_execution(), cleanup_failed=True)
        )

        self.assertEqual(evidence.status, "incompatible")
        self.assertEqual(evidence.failure.code, "cleanup_failed")

    def test_atomic_writer_round_trips_canonical_evidence_and_refuses_overwrite(self) -> None:
        evidence = self._run(successful_execution())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "compatibility" / "evidence.json"

            write_compatibility_evidence(path, evidence)

            encoded = path.read_bytes()
            self.assertEqual(encoded, json.dumps(
                evidence.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"))
            with self.assertRaisesRegex(ContractError, "exists"):
                write_compatibility_evidence(path, evidence)

    def test_writer_rejects_private_paths_in_public_summaries(self) -> None:
        evidence = self._run(successful_execution())
        unsafe = replace(
            evidence,
            checks=(
                replace(
                    evidence.checks[0],
                    summary="source copied from /Users/private/workspace",
                ),
                *evidence.checks[1:],
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ContractError, "sensitive"):
                write_compatibility_evidence(
                    Path(tmp) / "evidence.json",
                    unsafe,
                )


if __name__ == "__main__":
    unittest.main()
