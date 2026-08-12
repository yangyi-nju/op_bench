from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.backends import (
    LocalProcessBackend,
    RuntimeCommandResult,
    RuntimeTargetBinding,
    recover_remote_cleanup_resource,
)
from op_bench.runtime.resources import (
    RuntimeResourceHandle,
    runtime_raw_handle_hash,
    runtime_resource_id,
)
from op_bench.runtime.validation import ContractError
from tests.test_runtime_backends import LocalBackendFixture


class RuntimeBackendCleanupTests(unittest.TestCase):
    def test_cleanup_failure_is_terminal_public_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)

            with mock.patch(
                "op_bench.runtime.backends.shutil.rmtree",
                side_effect=OSError("private cleanup detail"),
            ):
                result = backend.cleanup(lease)

            self.assertFalse(result.report.all_released)
            self.assertEqual(result.report.entries[-1].status, "cleanup_failed")
            self.assertEqual(
                result.report.entries[-1].error_code,
                "workspace_remove_failed",
            )
            self.assertNotIn("private cleanup detail", repr(result.report.to_dict()))
            with self.assertRaisesRegex(ContractError, "terminal"):
                fixture.ledger.released(lease.handles[0].resource_id)

    def test_cleanup_rejects_handle_not_owned_by_exact_retry_store(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            private_path = fixture.evidence / "private_runtime_resources.json"
            private_path.unlink()

            with self.assertRaisesRegex(ContractError, "handle|lease"):
                backend.cleanup(lease)

    def test_remote_cleanup_recovery_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity_file = root / "identity"
            identity_file.write_text("fixture", encoding="utf-8")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=root,
                host_alias="exact.example.invalid",
                remote_user="runner",
                ssh_port=22,
                identity_file=identity_file,
                docker_binary="docker",
                ssh_binary="ssh",
                rsync_binary="rsync",
                remote_workspace_root="/srv/opbench/exact",
            )
            attempt_id = "attempt:v1:" + "a" * 64
            profile_hash = "sha256:" + "b" * 64

            def handle(resource_type: str, raw_handle: str) -> RuntimeResourceHandle:
                return RuntimeResourceHandle(
                    resource_id=runtime_resource_id(
                        attempt_id,
                        1,
                        profile_hash,
                        resource_type,
                        1,
                    ),
                    resource_type=resource_type,
                    ordinal=1,
                    raw_handle=raw_handle,
                    raw_handle_hash=runtime_raw_handle_hash(raw_handle),
                )

            commands: list[tuple[str, ...]] = []

            def runner(command, cwd, timeout_ms):
                del cwd, timeout_ms
                commands.append(command)
                return RuntimeCommandResult(
                    command=command,
                    cwd=".",
                    exit_code=(1 if "docker" in command[-1] else 0),
                    stdout="",
                    stderr=(
                        "Error: No such container"
                        if "docker" in command[-1]
                        else ""
                    ),
                    duration_ms=1,
                    timed_out=False,
                )

            container_name = "opbench-" + "a" * 20 + "-r0001"
            self.assertTrue(
                recover_remote_cleanup_resource(
                    binding,
                    handle("container", container_name),
                    attempt_id=attempt_id,
                    retry_index=1,
                    timeout_ms=1_000,
                    argv_runner=runner,
                )
            )
            remote_path = "/srv/opbench/exact/" + "a" * 64 + "/retry-0001/workspace"
            self.assertTrue(
                recover_remote_cleanup_resource(
                    binding,
                    handle(
                        "remote_workspace",
                        "runner@exact.example.invalid:" + remote_path,
                    ),
                    attempt_id=attempt_id,
                    retry_index=1,
                    timeout_ms=1_000,
                    argv_runner=runner,
                )
            )
            self.assertEqual(len(commands), 2)

            with self.assertRaisesRegex(ContractError, "does not match"):
                recover_remote_cleanup_resource(
                    binding,
                    handle("container", "opbench-unowned"),
                    attempt_id=attempt_id,
                    retry_index=1,
                    timeout_ms=1_000,
                    argv_runner=runner,
                )


if __name__ == "__main__":
    unittest.main()
