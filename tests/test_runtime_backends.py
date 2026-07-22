from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from op_bench.runtime.backends import (
    DockerRuntimeBackend,
    LocalProcessBackend,
    RemoteDockerRuntimeBackend,
    RuntimeAttemptContext,
    RuntimeCommandBackend,
    RuntimeLease,
    RuntimeTargetBinding,
    RuntimeBackendUnavailable,
    load_runtime_target_binding,
    ScriptedRuntimeBackend,
)
from op_bench.runtime.profiles import load_runtime_profile_registry
from op_bench.runtime.resources import (
    AttemptResourceLedger,
    RuntimeLeaseStore,
    verify_runtime_resource_ownership,
)
from op_bench.runtime.source_materialization import SourceMaterializationError
from op_bench.runtime.validation import ContractError
from tests.runtime_git_fixture import git, initialize_git_repo


ATTEMPT_ID = "attempt:v1:" + "a" * 64


class StepClock:
    def __init__(self) -> None:
        self.value = 1_000

    def __call__(self) -> int:
        current = self.value
        self.value += 1
        return current


class LocalBackendFixture:
    def __init__(self, root: Path, *, profile=None, binding=None) -> None:
        self.root = root
        self.source = root / "source"
        self.source_revision = initialize_git_repo(self.source)
        self.workspaces = root / "workspaces"
        self.workspaces.mkdir(exist_ok=True)
        self.evidence = root / "evidence"
        self.evidence.mkdir()
        self.profile = profile or load_runtime_profile_registry(
            Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
        ).get("local-cpu-process-v1")
        self.ledger = AttemptResourceLedger(
            self.evidence / "runtime_resources.jsonl",
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
            clock_ms=StepClock(),
        )
        self.store = RuntimeLeaseStore(
            self.evidence / "private_runtime_resources.json",
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
        )
        self.binding = binding or RuntimeTargetBinding(
            backend="local", local_workspace_parent=self.workspaces
        )
        self.context = RuntimeAttemptContext(
            attempt_id=ATTEMPT_ID,
            retry_index=1,
            runtime_profile_hash=self.profile.content_hash,
            frozen_source_directory=self.source,
            frozen_source_revision=self.source_revision,
            resource_ledger=self.ledger,
            lease_store=self.store,
            target_binding=self.binding,
        )


def pollute_source_after_frozen_commit(fixture: LocalBackendFixture) -> str:
    (fixture.source / ".gitignore").write_text(
        "ignored-cache/\n",
        encoding="utf-8",
    )
    git(fixture.source, "add", ".gitignore")
    git(fixture.source, "commit", "--quiet", "-m", "freeze ignore policy")
    revision = git(fixture.source, "rev-parse", "HEAD").stdout.decode().strip()
    fixture.context = replace(
        fixture.context,
        frozen_source_revision=revision,
    )
    (fixture.source / "ignored-cache").mkdir()
    (fixture.source / "ignored-cache" / "state.bin").write_bytes(b"ignored")
    (fixture.source / "untracked.txt").write_text("untracked\n", encoding="utf-8")
    (fixture.source / "src" / "operator.py").write_text(
        "VALUE = 999\n",
        encoding="utf-8",
    )
    return revision


class LocalProcessBackendTests(unittest.TestCase):
    def test_scripted_workspace_store_failure_closes_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = ScriptedRuntimeBackend.success()

            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=ContractError("private store failure"),
            ):
                with self.assertRaises(RuntimeBackendUnavailable) as raised:
                    backend.prepare(fixture.profile, fixture.context)

            self.assertEqual(
                raised.exception.reason_code,
                "workspace_registration_failed",
            )
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "create_failed"],
            )

    def test_materializer_cleanup_failure_is_attributed_to_owned_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()

            with (
                mock.patch(
                    "op_bench.runtime.source_materialization._stream_safe_git_archive",
                    side_effect=SourceMaterializationError("archive failure"),
                ),
                mock.patch(
                    "op_bench.runtime.source_materialization._remove_exact_workspace",
                    side_effect=OSError("materializer cleanup failure"),
                ),
                mock.patch(
                    "op_bench.runtime.backends.shutil.rmtree",
                    side_effect=OSError("backend cleanup failure"),
                ),
            ):
                with self.assertRaises(RuntimeBackendUnavailable) as raised:
                    backend.prepare(fixture.profile, fixture.context)

            self.assertEqual(raised.exception.reason_code, "prepare_cleanup_failed")
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "cleanup_failed"],
            )
            workspace = fixture.binding.local_workspace_parent / (
                fixture.context.attempt_id.removeprefix("attempt:v1:")
            ) / "retry-0001" / "workspace"
            self.assertTrue(workspace.is_dir())

    def test_process_store_failure_terminates_exact_started_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            original_put = fixture.store.put_exact

            def fail_process_store(resource_id, resource_type, ordinal, raw_handle):
                if resource_type == "process":
                    raise ContractError("private store failure")
                return original_put(resource_id, resource_type, ordinal, raw_handle)

            process = mock.Mock(pid=4242)
            with (
                mock.patch(
                    "op_bench.runtime.backends.subprocess.Popen",
                    return_value=process,
                ),
                mock.patch.object(
                    fixture.store,
                    "put_exact",
                    side_effect=fail_process_store,
                ),
                mock.patch(
                    "op_bench.runtime.backends._terminate_exact_process_group"
                ) as terminate,
            ):
                with self.assertRaises(RuntimeBackendUnavailable) as raised:
                    backend.run(
                        lease,
                        (sys.executable, "-c", "pass"),
                        ".",
                        1_000,
                    )

            self.assertEqual(raised.exception.reason_code, "process_registration_failed")
            terminate.assert_called_once_with(
                process,
                grace_ms=fixture.profile.cleanup_policy.grace_ms,
            )
            process_records = [
                record
                for record in fixture.ledger.records
                if record.resource_type == "process"
            ]
            self.assertEqual(
                [record.transition for record in process_records],
                ["declared", "created", "released"],
            )
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_process_store_interrupt_terminates_before_interrupt_propagates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            process = mock.Mock(pid=4242)

            with (
                mock.patch(
                    "op_bench.runtime.backends.subprocess.Popen",
                    return_value=process,
                ),
                mock.patch.object(
                    fixture.store,
                    "put_exact",
                    side_effect=KeyboardInterrupt,
                ),
                mock.patch(
                    "op_bench.runtime.backends._terminate_exact_process_group"
                ) as terminate,
            ):
                with self.assertRaises(KeyboardInterrupt):
                    backend.run(
                        lease,
                        (sys.executable, "-c", "pass"),
                        ".",
                        1_000,
                    )

            terminate.assert_called_once_with(
                process,
                grace_ms=fixture.profile.cleanup_policy.grace_ms,
            )
            process_records = [
                record
                for record in fixture.ledger.records
                if record.resource_type == "process"
            ]
            self.assertEqual(
                [record.transition for record in process_records],
                ["declared", "created", "released"],
            )
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_post_materialization_store_failure_removes_owned_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=ContractError("private store failure"),
            ):
                with self.assertRaises(RuntimeBackendUnavailable) as raised:
                    LocalProcessBackend().prepare(
                        fixture.profile,
                        fixture.context,
                    )

            self.assertEqual(raised.exception.reason_code, "workspace_prepare_failed")
            self.assertEqual(list(fixture.workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "released"],
            )

    def test_attempt_context_requires_explicit_frozen_source_revision(self) -> None:
        parameters = inspect.signature(RuntimeAttemptContext).parameters

        self.assertIn("frozen_source_revision", parameters)
        self.assertIs(
            parameters["frozen_source_revision"].default,
            inspect.Parameter.empty,
        )

    def test_prepare_materializes_only_the_frozen_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            pollute_source_after_frozen_commit(fixture)

            lease = LocalProcessBackend().prepare(fixture.profile, fixture.context)
            workspace = Path(lease.handles[0].raw_handle)

            self.assertEqual(
                (workspace / "src" / "operator.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            self.assertFalse((workspace / "ignored-cache").exists())
            self.assertFalse((workspace / "untracked.txt").exists())
            status = git(
                workspace,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            ).stdout
            self.assertEqual(status, b"")

    def test_runtime_process_does_not_inherit_controller_python_or_secret_env(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)

            with mock.patch.dict(
                os.environ,
                {
                    "PYTHONPATH": "controller-relative-path",
                    "OPBENCH_PRIVATE_SENTINEL": "controller-secret",
                },
                clear=False,
            ):
                result = backend.run(
                    lease,
                    (
                        sys.executable,
                        "-c",
                        (
                            "import os;"
                            "print('PYTHONPATH' in os.environ);"
                            "print('OPBENCH_PRIVATE_SENTINEL' in os.environ)"
                        ),
                    ),
                    ".",
                    5_000,
                )

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "False\nFalse\n")
            backend.cleanup(lease)

    def test_prepare_run_collect_cleanup_has_exact_resource_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()

            lease = backend.prepare(fixture.profile, fixture.context)
            result = backend.run(
                lease,
                (sys.executable, "-c", "print('runtime-ok')"),
                ".",
                5_000,
            )
            evidence = backend.collect(lease)
            cleanup = backend.cleanup(lease)
            repeated = backend.cleanup(lease)

            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.stdout, "runtime-ok\n")
            self.assertFalse(result.timed_out)
            self.assertTrue(cleanup.report.all_released)
            self.assertEqual(repeated, cleanup)
            self.assertNotIn(str(fixture.workspaces), repr(evidence.to_dict()))
            self.assertNotIn("pid:", repr(evidence.to_dict()))
            self.assertEqual(
                [(item.resource_type, item.transition) for item in fixture.ledger.verify()],
                [
                    ("workspace", "declared"),
                    ("workspace", "created"),
                    ("process", "declared"),
                    ("process", "created"),
                    ("process", "released"),
                    ("workspace", "released"),
                ],
            )
            self.assertEqual(list(fixture.workspaces.iterdir()), [])

    def test_timeout_targets_only_spawned_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)

            with mock.patch(
                "op_bench.runtime.backends.os.killpg",
                wraps=os.killpg,
            ) as kill_group:
                result = backend.run(
                    lease,
                    (sys.executable, "-c", "import time; time.sleep(30)"),
                    ".",
                    50,
                )

            self.assertTrue(result.timed_out)
            self.assertGreaterEqual(kill_group.call_count, 1)
            self.assertIsInstance(kill_group.call_args_list[0].args[0], int)
            backend.cleanup(lease)

    def test_unknown_lease_and_unsafe_cwd_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            unknown = replace(lease, attempt_id="attempt:v1:" + "b" * 64)

            with self.assertRaisesRegex(ContractError, "lease"):
                backend.run(unknown, (sys.executable, "-V"), ".", 1_000)
            with self.assertRaisesRegex(ContractError, "cwd"):
                backend.run(lease, (sys.executable, "-V"), "../outside", 1_000)
            backend.cleanup(lease)

    def test_command_adapter_preserves_canonical_action_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            backend = LocalProcessBackend()
            lease = backend.prepare(fixture.profile, fixture.context)
            adapter = RuntimeCommandBackend(backend, lease)

            result = adapter.run(
                (sys.executable, "-c", "print('adapter-ok')"),
                ".",
                5_000,
            )

            self.assertEqual(result.stdout, "adapter-ok\n")
            self.assertEqual(result.command[0], sys.executable)
            backend.cleanup(lease)

    def test_backend_source_has_no_global_discovery_or_cleanup_primitive(self) -> None:
        source = inspect.getsource(__import__("op_bench.runtime.backends", fromlist=["*"]))
        for forbidden in (
            "pkill",
            "killall",
            "docker ps",
            "ps aux",
            "host discovery",
            "port scan",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source.lower())

    def test_scripted_backend_records_the_same_resource_transitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = LocalBackendFixture(Path(temporary))
            scripted = ScriptedRuntimeBackend.success(
                stdout="scripted-ok\n",
                exit_code=0,
            )

            lease = scripted.prepare(fixture.profile, fixture.context)
            result = scripted.run(lease, ("fixture",), ".", 1_000)
            cleanup = scripted.cleanup(lease)

            self.assertEqual(result.stdout, "scripted-ok\n")
            self.assertTrue(cleanup.report.all_released)
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "declared", "created", "released", "released"],
            )


class RecordingArgvRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], Path | None, int]] = []

    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        from op_bench.runtime.backends import RuntimeCommandResult

        self.calls.append((command, cwd, timeout_ms))
        return RuntimeCommandResult(
            command=command,
            cwd="." if cwd is None else str(cwd),
            exit_code=0,
            stdout="",
            stderr="",
            duration_ms=1,
            timed_out=False,
        )


class StartFailureArgvRunner(RecordingArgvRunner):
    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        result = super().__call__(command, cwd, timeout_ms)
        if len(command) >= 2 and command[1] == "start":
            return replace(result, exit_code=1, stderr="private start failure")
        return result


class SelectiveExceptionArgvRunner(RecordingArgvRunner):
    def __init__(self, predicate) -> None:
        super().__init__()
        self.predicate = predicate

    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        if self.predicate(command):
            self.calls.append((command, cwd, timeout_ms))
            raise RuntimeBackendUnavailable("runtime_command_unavailable")
        return super().__call__(command, cwd, timeout_ms)


class ExistingRemoteWorkspaceArgvRunner(RecordingArgvRunner):
    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        result = super().__call__(command, cwd, timeout_ms)
        if (
            command[0] == "ssh-fixture"
            and "mkdir --" in command[-1]
            and "mkdir -p --" not in command[-1]
        ):
            return replace(result, exit_code=1, stderr="workspace already exists")
        return result


class RsyncAndRemoteCleanupFailureArgvRunner(RecordingArgvRunner):
    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        result = super().__call__(command, cwd, timeout_ms)
        if command[0] == "rsync-fixture":
            return replace(result, exit_code=1, stderr="sync failed")
        if command[0] == "ssh-fixture" and "rm -rf --" in command[-1]:
            return replace(result, exit_code=1, stderr="cleanup failed")
        return result


class TransientRsyncFailureArgvRunner(RecordingArgvRunner):
    def __init__(self) -> None:
        super().__init__()
        self.rsync_attempts = 0

    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        result = super().__call__(command, cwd, timeout_ms)
        if command[0] != "rsync-fixture":
            return result
        self.rsync_attempts += 1
        if self.rsync_attempts < 3:
            return replace(
                result,
                exit_code=255,
                stderr="transient ssh transport interruption",
            )
        return result


class TransientRemoteCleanupFailureArgvRunner(RecordingArgvRunner):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_attempts: dict[str, int] = {}

    def __call__(
        self,
        command: tuple[str, ...],
        cwd: Path | None,
        timeout_ms: int,
    ):
        result = super().__call__(command, cwd, timeout_ms)
        if command[0] != "ssh-fixture":
            return result
        remote_command = command[-1]
        if not (
            "docker-fixture rm --force" in remote_command
            or "rm -rf --" in remote_command
        ):
            return result
        attempts = self.cleanup_attempts.get(remote_command, 0) + 1
        self.cleanup_attempts[remote_command] = attempts
        if attempts == 1:
            return replace(result, exit_code=255, stderr="transient ssh failure")
        return result


class ContainerBackendCommandTests(unittest.TestCase):
    def test_docker_fresh_evaluation_uses_distinct_exact_owned_containers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()

            agent_backend = DockerRuntimeBackend(argv_runner=runner)
            agent_lease = agent_backend.prepare(profile, fixture.context)
            self.assertTrue(agent_backend.cleanup(agent_lease).report.all_released)

            evaluation_backend = DockerRuntimeBackend(argv_runner=runner)
            evaluation_lease = evaluation_backend.prepare(profile, fixture.context)
            self.assertTrue(
                evaluation_backend.cleanup(evaluation_lease).report.all_released
            )

            verify_runtime_resource_ownership(
                fixture.ledger.records,
                fixture.store.active_handles,
            )
            container_handles = [
                handle.raw_handle
                for handle in fixture.store.active_handles
                if handle.resource_type == "container"
            ]
            self.assertEqual(len(container_handles), 2)
            self.assertEqual(len(set(container_handles)), 2)

    def test_remote_fresh_evaluation_uses_distinct_exact_owned_handles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()

            agent_backend = RemoteDockerRuntimeBackend(argv_runner=runner)
            agent_lease = agent_backend.prepare(profile, fixture.context)
            self.assertTrue(agent_backend.cleanup(agent_lease).report.all_released)

            evaluation_backend = RemoteDockerRuntimeBackend(argv_runner=runner)
            evaluation_lease = evaluation_backend.prepare(profile, fixture.context)
            self.assertTrue(
                evaluation_backend.cleanup(evaluation_lease).report.all_released
            )

            verify_runtime_resource_ownership(
                fixture.ledger.records,
                fixture.store.active_handles,
            )
            for resource_type in ("remote_workspace", "container"):
                raw_handles = [
                    handle.raw_handle
                    for handle in fixture.store.active_handles
                    if handle.resource_type == resource_type
                ]
                self.assertEqual(len(raw_handles), 2)
                self.assertEqual(len(set(raw_handles)), 2)

    def test_remote_ordinal_read_failure_releases_local_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            original_records = AttemptResourceLedger.records.fget
            assert original_records is not None
            reads = 0

            def fail_remote_ordinal_read(instance):
                nonlocal reads
                reads += 1
                if reads == 2:
                    raise ContractError("fixture ledger read failure")
                return original_records(instance)

            with (
                mock.patch.object(
                    AttemptResourceLedger,
                    "records",
                    new=property(fail_remote_ordinal_read),
                ),
                self.assertRaises(RuntimeBackendUnavailable) as raised,
            ):
                RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                    profile,
                    fixture.context,
                )

            self.assertEqual(raised.exception.reason_code, "runtime_prepare_failed")
            self.assertEqual(runner.calls, [])
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "released"],
            )

    def test_docker_prepare_interrupt_cleans_exact_owned_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            recorded = RecordingArgvRunner()

            def interrupt_start(command, cwd, timeout_ms):
                result = recorded(command, cwd, timeout_ms)
                if command[:2] == ("docker-fixture", "start"):
                    raise KeyboardInterrupt
                return result

            with self.assertRaises(KeyboardInterrupt):
                DockerRuntimeBackend(argv_runner=interrupt_start).prepare(
                    profile,
                    fixture.context,
                )

            commands = [command for command, _, _ in recorded.calls]
            self.assertEqual(commands[-1][1:3], ("rm", "--force"))
            self.assertEqual(list(workspaces.iterdir()), [])
            final = {
                record.resource_type: record.transition
                for record in fixture.ledger.records
            }
            self.assertEqual(final, {"workspace": "released", "container": "released"})

    def test_docker_post_workspace_declare_failure_releases_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            original_declare = fixture.ledger.declare

            def fail_container_declare(resource_type, ordinal):
                if resource_type == "container":
                    raise ContractError("ledger declare failure")
                return original_declare(resource_type, ordinal)

            runner = RecordingArgvRunner()
            with mock.patch.object(
                fixture.ledger,
                "declare",
                side_effect=fail_container_declare,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    DockerRuntimeBackend(argv_runner=runner).prepare(
                        profile,
                        fixture.context,
                    )

            self.assertEqual(runner.calls, [])
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "released"],
            )

    def test_remote_post_leaf_container_declare_failure_unwinds_all_owned_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            original_declare = fixture.ledger.declare

            def fail_container_declare(resource_type, ordinal):
                if resource_type == "container":
                    raise ContractError("ledger declare failure")
                return original_declare(resource_type, ordinal)

            runner = RecordingArgvRunner()
            with mock.patch.object(
                fixture.ledger,
                "declare",
                side_effect=fail_container_declare,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                        profile,
                        fixture.context,
                    )

            commands = [command for command, _, _ in runner.calls]
            self.assertTrue(any(command[0] == "rsync-fixture" for command in commands))
            self.assertIn("rm -rf --", commands[-1][-1])
            self.assertNotIn("docker-fixture create", "\n".join(c[-1] for c in commands))
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                ["declared", "created", "declared", "created", "released", "released"],
            )

    def test_remote_prepare_cleanup_failure_is_not_masked_by_sync_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RsyncAndRemoteCleanupFailureArgvRunner()

            with self.assertRaises(RuntimeBackendUnavailable) as raised:
                RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                    profile,
                    fixture.context,
                )

            self.assertEqual(raised.exception.reason_code, "prepare_cleanup_failed")
            self.assertEqual(list(workspaces.iterdir()), [])
            final = {
                record.resource_type: record.transition
                for record in fixture.ledger.records
            }
            self.assertEqual(final["workspace"], "released")
            self.assertEqual(final["remote_workspace"], "cleanup_failed")

    def test_remote_prepare_continues_partial_sync_after_transient_transport_failures(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = TransientRsyncFailureArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(runner.rsync_attempts, 3)
            self.assertEqual(len(sync_commands), 3)
            self.assertEqual(sync_commands[0], sync_commands[1])
            self.assertEqual(sync_commands[1], sync_commands[2])
            self.assertIn("--partial", sync_commands[0])
            self.assertFalse(
                any(
                    command[0] == "ssh-fixture" and "rm -rf --" in command[-1]
                    for command, _, _ in runner.calls
                )
            )
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_docker_process_store_failure_never_executes_container_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = DockerRuntimeBackend(argv_runner=runner)
            lease = backend.prepare(profile, fixture.context)
            original_put = fixture.store.put_exact

            def fail_process_store(resource_id, resource_type, ordinal, raw_handle):
                if resource_type == "process":
                    raise ContractError("private store failure")
                return original_put(resource_id, resource_type, ordinal, raw_handle)

            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=fail_process_store,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    backend.run(lease, ("python", "-V"), ".", 1_000)

            self.assertFalse(
                any(len(command) > 1 and command[1] == "exec" for command, _, _ in runner.calls)
            )
            process_records = [
                record for record in fixture.ledger.records if record.resource_type == "process"
            ]
            self.assertEqual(
                [record.transition for record in process_records],
                ["declared", "create_failed"],
            )
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_process_store_failure_never_executes_remote_process(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)
            lease = backend.prepare(profile, fixture.context)
            original_put = fixture.store.put_exact
            calls_before_run = len(runner.calls)

            def fail_process_store(resource_id, resource_type, ordinal, raw_handle):
                if resource_type == "process":
                    raise ContractError("private store failure")
                return original_put(resource_id, resource_type, ordinal, raw_handle)

            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=fail_process_store,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    backend.run(lease, ("python", "-V"), ".", 1_000)

            self.assertEqual(len(runner.calls), calls_before_run)
            process_records = [
                record for record in fixture.ledger.records if record.resource_type == "process"
            ]
            self.assertEqual(
                [record.transition for record in process_records],
                ["declared", "create_failed"],
            )
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_docker_post_create_store_failure_removes_exact_owned_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            original_put = fixture.store.put_exact

            def fail_container_store(resource_id, resource_type, ordinal, raw_handle):
                if resource_type == "container":
                    raise ContractError("private store failure")
                return original_put(resource_id, resource_type, ordinal, raw_handle)

            runner = RecordingArgvRunner()
            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=fail_container_store,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    DockerRuntimeBackend(argv_runner=runner).prepare(
                        profile,
                        fixture.context,
                    )

            commands = [command for command, _, _ in runner.calls]
            container_name = commands[0][commands[0].index("--name") + 1]
            self.assertEqual(
                commands[-1],
                ("docker-fixture", "rm", "--force", container_name),
            )
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                [
                    "declared",
                    "created",
                    "declared",
                    "created",
                    "released",
                    "released",
                ],
            )

    def test_remote_post_leaf_store_failure_removes_only_exact_owned_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            original_put = fixture.store.put_exact

            def fail_remote_store(resource_id, resource_type, ordinal, raw_handle):
                if resource_type == "remote_workspace":
                    raise ContractError("private store failure")
                return original_put(resource_id, resource_type, ordinal, raw_handle)

            runner = RecordingArgvRunner()
            with mock.patch.object(
                fixture.store,
                "put_exact",
                side_effect=fail_remote_store,
            ):
                with self.assertRaises(RuntimeBackendUnavailable):
                    RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                        profile,
                        fixture.context,
                    )

            commands = [command for command, _, _ in runner.calls]
            self.assertEqual(len(commands), 3)
            self.assertIn("mkdir -p --", commands[0][-1])
            self.assertIn("mkdir --", commands[1][-1])
            self.assertIn("rm -rf --", commands[2][-1])
            self.assertFalse(any(command[0] == "rsync-fixture" for command in commands))
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [record.transition for record in fixture.ledger.records],
                [
                    "declared",
                    "created",
                    "declared",
                    "created",
                    "released",
                    "released",
                ],
            )

    def test_docker_materializes_only_the_frozen_revision_before_mount(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = self.profile("docker")
            workspaces = root / "workspaces"
            workspaces.mkdir()
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            pollute_source_after_frozen_commit(fixture)
            runner = RecordingArgvRunner()
            backend = DockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)

            workspace = next(
                handle for handle in lease.handles if handle.resource_type == "workspace"
            )
            snapshot = Path(workspace.raw_handle)
            self.assertEqual(
                (snapshot / "src" / "operator.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            self.assertFalse((snapshot / "ignored-cache").exists())
            self.assertFalse((snapshot / "untracked.txt").exists())
            self.assertIn(f"{snapshot}:/workspace:rw", runner.calls[0][0])
            backend.cleanup(lease)

    def test_remote_materializes_only_the_frozen_revision_before_sync(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            pollute_source_after_frozen_commit(fixture)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)

            workspace = next(
                handle for handle in lease.handles if handle.resource_type == "workspace"
            )
            snapshot = Path(workspace.raw_handle)
            self.assertEqual(
                (snapshot / "src" / "operator.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )
            self.assertFalse((snapshot / "ignored-cache").exists())
            self.assertFalse((snapshot / "untracked.txt").exists())
            initial_sync = next(
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            )
            self.assertEqual(initial_sync[-2], str(snapshot) + "/")
            backend.cleanup(lease)

    def test_remote_invalid_revision_fails_before_any_target_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            with self.assertRaises(RuntimeBackendUnavailable) as raised:
                backend.prepare(
                    profile,
                    replace(fixture.context, frozen_source_revision="f" * 40),
                )

            self.assertEqual(raised.exception.reason_code, "workspace_prepare_failed")
            self.assertEqual(
                [
                    (record.resource_type, record.transition)
                    for record in fixture.ledger.records
                ],
                [("workspace", "declared"), ("workspace", "create_failed")],
            )
            self.assertEqual(runner.calls, [])
            self.assertEqual(list(workspaces.iterdir()), [])

    def test_docker_command_exception_during_create_releases_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = SelectiveExceptionArgvRunner(
                lambda command: len(command) > 1 and command[1] == "create"
            )

            with self.assertRaisesRegex(
                RuntimeBackendUnavailable,
                "runtime_command_unavailable",
            ):
                DockerRuntimeBackend(argv_runner=runner).prepare(
                    profile,
                    fixture.context,
                )

            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [(record.resource_type, record.transition) for record in fixture.ledger.records],
                [
                    ("workspace", "declared"),
                    ("workspace", "created"),
                    ("container", "declared"),
                    ("container", "create_failed"),
                    ("workspace", "released"),
                ],
            )

    def test_cleanup_command_exceptions_do_not_skip_remaining_exact_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = SelectiveExceptionArgvRunner(lambda command: False)
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)
            lease = backend.prepare(profile, fixture.context)
            runner.predicate = lambda command: (
                command[0] == "ssh-fixture"
                and ("docker-fixture rm --force" in command[-1] or "rm -rf" in command[-1])
            )

            cleanup = backend.cleanup(lease)

            self.assertFalse(cleanup.report.all_released)
            self.assertFalse(
                Path(
                    next(
                        handle.raw_handle
                        for handle in lease.handles
                        if handle.resource_type == "workspace"
                    )
                ).exists()
            )
            self.assertEqual(
                sorted((entry.resource_type, entry.status) for entry in cleanup.report.entries),
                [
                    ("container", "cleanup_failed"),
                    ("remote_workspace", "cleanup_failed"),
                    ("workspace", "released"),
                ],
            )

    def test_remote_cleanup_retries_the_same_exact_resources_after_transient_ssh_failure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = TransientRemoteCleanupFailureArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)
            lease = backend.prepare(profile, fixture.context)

            cleanup = backend.cleanup(lease)

            self.assertTrue(cleanup.report.all_released)
            container = next(
                handle for handle in lease.handles if handle.resource_type == "container"
            )
            remote = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "remote_workspace"
            )
            self.assertEqual(
                runner.cleanup_attempts,
                {
                    f"docker-fixture rm --force {container.raw_handle}": 2,
                    f"rm -rf -- {remote.raw_handle.split(':', 1)[1]}": 2,
                },
            )

    @staticmethod
    def profile(backend: str, *, gpu: bool = False):
        base = load_runtime_profile_registry(
            Path(__file__).resolve().parents[1] / "configs" / "runtime_profiles.v1.json"
        ).get("local-cpu-process-v1")
        return replace(
            base,
            backend=backend,
            requires_gpu=gpu,
            network_policy="denied",
            mount_policy=replace(
                base.mount_policy,
                workspace_target="/workspace",
                root_filesystem="read_only_container",
            ),
            resource_policy=replace(
                base.resource_policy,
                cpu_millis=2_000,
                memory_bytes=4_096,
                pids_limit=64,
                gpu_count=1 if gpu else 0,
            ),
            cleanup_policy=replace(
                base.cleanup_policy,
                remove_container=True,
            ),
        )

    def test_docker_uses_exact_name_mount_limits_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker", gpu=True)
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = DockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            result = backend.run(lease, ("python", "-V"), "src", 1_000)
            execute = next(
                call[0]
                for call in runner.calls
                if len(call[0]) >= 2 and call[0][1] == "exec"
            )
            cleanup = backend.cleanup(lease)

            workspace = next(h for h in lease.handles if h.resource_type == "workspace")
            container = next(h for h in lease.handles if h.resource_type == "container")
            create = runner.calls[0][0]
            self.assertEqual(create[:2], ("docker-fixture", "create"))
            self.assertIn(container.raw_handle, create)
            self.assertIn(f"opbench.resource-id={container.resource_id}", create)
            self.assertIn(f"{workspace.raw_handle}:/workspace:rw", create)
            for expected in (
                "--network",
                "none",
                "--cpus",
                "2",
                "--memory",
                "4096",
                "--pids-limit",
                "64",
                "--gpus",
                "1",
                "--read-only",
            ):
                self.assertIn(expected, create)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("GIT_CONFIG_KEY_0=safe.directory", execute)
            self.assertIn("GIT_CONFIG_VALUE_0=/workspace", execute)
            for expected in (
                "XDG_CACHE_HOME=/tmp/op_bench_runtime/xdg-cache",
                "TRITON_CACHE_DIR=/tmp/op_bench_runtime/triton-cache",
                "TORCHINDUCTOR_CACHE_DIR=/tmp/op_bench_runtime/torchinductor-cache",
                "CCACHE_DIR=/tmp/op_bench_runtime/ccache",
                "CCACHE_MAXSIZE=2G",
            ):
                self.assertIn(expected, execute)
            self.assertTrue(cleanup.report.all_released)
            self.assertEqual(
                runner.calls[-1][0],
                ("docker-fixture", "rm", "--force", container.raw_handle),
            )

    def test_docker_start_failure_removes_exact_container_and_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("docker")
            binding = RuntimeTargetBinding(
                backend="docker",
                local_workspace_parent=workspaces,
                docker_binary="docker-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = StartFailureArgvRunner()
            backend = DockerRuntimeBackend(argv_runner=runner)

            with self.assertRaisesRegex(
                __import__(
                    "op_bench.runtime.backends",
                    fromlist=["RuntimeBackendUnavailable"],
                ).RuntimeBackendUnavailable,
                "container_start_failed",
            ):
                backend.prepare(profile, fixture.context)

            commands = [call[0] for call in runner.calls]
            container_name = commands[0][commands[0].index("--name") + 1]
            self.assertEqual(
                commands[-1],
                ("docker-fixture", "rm", "--force", container_name),
            )
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [(record.resource_type, record.transition) for record in fixture.ledger.records],
                [
                    ("workspace", "declared"),
                    ("workspace", "created"),
                    ("container", "declared"),
                    ("container", "created"),
                    ("container", "released"),
                    ("workspace", "released"),
                ],
            )

    def test_remote_docker_uses_only_explicit_target_and_direct_operations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                ssh_port=2222,
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            workspace = next(
                h for h in lease.handles if h.resource_type == "workspace"
            )
            self.assertTrue(Path(workspace.raw_handle).is_dir())
            (Path(workspace.raw_handle) / "src" / "operator.py").write_text(
                "VALUE = 2\n", encoding="utf-8"
            )
            backend.run(lease, ("python", "-V"), ".", 1_000)
            cleanup = backend.cleanup(lease)

            remote = next(
                h for h in lease.handles if h.resource_type == "remote_workspace"
            )
            container = next(h for h in lease.handles if h.resource_type == "container")
            commands = [call[0] for call in runner.calls]
            self.assertEqual(commands[0][0], "ssh-fixture")
            self.assertIn("runner@gpu-exact-fixture", commands[0])
            self.assertIn("mkdir -p --", commands[0][-1])
            self.assertIn("mkdir --", commands[1][-1])
            sync_commands = [
                command for command in commands if command[0] == "rsync-fixture"
            ]
            self.assertEqual(sync_commands[0][-1], remote.raw_handle + "/")
            self.assertEqual(sync_commands[-1][-1], remote.raw_handle + "/")
            self.assertFalse(
                any(
                    argument.startswith("--exclude=")
                    for command in sync_commands
                    for argument in command
                )
            )
            remote_execute = next(
                command
                for command in commands
                if command[0] == "ssh-fixture"
                and "docker-fixture exec" in command[-1]
            )
            self.assertIn("GIT_CONFIG_KEY_0=safe.directory", remote_execute[-1])
            self.assertIn("GIT_CONFIG_VALUE_0=/workspace", remote_execute[-1])
            for expected in (
                "XDG_CACHE_HOME=/tmp/op_bench_runtime/xdg-cache",
                "TRITON_CACHE_DIR=/tmp/op_bench_runtime/triton-cache",
                "TORCHINDUCTOR_CACHE_DIR=/tmp/op_bench_runtime/torchinductor-cache",
                "CCACHE_DIR=/tmp/op_bench_runtime/ccache",
                "CCACHE_MAXSIZE=2G",
            ):
                self.assertIn(expected, remote_execute[-1])
            self.assertTrue(cleanup.report.all_released)
            self.assertFalse(Path(workspace.raw_handle).exists())
            self.assertIn(
                f"docker-fixture rm --force {container.raw_handle}",
                commands[-2][-1],
            )
            flattened = "\n".join(" ".join(command) for command in commands).lower()
            for forbidden in (
                "ping ",
                "docker ps",
                " ps ",
                "--filter",
                "nmap",
                "*",
                "192.168.",
            ):
                with self.subTest(forbidden=forbidden):
                    self.assertNotIn(forbidden, flattened)

    def test_remote_sync_fingerprint_ignores_ambient_git_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            decoy = root / "decoy"
            initialize_git_repo(decoy)
            pollution = {
                "GIT_DIR": str(decoy / ".git"),
                "GIT_WORK_TREE": str(decoy),
                "GIT_INDEX_FILE": str(decoy / ".git" / "index"),
                "GIT_OBJECT_DIRECTORY": str(decoy / ".git" / "objects"),
                "GIT_ALTERNATE_OBJECT_DIRECTORIES": str(
                    fixture.source / ".git" / "objects"
                ),
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "core.hooksPath",
                "GIT_CONFIG_VALUE_0": str(root / "foreign-hooks"),
            }
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            with mock.patch.dict(os.environ, pollution, clear=False):
                lease = backend.prepare(profile, fixture.context)
                workspace = next(
                    handle
                    for handle in lease.handles
                    if handle.resource_type == "workspace"
                )
                (Path(workspace.raw_handle) / "src" / "operator.py").write_text(
                    "VALUE = 2\n",
                    encoding="utf-8",
                )
                backend.run(lease, ("python", "-V"), ".", 1_000)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(len(sync_commands), 2)
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_sync_fingerprint_includes_ignored_workspace_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            (fixture.source / ".gitignore").write_text(
                "controller-generated/\n",
                encoding="utf-8",
            )
            git(fixture.source, "add", ".gitignore")
            git(fixture.source, "commit", "--quiet", "-m", "freeze ignore rule")
            fixture.context = replace(
                fixture.context,
                frozen_source_revision=git(
                    fixture.source,
                    "rev-parse",
                    "HEAD",
                ).stdout.decode().strip(),
            )
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            workspace = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "workspace"
            )
            ignored = Path(workspace.raw_handle) / "controller-generated" / "asset.bin"
            ignored.parent.mkdir()
            ignored.write_bytes(b"must-sync")
            backend.run(lease, ("python", "-V"), ".", 1_000)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(len(sync_commands), 2)
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_sync_fingerprint_frames_untracked_paths_and_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            workspace = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "workspace"
            )
            first = Path(workspace.raw_handle) / "src" / "a"
            first.write_bytes(b"bc")
            backend.run(lease, ("python", "-V"), ".", 1_000)
            first.unlink()
            (Path(workspace.raw_handle) / "src" / "ab").write_bytes(b"c")
            backend.run(lease, ("python", "-V"), ".", 1_000)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(len(sync_commands), 3)
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_sync_fingerprint_includes_untracked_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            workspace = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "workspace"
            )
            executable = Path(workspace.raw_handle) / "src" / "tool.sh"
            executable.write_bytes(b"#!/bin/sh\nexit 0\n")
            executable.chmod(0o644)
            backend.run(lease, ("python", "-V"), ".", 1_000)
            executable.chmod(0o755)
            backend.run(lease, ("python", "-V"), ".", 1_000)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(len(sync_commands), 3)
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_sync_fingerprint_includes_directory_and_tracked_file_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            workspace = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "workspace"
            )
            workspace_path = Path(workspace.raw_handle)
            empty = workspace_path / "controller-empty"
            empty.mkdir(mode=0o755)
            backend.run(lease, ("python", "-V"), ".", 1_000)
            empty.chmod(0o700)
            backend.run(lease, ("python", "-V"), ".", 1_000)
            tracked = workspace_path / "src" / "operator.py"
            tracked.chmod(0o600)
            backend.run(lease, ("python", "-V"), ".", 1_000)
            workspace_path.chmod(0o700)
            backend.run(lease, ("python", "-V"), ".", 1_000)

            sync_commands = [
                command
                for command, _, _ in runner.calls
                if command[0] == "rsync-fixture"
            ]
            self.assertEqual(len(sync_commands), 5)
            self.assertTrue(backend.cleanup(lease).report.all_released)

    def test_remote_inplace_build_copies_exact_ccache_seed_into_attempt_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = replace(
                self.profile("remote_docker", gpu=True),
                source_loading_mode="inplace_build",
            )
            seed = "/srv/opbench/cache/ccache/frozen-cuda-environment"
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
                remote_ccache_seed=seed,
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = RecordingArgvRunner()
            backend = RemoteDockerRuntimeBackend(argv_runner=runner)

            lease = backend.prepare(profile, fixture.context)
            backend.run(lease, ("python", "-V"), ".", 1_000)
            cleanup = backend.cleanup(lease)

            remote = next(
                handle
                for handle in lease.handles
                if handle.resource_type == "remote_workspace"
            )
            remote_path = remote.raw_handle.split(":", 1)[1]
            commands = [call[0] for call in runner.calls]
            seed_copy = next(
                command
                for command in commands
                if command[0] == "ssh-fixture" and seed in command[-1]
            )
            self.assertIn("cp -a --reflink=auto --", seed_copy[-1])
            self.assertIn(f"{remote_path}/.ccache", seed_copy[-1])
            initial_sync = next(command for command in commands if command[0] == "rsync-fixture")
            self.assertIn("--exclude=/.ccache/", initial_sync)
            self.assertNotIn("--exclude=.ccache/", initial_sync)
            remote_execute = next(
                command
                for command in commands
                if command[0] == "ssh-fixture"
                and "docker-fixture exec" in command[-1]
            )
            self.assertIn("CCACHE_DIR=/workspace/.ccache", remote_execute[-1])
            self.assertNotIn("CCACHE_DIR=/tmp/op_bench_runtime/ccache", remote_execute[-1])
            self.assertIn("PYTHONPATH=/workspace", remote_execute[-1])
            self.assertTrue(cleanup.report.all_released)
            cleanup_commands = "\n".join(command[-1] for command in commands[-2:])
            self.assertNotIn(seed, cleanup_commands)

    def test_remote_inplace_seed_refuses_a_frozen_root_ccache_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = replace(
                self.profile("remote_docker", gpu=True),
                source_loading_mode="inplace_build",
            )
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
                remote_ccache_seed="/srv/opbench/cache/ccache/frozen-cuda-environment",
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            (fixture.source / ".ccache").mkdir()
            (fixture.source / ".ccache" / "tracked.bin").write_bytes(b"source-owned")
            git(fixture.source, "add", ".ccache/tracked.bin")
            git(fixture.source, "commit", "--quiet", "-m", "track root ccache")
            fixture.context = replace(
                fixture.context,
                frozen_source_revision=git(
                    fixture.source,
                    "rev-parse",
                    "HEAD",
                ).stdout.decode().strip(),
            )
            runner = RecordingArgvRunner()

            with self.assertRaisesRegex(
                RuntimeBackendUnavailable,
                "workspace_ccache_seed_collision",
            ):
                RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                    profile,
                    fixture.context,
                )

            self.assertEqual(runner.calls, [])
            self.assertEqual(list(workspaces.iterdir()), [])

    def test_remote_prepare_refuses_to_claim_an_existing_workspace_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "id_fixture"
            identity_file.write_text("fixture", encoding="utf-8")
            profile = replace(
                self.profile("remote_docker", gpu=True),
                source_loading_mode="inplace_build",
            )
            seed = "/srv/opbench/cache/ccache/frozen-cuda-environment"
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
                host_alias="gpu-exact-fixture",
                remote_user="runner",
                identity_file=identity_file,
                docker_binary="docker-fixture",
                ssh_binary="ssh-fixture",
                rsync_binary="rsync-fixture",
                remote_ccache_seed=seed,
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            runner = ExistingRemoteWorkspaceArgvRunner()

            with self.assertRaisesRegex(
                RuntimeBackendUnavailable,
                "remote_workspace_create_failed",
            ):
                RemoteDockerRuntimeBackend(argv_runner=runner).prepare(
                    profile,
                    fixture.context,
                )

            commands = [call[0] for call in runner.calls]
            self.assertFalse(any(command[0] == "rsync-fixture" for command in commands))
            flattened = "\n".join(command[-1] for command in commands)
            self.assertNotIn(seed, flattened)
            self.assertNotIn("docker-fixture", flattened)
            self.assertNotIn("rm -rf", flattened)
            self.assertEqual(list(workspaces.iterdir()), [])
            self.assertEqual(
                [
                    (record.resource_type, record.transition)
                    for record in fixture.ledger.records
                ],
                [
                    ("workspace", "declared"),
                    ("workspace", "created"),
                    ("remote_workspace", "declared"),
                    ("remote_workspace", "create_failed"),
                    ("workspace", "released"),
                ],
            )

    def test_remote_target_is_required_before_any_resource_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            profile = self.profile("remote_docker")
            binding = RuntimeTargetBinding(
                backend="remote_docker",
                local_workspace_parent=workspaces,
            )
            fixture = LocalBackendFixture(root, profile=profile, binding=binding)
            backend = RemoteDockerRuntimeBackend(argv_runner=RecordingArgvRunner())

            with self.assertRaisesRegex(ContractError, "explicit remote target"):
                backend.prepare(profile, fixture.context)

            self.assertEqual(fixture.ledger.records, ())

    def test_private_single_host_config_loads_exactly_without_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspaces = root / "workspaces"
            workspaces.mkdir()
            identity_file = root / "identity"
            identity_file.write_text("fixture", encoding="utf-8")
            target = root / "target.json"
            target.write_text(
                json.dumps(
                    {
                        "hosts": {
                            "only-explicit-target": {
                                "hostname": "exact.example.invalid",
                                "user": "runner",
                                "port": 2222,
                            "identity_file": str(identity_file),
                            "remote_workspace_root": "/srv/opbench/exact",
                            "remote_ccache_seed": "/srv/opbench/cache/ccache/frozen",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            binding = load_runtime_target_binding(
                target,
                local_workspace_parent=workspaces,
            )

            self.assertEqual(binding.backend, "remote_docker")
            self.assertEqual(binding.host_alias, "exact.example.invalid")
            self.assertEqual(binding.remote_workspace_root, "/srv/opbench/exact")
            self.assertEqual(
                binding.remote_ccache_seed,
                "/srv/opbench/cache/ccache/frozen",
            )


if __name__ == "__main__":
    unittest.main()
