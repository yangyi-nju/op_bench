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
from op_bench.runtime.resources import AttemptResourceLedger, RuntimeLeaseStore
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


class ContainerBackendCommandTests(unittest.TestCase):
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
            self.assertEqual(runner.calls[1][0][-2], str(snapshot) + "/")
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
            self.assertIn("mkdir", commands[0][-1])
            self.assertEqual(commands[1][0], "rsync-fixture")
            self.assertEqual(commands[1][-1], remote.raw_handle + "/")
            self.assertEqual(commands[4][0], "rsync-fixture")
            self.assertEqual(commands[4][-1], remote.raw_handle + "/")
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


if __name__ == "__main__":
    unittest.main()
