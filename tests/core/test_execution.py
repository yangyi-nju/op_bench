"""Benign Docker lifecycle faults, including lost creation acknowledgements."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from op_bench.runtime.execution import (
    DockerSession, ExecutionError, create_session, materialize_source,
    run_observation_process, run_process,
)
from op_bench.data.task import TaskSpec
from _support import make_task


class ProcessTransportTests(unittest.TestCase):
    def test_command_merges_logs_and_observation_exchanges_large_input_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = run_process([sys.executable, "-c",
                "import sys; print('output', flush=True); print('diagnostic', file=sys.stderr)"],
                cwd=root, timeout_sec=5, log_path=root / "command.log")
            self.assertEqual(0, command.exit_code)
            self.assertEqual("output\ndiagnostic\n", (root / "command.log").read_text())
            observed = run_observation_process([sys.executable, "-c",
                "import sys; print(len(sys.stdin.buffer.read())); print('diagnostic', file=sys.stderr)"],
                cwd=root, timeout_sec=5, input_bytes=b"x" * 262144,
                output_path=root / "observation.json", log_path=root / "observation.log")
            self.assertEqual(0, observed.exit_code)
            self.assertFalse(observed.timed_out)
            self.assertEqual("262144\n", (root / "observation.json").read_text())
            self.assertEqual("diagnostic\n", (root / "observation.log").read_text())

    def test_both_transports_timeout_and_reap_background_pipe_holders(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for observation in (False, True):
                for background in (False, True):
                    with self.subTest(observation=observation, background=background):
                        command = ([sys.executable, "-c",
                            "import subprocess,sys; subprocess.Popen([sys.executable, '-c', "
                            "'import time; time.sleep(30)']); print('complete', flush=True)"]
                            if background else [sys.executable, "-c", "import time; time.sleep(30)"])
                        kwargs = {"cwd": root, "timeout_sec": 2 if background else .1,
                                  "log_path": root / "command.log"}
                        if observation:
                            result = run_observation_process(command, **kwargs, input_bytes=b"",
                                output_path=root / "observation.json")
                        else:
                            result = run_process(command, **kwargs)
                        self.assertEqual(not background, result.timed_out)
                        if background:
                            self.assertEqual(0, result.exit_code)
                            output = root / ("observation.json" if observation else "command.log")
                            self.assertEqual("complete\n", output.read_text())


class FakeDaemon:
    """Stateful resource operations; failures can follow a completed mutation."""
    def __init__(self, fail=None, after_mutation=True):
        self.containers = {}
        self.volumes = set()
        self.calls = []
        self.fail = fail
        self.after_mutation = after_mutation
        self.failed = False

    def control(self, session, argv, timeout=60):
        self.calls.append(list(argv))
        trigger = not self.failed and self.fail is not None and self.fail(session, argv)
        if trigger:
            self.failed = True
            if not self.after_mutation:
                raise ExecutionError("fixture CLI timeout before acknowledgement")
        stdout = ""
        if argv[:2] == ["image", "inspect"]:
            stdout = json.dumps([{"Id": "sha256:fixture", "Os": "linux", "Architecture": "arm64"}])
        elif argv[0] == "info":
            stdout = json.dumps({"OSType": "linux", "Architecture": "aarch64"})
        elif argv[:2] == ["volume", "create"]:
            self.volumes.add(argv[-1])
        elif argv[:2] == ["volume", "rm"]:
            if argv[-1] not in self.volumes:
                raise ExecutionError("No such volume: " + argv[-1])
            self.volumes.remove(argv[-1])
        elif argv[0] == "run":
            self.containers[argv[argv.index("--name") + 1]] = True
        elif argv[0] == "inspect":
            if argv[-1] not in self.containers:
                raise ExecutionError("No such object: " + argv[-1])
            stdout = ("true" if self.containers[argv[-1]] else "false") if "--format" in argv else json.dumps([
                {"HostConfig": {"NetworkMode": "none"}, "Mounts": []}])
        elif argv[0] == "kill":
            self.containers[argv[-1]] = False
        elif argv[0] == "rm":
            if argv[-1] not in self.containers:
                raise ExecutionError("No such container: " + argv[-1])
            del self.containers[argv[-1]]
        if trigger:
            raise ExecutionError("fixture CLI timeout after daemon operation")
        return subprocess.CompletedProcess(argv, 0, stdout, "")


class ExecutionLifecycleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="opbench-lifecycle-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        task = TaskSpec.load(make_task(self.root / "task"))
        self.task = replace(task, environment=replace(task.environment, backend="docker", image="fixture", python="python3"))
        self.workspace = self.root / "workspace"
        materialize_source(self.task, self.workspace)

    def test_lost_volume_helper_and_main_creation_responses_clean_known_names(self):
        faults = (
            lambda session, argv: argv[:2] == ["volume", "create"],
            lambda session, argv: argv[:2] == ["volume", "create"] and argv[-1] == session.grader_volume,
            lambda session, argv: argv[0] == "run" and argv[argv.index("--name") + 1] == session.transfer_container,
            lambda session, argv: argv[0] == "run" and argv[argv.index("--name") + 1] == session.container,
        )
        for fault in faults:
            for after_mutation in (False, True):
                with self.subTest(fault=fault, after_mutation=after_mutation):
                    daemon = FakeDaemon(fault, after_mutation)
                    def control(session, argv, timeout=60):
                        return daemon.control(session, argv, timeout)
                    with patch.object(DockerSession, "_control", control), self.assertRaises(ExecutionError) as raised:
                        with create_session(self.task, self.workspace, self.task.grader_dir):
                            self.fail("Faulted setup must not expose an execution session")
                    self.assertEqual(set(), daemon.volumes)
                    self.assertEqual({}, daemon.containers)
                    self.assertTrue(raised.exception.execution_session_info["cleanup_complete"])
                    self.assertEqual([], raised.exception.execution_session_info["cleanup_errors"])

    def test_partial_second_upload_invalidates_prior_workspace_readiness(self):
        session = DockerSession(self.task, self.workspace, None)
        daemon = FakeDaemon()
        with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)):
            session.start()
            self.assertTrue(session._workspace_uploaded)
            daemon.fail = lambda current, argv: argv[0] == "cp" and ":" not in argv[1]
            with self.assertRaises(ExecutionError):
                session.sync_to_container()
            self.assertFalse(session._workspace_uploaded)
            with patch.object(session, "sync_from_container") as download:
                session.close()
                download.assert_not_called()
        self.assertFalse(session.workspace_capture_ready)
        self.assertTrue(session.cleanup_complete)

    def test_initializer_removal_failure_prevents_main_start_and_capture(self):
        session = DockerSession(self.task, self.workspace, None)
        daemon = FakeDaemon()
        def control(argv, timeout=60):
            if argv[:2] == ["rm", "--force"] and argv[-1] == session.transfer_container:
                raise ExecutionError("fixture helper termination unavailable")
            return daemon.control(session, argv, timeout)
        with patch.object(session, "_control", control):
            with self.assertRaisesRegex(ExecutionError, "transfer container removal"):
                session.start()
            self.assertNotIn(session.container, daemon.containers)
            self.assertFalse(session._workspace_uploaded)
            with patch.object(session, "sync_from_container") as download:
                session.close()
                download.assert_not_called()
        self.assertFalse(session.workspace_capture_ready)
        self.assertFalse(session.cleanup_complete)
        with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)):
            session.close()
        self.assertTrue(session.cleanup_complete)

    def test_residual_initializer_is_removed_before_freezing_main_workspace(self):
        session = DockerSession(self.task, self.workspace, None)
        session._container_created = session._transfer_created = session._workspace_uploaded = True
        daemon = FakeDaemon()
        daemon.containers = {session.container: True, session.transfer_container: True}
        def download():
            self.assertNotIn(session.transfer_container, daemon.containers)
            self.assertFalse(daemon.containers[session.container])
        with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)), \
             patch.object(session, "sync_from_container", side_effect=download):
            session.close()
        self.assertTrue(session.workspace_capture_ready)
        self.assertTrue(session.cleanup_complete)

    def test_stop_failure_never_freezes_a_potentially_active_workspace(self):
        session = DockerSession(self.task, self.workspace, None)
        session._container_created = session._workspace_uploaded = True
        daemon = FakeDaemon(lambda current, argv: argv[0] == "kill", after_mutation=False)
        daemon.containers[session.container] = True
        with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)), \
             patch.object(session, "sync_from_container") as download:
            session.close()
            download.assert_not_called()
        self.assertFalse(session.workspace_capture_ready)
        self.assertTrue(session.execution_stopped)  # Final force-removal succeeded.
        self.assertTrue(session.cleanup_complete)
        self.assertTrue(any("Stop execution" in error for error in session.cleanup_errors))

    def test_unconfirmed_main_and_helper_termination_leave_capture_unavailable(self):
        for remaining in ("main", "helper"):
            with self.subTest(remaining=remaining):
                session = DockerSession(self.task, self.workspace, None)
                session._container_created = session._transfer_created = session._workspace_uploaded = True
                daemon = FakeDaemon()
                daemon.containers = {session.container: True, session.transfer_container: True}
                blocked_name = session.container if remaining == "main" else session.transfer_container
                def control(argv, timeout=60):
                    if argv[0] in {"kill", "rm"} and argv[-1] == blocked_name:
                        raise ExecutionError("fixture termination unavailable")
                    return daemon.control(session, argv, timeout)
                with patch.object(session, "_control", control), patch.object(session, "sync_from_container") as download:
                    session.close()
                    download.assert_not_called()
                self.assertTrue(daemon.containers[blocked_name])
                self.assertFalse(session.execution_stopped)
                self.assertFalse(session.workspace_capture_ready)
                self.assertFalse(session.cleanup_complete)
                with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)):
                    session.close()
                self.assertTrue(session.cleanup_complete)

    def test_failed_final_download_keeps_previous_host_workspace_and_unready_state(self):
        session = DockerSession(self.task, self.workspace, None)
        session._container_created = session._workspace_uploaded = True
        marker = self.workspace / "previous-good-download"
        marker.write_text("retained")
        daemon = FakeDaemon()
        daemon.containers[session.container] = True
        def control(argv, timeout=60):
            if argv[0] == "cp" and argv[1].startswith(session.container + ":"):
                (Path(argv[-1]) / "partial").write_text("incomplete transfer")
                raise ExecutionError("fixture download interrupted")
            return daemon.control(session, argv, timeout)
        with patch.object(session, "_control", control):
            session.close()
        self.assertEqual("retained", marker.read_text())
        self.assertFalse((self.workspace / "partial").exists())
        self.assertFalse(session.workspace_capture_ready)
        self.assertTrue(session.cleanup_complete)

    def test_observation_only_session_stops_and_releases_resources_without_download(self):
        session = DockerSession(self.task, self.workspace, None, capture_workspace=False)
        daemon = FakeDaemon()
        with patch.object(session, "_control", lambda argv, timeout=60: daemon.control(session, argv, timeout)):
            session.start()
            with patch.object(session, "sync_from_container") as download:
                session.close()
                download.assert_not_called()
        self.assertTrue(session.execution_stopped)
        self.assertTrue(session.cleanup_complete)
        self.assertFalse(session.workspace_capture_ready)
        self.assertFalse(session.info["workspace_capture_requested"])
        self.assertEqual({}, daemon.containers)
        self.assertEqual(set(), daemon.volumes)

    @unittest.skipUnless(os.environ.get("OPBENCH_TEST_DOCKER_IMAGE"), "Set OPBENCH_TEST_DOCKER_IMAGE for Docker integration")
    def test_real_created_volume_and_container_are_cleaned_after_lost_cli_response(self):
        task = replace(self.task, environment=replace(self.task.environment,
            image=os.environ["OPBENCH_TEST_DOCKER_IMAGE"]))
        original = DockerSession._control
        for operation in ("volume", "container"):
            state = {"failed": False}
            def control(session, argv, timeout=60):
                result = original(session, argv, timeout)
                selected = (argv[:2] == ["volume", "create"]) if operation == "volume" else (
                    argv[0] == "run" and argv[argv.index("--name") + 1] == session.container)
                if selected and not state["failed"]:
                    state["failed"] = True
                    raise ExecutionError("fixture discarded successful Docker acknowledgement")
                return result
            with patch.object(DockerSession, "_control", control), self.assertRaises(ExecutionError) as raised:
                with create_session(task, self.workspace):
                    self.fail("Lost creation acknowledgement must fail session setup")
            info = raised.exception.execution_session_info
            self.assertTrue(info["cleanup_complete"], info)
            self.assertEqual([], info["cleanup_errors"])
            for command in (["docker", "ps", "-a", "--filter", "name=" + info["session_id"], "--format", "{{.Names}}"],
                            ["docker", "volume", "ls", "--filter", "label=opbench.session=" + info["session_id"], "--format", "{{.Name}}"]):
                result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=20)
                self.assertEqual("", result.stdout.strip())
