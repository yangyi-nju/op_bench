"""Tests for op_bench.remote module."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from op_bench.remote import RemoteDockerExecutor, RemoteHost, load_hosts_config


class RemoteHostTests(unittest.TestCase):
    def test_ssh_command_prefix_default_port(self):
        host = RemoteHost(user="ubuntu", hostname="10.0.0.42")
        cmd = host.ssh_command_prefix()
        self.assertNotIn("-p", cmd)
        self.assertEqual(cmd[-1], "ubuntu@10.0.0.42")
        self.assertIn("-o", cmd)
        self.assertIn("StrictHostKeyChecking=accept-new", cmd)

    def test_ssh_command_prefix_custom_port_and_key(self):
        host = RemoteHost(
            user="ec2-user",
            hostname="example.com",
            port=2222,
            identity_file="~/.ssh/key.pem",
        )
        cmd = host.ssh_command_prefix()
        self.assertIn("-p", cmd)
        self.assertIn("2222", cmd)
        self.assertIn("-i", cmd)
        # path should be expanded
        idx = cmd.index("-i")
        self.assertNotIn("~", cmd[idx + 1])
        self.assertEqual(cmd[-1], "ec2-user@example.com")

    def test_from_dict(self):
        data = {
            "user": "ubuntu",
            "hostname": "10.0.0.42",
            "port": 22,
            "identity_file": "~/.ssh/gpu_key",
            "remote_workspace_root": "/data/op_bench",
        }
        host = RemoteHost.from_dict(data)
        self.assertEqual(host.user, "ubuntu")
        self.assertEqual(host.hostname, "10.0.0.42")
        self.assertEqual(host.remote_workspace_root, "/data/op_bench")

    def test_rsync_remote_path(self):
        host = RemoteHost(user="ubuntu", hostname="10.0.0.42")
        self.assertEqual(host.rsync_remote_path("/foo"), "ubuntu@10.0.0.42:/foo")


class LoadHostsConfigTests(unittest.TestCase):
    def test_load_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hosts.json"
            path.write_text(json.dumps({
                "hosts": {
                    "gpu-a100": {
                        "user": "ubuntu",
                        "hostname": "10.0.0.42",
                    },
                    "gpu-h100": {
                        "user": "root",
                        "hostname": "h100.example.com",
                        "port": 2222,
                    },
                }
            }))
            hosts = load_hosts_config(path)
            self.assertIn("gpu-a100", hosts)
            self.assertIn("gpu-h100", hosts)
            self.assertEqual(hosts["gpu-h100"].port, 2222)

    def test_load_missing_file_returns_empty(self):
        hosts = load_hosts_config(Path("/nonexistent/path/hosts.json"))
        self.assertEqual(hosts, {})

    def test_load_no_path_no_env(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(load_hosts_config(), {})


class RemoteDockerExecutorTests(unittest.TestCase):
    def setUp(self):
        self.host = RemoteHost(user="ubuntu", hostname="10.0.0.42")
        self.executor = RemoteDockerExecutor(
            host=self.host,
            image="op-bench/pytorch-cuda:torch2.6.0-cu124-py311",
            workspace_dir="/workspace",
            container_name="op-bench-test-abc123",
            gpus="all",
            labels={"op-bench.managed": "true"},
        )

    def test_remote_workspace_uses_container_name(self):
        self.assertEqual(
            self.executor.remote_workspace,
            "/tmp/op_bench_workspaces/op-bench-test-abc123",
        )

    def test_remote_workspace_explicit_override(self):
        executor = RemoteDockerExecutor(
            host=self.host,
            image="x",
            container_name="c",
            remote_workspace="/custom/path",
        )
        self.assertEqual(executor.remote_workspace, "/custom/path")

    def test_command_for_start_includes_gpus(self):
        cmd = self.executor.command_for_start()
        # ssh prefix
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("ubuntu@10.0.0.42", cmd)
        remote_command = cmd[-1]
        # docker portion
        self.assertIn("docker run --detach", remote_command)
        self.assertIn("--gpus all", remote_command)
        # labels
        self.assertIn("--label op-bench.managed=true", remote_command)
        # volume points to remote workspace
        self.assertIn(
            "/tmp/op_bench_workspaces/op-bench-test-abc123:/workspace",
            remote_command,
        )
        # image and tail at the end
        self.assertIn("op-bench/pytorch-cuda:torch2.6.0-cu124-py311", remote_command)
        self.assertTrue(remote_command.endswith("tail -f /dev/null"))

    def test_command_for_run_uses_docker_exec(self):
        cmd = self.executor.command_for_run(["python", "test/foo.py"])
        self.assertEqual(cmd[0], "ssh")
        self.assertIn("docker exec", cmd[-1])
        self.assertIn("op-bench-test-abc123", cmd[-1])
        self.assertIn("python test/foo.py", cmd[-1])

    def test_command_for_run_quotes_shell_sensitive_arguments(self):
        cmd = self.executor.command_for_run([
            "python",
            "-c",
            "import torch; print(torch.__version__)",
        ])

        self.assertIn("python -c 'import torch; print(torch.__version__)'", cmd[-1])

    def test_command_for_start_without_gpus(self):
        executor = RemoteDockerExecutor(
            host=self.host,
            image="op-bench/cpu:latest",
            container_name="c",
            gpus=None,
        )
        cmd = executor.command_for_start()
        self.assertNotIn("--gpus", cmd[-1])

    def test_command_for_start_mounts_persistent_ccache(self):
        executor = RemoteDockerExecutor(
            host=self.host,
            image="op-bench/cuda-devel:latest",
            workspace_dir="/workspace",
            container_name="c",
            persistent_ccache_key="pytorch-cuda-devel",
        )

        remote_command = executor.command_for_start()[-1]

        self.assertIn(
            "--volume /tmp/op_bench_workspaces/_cache/ccache/pytorch-cuda-devel:/workspace/.ccache",
            remote_command,
        )
        self.assertEqual(
            executor.remote_ccache_dir,
            "/tmp/op_bench_workspaces/_cache/ccache/pytorch-cuda-devel",
        )

    def test_persistent_ccache_key_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            RemoteDockerExecutor(
                host=self.host,
                image="op-bench/cuda-devel:latest",
                persistent_ccache_key="../shared",
            )

    def test_collect_environment_records_remote_host(self):
        evidence = self.executor.collect_environment()
        self.assertEqual(evidence["executor"], "remote_docker")
        self.assertEqual(evidence["remote_host"], "10.0.0.42")
        self.assertEqual(evidence["remote_user"], "ubuntu")
        self.assertEqual(evidence["gpus"], "all")

    @mock.patch("op_bench.remote.subprocess.run")
    def test_sync_to_remote_runs_mkdir_then_rsync(self, mock_run):
        # Both calls return success
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        with tempfile.TemporaryDirectory() as tmp:
            result = self.executor.sync_to_remote(Path(tmp), timeout_sec=60)
        self.assertEqual(result.exit_code, 0)
        # Verify mkdir was called first, then rsync
        self.assertEqual(mock_run.call_count, 2)
        first_call_cmd = mock_run.call_args_list[0][0][0]
        second_call_cmd = mock_run.call_args_list[1][0][0]
        self.assertIn("mkdir -p", first_call_cmd[-1])
        self.assertEqual(second_call_cmd[0], "rsync")
        self.assertIn("-az", second_call_cmd)
        self.assertIn("--delete", second_call_cmd)

    @mock.patch("op_bench.remote.time.sleep")
    @mock.patch("op_bench.remote.subprocess.run")
    def test_sync_to_remote_retries_partial_transfer(self, mock_run, mock_sleep):
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(returncode=23, stdout="", stderr="vanished source file"),
            mock.Mock(returncode=0, stdout="", stderr=""),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = self.executor.sync_to_remote(Path(tmp), timeout_sec=60)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(mock_run.call_count, 3)
        mock_sleep.assert_called_once_with(2)

    @mock.patch("op_bench.remote.time.sleep")
    @mock.patch("op_bench.remote.subprocess.run")
    def test_sync_to_remote_does_not_retry_nontransient_failure(self, mock_run, mock_sleep):
        mock_run.side_effect = [
            mock.Mock(returncode=0, stdout="", stderr=""),
            mock.Mock(returncode=3, stdout="", stderr="permission denied"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            result = self.executor.sync_to_remote(Path(tmp), timeout_sec=60)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(mock_run.call_count, 2)
        mock_sleep.assert_not_called()

    @mock.patch("op_bench.remote.subprocess.run")
    def test_close_removes_remote_container(self, mock_run):
        mock_run.return_value = mock.Mock(returncode=0, stdout="", stderr="")
        result = self.executor.close()
        self.assertIsNotNone(result)
        # First call: docker rm -f
        first_call_cmd = mock_run.call_args_list[0][0][0]
        self.assertIn("docker rm -f op-bench-test-abc123", first_call_cmd[-1])


if __name__ == "__main__":
    unittest.main()
