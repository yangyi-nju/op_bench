from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import venv

from op_bench.runtime.actions import (
    CanonicalActionService,
    CommandExecution,
    RegisteredTest,
)
from op_bench.runtime.contracts import ActionRequest
from op_bench.runtime.orchestrator import _registered_tests
from op_bench.runtime.source_loading import (
    RuntimeSourcePreparation,
    build_runtime_source_preparation,
)
from op_bench.runtime.workspace import AuthoritativeWorkspace
from tests.runtime_git_fixture import initialize_git_repo
from tests.runtime_orchestrator_fixture import (
    build_orchestrator_fixture,
    request_for,
)
from tests.test_runtime_contracts import (
    SHA_A,
    budget_policy,
    capability_policy,
    identity,
    runtime_profile,
)
from tests.test_runtime_workspace import policy as workspace_policy


class RecordingBackend:
    def __init__(
        self,
        *,
        preparation_exit_code: int = 0,
        preparation_timed_out: bool = False,
    ) -> None:
        self.calls: list[tuple[tuple[str, ...], str, int]] = []
        self.preparation_exit_code = preparation_exit_code
        self.preparation_timed_out = preparation_timed_out

    def run(
        self,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> CommandExecution:
        self.calls.append((command, cwd, timeout_ms))
        preparing = command == ("python", "-I", "-c", "prepare-source")
        return CommandExecution(
            command=command,
            cwd=cwd,
            exit_code=self.preparation_exit_code if preparing else 0,
            stdout="prepared\n" if preparing else "selected\n",
            stderr="build failed\n" if preparing and self.preparation_exit_code else "",
            duration_ms=7,
            timed_out=self.preparation_timed_out if preparing else False,
        )


class SubprocessSourceBackend:
    def __init__(self, root: Path, source_loading_mode: str) -> None:
        self.root = root
        self.source_loading_mode = source_loading_mode
        self.calls: list[tuple[tuple[str, ...], str, int]] = []

    def run(
        self,
        command: tuple[str, ...],
        cwd: str,
        timeout_ms: int,
    ) -> CommandExecution:
        self.calls.append((command, cwd, timeout_ms))
        environment = dict(os.environ)
        if self.source_loading_mode == "python_overlay":
            environment["PYTHONPATH"] = "/tmp/op_bench_runtime/site-packages"
        elif self.source_loading_mode == "inplace_build":
            environment["PYTHONPATH"] = str(self.root)
        started = __import__("time").monotonic()
        try:
            result = subprocess.run(
                command,
                cwd=self.root / cwd,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_ms / 1_000,
            )
            timed_out = False
            exit_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = 124
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
        return CommandExecution(
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration_ms=max(
                0,
                round((__import__("time").monotonic() - started) * 1_000),
            ),
            timed_out=timed_out,
        )


class RuntimeAgentSourceLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        initialize_git_repo(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def service(
        self,
        backend: RecordingBackend,
        *,
        max_commands: int = 10,
    ) -> CanonicalActionService:
        return CanonicalActionService(
            session_id="source-loading-session",
            workspace=AuthoritativeWorkspace.open(
                self.root,
                source=identity("source", "fixture@source-loading", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("command_run", "test_run"),
                allowed_command_prefixes=("git diff",),
                registered_tests=("public::source",),
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=10_000,
                max_actions=10,
                max_tests=10,
                max_commands=max_commands,
                max_output_bytes=10_000,
            ),
            command_backend=backend,
            test_registry={
                "public::source": RegisteredTest(
                    selector_id="public::source",
                    command=("python", "-m", "unittest", "tests.test_source"),
                    cwd=".",
                    timeout_ms=500,
                    preparation=RuntimeSourcePreparation(
                        command=("python", "-I", "-c", "prepare-source"),
                        cwd=".",
                        timeout_ms=600,
                    ),
                )
            },
            clock_ms=lambda: 1_000,
        )

    def request(
        self,
        action_id: str,
        action_name: str,
        arguments: dict[str, object],
        sequence: int,
    ) -> ActionRequest:
        return ActionRequest(
            session_id="source-loading-session",
            action_id=action_id,
            action_name=action_name,
            arguments=arguments,
            client_sequence=sequence,
            deadline_ms=2_000,
        )

    def test_registered_test_prepares_source_before_selector(self) -> None:
        backend = RecordingBackend()
        service = self.service(backend)

        observation = service.execute(
            self.request(
                "test-source",
                "test_run",
                {"selector_id": "public::source"},
                1,
            )
        )

        self.assertTrue(observation.ok)
        self.assertEqual(
            [call[0] for call in backend.calls],
            [
                ("python", "-I", "-c", "prepare-source"),
                ("python", "-m", "unittest", "tests.test_source"),
            ],
        )
        self.assertEqual([call[2] for call in backend.calls], [600, 500])
        self.assertEqual(observation.budget_delta.tests, 1)
        self.assertEqual(observation.budget_delta.commands, 2)

    def test_failed_source_preparation_skips_selector(self) -> None:
        backend = RecordingBackend(preparation_exit_code=2)
        service = self.service(backend)

        observation = service.execute(
            self.request(
                "test-source-failed",
                "test_run",
                {"selector_id": "public::source"},
                1,
            )
        )

        self.assertFalse(observation.ok)
        self.assertEqual(observation.error_code, "runtime_error")
        self.assertEqual(observation.message, "test source preparation failed")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(observation.data["phase"], "source_preparation")
        self.assertEqual(observation.data["exit_code"], 2)
        self.assertEqual(observation.budget_delta.tests, 1)
        self.assertEqual(observation.budget_delta.commands, 1)

    def test_timed_out_source_preparation_skips_selector(self) -> None:
        backend = RecordingBackend(preparation_timed_out=True)
        service = self.service(backend)

        observation = service.execute(
            self.request(
                "test-source-timeout",
                "test_run",
                {"selector_id": "public::source"},
                1,
            )
        )

        self.assertFalse(observation.ok)
        self.assertEqual(observation.error_code, "timeout")
        self.assertEqual(observation.message, "test source preparation timed out")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(observation.data["phase"], "source_preparation")
        self.assertEqual(observation.budget_delta.tests, 1)
        self.assertEqual(observation.budget_delta.commands, 1)

    def test_command_run_does_not_prepare_source(self) -> None:
        backend = RecordingBackend()
        service = self.service(backend)

        observation = service.execute(
            self.request(
                "command-only",
                "command_run",
                {"command": ["git", "diff"]},
                1,
            )
        )

        self.assertTrue(observation.ok)
        self.assertEqual([call[0] for call in backend.calls], [("git", "diff")])
        self.assertEqual(observation.budget_delta.tests, 0)
        self.assertEqual(observation.budget_delta.commands, 1)

    def test_python_overlay_selector_observes_authoritative_workspace_source(
        self,
    ) -> None:
        package = "opbench_source_fixture"
        overlay_package = (
            Path("/tmp/op_bench_runtime/site-packages") / package
        )
        self.assertFalse(overlay_package.exists())
        venv_root = Path(self.temporary.name) / "venv"
        venv.EnvBuilder(with_pip=False).create(venv_root)
        venv_python = venv_root / "bin" / "python"
        purelib = Path(
            subprocess.run(
                (
                    str(venv_python),
                    "-I",
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()
        )
        installed = purelib / package
        installed.mkdir()
        (installed / "__init__.py").write_text(
            'VALUE = "installed"\n',
            encoding="utf-8",
        )
        authoritative = self.root / package
        authoritative.mkdir()
        (authoritative / "__init__.py").write_text(
            'VALUE = "authoritative"\n',
            encoding="utf-8",
        )
        from tests.runtime_git_fixture import git

        git(self.root, "add", f"{package}/__init__.py")
        git(self.root, "commit", "--quiet", "-m", "add overlay fixture")
        profile = replace(
            runtime_profile(),
            source_loading_mode="python_overlay",
            timeout_ms=5_000,
        )
        preparation = build_runtime_source_preparation(
            profile,
            str(venv_python),
            (f"{package}/__init__.py",),
        )
        self.assertIsNotNone(preparation)
        backend = SubprocessSourceBackend(self.root, "python_overlay")
        service = self._source_service(
            backend,
            RegisteredTest(
                selector_id="public::source",
                command=(
                    str(venv_python),
                    "-c",
                    f"import {package}; print({package}.VALUE)",
                ),
                cwd=".",
                timeout_ms=5_000,
                preparation=preparation,
            ),
        )
        try:
            observation = service.execute(
                self.request(
                    "overlay-source",
                    "test_run",
                    {"selector_id": "public::source"},
                    1,
                )
            )
        finally:
            if overlay_package.is_dir() and not overlay_package.is_symlink():
                shutil.rmtree(overlay_package)

        self.assertTrue(observation.ok, observation.to_dict())
        self.assertEqual(observation.data["stdout"], "authoritative\n")
        self.assertEqual(len(backend.calls), 2)

    def test_inplace_selector_runs_build_before_import(self) -> None:
        package = "opbench_inplace_fixture"
        authoritative = self.root / package
        authoritative.mkdir()
        (authoritative / "__init__.py").write_text(
            'VALUE = "before-build"\n',
            encoding="utf-8",
        )
        (self.root / "setup.py").write_text(
            (
                "import pathlib,sys\n"
                "if sys.argv[1:] != ['build_ext', '--inplace']:\n"
                "    raise SystemExit(2)\n"
                f"pathlib.Path('{package}/__init__.py').write_text("
                "'VALUE = \"after-build\"\\n', encoding='utf-8')\n"
            ),
            encoding="utf-8",
        )
        from tests.runtime_git_fixture import git

        git(self.root, "add", "setup.py", f"{package}/__init__.py")
        git(self.root, "commit", "--quiet", "-m", "add inplace fixture")
        profile = replace(
            runtime_profile(),
            source_loading_mode="inplace_build",
            timeout_ms=5_000,
        )
        preparation = build_runtime_source_preparation(
            profile,
            sys.executable,
            (),
        )
        self.assertIsNotNone(preparation)
        backend = SubprocessSourceBackend(self.root, "inplace_build")
        service = self._source_service(
            backend,
            RegisteredTest(
                selector_id="public::source",
                command=(
                    sys.executable,
                    "-c",
                    f"import {package}; print({package}.VALUE)",
                ),
                cwd=".",
                timeout_ms=5_000,
                preparation=preparation,
            ),
        )

        observation = service.execute(
            self.request(
                "inplace-source",
                "test_run",
                {"selector_id": "public::source"},
                1,
            )
        )

        self.assertTrue(observation.ok, observation.to_dict())
        self.assertEqual(observation.data["stdout"], "after-build\n")
        self.assertEqual([call[0][0] for call in backend.calls], ["bash", sys.executable])

    def test_source_preparation_builder_is_strict_and_deterministic(self) -> None:
        profile = replace(
            runtime_profile(),
            source_loading_mode="python_overlay",
            timeout_ms=12_345,
        )

        preparation = build_runtime_source_preparation(
            profile,
            "python-fixture",
            (
                "torch/nn/modules/module.py",
                "torch/testing/_internal/common_nn.py",
            ),
        )

        self.assertIsNotNone(preparation)
        assert preparation is not None
        self.assertEqual(preparation.command[:3], ("python-fixture", "-I", "-c"))
        self.assertEqual(
            json.loads(preparation.command[-1]),
            {
                "package": "torch",
                "paths": [
                    "torch/nn/modules/module.py",
                    "torch/testing/_internal/common_nn.py",
                ],
            },
        )
        self.assertEqual(preparation.cwd, ".")
        self.assertEqual(preparation.timeout_ms, 12_345)
        self.assertIsNone(
            build_runtime_source_preparation(
                replace(profile, source_loading_mode="none"),
                "python-fixture",
                (),
            )
        )
        with self.assertRaisesRegex(
            Exception,
            "source_overlay_package_ambiguous",
        ):
            build_runtime_source_preparation(
                profile,
                "python-fixture",
                ("torch/a.py", "torchvision/b.py"),
            )

    def test_orchestrator_registers_only_selected_profile_source_preparation(
        self,
    ) -> None:
        fixture = build_orchestrator_fixture(
            Path(self.temporary.name) / "orchestrator"
        )
        request = request_for(fixture)
        profile = replace(
            fixture.profile,
            source_loading_mode="python_overlay",
        )

        registered = _registered_tests(
            request,
            fixture.manifest.tasks[0],
            "python-fixture",
            runtime_profile=profile,
            source_overlay_paths=("calc/__init__.py",),
        )

        selected = registered[next(iter(registered))]
        self.assertIsNotNone(selected.preparation)
        assert selected.preparation is not None
        self.assertEqual(
            json.loads(selected.preparation.command[-1]),
            {
                "package": "calc",
                "paths": ["calc/__init__.py"],
            },
        )
        without_loading = _registered_tests(
            request,
            fixture.manifest.tasks[0],
            "python-fixture",
            runtime_profile=replace(profile, source_loading_mode="none"),
            source_overlay_paths=(),
        )
        self.assertIsNone(without_loading[next(iter(without_loading))].preparation)

    def _source_service(
        self,
        backend: SubprocessSourceBackend,
        registered: RegisteredTest,
    ) -> CanonicalActionService:
        return CanonicalActionService(
            session_id="source-loading-session",
            workspace=AuthoritativeWorkspace.open(
                self.root,
                source=identity("source", "fixture@source-loading", SHA_A),
                policy=workspace_policy(),
            ),
            capability_policy=replace(
                capability_policy(),
                allowed_actions=("test_run",),
                registered_tests=("public::source",),
            ),
            budget_policy=replace(
                budget_policy(),
                wall_clock_ms=20_000,
                max_actions=10,
                max_tests=10,
                max_commands=10,
                max_output_bytes=10_000,
            ),
            command_backend=backend,
            test_registry={"public::source": registered},
            clock_ms=lambda: 1_000,
        )


if __name__ == "__main__":
    unittest.main()
