"""Execution sessions shared by patch evaluation and optional agent runners.

The local backend is for development only: it is not a security boundary.
Docker configuration is defense in depth, not a claim of verified isolation.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import re
import shutil
import signal
import selectors
import subprocess
import tarfile
import tempfile
import time
from typing import Iterator, Sequence
import uuid

from op_bench.data.task import TaskSpec


_UNITTEST_BOOTSTRAP = r'''
import json, os, pathlib, runpy, sys, unittest

witness = pathlib.Path(sys.argv[1])
script = os.path.abspath(sys.argv[2])
summary = {"schema_version": 1, "runner_invocations": 0, "tests_run": 0,
           "failures": 0, "errors": 0, "skipped": 0,
           "expected_failures": 0, "unexpected_successes": 0}
original_run = unittest.TextTestRunner.run

def record_run(self, test):
    result = original_run(self, test)
    summary["runner_invocations"] += 1
    summary["tests_run"] += result.testsRun
    for output, attribute in (("failures", "failures"), ("errors", "errors"),
                              ("skipped", "skipped"), ("expected_failures", "expectedFailures"),
                              ("unexpected_successes", "unexpectedSuccesses")):
        summary[output] += len(getattr(result, attribute, ()))
    witness.write_text(json.dumps(summary), encoding="utf-8")
    return result

unittest.TextTestRunner.run = record_run
sys.argv = sys.argv[2:]
# Bootstrap imports use isolated mode; the declared script still receives its
# normal script-directory and PYTHONPATH imports after unittest is initialized.
public_paths = []
if "PYTHONPATH" in os.environ:
    public_paths = [os.path.abspath(value or os.curdir)
                    for value in os.environ["PYTHONPATH"].split(os.pathsep)]
sys.path[0:0] = [os.path.dirname(script), *public_paths]
runpy.run_path(script, run_name="__main__")
'''


class ExecutionError(RuntimeError):
    """An execution resource could not be created or used."""


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    timed_out: bool
    duration_sec: float
    output_limited: bool = False

    def to_dict(self) -> dict:
        return {"exit_code": self.exit_code, "timed_out": self.timed_out,
                "duration_sec": self.duration_sec, "output_limited": self.output_limited}


class WitnessCollectionError(ExecutionError):
    """A process ran, but the controller could not collect its test evidence."""

    def __init__(self, message: str, command_result: CommandResult):
        super().__init__(message)
        self.command_result = command_result


def _kill_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # macOS can briefly return EPERM for an already-exiting process group
        # whose leader is not yet waitable. Reap it and confirm the group is
        # gone; never treat a still-running, unsignalable process as stopped.
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            raise error
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_process(argv: Sequence[str], *, cwd: Path, timeout_sec: float,
                 log_path: Path, environment: dict[str, str] | None,
                 max_output_bytes: int, input_bytes: bytes | None = None,
                 output_path: Path | None = None,
                 max_log_bytes: int = 16 * 1024 * 1024) -> CommandResult:
    """One process lifecycle for command logs and bounded observation exchange."""
    for limit in (max_output_bytes, max_log_bytes):
        if type(limit) is not int or limit < 1:
            raise ExecutionError("Process output limits must be positive integers")
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        log = stack.enter_context(log_path.open("wb"))
        output = log
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = stack.enter_context(output_path.open("wb"))
        try:
            process = subprocess.Popen(
                list(argv), cwd=cwd, env=environment,
                stdin=subprocess.DEVNULL if input_bytes is None else subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if output_path is None else subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            log.write(f"Could not start command: {exc}\n".encode())
            raise ExecutionError(str(exc)) from exc
        timed_out = output_limited = False
        sent = 0
        totals = {"stdout": 0, "stderr": 0}
        streams = {"stdout": (output, max_output_bytes), "stderr": (log, max_log_bytes)}
        assert process.stdout is not None
        try:
            with selectors.DefaultSelector() as selector:
                if process.stdin is not None:
                    os.set_blocking(process.stdin.fileno(), False)
                    selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                if process.stderr is not None:
                    selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map() or process.poll() is None:
                    remaining = timeout_sec - (time.monotonic() - started)
                    if remaining <= 0:
                        timed_out = True
                        break
                    # A child can exit while a descendant still holds its pipes.
                    # Reap the whole group before waiting for their final EOF.
                    if process.poll() is not None:
                        _kill_process_group(process)
                    for key, _ in selector.select(min(.05, remaining)):
                        if key.data == "stdin":
                            assert input_bytes is not None and process.stdin is not None
                            try:
                                sent += os.write(key.fd, input_bytes[sent:sent + 65536])
                            except BrokenPipeError:
                                sent = len(input_bytes)
                            except BlockingIOError:
                                continue
                            if sent == len(input_bytes):
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                            continue
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        target, limit = streams[key.data]
                        available = limit - totals[key.data]
                        target.write(chunk[:available])
                        totals[key.data] += min(len(chunk), available)
                        if totals[key.data] >= limit:
                            output_limited = True
                            break
                    if output_limited:
                        break
        finally:
            _kill_process_group(process)
            process.wait()
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
    return CommandResult(process.returncode, timed_out, time.monotonic() - started, output_limited)


def run_process(argv: Sequence[str], *, cwd: Path, timeout_sec: float,
                log_path: Path, environment: dict[str, str] | None = None,
                max_output_bytes: int = 16 * 1024 * 1024) -> CommandResult:
    """Run one argv command with a combined bounded log and reap its descendants."""
    return _run_process(argv, cwd=cwd, timeout_sec=timeout_sec, log_path=log_path,
                        environment=environment, max_output_bytes=max_output_bytes)


def run_observation_process(argv: Sequence[str], *, cwd: Path, timeout_sec: float,
                            input_bytes: bytes, output_path: Path, log_path: Path,
                            environment: dict[str, str] | None = None,
                            max_output_bytes: int = 8 * 1024 * 1024,
                            max_log_bytes: int = 16 * 1024 * 1024) -> CommandResult:
    """Exchange one bounded input; retain stdout separately from diagnostic stderr.

    No private expected values or judge code are supplied to this process.
    """
    return _run_process(argv, cwd=cwd, timeout_sec=timeout_sec, log_path=log_path,
                        environment=environment, max_output_bytes=max_output_bytes,
                        input_bytes=input_bytes, output_path=output_path, max_log_bytes=max_log_bytes)


_PROCESS_WRAPPER = (
    "import os,signal,subprocess,sys\n"
    "p=subprocess.Popen(sys.argv[1:],start_new_session=True)\n"
    "try:\n c=p.wait()\n"
    "finally:\n"
    " try: os.killpg(p.pid,signal.SIGKILL)\n"
    " except ProcessLookupError: pass\n"
    " p.wait()\n"
    "sys.exit(c if c>=0 else 128-c)\n"
)


def resolve_image_id(image: str, *, timeout_sec: float = 60) -> str:
    """Resolve an installed image without pulling or creating an execution session."""
    try:
        inspected = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, check=True, timeout=timeout_sec)
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ExecutionError(f"Cannot resolve task Docker image: {detail}") from exc
    image_id = inspected.stdout.strip()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ExecutionError("Docker image inspect did not return a valid immutable image ID")
    return image_id


def _git(source: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source), *args], text=True, capture_output=True,
            check=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ExecutionError(f"Cannot read source Git checkout: {exc}") from exc
    return result.stdout.strip()


def source_identity(task: TaskSpec) -> dict:
    """Describe the source revision shared by exports and baseline reuse."""
    source = task.source.path.resolve()
    if task.source.revision is not None or (source / ".git").exists():
        if Path(_git(source, "rev-parse", "--show-toplevel")).resolve() != source:
            raise ExecutionError("A Git source.path must point to its repository root")
        revision = _git(source, "rev-parse", "--verify", "--end-of-options",
                        (task.source.revision or "HEAD") + "^{commit}")
        return {"kind": "git", "revision": revision}
    return {"kind": "directory_snapshot", "revision": None, "path": str(source)}


def _export_git(source: Path, destination: Path, commit: str) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        try:
            subprocess.run(
                ["git", "-C", str(source), "archive", "--format=tar", commit],
                stdout=archive, stderr=subprocess.PIPE, check=True, timeout=120,
            )
            archive.seek(0)
            with tarfile.open(fileobj=archive, mode="r:") as tar:
                tar.extractall(destination, filter="data")
            tree = subprocess.run(
                ["git", "-C", str(source), "ls-tree", "-rz", commit],
                capture_output=True, check=True, timeout=60,
            ).stdout
        except (OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
            raise ExecutionError(f"Cannot export source revision: {exc}") from exc
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        details, raw_path = entry.split(b"\t", 1)
        mode, _, object_id = details.split(b" ")
        if mode != b"160000":
            continue
        relative_path = Path(os.fsdecode(raw_path))
        checkout = source / relative_path
        expected = object_id.decode("ascii")
        if not checkout.is_dir() or not (checkout / ".git").exists():
            raise ExecutionError(f"Prepare required submodule {relative_path} at commit {expected} before evaluation")
        if _git(checkout, "rev-parse", "HEAD") != expected:
            raise ExecutionError(f"Submodule {relative_path} is not checked out at required commit {expected}")
        target = destination / relative_path
        if not target.resolve().is_relative_to(destination.resolve()):
            raise ExecutionError("Submodule export escapes the source workspace")
        _export_git(checkout, target, expected)


def materialize_source(task: TaskSpec, destination: Path) -> dict:
    """Export a fresh source tree without its Git history or private task assets.

    Git sources use the specified revision, or HEAD. Directory sources use a
    snapshot; callers should record how that directory was obtained.
    """
    destination = Path(destination).resolve()
    source = task.source.path.resolve()
    if not source.is_dir():
        raise ExecutionError(f"Source directory is not prepared: {source}. Explicitly provision the task source before evaluation")
    if destination.exists():
        raise ExecutionError(f"Source destination already exists: {destination}")
    if destination.is_relative_to(source):
        raise ExecutionError("Source destination must be outside the original source")
    if task.task_dir.resolve().is_relative_to(source) or (
        task.grader_dir is not None and (
            task.grader_dir.resolve().is_relative_to(source)
            or source.is_relative_to(task.grader_dir.resolve())
        )
    ):
        raise ExecutionError("Source overlaps private task assets")
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = source_identity(task)
    if info["kind"] == "git":
        _export_git(source, destination, info["revision"])
    else:
        shutil.copytree(source, destination, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    # Internal symlinks are common in real projects. External links would expose
    # files beyond the promised source tree when using the local backend.
    for item in destination.rglob("*"):
        if item.name == ".git":
            raise ExecutionError("Source snapshot contains nested Git history")
        if item.is_symlink() and not item.resolve().is_relative_to(destination):
            raise ExecutionError(f"Source contains an external symlink: {item.relative_to(destination)}")
        if not item.is_symlink() and not item.is_file() and not item.is_dir():
            raise ExecutionError(f"Unsupported source file type: {item.relative_to(destination)}")
    return info


class ExecutionSession:
    def __init__(self, task: TaskSpec, workspace: Path, grader_dir: Path | None,
                 *, capture_workspace: bool = True):
        self.task = task
        self.workspace = workspace.resolve()
        self.grader_dir = grader_dir.resolve() if grader_dir is not None else None
        self.session_id = "opbench-local-" + uuid.uuid4().hex
        self.closed = False
        self.execution_stopped = False
        self.cleanup_errors: list[str] = []
        self.capture_workspace = capture_workspace
        self.workspace_capture_ready = capture_workspace
        self.initial_workspace_upload = {"status": "not_applicable", "duration_sec": 0.0}

    @property
    def info(self) -> dict:
        return {"backend": self.task.environment.backend,
                "session_id": self.session_id, "closed": self.closed,
                "execution_stopped": self.execution_stopped,
                "grader_access": self.grader_dir is not None,
                "development_only": self.task.environment.backend == "local",
                "image": self.task.environment.image,
                "resource_enforcement": "docker" if self.task.environment.backend == "docker" else "not_enforced",
                "network": "host" if self.task.environment.backend == "local" else "none",
                "network_policy": "unrestricted_development" if self.task.environment.backend == "local" else "disabled",
                "cleanup_errors": self.cleanup_errors,
                "workspace_capture_requested": self.capture_workspace,
                "initial_workspace_upload": dict(self.initial_workspace_upload),
                "workspace_capture_ready": self.workspace_capture_ready,
                "resources": {"cpus": self.task.environment.cpus,
                              "memory": self.task.environment.memory,
                              "pids_limit": self.task.environment.pids_limit,
                              "gpus": self.task.environment.gpus}}

    def expand(self, argv: Sequence[str]) -> list[str]:
        replacements = {
            "{workspace}": str(self.workspace),
            "{python}": self.task.environment.python,
        }
        if self.grader_dir is not None:
            replacements["{grader}"] = str(self.grader_dir)
        expanded = []
        for token in argv:
            if "{grader}" in token and self.grader_dir is None:
                raise ExecutionError("Command requires grader but this session has no grader access")
            for placeholder, value in replacements.items():
                token = token.replace(placeholder, value)
            expanded.append(token)
        return expanded

    def run(self, argv: Sequence[str], timeout_sec: float, log_path: Path,
            *, max_output_bytes: int = 16 * 1024 * 1024) -> CommandResult:
        if self.closed:
            raise ExecutionError("Execution session is closed")
        return run_process(self.expand(argv), cwd=self.workspace, timeout_sec=timeout_sec,
                           log_path=log_path, environment=self._local_environment(), max_output_bytes=max_output_bytes)

    def _local_environment(self) -> dict[str, str]:
        env = {key: value for key, value in os.environ.items()
               if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT"}}
        env.update(self.task.environment.environment)
        return env

    def close(self) -> None:
        self.closed = True
        self.execution_stopped = True

    def observe(self, argv: Sequence[str], timeout_sec: float, input_bytes: bytes,
                output_path: Path, log_path: Path, *, max_output_bytes: int = 8 * 1024 * 1024,
                max_log_bytes: int = 16 * 1024 * 1024) -> CommandResult:
        if self.closed:
            raise ExecutionError("Execution session is closed")
        return run_observation_process(self.expand(argv), cwd=self.workspace, timeout_sec=timeout_sec,
            input_bytes=input_bytes, output_path=output_path, log_path=log_path, environment=self._local_environment(),
            max_output_bytes=max_output_bytes, max_log_bytes=max_log_bytes)

    def run_unittest(self, argv: Sequence[str], timeout_sec: float, log_path: Path,
                     *, max_output_bytes: int = 16 * 1024 * 1024) -> tuple[CommandResult, dict | None]:
        """Collect actual stdlib TextTestRunner counts outside the workspace.

        This detects absent execution, empty suites and early normal exits. It
        does not prove that arbitrary malicious code sharing the Python process
        cannot manipulate unittest or its evidence.
        """
        with tempfile.TemporaryDirectory(prefix="opbench-unittest-") as temporary:
            host_witness = Path(temporary) / "witness.json"
            witness_path = self._unittest_witness_path(host_witness)
            result = self.run([argv[0], "-I", "-c", _UNITTEST_BOOTSTRAP, witness_path, *argv[1:]],
                              timeout_sec, log_path, max_output_bytes=max_output_bytes)
            try:
                witness = self._read_unittest_witness(witness_path, host_witness)
            except ExecutionError as exc:
                raise WitnessCollectionError(str(exc), result) from exc
            return result, witness

    def _unittest_witness_path(self, host_witness: Path) -> str:
        return str(host_witness)

    def _read_unittest_witness(self, witness_path: str, host_witness: Path) -> dict | None:
        if not host_witness.exists():
            return None
        try:
            if host_witness.stat().st_size > 16384:
                return {"error": "unittest witness exceeds its size limit"}
            value = json.loads(host_witness.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {"error": "unittest witness is not an object"}
        except OSError as exc:
            raise ExecutionError(f"Cannot collect local unittest witness: {exc}") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            return {"error": f"Cannot read unittest witness: {exc}"}

    def sync_to_container(self) -> None:
        """Local sessions already operate directly on the workspace."""

    def sync_from_container(self) -> None:
        """Local sessions already operate directly on the workspace."""


class DockerSession(ExecutionSession):
    def __init__(self, task: TaskSpec, workspace: Path, grader_dir: Path | None,
                 *, capture_workspace: bool = True):
        super().__init__(task, workspace, grader_dir, capture_workspace=capture_workspace)
        self.observed_network_mode: str | None = None
        self.container = "opbench-" + uuid.uuid4().hex
        self.session_id = self.container
        self.image_id: str | None = None
        self.image_os: str | None = None
        self.image_architecture: str | None = None
        self.image_variant: str | None = None
        self.daemon_os: str | None = None
        self.daemon_architecture: str | None = None
        self.platform_detection_error: str | None = None
        self.workspace_volume = self.container + "-workspace"
        self.grader_volume = self.container + "-grader"
        self.transfer_container = self.container + "-transfer"
        self._volumes: set[str] = set()
        self._container_created = False
        self._transfer_created = False
        self._workspace_uploaded = False
        self.cleanup_complete = False
        self.workspace_capture_ready = False
        self.initial_workspace_upload = {"status": "not_started", "duration_sec": None}

    @property
    def info(self) -> dict:
        image_arch = self._normalize_architecture(self.image_architecture)
        daemon_arch = self._normalize_architecture(self.daemon_architecture)
        possible_emulation = image_arch != daemon_arch if image_arch and daemon_arch else None
        return {**super().info, "image_id": self.image_id,
                "image_os": self.image_os, "image_architecture": self.image_architecture,
                "image_variant": self.image_variant, "daemon_os": self.daemon_os,
                "daemon_architecture": self.daemon_architecture,
                "possible_emulation": possible_emulation,
                "platform_detection_error": self.platform_detection_error,
                "observed_network_mode": self.observed_network_mode,
                "read_only_root": True, "isolation_validation": "not_attested",
                "workspace_transport": "docker_volume_copy",
                "resources_remaining": {
                    "container": self.container if self._container_created else None,
                    "transfer_container": self.transfer_container if self._transfer_created else None,
                    "volumes": sorted(self._volumes),
                },
                "cleanup_complete": self.cleanup_complete}

    @staticmethod
    def _normalize_architecture(value: str | None) -> str | None:
        if not value:
            return None
        normalized = value.lower()
        return {"x86_64": "amd64", "aarch64": "arm64", "arm64/v8": "arm64",
                "i386": "386", "i686": "386", "armv7l": "arm", "armv6l": "arm"}.get(normalized, normalized)

    def _inspect_platform(self) -> None:
        """Record the actual image/daemon pair; architecture mismatch is evidence
        of possible emulation, not a measurement of its performance overhead.
        """
        inspected = self._control(["image", "inspect", self.task.environment.image])
        try:
            image = json.loads(inspected.stdout)[0]
            image_id = image["Id"]
            if not isinstance(image_id, str) or not image_id:
                raise ValueError("Image ID is missing")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise ExecutionError(f"Cannot read inspected image identity: {exc}") from exc
        self.image_id = image_id
        self.image_os = image.get("Os")
        self.image_architecture = image.get("Architecture")
        self.image_variant = image.get("Variant") or None
        try:
            daemon = json.loads(self._control(["info", "--format", "{{json .}}"]).stdout)
            self.daemon_os = daemon.get("OSType")
            self.daemon_architecture = daemon.get("Architecture")
        except (ExecutionError, json.JSONDecodeError, AttributeError) as exc:
            # Some restricted contexts allow image/run operations but not info.
            # Unknown architecture is recorded explicitly, never called native.
            self.platform_detection_error = str(exc)

    def start(self) -> None:
        env = self.task.environment
        # Do not pull implicitly: preparation must provision the declared image.
        self._inspect_platform()
        for volume in ([self.workspace_volume, self.grader_volume] if self.grader_dir else [self.workspace_volume]):
            # A lost CLI response does not mean the daemon rejected creation.
            # Register ownership before sending any mutating request.
            self._volumes.add(volume)
            self._control(["volume", "create", "--label", "opbench.session=" + self.container, volume])
        upload_started = time.monotonic()
        self.initial_workspace_upload = {"status": "running", "duration_sec": None}
        try:
            self._upload(include_grader=self.grader_dir is not None)
        except BaseException:
            self.initial_workspace_upload = {"status": "failed", "duration_sec": None,
                                             "elapsed_sec": time.monotonic() - upload_started}
            raise
        self.initial_workspace_upload = {"status": "completed", "duration_sec": time.monotonic() - upload_started}
        argv = [
            "run", "--detach", "--pull=never", "--name", self.container,
            "--read-only", "--network", "none", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--cpus", str(env.cpus),
            "--memory", env.memory, "--pids-limit", str(env.pids_limit),
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,exec,nosuid,size=1g,mode=1777",
            "--workdir", "/workspace", "--env", "HOME=/tmp",
            "--mount", f"type=volume,source={self.workspace_volume},target=/workspace,volume-nocopy",
        ]
        if self.grader_dir is not None:
            argv.extend(["--mount", f"type=volume,source={self.grader_volume},target=/grader,readonly,volume-nocopy"])
        if env.gpus is not None:
            argv.extend(["--gpus", env.gpus])
        for key, value in env.environment.items():
            argv.extend(["--env", f"{key}={value}"])
        entrypoint = "import time; time.sleep(31536000)"
        argv.extend(["--entrypoint", env.python, self.image_id, "-I", "-c", entrypoint])
        self._container_created = True
        self._control(argv)
        self._observe_container()

    def _observe_container(self) -> None:
        """Retain two actual Docker facts, without general isolation attestation."""
        inspected = self._control(["inspect", self.container])
        try:
            container = json.loads(inspected.stdout)[0]
            mode = container["HostConfig"]["NetworkMode"]
            if not isinstance(mode, str):
                raise ValueError("Container network mode is unavailable")
            self.observed_network_mode = mode
            if mode != "none":
                raise ValueError(f"Container network mode {mode!r} differs from required 'none'")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
            raise ExecutionError(f"Cannot confirm execution container configuration: {exc}") from exc

    def _upload(self, *, include_grader: bool = False) -> None:
        """Replace volume contents through the Docker API, including deletions.

        Only this short-lived initializer mounts the grader volume writable. It
        runs trusted Python in isolated mode, never task/agent code. Named volumes
        work with remote contexts where the daemon cannot see host paths.

        The declared task image supplies Python, avoiding an undeclared helper
        image or an implicit network pull. Its recorded architecture also applies
        to initialization: an emulated task image may fail before a patch runs.
        """
        # A partially replaced volume is not the previous valid workspace. This
        # remains false if upload/initialization or helper termination fails.
        self._workspace_uploaded = False
        self.workspace_capture_ready = False
        uploads = [(self.workspace, self.workspace_volume, "/workspace")]
        if include_grader and self.grader_dir is not None:
            uploads.append((self.grader_dir, self.grader_volume, "/grader"))
        argv = ["run", "--detach", "--pull=never", "--name", self.transfer_container,
                "--read-only", "--network=none", "--cap-drop=ALL",
                "--cap-add=CHOWN", "--cap-add=DAC_OVERRIDE", "--cap-add=FOWNER",
                "--security-opt=no-new-privileges", "--user", "0:0"]
        for _, volume, target in uploads:
            argv.extend(["--mount", f"type=volume,source={volume},target={target},volume-nocopy"])
        argv.extend(["--entrypoint", self.task.environment.python, self.image_id,
                     "-I", "-c", "import time; time.sleep(31536000)"])
        self._transfer_created = True
        try:
            self._control(argv)
            for host_path, _, target in uploads:
                clear = (
                    "import pathlib,shutil,sys\n"
                    "for p in pathlib.Path(sys.argv[1]).iterdir():\n"
                    " if p.is_dir() and not p.is_symlink(): shutil.rmtree(p)\n"
                    " else: p.unlink()\n"
                )
                self._control(["exec", self.transfer_container, self.task.environment.python,
                               "-I", "-c", clear, target], timeout=300)
                self._control(["cp", str(host_path) + "/.", self.transfer_container + ":" + target], timeout=600)
                ownership = (
                    "import os,sys\n"
                    "uid,gid=int(sys.argv[2]),int(sys.argv[3])\n"
                    "os.chown(sys.argv[1],uid,gid)\n"
                    "for root,dirs,files in os.walk(sys.argv[1]):\n"
                    " for name in dirs+files: os.lchown(os.path.join(root,name),uid,gid)\n"
                )
                self._control(["exec", self.transfer_container, self.task.environment.python,
                               "-I", "-c", ownership, target, str(os.getuid()), str(os.getgid())], timeout=300)
        finally:
            if self._transfer_created:
                self._remove_container(self.transfer_container)
        if self._transfer_created:
            raise ExecutionError("Workspace initialization cannot continue: transfer container removal is unconfirmed")
        self._workspace_uploaded = True

    def sync_to_container(self) -> None:
        if self.closed:
            raise ExecutionError("Cannot upload to a stopped execution session")
        self._upload()

    def sync_from_container(self) -> None:
        if not self._container_created or not self._workspace_uploaded:
            raise ExecutionError("No initialized container workspace is available to download")
        # Download into a new host directory first. A failed transfer must not
        # erase the last successfully downloaded source/patch workspace.
        with tempfile.TemporaryDirectory(prefix="opbench-download-", dir=self.workspace.parent) as temporary:
            staging = Path(temporary) / "workspace"
            staging.mkdir()
            self._control(["cp", self.container + ":/workspace/.", str(staging)], timeout=600)
            previous = Path(temporary) / "previous"
            if self.workspace.exists():
                self.workspace.rename(previous)
            try:
                staging.rename(self.workspace)
            except BaseException:
                if previous.exists():
                    previous.rename(self.workspace)
                raise

    def _unittest_witness_path(self, host_witness: Path) -> str:
        return "/tmp/opbench-unittest-witness-" + uuid.uuid4().hex + ".json"

    def _read_unittest_witness(self, witness_path: str, host_witness: Path) -> dict | None:
        if self.closed:
            return None
        # Docker cp can see a different filesystem view for a running tmpfs
        # mount. Read the bounded artifact inside the running container instead.
        reader = (
            "import base64,json,sys\n"
            "try:\n"
            " with open(sys.argv[1],'rb') as stream: data=stream.read(16385)\n"
            "except FileNotFoundError:\n response={'state':'missing'}\n"
            "else:\n response={'state':'present','data':base64.b64encode(data).decode('ascii')}\n"
            "print(json.dumps(response))\n"
        )
        response = self._control(["exec", self.container, self.task.environment.python,
                                  "-I", "-c", reader, witness_path])
        try:
            envelope = json.loads(response.stdout)
            if envelope.get("state") == "missing":
                return None
            if envelope.get("state") != "present":
                raise ValueError("Unknown witness collection response")
            payload = base64.b64decode(envelope["data"], validate=True)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ExecutionError(f"Cannot decode the unittest collection response: {exc}") from exc
        if len(payload) > 16384:
            return {"error": "unittest witness exceeds its size limit"}
        try:
            host_witness.write_bytes(payload)
        except OSError as exc:
            raise ExecutionError(f"Cannot retain the collected unittest witness: {exc}") from exc
        return super()._read_unittest_witness(witness_path, host_witness)

    def _control(self, argv: Sequence[str], timeout: float = 60) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(["docker", *argv], capture_output=True, text=True,
                                  timeout=timeout, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            raise ExecutionError(f"Docker resource operation failed: {detail}") from exc

    @staticmethod
    def _not_found(error: ExecutionError, kind: str) -> bool:
        # Only a response about the single previously named resource establishes
        # absence. Transport/permission failures remain cleanup failures.
        return f"no such {kind}" in str(error).lower()

    def _remove_container(self, name: str) -> bool:
        try:
            self._control(["rm", "--force", name])
        except ExecutionError as exc:
            if not self._not_found(exc, "container"):
                role = "transfer" if name == self.transfer_container else "execution"
                self.cleanup_errors.append(f"Remove {role} container: {exc}")
                return False
        if name == self.transfer_container:
            self._transfer_created = False
        else:
            self._container_created = False
        return True

    def expand(self, argv: Sequence[str]) -> list[str]:
        expanded = []
        for token in argv:
            if "{grader}" in token and self.grader_dir is None:
                raise ExecutionError("Command requires grader but this session has no grader access")
            expanded.append(token.replace("{workspace}", "/workspace")
                            .replace("{grader}", "/grader")
                            .replace("{python}", self.task.environment.python))
        return expanded

    def run(self, argv: Sequence[str], timeout_sec: float, log_path: Path,
            *, max_output_bytes: int = 16 * 1024 * 1024) -> CommandResult:
        if self.closed:
            raise ExecutionError("Execution session is closed")
        try:
            # Reap background descendants left by an otherwise successful test.
            # A timed-out wrapper is handled by removing the entire container.
            result = run_process(
                ["docker", "exec", "--workdir", "/workspace", self.container,
                 self.task.environment.python, "-I", "-c", _PROCESS_WRAPPER, *self.expand(argv)],
                cwd=self.workspace, timeout_sec=timeout_sec, log_path=log_path,
                max_output_bytes=max_output_bytes,
            )
        except BaseException:
            self.close()
            raise
        if result.timed_out or result.output_limited:
            # Killing docker exec's client does not kill its container processes.
            # Stop all processes and retain the workspace before removing the
            # container/volumes. Native runners still need to freeze this patch.
            self.close()
        return result

    def observe(self, argv: Sequence[str], timeout_sec: float, input_bytes: bytes,
                output_path: Path, log_path: Path, *, max_output_bytes: int = 8 * 1024 * 1024,
                max_log_bytes: int = 16 * 1024 * 1024) -> CommandResult:
        if self.closed:
            raise ExecutionError("Execution session is closed")
        try:
            result = run_observation_process(
                ["docker", "exec", "-i", "--workdir", "/workspace", self.container,
                 self.task.environment.python, "-I", "-c", _PROCESS_WRAPPER, *self.expand(argv)],
                cwd=self.workspace, timeout_sec=timeout_sec, input_bytes=input_bytes,
                output_path=output_path, log_path=log_path,
                max_output_bytes=max_output_bytes, max_log_bytes=max_log_bytes)
        except BaseException:
            self.close()
            raise
        if result.timed_out or result.output_limited:
            self.close()
        return result

    def close(self) -> None:
        if self.cleanup_complete:
            return
        self.closed = True
        # Initializers can write the same workspace volume. Their lifetime is
        # part of the freeze boundary even when the main container is stopped.
        if self._transfer_created:
            self._remove_container(self.transfer_container)
        if self._container_created:
            stopped = False
            try:
                running = self._control(["inspect", "--format", "{{.State.Running}}", self.container]).stdout.strip()
                if running not in {"true", "false"}:
                    raise ExecutionError("Docker did not return a boolean container running state")
                if running == "true":
                    self._control(["kill", self.container])
                stopped = True
            except ExecutionError as exc:
                if self._not_found(exc, "container") or self._not_found(exc, "object"):
                    self._container_created = False
                    stopped = True
                else:
                    self.cleanup_errors.append(f"Stop execution container: {exc}")
            self.execution_stopped = stopped and not self._transfer_created
            if (self.capture_workspace and self._container_created and self.execution_stopped
                    and self._workspace_uploaded and not self.workspace_capture_ready):
                try:
                    self.sync_from_container()
                    self.workspace_capture_ready = True
                except (ExecutionError, OSError) as exc:
                    self.cleanup_errors.append(f"Download final workspace: {exc}")
            if self._container_created and self._remove_container(self.container):
                self.execution_stopped = not self._transfer_created
        for volume in tuple(self._volumes):
            try:
                self._control(["volume", "rm", volume])
                self._volumes.remove(volume)
            except ExecutionError as exc:
                if self._not_found(exc, "volume"):
                    self._volumes.remove(volume)
                else:
                    self.cleanup_errors.append(f"Remove volume {volume}: {exc}")
        if not self._container_created and not self._transfer_created:
            self.execution_stopped = True
        self.cleanup_complete = not self._container_created and not self._transfer_created and not self._volumes


@contextmanager
def create_session(task: TaskSpec, workspace: Path, grader_dir: Path | None = None,
                   *, capture_workspace: bool = True) -> Iterator[ExecutionSession]:
    """Create an execution context. Omit grader_dir for all solver sessions.

    Grading workers only return bounded observations/logs, so their modified
    workspaces need not be downloaded. They must still stop and release every
    resource. Build and solver sessions retain the default final capture.
    """
    session: ExecutionSession
    if task.environment.backend == "docker":
        session = DockerSession(task, workspace, grader_dir, capture_workspace=capture_workspace)
        try:
            session.start()
        except BaseException as exc:
            # A failed start can still have allocated a named container.
            try:
                session.close()
            except ExecutionError as cleanup_exc:
                session.cleanup_errors.append(str(cleanup_exc))
            for error in session.cleanup_errors:
                exc.add_note(error)
            exc.execution_session_info = session.info
            raise
    else:
        session = ExecutionSession(task, workspace, grader_dir, capture_workspace=capture_workspace)
    try:
        yield session
    finally:
        try:
            session.close()
        except Exception as exc:
            # Cleanup diagnostics must not replace an already known command or
            # scoring result. Callers requiring workspace capture check the
            # explicit readiness state before consuming it.
            session.cleanup_errors.append(f"Close execution session: {type(exc).__name__}: {exc}")
