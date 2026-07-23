from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import threading
import time
from typing import Final

from op_bench.runtime.adapters import AdapterActionClient
from op_bench.runtime.canonical import canonical_json
from op_bench.runtime.contracts import ActionObservation, ActionRequest
from op_bench.runtime.validation import ContractError, require_int, require_str


_CLIENT_FILENAME: Final = "opbench_action.py"
_REQUEST_DIRECTORY: Final = "requests"
_RESPONSE_DIRECTORY: Final = "responses"
_MAX_MESSAGE_BYTES: Final = 1_048_576


_CLIENT_SOURCE = r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import hmac
import json
import os
import stat
import sys
import time
import uuid


ROOT = __ROOT__
REQUEST_DIRECTORY = __REQUEST_DIRECTORY__
RESPONSE_DIRECTORY = __RESPONSE_DIRECTORY__
SESSION_ID = __SESSION_ID__
DEADLINE_MS = __DEADLINE_MS__
TIMEOUT_MS = __TIMEOUT_MS__
ROOT_IDENTITY = __ROOT_IDENTITY__
REQUEST_IDENTITY = __REQUEST_IDENTITY__
RESPONSE_IDENTITY = __RESPONSE_IDENTITY__
TRANSPORT_TOKEN_DIGEST = __TRANSPORT_TOKEN_DIGEST__
STATE_FILENAME = "client_sequence.json"
LOCK_FILENAME = "client_sequence.lock"
COMMANDS = (
    "workspace_list",
    "workspace_search",
    "workspace_read",
    "workspace_write",
    "workspace_apply_patch",
    "command_run",
    "test_run",
    "vcs_diff",
    "session_finish",
)
REQUEST_FIELDS = {
    "contract_type",
    "schema_version",
    "session_id",
    "action_id",
    "action_name",
    "arguments",
    "client_sequence",
    "deadline_ms",
}
OBSERVATION_FIELDS = {
    "contract_type",
    "schema_version",
    "session_id",
    "action_id",
    "ok",
    "error_code",
    "message",
    "data",
    "started_at_ms",
    "ended_at_ms",
    "budget_delta",
    "mutation_state",
}
DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _canonical(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _open_bound_directory(path, expected):
    descriptor = os.open(path, DIRECTORY_FLAGS)
    observed = os.fstat(descriptor)
    if not stat.S_ISDIR(observed.st_mode):
        os.close(descriptor)
        raise RuntimeError("action exchange directory is not a directory")
    if (observed.st_dev, observed.st_ino) != tuple(expected):
        os.close(descriptor)
        raise RuntimeError("action exchange directory binding changed")
    return descriptor


def _read_regular(descriptor, filename, *, maximum=1048576):
    file_descriptor = os.open(filename, os.O_RDONLY | NOFOLLOW, dir_fd=descriptor)
    try:
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("action exchange message is not a regular file")
        if metadata.st_size > maximum:
            raise RuntimeError("action exchange message is too large")
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(file_descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > maximum:
            raise RuntimeError("action exchange message is too large")
        return payload
    finally:
        os.close(file_descriptor)


def _atomic_write(descriptor, filename, payload):
    temporary = ".tmp-" + uuid.uuid4().hex
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | NOFOLLOW
    file_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(file_descriptor, view)
            view = view[written:]
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    try:
        os.replace(
            temporary,
            filename,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except OSError:
            pass
        raise


def _next_sequence(root_descriptor):
    lock_descriptor = os.open(
        LOCK_FILENAME,
        os.O_RDWR | os.O_CREAT | NOFOLLOW,
        0o600,
        dir_fd=root_descriptor,
    )
    try:
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            raise RuntimeError("action client lock is not a regular file")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        sequence = 0
        try:
            encoded = _read_regular(root_descriptor, STATE_FILENAME, maximum=128)
        except FileNotFoundError:
            encoded = b""
        if encoded:
            value = json.loads(encoded.decode("utf-8"))
            if set(value) != {"client_sequence"}:
                raise RuntimeError("action client sequence state is malformed")
            sequence = value["client_sequence"]
            if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
                raise RuntimeError("action client sequence state is malformed")
        sequence += 1
        _atomic_write(
            root_descriptor,
            STATE_FILENAME,
            _canonical({"client_sequence": sequence}).encode("utf-8"),
        )
        return sequence
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)


def _request_from_command(command, arguments, root_descriptor):
    return {
        "contract_type": "action_request",
        "schema_version": "v1",
        "session_id": SESSION_ID,
        "action_id": "action-" + uuid.uuid4().hex,
        "action_name": command,
        "arguments": arguments,
        "client_sequence": _next_sequence(root_descriptor),
        "deadline_ms": DEADLINE_MS,
    }


def _validate_request(value):
    if not isinstance(value, dict) or set(value) != REQUEST_FIELDS:
        raise RuntimeError("request is not a canonical ActionRequest")
    if value["contract_type"] != "action_request" or value["schema_version"] != "v1":
        raise RuntimeError("request is not a canonical ActionRequest")
    if value["session_id"] != SESSION_ID or value["deadline_ms"] != DEADLINE_MS:
        raise RuntimeError("request does not match the fixed action exchange")
    if not isinstance(value["action_id"], str) or not value["action_id"]:
        raise RuntimeError("request action_id is invalid")
    if value["action_name"] not in COMMANDS or not isinstance(value["arguments"], dict):
        raise RuntimeError("request action is invalid")
    sequence = value["client_sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise RuntimeError("request client_sequence is invalid")
    _canonical(value)
    return value


def _validate_observation(value, request):
    if not isinstance(value, dict) or set(value) != OBSERVATION_FIELDS:
        raise RuntimeError("response is not a canonical ActionObservation")
    if value["contract_type"] != "action_observation" or value["schema_version"] != "v1":
        raise RuntimeError("response is not a canonical ActionObservation")
    if value["session_id"] != request["session_id"] or value["action_id"] != request["action_id"]:
        raise RuntimeError("response identity does not match request")
    _canonical(value)
    return value


def _parse_json_object(encoded, label):
    try:
        value = json.loads(encoded)
    except (TypeError, ValueError):
        raise RuntimeError(label + " is not valid JSON") from None
    if not isinstance(value, dict):
        raise RuntimeError(label + " must be a JSON object")
    return value


def _invoke(request):
    request = _validate_request(request)
    request_descriptor = _open_bound_directory(REQUEST_DIRECTORY, REQUEST_IDENTITY)
    response_descriptor = _open_bound_directory(RESPONSE_DIRECTORY, RESPONSE_IDENTITY)
    transport_id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()
    request_filename = "request-" + transport_id + ".json"
    response_filename = "response-" + transport_id + ".json"
    try:
        _atomic_write(
            request_descriptor,
            request_filename,
            _canonical(request).encode("utf-8"),
        )
        expires = time.monotonic() + TIMEOUT_MS / 1000.0
        while True:
            try:
                encoded = _read_regular(response_descriptor, response_filename)
            except FileNotFoundError:
                if time.monotonic() >= expires:
                    raise TimeoutError("action exchange timed out") from None
                time.sleep(0.005)
                continue
            try:
                os.unlink(response_filename, dir_fd=response_descriptor)
            except FileNotFoundError:
                pass
            value = _parse_json_object(encoded.decode("utf-8"), "response")
            return _validate_observation(value, request)
    finally:
        os.close(request_descriptor)
        os.close(response_descriptor)


def main():
    if TRANSPORT_TOKEN_DIGEST is not None:
        token = os.environ.get("OPBENCH_ACTION_TRANSPORT_TOKEN", "")
        observed = hashlib.sha256(token.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(observed, TRANSPORT_TOKEN_DIGEST):
            raise RuntimeError("action transport authentication failed")
    parser = argparse.ArgumentParser(prog="opbench_action.py")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--arguments", required=True)
    json_parser = subparsers.add_parser("json")
    json_parser.add_argument("--request", required=True)
    args = parser.parse_args()

    root_descriptor = _open_bound_directory(ROOT, ROOT_IDENTITY)
    try:
        if args.command == "json":
            request = _parse_json_object(args.request, "request")
        else:
            arguments = _parse_json_object(args.arguments, "arguments")
            request = _request_from_command(args.command, arguments, root_descriptor)
    finally:
        os.close(root_descriptor)

    response = _invoke(request)
    sys.stdout.write(_canonical(response) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        message = str(exc) or type(exc).__name__
        sys.stderr.write("opbench action failed: " + message + "\n")
        raise SystemExit(1)
'''


def _directory_identity(path: Path) -> tuple[int, int]:
    metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise ContractError("action exchange directory is not a regular directory")
    return metadata.st_dev, metadata.st_ino


def _open_bound_directory(path: Path, identity: tuple[int, int]) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ContractError("action exchange directory binding changed") from exc
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode) or (metadata.st_dev, metadata.st_ino) != identity:
        os.close(descriptor)
        raise ContractError("action exchange directory binding changed")
    return descriptor


def _read_regular(descriptor: int, filename: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(filename, flags, dir_fd=descriptor)
    try:
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("action exchange message is not a regular file")
        if metadata.st_size > _MAX_MESSAGE_BYTES:
            raise ContractError("action exchange message is too large")
        payload = bytearray()
        while len(payload) <= _MAX_MESSAGE_BYTES:
            chunk = os.read(file_descriptor, min(65_536, _MAX_MESSAGE_BYTES + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > _MAX_MESSAGE_BYTES:
            raise ContractError("action exchange message is too large")
        return bytes(payload)
    finally:
        os.close(file_descriptor)


def _write_regular(descriptor: int, filename: str, payload: bytes) -> None:
    temporary = f".tmp-{threading.get_ident()}-{time.monotonic_ns()}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(temporary, flags, 0o600, dir_fd=descriptor)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(file_descriptor, view)
            view = view[written:]
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    try:
        os.replace(
            temporary,
            filename,
            src_dir_fd=descriptor,
            dst_dir_fd=descriptor,
        )
        os.fsync(descriptor)
    except BaseException:
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except OSError:
            pass
        raise


class ProcessActionExchange:
    """Descriptor-bound JSON transport for a process-isolated canonical Adapter."""

    def __init__(
        self,
        *,
        action_client: AdapterActionClient,
        session_id: str,
        exchange_root: Path,
        timeout_ms: int,
        deadline_ms: int,
        transport_token: str | None = None,
    ) -> None:
        if not isinstance(action_client, AdapterActionClient):
            raise ContractError("action_client: expected AdapterActionClient")
        self._action_client = action_client
        self.session_id = require_str(session_id, "session_id")
        if not isinstance(exchange_root, Path):
            raise ContractError("exchange_root: expected Path")
        if not exchange_root.is_absolute():
            raise ContractError("exchange_root: expected absolute path")
        self.exchange_root = exchange_root
        self.timeout_ms = require_int(timeout_ms, "timeout_ms", minimum=1)
        self.deadline_ms = require_int(deadline_ms, "deadline_ms", minimum=1)
        if transport_token is None:
            self._transport_token_digest = None
        else:
            selected_token = require_str(transport_token, "transport_token")
            self._transport_token_digest = hashlib.sha256(
                selected_token.encode("utf-8")
            ).hexdigest()
        self.request_directory = exchange_root / _REQUEST_DIRECTORY
        self.response_directory = exchange_root / _RESPONSE_DIRECTORY
        self.client_path = exchange_root / _CLIENT_FILENAME
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._request_identity: tuple[int, int] | None = None
        self._response_identity: tuple[int, int] | None = None
        self._observation_count = 0
        self._finish_count = 0
        self._counts_lock = threading.Lock()
        self._server_failure: str | None = None

    @property
    def observation_count(self) -> int:
        with self._counts_lock:
            return self._observation_count

    @property
    def finish_count(self) -> int:
        with self._counts_lock:
            return self._finish_count

    @property
    def server_failure(self) -> str | None:
        return self._server_failure

    def start(self) -> "ProcessActionExchange":
        if self._thread is not None:
            return self
        try:
            self.exchange_root.mkdir(mode=0o700, parents=False, exist_ok=False)
            self.request_directory.mkdir(mode=0o700)
            self.response_directory.mkdir(mode=0o700)
        except OSError as exc:
            raise ContractError("action exchange root could not be created") from exc
        root_identity = _directory_identity(self.exchange_root)
        self._request_identity = _directory_identity(self.request_directory)
        self._response_identity = _directory_identity(self.response_directory)
        source = (
            _CLIENT_SOURCE.replace("__ROOT__", repr(str(self.exchange_root)))
            .replace("__REQUEST_DIRECTORY__", repr(str(self.request_directory)))
            .replace("__RESPONSE_DIRECTORY__", repr(str(self.response_directory)))
            .replace("__SESSION_ID__", repr(self.session_id))
            .replace("__DEADLINE_MS__", repr(self.deadline_ms))
            .replace("__TIMEOUT_MS__", repr(self.timeout_ms))
            .replace("__ROOT_IDENTITY__", repr(root_identity))
            .replace("__REQUEST_IDENTITY__", repr(self._request_identity))
            .replace("__RESPONSE_IDENTITY__", repr(self._response_identity))
            .replace(
                "__TRANSPORT_TOKEN_DIGEST__",
                repr(self._transport_token_digest),
            )
        )
        descriptor = os.open(
            self.client_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o700,
        )
        try:
            encoded = source.encode("utf-8")
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._thread = threading.Thread(
            target=self._serve,
            name="opbench-process-actions",
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self, *, cleanup: bool = True) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=max(1.0, self.timeout_ms / 1000.0 + 0.5))
            self._thread = None
        if cleanup and self.exchange_root.exists():
            shutil.rmtree(self.exchange_root)

    def __enter__(self) -> "ProcessActionExchange":
        return self.start()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _serve(self) -> None:
        assert self._request_identity is not None
        assert self._response_identity is not None
        try:
            request_descriptor = _open_bound_directory(
                self.request_directory,
                self._request_identity,
            )
            response_descriptor = _open_bound_directory(
                self.response_directory,
                self._response_identity,
            )
        except ContractError:
            self._server_failure = "exchange_binding_changed"
            return
        processed: set[str] = set()
        try:
            while not self._stop.is_set():
                names = sorted(
                    name
                    for name in os.listdir(request_descriptor)
                    if name.startswith("request-") and name.endswith(".json")
                )
                pending = [name for name in names if name not in processed]
                if not pending:
                    self._stop.wait(0.005)
                    continue
                for filename in pending:
                    if self._stop.is_set():
                        break
                    processed.add(filename)
                    try:
                        self._serve_one(
                            request_descriptor,
                            response_descriptor,
                            filename,
                        )
                    except BaseException:  # noqa: BLE001 - fixed boundary state only.
                        self._server_failure = "malformed_action_exchange"
                    finally:
                        try:
                            os.unlink(filename, dir_fd=request_descriptor)
                        except OSError:
                            pass
        finally:
            os.close(request_descriptor)
            os.close(response_descriptor)

    def _serve_one(
        self,
        request_descriptor: int,
        response_descriptor: int,
        filename: str,
    ) -> None:
        encoded = _read_regular(request_descriptor, filename)
        try:
            value = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ContractError("action exchange request is invalid JSON") from None
        request = ActionRequest.from_dict(value)
        if request.session_id != self.session_id or request.deadline_ms != self.deadline_ms:
            raise ContractError("action exchange request binding mismatch")
        canonical_request = request.to_dict()
        if encoded != canonical_json(canonical_request).encode("utf-8"):
            raise ContractError("action exchange request is not canonical JSON")
        response = self._action_client.execute(canonical_request)
        observation = ActionObservation.from_dict(response)
        if (
            observation.session_id != request.session_id
            or observation.action_id != request.action_id
        ):
            raise ContractError("action exchange observation binding mismatch")
        transport_id = filename[len("request-") : -len(".json")]
        response_filename = f"response-{transport_id}.json"
        _write_regular(
            response_descriptor,
            response_filename,
            canonical_json(observation.to_dict()).encode("utf-8"),
        )
        with self._counts_lock:
            self._observation_count += 1
            if request.action_name == "session_finish":
                self._finish_count += 1


__all__ = ["ProcessActionExchange"]
