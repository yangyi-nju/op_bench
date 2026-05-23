from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from op_bench.actions import WorkspaceActions
from op_bench.progress import Progress, format_command, format_duration, noop_progress


class ActionBridgeServer:
    def __init__(
        self,
        actions: WorkspaceActions,
        log_path: Path,
        exchange_dir: Path,
        progress: Progress | None = None,
    ) -> None:
        self.actions = actions
        self.log_path = log_path
        self.exchange_dir = exchange_dir
        self.progress = progress or noop_progress
        self.requests_dir = exchange_dir / "requests"
        self.responses_dir = exchange_dir / "responses"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(
            {
                "event": "session_start",
                "time": time.time(),
                "initial_digest": self.actions.workspace_state_digest(),
            }
        )

    def __enter__(self) -> "ActionBridgeServer":
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _serve(self) -> None:
        while not self._stop.is_set():
            handled = False
            for request_path in sorted(self.requests_dir.glob("*.json")):
                handled = True
                self._handle_request(request_path)
            if not handled:
                self._stop.wait(0.05)

    def _handle_request(self, request_path: Path) -> None:
        processing_path = request_path.with_suffix(".processing")
        try:
            request_path.rename(processing_path)
        except FileNotFoundError:
            return
        try:
            action = json.loads(processing_path.read_text(encoding="utf-8"))
            result = self.execute(action)
        except Exception as exc:  # noqa: BLE001 - action failures are returned to the agent.
            result = {"error": str(exc)}
        response_path = self.responses_dir / request_path.name
        temp_response_path = response_path.with_suffix(".tmp")
        temp_response_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temp_response_path.rename(response_path)
        processing_path.unlink(missing_ok=True)

    def execute(self, action: object) -> dict[str, Any]:
        if not isinstance(action, dict):
            return {"error": "action must be a JSON object"}
        before_digest = self.actions.workspace_state_digest()
        start = time.monotonic()
        action_summary = _action_summary(action)
        self.progress(f"action bridge call: {action_summary}")
        try:
            result = execute_workspace_action(action, self.actions)
        except Exception as exc:  # noqa: BLE001 - action errors are observations for the agent.
            result = {"error": str(exc)}
        after_digest = self.actions.workspace_state_digest()
        self._append_log(
            {
                "event": "tool_call",
                "time": time.time(),
                "action": _safe_json(action),
                "result": _safe_json(result),
                "duration_sec": time.monotonic() - start,
                "before_digest": before_digest,
                "after_digest": after_digest,
            }
        )
        status = _result_status(result)
        self.progress(f"action bridge done: {action_summary}, {status}, duration={format_duration(time.monotonic() - start)}")
        return result

    def _append_log(self, entry: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def execute_workspace_action(action: dict[str, Any], actions: WorkspaceActions) -> dict[str, Any]:
    action = _normalize_action(action)
    action_name = str(action.get("action", ""))
    if action_name == "read_file":
        return {"content": actions.read_file(_path_argument(action))}
    if action_name == "write_file":
        actions.write_file(_path_argument(action), str(action.get("content", "")))
        return {"ok": True}
    if action_name == "apply_patch":
        return actions.apply_patch(str(action.get("patch", ""))).to_dict()
    if action_name == "run_command":
        command = action.get("command")
        if isinstance(command, str):
            command = ["bash", "-lc", command]
        if not isinstance(command, list):
            return {"error": "run_command.command must be a list or string"}
        timeout_value = action.get("timeout_sec")
        timeout_sec = int(timeout_value) if timeout_value is not None else None
        return actions.run_command([str(part) for part in command], timeout_sec=timeout_sec).to_dict()
    if action_name == "run_test":
        return actions.run_test(_test_name_argument(action)).to_dict()
    if action_name == "git_diff":
        return {"diff": actions.git_diff()}
    return {"error": f"unsupported action: {action_name}"}


def build_action_cli(exchange_dir: Path, python_executable: str) -> str:
    return f"""#!{python_executable}
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

EXCHANGE_DIR = Path({str(exchange_dir)!r})
REQUESTS_DIR = EXCHANGE_DIR / "requests"
RESPONSES_DIR = EXCHANGE_DIR / "responses"


def main(argv):
    if not argv or argv[0] in {{"help", "--help", "-h"}}:
        print("usage: opbench_action.py <read_file|write_file|apply_patch|run_test|run_command|git_diff|json> ...", file=sys.stderr)
        print("run_command with one argument is executed as a shell command; multiple arguments are executed as argv.", file=sys.stderr)
        return 2
    command = argv[0]
    if command == "json":
        if len(argv) != 2:
            print("json mode requires one JSON object argument", file=sys.stderr)
            return 2
        action = json.loads(argv[1])
    elif command == "read_file":
        action = {{"action": "read_file", "path": argv[1]}}
    elif command == "write_file":
        action = {{"action": "write_file", "path": argv[1], "content": sys.stdin.read()}}
    elif command == "apply_patch":
        action = {{"action": "apply_patch", "patch": sys.stdin.read()}}
    elif command == "run_test":
        action = {{"action": "run_test", "test_name": argv[1]}}
    elif command == "run_command":
        if len(argv) < 2:
            print("run_command requires a command after --", file=sys.stderr)
            return 2
        parts = argv[1:]
        if parts and parts[0] == "--":
            parts = parts[1:]
        action = {{"action": "run_command", "command": parts[0] if len(parts) == 1 else parts}}
    elif command == "git_diff":
        action = {{"action": "git_diff"}}
    else:
        print(f"unknown action command: {{command}}", file=sys.stderr)
        return 2
    response = call(action)
    sys.stdout.write(json.dumps(response, indent=2, ensure_ascii=False, sort_keys=True))
    sys.stdout.write("\\n")
    return 0


def call(action):
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    request_id = uuid.uuid4().hex
    request_path = REQUESTS_DIR / f"{{request_id}}.json"
    temp_request_path = REQUESTS_DIR / f"{{request_id}}.tmp"
    response_path = RESPONSES_DIR / f"{{request_id}}.json"
    temp_request_path.write_text(json.dumps(action, ensure_ascii=False), encoding="utf-8")
    temp_request_path.rename(request_path)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        if response_path.exists():
            data = json.loads(response_path.read_text(encoding="utf-8"))
            response_path.unlink(missing_ok=True)
            return data
        time.sleep(0.05)
    raise TimeoutError("timed out waiting for op_bench action response")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
"""


def action_log_count(action_log_path: Path) -> int:
    if not action_log_path.exists():
        return 0
    return sum(
        1
        for line in action_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "tool_call"
    )


def action_log_integrity(action_log_path: Path, initial_digest: str, final_digest: str) -> dict[str, object]:
    errors: list[str] = []
    previous_digest = initial_digest
    if not action_log_path.exists():
        errors.append("action log was not created")
        return {"status": "workspace_changed_outside_actions" if final_digest != initial_digest else "no_actions", "errors": errors}

    for line_number, line in enumerate(action_log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("event") == "session_start":
            logged_initial = entry.get("initial_digest")
            if logged_initial != initial_digest:
                errors.append(f"initial digest mismatch at log line {line_number}")
            previous_digest = str(logged_initial)
            continue
        if entry.get("event") != "tool_call":
            continue
        before_digest = str(entry.get("before_digest"))
        after_digest = str(entry.get("after_digest"))
        if before_digest != previous_digest:
            errors.append(f"workspace changed outside actions before log line {line_number}")
        previous_digest = after_digest

    if final_digest != previous_digest:
        errors.append("workspace changed outside actions after last logged tool call")
    return {"status": "clean" if not errors else "workspace_changed_outside_actions", "errors": errors}


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    args = action.get("args")
    if not isinstance(args, dict):
        return action
    normalized = {key: value for key, value in action.items() if key != "args"}
    for key, value in args.items():
        normalized.setdefault(str(key), value)
    return normalized


def _path_argument(action: dict[str, Any]) -> str:
    value = action.get("path", action.get("file_path"))
    if not isinstance(value, str) or not value:
        raise ValueError("action requires a non-empty 'path'")
    return value


def _test_name_argument(action: dict[str, Any]) -> str:
    value = action.get("test_name", action.get("test", action.get("name")))
    if not isinstance(value, str) or not value:
        raise ValueError("run_test requires a non-empty 'test_name'")
    return value


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dataclass_fields__"):
            return asdict(value)
        return repr(value)


def _action_summary(action: object) -> str:
    if not isinstance(action, dict):
        return "invalid-action"
    action = _normalize_action(action)
    name = str(action.get("action", ""))
    if name in {"read_file", "write_file"}:
        return f"{name} {action.get('path', action.get('file_path', ''))}"
    if name == "apply_patch":
        patch = str(action.get("patch", ""))
        return f"apply_patch chars={len(patch)}"
    if name == "run_test":
        return f"run_test {action.get('test_name', action.get('test', action.get('name', '')))}"
    if name == "run_command":
        command = action.get("command")
        if isinstance(command, list):
            return f"run_command {format_command([str(part) for part in command])}"
        return f"run_command {str(command)[:220]}"
    return name or "unknown-action"


def _result_status(result: dict[str, Any]) -> str:
    if "error" in result:
        return "error"
    if {"exit_code", "timed_out"}.issubset(result):
        suffix = " timeout" if result.get("timed_out") else ""
        return f"exit={result.get('exit_code')}{suffix}"
    return "ok"
