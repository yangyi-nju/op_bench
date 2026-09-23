"""One controlled tool implementation shared by the fixed harness and MCP."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


TOOL_VERSION = "controlled-tools-2"


def _definition(name, description, properties=None, required=()):
    fields = {key: {"maxLength": 1000000, **field} if field["type"] == "string" else field
              for key, field in (properties or {}).items()}
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": fields,
                       "required": list(required), "additionalProperties": False}}}


_PATH = {"type": "string", "description": "Relative workspace path."}
DEFINITIONS = [
    _definition("list_files", "List public workspace files; paginate with next_offset.",
                {"path": _PATH, "offset": {"type": "integer", "minimum": 0},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 200}}),
    _definition("read_file", "Read a UTF-8 source file with line numbers.",
                {"path": _PATH, "start_line": {"type": "integer", "minimum": 1},
                 "max_lines": {"type": "integer", "minimum": 1, "maximum": 500}}, ["path"]),
    _definition("search", "Search literal text in public UTF-8 workspace files.",
                {"path": _PATH, "query": {"type": "string", "minLength": 1},
                 "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ["query"]),
    _definition("write_file", "Create or replace a UTF-8 source or public test file.",
                {"path": _PATH, "content": {"type": "string"}}, ["path", "content"]),
    _definition("replace_text", "Replace exactly one literal match in a UTF-8 file.",
                {"path": _PATH, "old_text": {"type": "string", "minLength": 1},
                 "new_text": {"type": "string"}}, ["path", "old_text", "new_text"]),
    _definition("delete_file", "Delete a source file.", {"path": _PATH}, ["path"]),
    _definition("build", "Run the task's declared build commands in the workspace."),
    _definition("run_public_tests", "Run the public task checks. These are not final grading."),
    _definition("diff", "Show current source changes and untracked file names."),
    _definition("finish", "Finish solving and submit current workspace changes for independent grading.",
                {"summary": {"type": "string"}}, ["summary"]),
]
PARAMETERS = {item["function"]["name"]: item["function"]["parameters"] for item in DEFINITIONS}


def validate_arguments(arguments, schema):
    """Validate the flat string/integer arguments used by the fixed tools."""
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    properties = schema["properties"]
    if set(arguments) - properties.keys() or set(schema["required"]) - arguments.keys():
        raise ValueError("unexpected or missing tool argument")
    for key, value in arguments.items():
        field = properties[key]
        if field["type"] == "integer":
            if type(value) is not int or value < field.get("minimum", 0) or value > field.get("maximum", 10000000):
                raise ValueError(f"invalid integer argument: {key}")
        elif field["type"] == "string":
            if not isinstance(value, str) or "\0" in value or len(value) < field.get("minLength", 0) or len(value) > field.get("maxLength", 1000000):
                raise ValueError(f"invalid text argument: {key}")
        else:
            raise ValueError(f"unsupported tool argument type: {key}")


class ToolExecutor:
    """Receives a session capability and public commands, never a grader path."""
    definitions = DEFINITIONS

    def __init__(self, session, output_dir: Path, *, build_commands=(), public_commands=(),
                 reserved=(".opbench",), per_tool_timeout_sec=120):
        if session.grader_dir is not None:
            raise ValueError("Solving tools cannot receive a grading session")
        self.session, self.output = session, Path(output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.commands = {"build": build_commands, "run_public_tests": public_commands}
        if any("{grader}" in arg for commands in self.commands.values() for cmd in commands for arg in cmd):
            raise ValueError("Public tools cannot invoke private graders")
        self.reserved = tuple(reserved)
        self.per_tool_timeout_sec = per_tool_timeout_sec
        self.finished = False
        self.sequence = 0

    def call(self, name, arguments, timeout_sec):
        self.sequence += 1
        try:
            if not isinstance(name, str) or name not in PARAMETERS:
                raise ValueError("unknown tool")
            validate_arguments(arguments, PARAMETERS[name])
            if self.finished:
                raise ValueError("attempt already finished")
            if not math.isfinite(timeout_sec) or timeout_sec <= 0:
                raise ValueError("tool budget exhausted")
        except (TypeError, ValueError) as exc:
            return {"error": str(exc), "is_error": True}
        if name == "finish":
            self.finished = True
            return {"submitted": True, "summary": arguments["summary"]}
        deadline = time.monotonic() + min(timeout_sec, self.per_tool_timeout_sec)
        if name in self.commands:
            results = []
            for index, command in enumerate(self.commands[name]):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"results": results, "timed_out": True, "is_error": True}
                log = self.output / f"tool-{self.sequence:04d}-{index:03d}.log"
                outcome = self.session.run(command, remaining, log, max_output_bytes=16 * 1024 * 1024)
                text = log.read_text(errors="replace")
                feedback = text if len(text) <= 32768 else text[:8192] + "\n[... log truncated for model ...]\n" + text[-24576:]
                results.append({**outcome.to_dict(), "output": feedback,
                                "feedback_truncated": len(text) > 32768})
                if outcome.exit_code != 0 or outcome.timed_out or outcome.output_limited:
                    break
            return {"results": results, "no_commands": not self.commands[name],
                    "session_closed": self.session.closed,
                    "timed_out": any(row["timed_out"] for row in results)}
        log = self.output / f"tool-{self.sequence:04d}.log"
        helper = Path(__file__).with_name("_workspace_tools.py").read_text()
        workspace = "/workspace" if self.session.info["backend"] == "docker" else str(self.session.workspace)
        request = json.dumps({"workspace": workspace, "name": name, "arguments": arguments,
                              "reserved": self.reserved}, ensure_ascii=True)
        observation = self.output / f"tool-{self.sequence:04d}.json"
        outcome = self.session.observe(["{python}", "-I", "-c", helper],
            max(0.001, deadline - time.monotonic()), request.encode(), observation, log,
            max_output_bytes=65536, max_log_bytes=65536)
        if outcome.exit_code != 0 or outcome.timed_out or outcome.output_limited:
            return {**outcome.to_dict(), "error": "workspace tool did not complete", "is_error": True,
                    "session_closed": self.session.closed}
        try:
            value = json.loads(observation.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected object")
            if value.get("error"):
                value["is_error"] = True
            return value
        except (ValueError, OSError):
            return {"error": "invalid workspace tool response", "is_error": True}
