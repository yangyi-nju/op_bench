"""Model transports for the fixed OpBench tool loop.

Clients return decisions; they never receive a task workspace or execute tools.
Codex CLI is a local inference smoke adapter, not a native Agent integration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import http.client
import json
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any
from urllib.parse import urlsplit

from op_bench.io import write_json


class ModelError(RuntimeError):
    """A credential-safe transport/protocol error with a stable reason."""
    def __init__(self, reason: str, *, timeout: bool = False,
                 invalid_output: bool = False, evidence_path: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.timeout = timeout
        self.invalid_output = invalid_output
        self.evidence_path = evidence_path


@dataclass(frozen=True)
class ModelSpec:
    """Recorded model settings, without credential values.

    model_id labels an experiment entry; model selects the provider's model.
    api_key_env names a controller environment variable, not the key itself.
    """
    model_id: str
    backend: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    codex_binary: str = "codex"
    reasoning_effort: str | None = None
    max_completion_tokens: int = 4096
    max_response_bytes: int = 8388608

    def __post_init__(self):
        for key in ("model_id", "model", "codex_binary"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
                raise ValueError(f"{key} must be nonempty text without control characters")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.model_id):
            raise ValueError("model_id must be a filesystem-safe identifier")
        if self.backend not in {"codex_cli", "chat_completions"}:
            raise ValueError("model backend must be codex_cli or chat_completions")
        if self.reasoning_effort is not None and self.reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("unsupported reasoning_effort")
        for key in ("max_completion_tokens", "max_response_bytes"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if self.backend == "chat_completions":
            _endpoint(self.base_url)
            if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)):
                raise ValueError("api_key_env must name an environment variable")
        elif self.base_url is not None or self.api_key_env is not None:
            raise ValueError("codex_cli uses its existing login, not base_url or api_key_env")

    @classmethod
    def from_dict(cls, value: dict) -> "ModelSpec":
        if not isinstance(value, dict):
            raise ValueError("model configuration must be an object")
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("unknown model configuration fields: " + ", ".join(sorted(unknown)))
        try:
            return cls(**value)
        except TypeError as exc:
            raise ValueError("model_id, backend and model are required") from exc

    def to_dict(self) -> dict:
        return asdict(self)


def _endpoint(base_url: str | None):
    if not isinstance(base_url, str):
        raise ValueError("chat_completions requires an explicit base_url")
    parsed = urlsplit(base_url)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("invalid model endpoint port") from None
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or any(ord(c) <= 32 for c in base_url)):
        raise ValueError("base_url must be HTTP(S), without credentials, query or fragment")
    return parsed, port


def _object(text: str | bytes) -> dict:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON number")
    result = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("JSON object required")
    return result


def _usage(raw: Any, *, codex: bool = False) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    keys = {"input_tokens": "input_tokens" if codex else "prompt_tokens",
            "output_tokens": "output_tokens" if codex else "completion_tokens",
            "total_tokens": "total_tokens"}
    result = {key: raw.get(source) if type(raw.get(source)) is int and raw[source] >= 0 else None
              for key, source in keys.items()}
    if result["total_tokens"] is None and result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    result.update({"source": "cli_reported" if codex else "provider_reported", "cost_usd": None})
    return result


class ChatCompletionsClient:
    """Translate a compatible HTTP API into decisions for OpBench's loop."""
    def __init__(self, spec: ModelSpec, output_dir: Path):
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.parsed, self.port = _endpoint(spec.base_url)
        self.api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        if spec.api_key_env and not self.api_key:
            raise ModelError("missing_api_key_environment_variable")
        if any(ord(c) < 32 for c in self.api_key):
            raise ModelError("invalid_api_key")

    def complete(self, messages: list[dict], tools: list[dict], timeout_sec: float) -> dict:
        """Return tool_calls, content, usage and transport metadata for one turn.

        The caller owns conversation history and tool execution. This client
        records API payloads and normalizes responses; it neither edits the
        workspace nor interprets a model's completion as a successful repair.
        """
        if timeout_sec <= 0:
            raise ModelError("model_deadline", timeout=True)
        payload = {"model": self.spec.model, "messages": messages, "tools": tools,
                   "tool_choice": "auto", "parallel_tool_calls": False,
                   "max_completion_tokens": self.spec.max_completion_tokens, "stream": False}
        if self.spec.reasoning_effort is not None:
            payload["reasoning_effort"] = self.spec.reasoning_effort
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.calls += 1
        request_dir = self.output_dir / f"request-{self.calls:04d}"
        request_dir.mkdir(exist_ok=False)
        write_json(request_dir / "request.json", payload)
        connection_type = http.client.HTTPSConnection if self.parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self.parsed.hostname, self.port, timeout=timeout_sec)
        response = None
        deadline = time.monotonic() + timeout_sec
        expired = threading.Event()
        def expire():
            expired.set()
            if connection.sock:
                try:
                    connection.sock.shutdown(2)
                except OSError:
                    pass
        watchdog = threading.Timer(timeout_sec, expire)
        watchdog.daemon = True
        watchdog.start()
        def remaining():
            left = deadline - time.monotonic()
            if expired.is_set() or left <= 0:
                raise ModelError("model_deadline", timeout=True)
            connection.timeout = left
            if connection.sock:
                connection.sock.settimeout(left)
        try:
            headers = {"Content-Type": "application/json", "Accept-Encoding": "identity"}
            if self.api_key:
                headers["Authorization"] = "Bearer " + self.api_key
            # base_url is the API prefix (for example /v1), not a full endpoint.
            connection.request("POST", self.parsed.path.rstrip("/") + "/chat/completions", body=body, headers=headers)
            remaining()
            response = connection.getresponse()
            if response.status != 200:
                raise ModelError(f"model_http_status_{response.status}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json" or response.getheader("Content-Encoding", "identity") != "identity":
                raise ModelError("model_response_type")
            chunks, total = [], 0
            while True:
                remaining()
                chunk = response.read1(min(65536, self.spec.max_response_bytes - total + 1))
                remaining()
                if not chunk:
                    if response.length not in (None, 0):
                        raise ModelError("model_incomplete_response")
                    break
                total += len(chunk)
                if total > self.spec.max_response_bytes:
                    raise ModelError("model_response_size_limit")
                chunks.append(chunk)
            try:
                raw = _object(b"".join(chunks))
                if raw.get("error"):
                    raise ModelError("model_service_error_envelope")
                choices = raw["choices"]
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise ValueError("one choice required")
                choice = choices[0]
                message = choice["message"]
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    raise ValueError("invalid assistant envelope")
                finish_reason = choice.get("finish_reason")
                if finish_reason not in {"stop", "tool_calls", "length", "content_filter"}:
                    raise ValueError("unknown finish reason")
            except (KeyError, TypeError, ValueError, UnicodeError, RecursionError):
                raise ModelError("model_invalid_response_envelope") from None
            # Keep the model's public response for protocol-failure review. No
            # headers, authentication values or provider error bodies are saved.
            evidence = request_dir / "response.json"
            write_json(evidence, {key: raw[key] for key in ("id", "model", "choices", "usage") if key in raw})
            if finish_reason in {"length", "content_filter"}:
                raise ModelError("model_completion_" + finish_reason, invalid_output=True,
                                 evidence_path=str(evidence))
            if message.get("refusal"):
                raise ModelError("model_refusal", invalid_output=True, evidence_path=str(evidence))
            try:
                content = message.get("content")
                if content is None:
                    content = ""
                raw_calls = message.get("tool_calls", [])
                if not isinstance(content, str) or not isinstance(raw_calls, list):
                    raise ValueError("invalid assistant content envelope")
                calls = []
                for call in raw_calls:
                    if (not isinstance(call, dict) or call.get("type") != "function"
                            or not isinstance(call.get("id"), str) or not isinstance(call.get("function"), dict)
                            or not isinstance(call["function"].get("name"), str)
                            or not isinstance(call["function"].get("arguments"), str)):
                        raise ValueError("invalid function call envelope")
                    try:
                        arguments = _object(call["function"]["arguments"])
                    except (ValueError, UnicodeError, RecursionError):
                        raise ModelError("model_invalid_tool_arguments", invalid_output=True,
                                         evidence_path=str(evidence)) from None
                    calls.append({"id": call["id"], "name": call["function"]["name"], "arguments": arguments})
            except (KeyError, TypeError, ValueError, RecursionError):
                raise ModelError("model_invalid_response_envelope", evidence_path=str(evidence)) from None
            return {"tool_calls": calls, "content": content, "usage": _usage(raw.get("usage")),
                    "backend_metadata": {"backend": "chat_completions", "model": self.spec.model,
                                         "returned_model": raw.get("model"), "response_id": raw.get("id"),
                                         "request_path": str(request_dir)}}
        except ModelError:
            raise
        except (OSError, TimeoutError, http.client.HTTPException):
            raise ModelError("model_deadline" if expired.is_set() or time.monotonic() >= deadline else "model_transport_error",
                             timeout=expired.is_set() or time.monotonic() >= deadline) from None
        finally:
            watchdog.cancel()
            if response is not None:
                response.close()
            connection.close()


CODEX_INSTRUCTIONS = """You are an inference component in OpBench's fixed tool loop.
Return exactly one JSON decision matching the provided schema. Do not execute any
native Codex tools, inspect the local machine, browse, or delegate. The listed
OpBench tools are descriptions of actions the outer controller may perform after
your response; express those actions only inside the tool_calls JSON array.
Use only the public task and prior tool results in the supplied conversation.
Do not invent tool results. Use finish when the repair is ready.
"""


def _strict_schema(schema: dict) -> dict:
    result = copy.deepcopy(schema)
    if result.get("type") == "object":
        properties = result.get("properties", {})
        required = set(result.get("required", []))
        result["properties"] = {
            key: _strict_schema(value) if key in required else {"anyOf": [_strict_schema(value), {"type": "null"}]}
            for key, value in properties.items()}
        result["required"] = list(properties)
        result["additionalProperties"] = False
    elif result.get("type") == "array":
        result["items"] = _strict_schema(result["items"])
    return result


def _decision_schema(tools: list[dict]) -> dict:
    variants = []
    for tool in tools:
        function = tool["function"]
        variants.append({"type": "object", "properties": {
            "id": {"type": "string"}, "name": {"type": "string", "enum": [function["name"]]},
            "arguments": _strict_schema(function["parameters"])},
            "required": ["id", "name", "arguments"], "additionalProperties": False})
    return {"type": "object", "properties": {"tool_calls": {"type": "array", "items": {"anyOf": variants}},
            "content": {"type": "string"}}, "required": ["tool_calls", "content"], "additionalProperties": False}


def _kill(process: subprocess.Popen):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class CodexCLIClient:
    """One ephemeral CLI turn per decision, with no OpBench workspace access API."""
    def __init__(self, spec: ModelSpec, output_dir: Path):
        self.spec = spec
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.codex_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).resolve()
        try:
            cache = _object((self.codex_home / "models_cache.json").read_text())
            entries = [entry for entry in cache["models"] if entry.get("slug") == spec.model]
            if len(entries) != 1:
                raise ValueError("selected model not in local catalog")
            self.model_catalog = copy.deepcopy(entries[0])
        except (OSError, ValueError, KeyError, TypeError):
            raise ModelError("codex_selected_model_missing_from_local_catalog") from None
        self.model_catalog.update({"shell_type": "disabled", "apply_patch_tool_type": None,
                                   "experimental_supported_tools": [], "supports_search_tool": False,
                                   "include_skills_usage_instructions": False,
                                   "include_plugin_usage_instructions": False,
                                   "include_apps_usage_instructions": False, "node_repl_disabled": True})

    def _command(self, scratch: Path) -> list[str]:
        args = [self.spec.codex_binary, "-a", "never", "exec", "--ephemeral", "--ignore-user-config",
                "--ignore-rules", "--skip-git-repo-check", "--sandbox", "read-only", "--json",
                "--color", "never", "--model", self.spec.model, "--cd", str(scratch),
                "--output-schema", str(scratch / "decision-schema.json"),
                "--output-last-message", str(scratch / "decision.json")]
        # Built-in provider entries are not generally overridable in Codex.
        # A local alias preserves OpenAI's normal auth/endpoint selection while
        # selecting HTTPS immediately instead of paying WebSocket retries on
        # every independent inference decision. No credential is copied here.
        config = {"model_provider": "opbench_openai_https",
                  "model_providers.opbench_openai_https.name": "OpenAI",
                  "model_providers.opbench_openai_https.wire_api": "responses",
                  "model_providers.opbench_openai_https.requires_openai_auth": True,
                  "model_providers.opbench_openai_https.supports_websockets": False,
                  "model_providers.opbench_openai_https.request_max_retries": 0,
                  "model_providers.opbench_openai_https.stream_max_retries": 0,
                  "model_catalog_json": str(scratch / "model-catalog.json"),
                  "model_instructions_file": str(scratch / "instructions.txt"),
                  "developer_instructions": CODEX_INSTRUCTIONS,
                  "web_search": "disabled", "project_doc_max_bytes": 0,
                  "include_environment_context": False, "include_apps_instructions": False,
                  "include_collaboration_mode_instructions": False,
                  "skills.include_instructions": False, "skills.bundled.enabled": False,
                  "tools.update_plan.enabled": False, "tools.experimental_request_user_input.enabled": False,
                  "shell_environment_policy.inherit": "none", "shell_environment_policy.experimental_use_profile": False}
        for feature in ("shell_tool", "unified_exec", "shell_snapshot", "apps", "plugins", "remote_plugin",
                        "browser_use", "browser_use_external", "in_app_browser", "computer_use", "image_generation",
                        "view_image", "memories", "skill_search", "tool_suggest", "multi_agent", "multi_agent_v2",
                        "hooks", "code_mode", "code_mode_host", "code_mode_only", "sleep_tool", "goals",
                        "request_permissions_tool", "workspace_dependencies", "unbounded_connection_retries"):
            config["features." + feature] = False
        config["features.skip_host_skill_discovery"] = True
        if self.spec.reasoning_effort is not None:
            config["model_reasoning_effort"] = self.spec.reasoning_effort
        for key, value in config.items():
            args.extend(["-c", key + "=" + json.dumps(value)])
        return args + ["-"]

    def complete(self, messages: list[dict], tools: list[dict], timeout_sec: float) -> dict:
        if timeout_sec <= 0:
            raise ModelError("model_deadline", timeout=True)
        self.calls += 1
        request_dir = self.output_dir / f"request-{self.calls:04d}"
        request_dir.mkdir(exist_ok=False)
        prompt = json.dumps({"conversation": messages, "available_opbench_tools": tools}, ensure_ascii=False, allow_nan=False)
        deadline = time.monotonic() + timeout_sec
        retained, completed = 0, False
        process = None
        usage = None
        with tempfile.TemporaryDirectory(prefix="opbench-inference-") as temporary:
            scratch = Path(temporary)
            write_json(scratch / "decision-schema.json", _decision_schema(tools))
            write_json(scratch / "model-catalog.json", {"models": [self.model_catalog]})
            (scratch / "instructions.txt").write_text(CODEX_INSTRUCTIONS)
            command = self._command(scratch)
            # Preserve the normal login store; never copy credentials into artifacts.
            allowed_env = {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR", "CURL_CA_BUNDLE"}
            env = {key: value for key, value in os.environ.items() if key in allowed_env}
            env["CODEX_HOME"] = str(self.codex_home)
            write_json(request_dir / "request.json", {"model": self.spec.model, "backend": "codex_cli",
                       "reasoning_effort": self.spec.reasoning_effort, "mode": "inference_smoke",
                       "transport": "https", "provider_alias": "opbench_openai_https",
                       "native_tools": "disabled_and_event_checked", "conversation": messages, "tools": tools})
            try:
                with (scratch / "prompt.stdin").open("w+", encoding="utf-8") as stdin, (request_dir / "events.jsonl").open("wb") as events, (request_dir / "stderr.log").open("wb") as errors:
                    stdin.write(prompt)
                    stdin.seek(0)
                    process = subprocess.Popen(command, cwd=scratch, env=env, stdin=stdin,
                                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
                    assert process.stdout is not None and process.stderr is not None
                    pending = bytearray()
                    def event_line(line):
                        nonlocal usage, completed
                        try:
                            row = _object(line)
                        except (ValueError, UnicodeError):
                            raise ModelError("codex_invalid_event") from None
                        event_type = row.get("type")
                        if event_type in {"item.started", "item.updated", "item.completed"}:
                            item = row.get("item")
                            if not isinstance(item, dict) or item.get("type") not in {"agent_message", "reasoning", "error"}:
                                raise ModelError("codex_native_tool_attempt")
                        elif event_type == "turn.completed":
                            completed = True
                            usage = row.get("usage")
                        elif event_type == "turn.failed":
                            raise ModelError("codex_turn_failed")
                        elif event_type not in {"thread.started", "turn.started", "error"}:
                            raise ModelError("codex_unknown_event")
                    with selectors.DefaultSelector() as selector:
                        for stream in (process.stdout, process.stderr):
                            os.set_blocking(stream.fileno(), False)
                            selector.register(stream, selectors.EVENT_READ)
                        while selector.get_map():
                            left = deadline - time.monotonic()
                            if left <= 0:
                                raise ModelError("model_deadline", timeout=True)
                            for key, _ in selector.select(min(left, 0.05)):
                                chunk = os.read(key.fd, 65536)
                                if not chunk:
                                    selector.unregister(key.fileobj)
                                    continue
                                retained += len(chunk)
                                if retained > self.spec.max_response_bytes:
                                    raise ModelError("codex_response_size_limit")
                                if key.fileobj is process.stderr:
                                    errors.write(chunk)
                                else:
                                    events.write(chunk)
                                    events.flush()
                                    pending.extend(chunk)
                                    while b"\n" in pending:
                                        line, _, tail = pending.partition(b"\n")
                                        pending = bytearray(tail)
                                        if line.strip():
                                            event_line(line)
                    if pending.strip():
                        event_line(pending)
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                    if process.returncode:
                        raise ModelError("codex_nonzero_exit")
                if not completed:
                    raise ModelError("codex_missing_completed_turn")
                result_path = scratch / "decision.json"
                if result_path.stat().st_size > self.spec.max_response_bytes:
                    raise ModelError("codex_response_size_limit")
                raw_decision = result_path.read_bytes()
                evidence = request_dir / "decision.raw.txt"
                evidence.write_bytes(raw_decision)
                try:
                    decision = _object(raw_decision)
                    if (set(decision) != {"tool_calls", "content"}
                            or not isinstance(decision["tool_calls"], list) or not isinstance(decision["content"], str)):
                        raise ValueError("invalid decision shape")
                    definitions = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
                    for call in decision["tool_calls"]:
                        if (not isinstance(call, dict) or set(call) != {"id", "name", "arguments"}
                                or not isinstance(call["arguments"], dict) or not isinstance(call["name"], str)):
                            raise ValueError("invalid tool call shape")
                        schema = definitions.get(call["name"], {})
                        call["arguments"] = {key: value for key, value in call["arguments"].items()
                                             if value is not None or key in schema.get("required", [])}
                except (ValueError, UnicodeError, TypeError, KeyError, RecursionError):
                    raise ModelError("codex_invalid_decision", invalid_output=True,
                                     evidence_path=str(evidence)) from None
                decision.update({"usage": _usage(usage, codex=True), "backend_metadata": {
                    "backend": "codex_cli", "model": self.spec.model, "mode": "inference_smoke",
                    "transport": "https", "provider_alias": "opbench_openai_https",
                    "native_tool_events": 0, "request_path": str(request_dir),
                    "max_completion_tokens_enforcement": "not_supported_by_codex_cli"}})
                write_json(request_dir / "response.json", decision)
                return decision
            except ModelError:
                raise
            except subprocess.TimeoutExpired:
                raise ModelError("model_deadline", timeout=True) from None
            except (OSError, ValueError, TypeError, KeyError, RecursionError):
                raise ModelError("codex_start_or_response_error") from None
            finally:
                if process is not None:
                    _kill(process)
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    for stream in (process.stdout, process.stderr):
                        if stream is not None:
                            stream.close()


def create_client(spec: ModelSpec, output_dir: str | Path):
    return CodexCLIClient(spec, Path(output_dir)) if spec.backend == "codex_cli" else ChatCompletionsClient(spec, Path(output_dir))
